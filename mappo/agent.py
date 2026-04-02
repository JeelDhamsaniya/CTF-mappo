# mappo/agent.py
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mappo.actor import ActorNetwork
from mappo.critic import CriticNetwork
from mappo.buffer import RolloutBuffer


class MAPPOAgent:
    """
    Manages one team of 2 agents using MAPPO.
    One ActorNetwork per agent (decentralized execution).
    One shared CriticNetwork per team (centralized training, joint obs input).
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_shape: Tuple[int, ...],
        action_size: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
    ):
        self.agent_ids   = agent_ids
        obs_size         = int(np.prod(obs_shape))
        self.obs_size    = obs_size
        self.action_size = action_size
        self.device      = torch.device("cpu")

        self.actors: Dict[str, ActorNetwork] = {
            aid: ActorNetwork(obs_size, action_size) for aid in agent_ids
        }
        # Centralized critic: sees both agents' obs concatenated
        self.critic = CriticNetwork(obs_size * len(agent_ids))

        self.actor_optimizers: Dict[str, torch.optim.Adam] = {
            aid: torch.optim.Adam(self.actors[aid].parameters(), lr=lr_actor)
            for aid in agent_ids
        }
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

    def to(self, device: torch.device):
        self.device = device
        for actor in self.actors.values():
            actor.to(device)
        self.critic.to(device)
        return self

    # ── Joint obs helper ──────────────────────────────────────────────────────

    def _joint_obs(self, obs_dict: dict) -> torch.Tensor:
        """Concatenate all agents' observations into one tensor for the critic."""
        parts = [
            torch.tensor(obs_dict[aid], dtype=torch.float32)
            for aid in self.agent_ids
        ]
        return torch.cat(parts, dim=-1).unsqueeze(0).to(self.device)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(
        self, agent_id: str, obs: np.ndarray, all_obs_dict: dict = None
    ) -> Tuple[int, float, float]:
        """Stochastic action — used during training rollouts."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _ = self.actors[agent_id].get_action(obs_t)
            if all_obs_dict is not None:
                joint = self._joint_obs(all_obs_dict)
                value = self.critic(joint).squeeze().item()
            else:
                value = 0.0
        lp = log_prob.item() if torch.is_tensor(log_prob) else float(log_prob)
        return action, lp, value

    def select_action_greedy(
        self, agent_id: str, obs: np.ndarray, temperature: float = 0.1
    ) -> int:
        """Near-greedy action via Boltzmann with low temperature."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actors[agent_id](obs_t) / temperature
        action = Categorical(logits=logits).sample().item()
        return action

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(
        self,
        buffers: Dict[str, RolloutBuffer],
        clip_ratio: float   = 0.2,
        value_coef: float   = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int     = 4,
        gamma: float        = 0.99,
        gae_lambda: float   = 0.95,
    ) -> Tuple[float, float]:
        total_pl, total_vl, count = 0.0, 0.0, 0

        # Build joint obs tensor for the critic: shape (T, obs_size * num_agents)
        # Stack each agent's obs trajectory side-by-side
        all_obs_np = {
            aid: np.array(buffers[aid].obs) for aid in self.agent_ids
        }
        joint_obs_t = torch.tensor(
            np.concatenate([all_obs_np[aid] for aid in self.agent_ids], axis=-1),
            dtype=torch.float32,
        ).to(self.device)   # shape (T, 204)

        # Pre-compute GAE for every agent
        agent_data: Dict[str, dict] = {}
        for aid in self.agent_ids:
            buf = buffers[aid]
            advantages, returns = buf.compute_returns_and_advantages(
                next_value=0.0, gamma=gamma, gae_lambda=gae_lambda
            )
            obs_t     = torch.tensor(all_obs_np[aid], dtype=torch.float32).to(self.device)
            actions_t = torch.tensor(buf.actions, dtype=torch.long).to(self.device)
            old_lp_t  = torch.tensor(
                [lp.item() if torch.is_tensor(lp) else float(lp) for lp in buf.log_probs],
                dtype=torch.float32,
            ).to(self.device)
            agent_data[aid] = dict(
                obs=obs_t,
                actions=actions_t,
                old_lp=old_lp_t,
                adv=advantages.to(self.device),
                ret=returns.to(self.device),
            )

        for _ in range(ppo_epochs):
            for aid in self.agent_ids:
                d = agent_data[aid]

                # Actor loss
                logits  = self.actors[aid](d["obs"])
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(d["actions"])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - d["old_lp"])
                surr1 = ratio * d["adv"]
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * d["adv"]
                pl    = -torch.min(surr1, surr2).mean()

                # Centralized critic loss (shared joint obs)
                val_pred = self.critic(joint_obs_t).squeeze(-1)
                vl       = nn.functional.mse_loss(val_pred, d["ret"])

                loss = pl + value_coef * vl - entropy_coef * entropy

                self.actor_optimizers[aid].zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actors[aid].parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizers[aid].step()
                self.critic_optimizer.step()

                total_pl += pl.item()
                total_vl += vl.item()
                count    += 1

        for buf in buffers.values():
            buf.clear()

        n = max(count, 1)
        return total_pl / n, total_vl / n

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "actors": {aid: self.actors[aid].state_dict() for aid in self.agent_ids},
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for aid in self.agent_ids:
            self.actors[aid].load_state_dict(ckpt["actors"][aid])
        self.critic.load_state_dict(ckpt["critic"])