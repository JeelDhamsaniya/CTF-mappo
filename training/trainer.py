import os
import time
from typing import Dict

import numpy as np

from env.ctf_env import CaptureTheFlagEnv
from mappo.agent import MAPPOAgent
from mappo.buffer import RolloutBuffer
from utils.plotter import plot_training_results


class MAPPOTrainer:
    """
    Orchestrates training of two MAPPO teams on the CTF environment.
    """

    def __init__(self, env: CaptureTheFlagEnv, config: dict, device):
        self.env    = env
        self.config = config
        self.device = device

        obs_shape   = (env.OBS_SIZE,)          # 104 after env update
        action_size = env.action_space.n        # 5

        self.team1_agent = MAPPOAgent(
            agent_ids=env.team1_ids, obs_shape=obs_shape,
            action_size=action_size,
            lr_actor=config["lr_actor"], lr_critic=config["lr_critic"],
        ).to(device)

        self.team2_agent = MAPPOAgent(
            agent_ids=env.team2_ids, obs_shape=obs_shape,
            action_size=action_size,
            lr_actor=config["lr_actor"], lr_critic=config["lr_critic"],
        ).to(device)

        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(config.get("log_dir", "logs"), exist_ok=True)

    # ── Episode collection ────────────────────────────────────────────────────

    def run_episode(self) -> Dict:
        obs_dict  = self.env.reset()
        max_steps = self.config["max_steps"]
        buffers   = {aid: RolloutBuffer() for aid in self.env.agent_ids}
        total_rew = {aid: 0.0 for aid in self.env.agent_ids}
        ep_len    = 0

        for _ in range(max_steps):
            actions, log_probs, values_ = {}, {}, {}

            for aid in self.env.team1_ids:
                a, lp, v = self.team1_agent.select_action(aid, obs_dict[aid])
                actions[aid], log_probs[aid], values_[aid] = a, lp, v

            for aid in self.env.team2_ids:
                a, lp, v = self.team2_agent.select_action(aid, obs_dict[aid])
                actions[aid], log_probs[aid], values_[aid] = a, lp, v

            next_obs, rewards, dones, _ = self.env.step(actions)
            done   = dones["__all__"]
            ep_len += 1

            for aid in self.env.agent_ids:
                buffers[aid].store(
                    obs=obs_dict[aid], action=actions[aid],
                    log_prob=log_probs[aid], reward=rewards[aid],
                    value=values_[aid], done=float(done),
                )
                total_rew[aid] += rewards[aid]

            obs_dict = next_obs
            if done:
                break

        scores = self.env.scores
        if scores["team1"] > scores["team2"]:
            winner = "team_1"
        elif scores["team2"] > scores["team1"]:
            winner = "team_2"
        else:
            winner = "draw"

        self._last_buffers = buffers
        return {
            "team_1_total_reward": sum(total_rew[a] for a in self.env.team1_ids),
            "team_2_total_reward": sum(total_rew[a] for a in self.env.team2_ids),
            "episode_length":      ep_len,
            "winner":              winner,
        }

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, num_episodes: int):
        log_every    = self.config.get("log_every",    10)
        save_every   = self.config.get("save_every",   500)
        gamma        = self.config["gamma"]
        gae_lambda   = self.config["gae_lambda"]
        clip_ratio   = self.config["clip_ratio"]
        value_coef   = self.config["value_coef"]
        entropy_coef = self.config["entropy_coef"]
        ppo_epochs   = self.config["ppo_epochs"]

        t1_rewards, t2_rewards   = [], []
        t1_wins_hist, t2_wins_hist = [], []
        pl_hist, vl_hist         = [], []
        recent_winners           = []

        print("\n" + "=" * 70)
        print("  MAPPO CTF — Training")
        print(f"  Episodes: {num_episodes}  |  Max steps/ep: {self.config['max_steps']}")
        print(f"  Entropy coef: {entropy_coef}  |  Progress reward: {self.config.get('progress_reward_coef', 1.0)}")
        print("=" * 70 + "\n")

        start = time.time()

        for ep in range(1, num_episodes + 1):
            result  = self.run_episode()
            buffers = self._last_buffers

            t1_rewards.append(result["team_1_total_reward"])
            t2_rewards.append(result["team_2_total_reward"])

            recent_winners.append(result["winner"])
            if len(recent_winners) > 100:
                recent_winners.pop(0)

            t1_wr = sum(1 for w in recent_winners if w == "team_1") / len(recent_winners) * 100
            t2_wr = sum(1 for w in recent_winners if w == "team_2") / len(recent_winners) * 100
            t1_wins_hist.append(t1_wr)
            t2_wins_hist.append(t2_wr)

            # PPO updates
            pl1, vl1 = self.team1_agent.update(
                {a: buffers[a] for a in self.env.team1_ids},
                clip_ratio=clip_ratio, value_coef=value_coef,
                entropy_coef=entropy_coef, ppo_epochs=ppo_epochs,
                gamma=gamma, gae_lambda=gae_lambda,
            )
            pl2, vl2 = self.team2_agent.update(
                {a: buffers[a] for a in self.env.team2_ids},
                clip_ratio=clip_ratio, value_coef=value_coef,
                entropy_coef=entropy_coef, ppo_epochs=ppo_epochs,
                gamma=gamma, gae_lambda=gae_lambda,
            )
            avg_pl = (pl1 + pl2) / 2.0
            avg_vl = (vl1 + vl2) / 2.0
            pl_hist.append(avg_pl)
            vl_hist.append(avg_vl)

            if ep % log_every == 0:
                elapsed  = time.time() - start
                eta      = elapsed / ep * (num_episodes - ep)
                eta_str  = f"{int(eta//60):02d}m{int(eta%60):02d}s"
                winner_s = {"team_1": "T1 ✓", "team_2": "T2 ✓", "draw": "draw"}[result["winner"]]
                print(
                    f"Ep {ep:5d}/{num_episodes} | {winner_s:5} | "
                    f"T1 Rwd:{result['team_1_total_reward']:8.1f}  "
                    f"T2 Rwd:{result['team_2_total_reward']:8.1f} | "
                    f"WR T1:{t1_wr:5.1f}% T2:{t2_wr:5.1f}% | "
                    f"Len:{result['episode_length']:4d} | "
                    f"PLoss:{avg_pl:7.4f} VLoss:{avg_vl:8.2f} | ETA:{eta_str}"
                )

            if ep % save_every == 0:
                self.team1_agent.save(os.path.join(self.checkpoint_dir, f"team1_ep{ep}.pt"))
                self.team2_agent.save(os.path.join(self.checkpoint_dir, f"team2_ep{ep}.pt"))
                print(f"  [Checkpoint saved @ ep {ep}]")

        # Final save
        self.team1_agent.save(os.path.join(self.checkpoint_dir, "team1_final.pt"))
        self.team2_agent.save(os.path.join(self.checkpoint_dir, "team2_final.pt"))
        print("\n[Training Complete] Final models saved.")

        # Plot
        log_dir = self.config.get("log_dir", "logs")
        plot_training_results(
            team1_rewards=t1_rewards, team2_rewards=t2_rewards,
            team1_wins=t1_wins_hist, team2_wins=t2_wins_hist,
            policy_losses=pl_hist, value_losses=vl_hist,
            save_path=os.path.join(log_dir, "training_results.png"),
        )
        print(f"[Plot] Saved → {log_dir}/training_results.png")
