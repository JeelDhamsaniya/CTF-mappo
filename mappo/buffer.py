import torch
import numpy as np


class RolloutBuffer:
    """
    Stores trajectory data for one full episode for a single agent.

    Fields stored per time step:
        obs       : agent's flattened observation
        actions   : integer action taken
        log_probs : log probability of the action under the policy at collection time
        rewards   : scalar reward received
        values    : critic's value estimate V(s) at this step
        dones     : whether this step was terminal
    """

    def __init__(self):
        self.obs: list = []
        self.actions: list = []
        self.log_probs: list = []
        self.rewards: list = []
        self.values: list = []
        self.dones: list = []

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob,
        reward: float,
        value,
        done: bool,
    ):
        """Append one transition to all lists."""
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float,
        gae_lambda: float,
    ):
        """
        Compute Generalized Advantage Estimation (GAE) returns.

        For each timestep t (iterating backwards):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            gae_t   = delta_t + gamma * gae_lambda * (1 - done_t) * gae_{t+1}

        Returns:
            advantages : Tensor of shape (T,) — normalized advantages
            returns    : Tensor of shape (T,) — targets for the value network
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        # Convert stored values to numpy scalars
        values_np = np.array(
            [v.item() if torch.is_tensor(v) else float(v) for v in self.values],
            dtype=np.float32,
        )
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values_np[t + 1]
            # Terminal states must not bootstrap from next state
            mask = 1.0 - dones_np[t]
            delta = rewards_np[t] + gamma * next_val * mask - values_np[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        values_tensor = torch.tensor(values_np, dtype=torch.float32)
        returns_tensor = advantages_tensor + values_tensor

        # Normalize advantages
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std() + 1e-8
        advantages_tensor = (advantages_tensor - adv_mean) / adv_std

        return advantages_tensor, returns_tensor

    def clear(self):
        """Reset all stored data."""
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
