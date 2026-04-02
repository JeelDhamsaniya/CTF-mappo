# mappo/critic.py
import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    Centralized critic for MAPPO.

    Improvement vs. original:
      • Accepts the *concatenated* observations of both teammates
        (global_obs_size = 2 × 102 = 204) — this is what "centralized"
        actually means in CTDE.
      • LayerNorm throughout for training stability.
      • Slightly deeper: 256 → 256 → 128 → 1.
    """

    def __init__(self, global_obs_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_obs_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_obs: shape (..., global_obs_size)  — concat of both agents' obs
        Returns:
            value: shape (..., 1)
        """
        return self.network(global_obs)