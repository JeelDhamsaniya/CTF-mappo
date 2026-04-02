# mappo/actor.py
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    """256→256 residual block with LayerNorm."""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))   # skip connection


class ActorNetwork(nn.Module):
    """
    Improved decentralized actor.

    Changes vs. original:
      • LayerNorm after every Linear (stabilises training, especially early on)
      • One ResidualBlock (256→256) for richer feature extraction without
        adding many parameters
      • Wider first layer kept (256) so the residual skip has matching dims
    """

    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.res = ResidualBlock(256)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.trunk(obs)
        x = self.res(x)
        return self.head(x)

    def get_action(self, obs: torch.Tensor):
        logits   = self.forward(obs)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action.item(), log_prob, entropy