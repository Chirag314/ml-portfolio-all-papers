from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class QNetworkConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 128


class QNetwork(nn.Module):
    """
    Simple MLP Q-network for CartPole inspired by DQN paper artichitecture"""

    def __init__(self, cfg: QNetworkConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def fprward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
