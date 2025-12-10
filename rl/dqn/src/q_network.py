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
    Simple MLP Q-network for CartPole.

    Input: state vector of shape (batch_size, obs_dim)
    Output: Q-values of shape (batch_size, action_dim)
    """

    def __init__(self, cfg: QNetworkConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, obs_dim)
        returns: (batch_size, action_dim)
        """
        return self.net(x)
