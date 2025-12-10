import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.q_network import QNetwork, QNetworkConfig  # noqa: E402


def test_q_network_output_shape():
    obs_dim = 4
    action_dim = 2
    cfg = QNetworkConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)
    net = QNetwork(cfg)

    x = torch.randn(5, obs_dim)
    q = net(x)
    assert q.shape == (5, action_dim)
