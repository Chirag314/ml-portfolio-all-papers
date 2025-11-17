import sys
from pathlib import Path

# Ensure the lenet folder (which contains src/) is on sys.path
LENET_ROOT = Path(__file__).resolve().parents[1]
if str(LENET_ROOT) not in sys.path:
    sys.path.insert(0, str(LENET_ROOT))

import torch
from src.model import LeNet5


def test_forward_shape():
    model = LeNet5()
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)
