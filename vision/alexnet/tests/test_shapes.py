import sys
from pathlib import Path

ALEXNET_ROOT = Path(__file__).resolve().parents[1]
if str(ALEXNET_ROOT) not in sys.path:
    sys.path.insert(0, str(ALEXNET_ROOT))

import torch
from src.model import AlexNet  # noqa: E402


def test_forward_shape():
    model = AlexNet(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
