import sys
from pathlib import Path

# Ensure lenet-like project root (vision/resnet) is on path
RESNET_ROOT = Path(__file__).resolve().parents[1]
if str(RESNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESNET_ROOT))

import torch
from src.model import resnet18


def test_forward_shape():
    model = resnet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
