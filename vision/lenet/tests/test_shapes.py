import torch

from src.model import LeNet5


def test_forward_shape():
    model = LeNet5()
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)
