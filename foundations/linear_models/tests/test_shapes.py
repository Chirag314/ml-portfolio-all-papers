import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.models import LinearRegression, LogisticRegression  # noqa: E402


def test_linear_regression_shape():
    model = LinearRegression(n_features=5)
    x = torch.randn(2, 5)
    y = model(x)
    assert y.shape == (2,)


def test_logistic_regression_shape():
    model = LogisticRegression(n_features=2)
    x = torch.randn(2, 2)
    logits = model(x)
    assert logits.shape == (2,)
