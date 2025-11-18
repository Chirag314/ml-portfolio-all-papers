import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # y = Xw + b
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        return logits  # sigmoid will be applied in loss
