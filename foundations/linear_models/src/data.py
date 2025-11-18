from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass
class RegressionDataConfig:
    n_samples: int = 1000
    n_features: int = 5
    noise_std: float = 0.5
    batch_size: int = 64
    val_split: float = 0.2


@dataclass
class ClassificationDataConfig:
    n_samples_per_class: int = 500
    batch_size: int = 64
    val_split: float = 0.2


def make_regression_loaders(cfg: RegressionDataConfig):
    """
    y = Xw_true + noise, with w_true random.
    """
    n = cfg.n_samples
    d = cfg.n_features

    X = torch.randn(n, d)
    w_true = torch.randn(d, 1)
    noise = cfg.noise_std * torch.randn(n, 1)
    y = X @ w_true + noise

    ds = TensorDataset(X, y)
    n_val = int(cfg.val_split * n)
    n_train = n - n_val

    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, w_true.squeeze()


def make_classification_loaders(cfg: ClassificationDataConfig):
    """
    Two Gaussian blobs in 2D, labels {0,1}.
    """
    n0 = cfg.n_samples_per_class
    n1 = cfg.n_samples_per_class

    mean0 = torch.tensor([-2.0, 0.0])
    mean1 = torch.tensor([2.0, 0.0])

    cov = torch.tensor([[1.0, 0.2], [0.2, 1.0]])

    L = torch.linalg.cholesky(cov)

    z0 = torch.randn(n0, 2) @ L.T + mean0
    z1 = torch.randn(n1, 2) @ L.T + mean1

    X = torch.cat([z0, z1], dim=0)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)], dim=0)

    ds = TensorDataset(X, y)
    n = len(ds)
    n_val = int(cfg.val_split * n)
    n_train = n - n_val

    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader
