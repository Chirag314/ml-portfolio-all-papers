from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class BlobConfig:
    n_samples: int = 1000
    n_features: int = 10
    n_clusters: int = 4
    cluster_std: float = 0.7


def make_blobs(
    cfg: BlobConfig, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Gaussian blobs in d dimensions with known labels.
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    torch.manual_seed(42)

    n_per_cluster = cfg.n_samples // cfg.n_clusters
    rest = cfg.n_samples - n_per_cluster * cfg.n_clusters

    centers = torch.randn(cfg.n_clusters, cfg.n_features) * 5.0
    all_x = []
    all_y = []

    for k in range(cfg.n_clusters):
        n_k = n_per_cluster + (1 if k < rest else 0)
        mean = centers[k]
        x_k = mean + cfg.cluster_std * torch.randn(n_k, cfg.n_features)
        y_k = torch.full((n_k,), k, dtype=torch.long)
        all_x.append(x_k)
        all_y.append(y_k)

    X = torch.cat(all_x, dim=0).to(device)
    y = torch.cat(all_y, dim=0).to(device)
    return X, y
