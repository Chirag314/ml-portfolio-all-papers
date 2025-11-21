from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class KMeansResult:
    centers: torch.Tensor  # (k, d)
    labels: torch.Tensor  # (n,)
    inertia: float  # sum of squared distances


def kmeans(
    X: torch.Tensor,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 5,
):
    # safety cast in case YAML gives strings
    max_iter = int(max_iter)
    n_init = int(n_init)
    tol = float(tol)

    """
    Simple k-means from scratch.
    """
    best_inertia = float("inf")
    best_centers = None
    best_labels = None

    n_samples, d = X.shape

    for _ in range(n_init):
        # random distinct indices
        perm = torch.randperm(n_samples)
        centers = X[perm[:n_clusters]].clone()

        for _ in range(max_iter):
            # compute squared distances to centers: (n, k)
            # broadcasting: X (n, d), centers (k, d)
            diff = X[:, None, :] - centers[None, :, :]
            dist2 = (diff**2).sum(dim=2)

            # assign
            labels = dist2.argmin(dim=1)

            # recompute centers
            new_centers = torch.stack(
                [
                    X[labels == j].mean(dim=0) if (labels == j).any() else centers[j]
                    for j in range(n_clusters)
                ],
                dim=0,
            )

            shift = (new_centers - centers).norm().item()
            centers = new_centers
            if shift < tol:
                break

        # final inertia
        diff = X[:, None, :] - centers[None, :, :]
        dist2 = (diff**2).sum(dim=2)
        inertia = dist2.min(dim=1).values.sum().item()

        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.clone()
            best_labels = labels.clone()

    return KMeansResult(
        centers=best_centers,
        labels=best_labels,
        inertia=best_inertia,
    )
