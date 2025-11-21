from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PCAResult:
    components: torch.Tensor  # (n_components, d)
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor
    mean: torch.Tensor  # (d,)


def pca(X: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, PCAResult]:
    """
    PCA via SVD: X_centered = U S V^T
    components = first n_components rows of V.
    Returns:
      X_proj: (n_samples, n_components)
      PCAResult
    """
    # Center
    mean = X.mean(dim=0)
    X_centered = X - mean

    # SVD
    # X_centered = U S Vh, Vh shape: (d, d)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # Eigenvalues (variance along PCs)
    n_samples = X.shape[0]
    eigenvalues = (S**2) / (n_samples - 1)

    # Sort in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    Vh = Vh[idx]

    components = Vh[:n_components]  # (n_components, d)
    explained_variance = eigenvalues[:n_components]
    total_var = eigenvalues.sum()
    explained_variance_ratio = explained_variance / total_var

    # Project
    X_proj = X_centered @ components.T

    result = PCAResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
    )
    return X_proj, result
