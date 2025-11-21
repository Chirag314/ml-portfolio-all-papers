import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.data import BlobConfig, make_blobs  # noqa: E402
from src.pca import pca  # noqa: E402
from src.kmeans import kmeans  # noqa: E402


def test_pca_and_kmeans_shapes():
    cfg = BlobConfig(n_samples=200, n_features=5, n_clusters=3, cluster_std=0.5)
    X, y = make_blobs(cfg)
    assert X.shape == (200, 5)
    assert y.shape == (200,)

    X_pca, res = pca(X, n_components=2)
    assert X_pca.shape == (200, 2)
    assert res.components.shape == (2, 5)

    km = kmeans(X_pca, n_clusters=3, max_iter=10, n_init=2)
    assert km.centers.shape == (3, 2)
    assert km.labels.shape == (200,)
