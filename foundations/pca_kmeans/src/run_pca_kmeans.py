import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from .path_setup import REPO_ROOT  # noqa: F401
from common.seed import set_seed
from common.logging import init_loggers
from .data import BlobConfig, make_blobs
from .pca import pca
from .kmeans import kmeans


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--use_mlflow", action="store_true")
    p.add_argument("--run_name", type=str, default="pca_kmeans")
    return p.parse_args()


def plot_clusters(X2d, labels, title, path: Path):
    plt.figure()
    scatter = plt.scatter(
        X2d[:, 0], X2d[:, 1], c=labels.cpu(), cmap="tab10", s=15, alpha=0.8
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    logger = init_loggers(
        project="ml-portfolio",
        run_name=args.run_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        tags={"paper": "pca-kmeans-foundations", **cfg},
    )

    data_cfg = BlobConfig(
        n_samples=cfg["data"]["n_samples"],
        n_features=cfg["data"]["n_features"],
        n_clusters=cfg["data"]["n_clusters"],
        cluster_std=cfg["data"]["cluster_std"],
    )
    X, y_true = make_blobs(data_cfg, device=device)

    # PCA
    X_pca, pca_res = pca(X, n_components=cfg["pca"]["n_components"])
    print("Explained variance ratio:", pca_res.explained_variance_ratio.cpu().numpy())

    # KMeans in original space
    km_orig = kmeans(
        X,
        n_clusters=cfg["kmeans"]["n_clusters"],
        max_iter=cfg["kmeans"]["max_iter"],
        tol=cfg["kmeans"]["tol"],
        n_init=cfg["kmeans"]["n_init"],
    )

    # KMeans in PCA space
    km_pca = kmeans(
        X_pca,
        n_clusters=cfg["kmeans"]["n_clusters"],
        max_iter=cfg["kmeans"]["max_iter"],
        tol=cfg["kmeans"]["tol"],
        n_init=cfg["kmeans"]["n_init"],
    )

    print(f"Inertia original space: {km_orig.inertia:.2f}")
    print(f"Inertia PCA space:      {km_pca.inertia:.2f}")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # plots
    plot_clusters(
        X_pca.detach().cpu(),
        y_true,
        "True labels in PCA(2D)",
        out_dir / "pca_true_labels.png",
    )
    plot_clusters(
        X_pca.detach().cpu(),
        km_pca.labels,
        "KMeans clusters in PCA(2D)",
        out_dir / "pca_kmeans_labels.png",
    )

    # save summary
    summary = {
        "explained_variance_ratio": pca_res.explained_variance_ratio.tolist(),
        "inertia_original": km_orig.inertia,
        "inertia_pca": km_pca.inertia,
    }
    (out_dir / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if logger:
        logger.log_metrics(
            {
                "pca/explained_var_ratio_0": float(pca_res.explained_variance_ratio[0]),
                "pca/explained_var_ratio_1": float(pca_res.explained_variance_ratio[1]),
                "kmeans/inertia_original": km_orig.inertia,
                "kmeans/inertia_pca": km_pca.inertia,
            },
            step=0,
        )
        logger.finish()


if __name__ == "__main__":
    main()
