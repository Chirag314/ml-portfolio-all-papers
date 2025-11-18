import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import yaml

from .path_setup import REPO_ROOT  # noqa: F401
from common.seed import set_seed
from common.logging import init_loggers
from .data import RegressionDataConfig, make_regression_loaders
from .models import LinearRegression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--use_mlflow", action="store_true")
    p.add_argument("--run_name", type=str, default="linear_regression_gd")
    return p.parse_args()


def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_cfg = RegressionDataConfig(
        n_samples=cfg["data"]["n_samples"],
        n_features=cfg["data"]["n_features"],
        noise_std=cfg["data"]["noise_std"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
    )

    train_loader, val_loader, w_true = make_regression_loaders(data_cfg)

    model = LinearRegression(n_features=cfg["model"]["n_features"]).to(device)

    optimizer_type = cfg["train"]["optimizer"].lower()
    if optimizer_type == "gd":
        # We'll emulate full-batch GD by using one batch per epoch
        full_X = torch.cat([x for x, _ in train_loader], dim=0).to(device)
        full_y = torch.cat([y for _, y in train_loader], dim=0).to(device)
        # overwrite train_loader to single batch
        train_loader = [(full_X, full_y)]

    opt = optim.SGD(model.parameters(), lr=cfg["train"]["lr"])

    logger = init_loggers(
        project="ml-portfolio",
        run_name=args.run_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        tags={"paper": "linear-regression-foundations", **cfg},
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    epochs = cfg["train"]["epochs"]
    ckpt_path = Path(cfg["train"]["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).squeeze()
            opt.zero_grad()
            pred = model(x)
            loss = mse_loss(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)

        # validation
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).squeeze()
                pred = model(x)
                loss = mse_loss(pred, y)
                val_loss_sum += loss.item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_loss_sum / val_n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if logger:
            logger.log_metrics(
                {"train/loss": train_loss, "val/loss": val_loss},
                step=epoch,
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} best_val={best_val:.4f}"
        )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "regression_history.json").write_text(json.dumps(history, indent=2))

    # Plot
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Linear regression — GD/SGD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "regression_loss.png")
    plt.close()

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
