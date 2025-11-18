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
from .data import ClassificationDataConfig, make_classification_loaders
from .models import LogisticRegression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--use_mlflow", action="store_true")
    p.add_argument("--run_name", type=str, default="logistic_regression_sgd")
    return p.parse_args()


def accuracy(logits, y):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == y).float().mean().item()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_cfg = ClassificationDataConfig(
        n_samples_per_class=cfg["data"]["n_samples_per_class"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
    )

    train_loader, val_loader = make_classification_loaders(data_cfg)

    model = LogisticRegression(n_features=cfg["model"]["n_features"]).to(device)

    opt = optim.SGD(model.parameters(), lr=cfg["train"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    logger = init_loggers(
        project="ml-portfolio",
        run_name=args.run_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        tags={"paper": "logistic-regression-foundations", **cfg},
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0

    epochs = cfg["train"]["epochs"]
    ckpt_path = Path(cfg["train"]["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_acc += accuracy(logits.detach(), y)
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        train_acc = total_acc / max(1, n_batches)

        # validation
        model.eval()
        val_loss_sum, val_acc_sum, val_n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item()
                val_acc_sum += accuracy(logits, y)
                val_n_batches += 1

        val_loss = val_loss_sum / max(1, val_n_batches)
        val_acc = val_acc_sum / max(1, val_n_batches)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if logger:
            logger.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                },
                step=epoch,
            )

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} best_val={best_val:.3f}"
        )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "classification_history.json").write_text(json.dumps(history, indent=2))

    # Plot loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("Logistic regression — loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "classification_loss.png")
    plt.close()

    # Plot accuracy
    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Logistic regression — accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "classification_acc.png")
    plt.close()

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
