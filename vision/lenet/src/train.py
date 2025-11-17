import sys
from pathlib import Path

# Ensure repo root (ml-portfolio-all-papers) is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from tqdm import tqdm
import yaml

# make sure we can import common.*
from .path_setup import REPO_ROOT  # noqa: F401  (side effect: modifies sys.path)

from common.seed import set_seed
from common.logging import init_loggers
from .data import get_loaders
from .model import LeNet5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--use_mlflow", action="store_true")
    p.add_argument("--run_name", type=str, default="lenet-dev")
    return p.parse_args()


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_plot(history, out_dir: Path, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["eval_loss"], label="eval")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{title} — loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png")
    plt.close()

    # acc
    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["eval_acc"], label="eval")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{title} — accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc.png")
    plt.close()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    out_path = Path(cfg["train"]["ckpt_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger = init_loggers(
        project="ml-portfolio",
        run_name=args.run_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        tags={"paper": "lenet", **cfg},
    )

    train_loader, test_loader = get_loaders(
        Path(cfg["data"]["root"]),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"].get("augment", False),
    )

    model = LeNet5(cfg["model"]["in_channels"], cfg["model"]["num_classes"]).to(device)

    if cfg["train"]["optimizer"].lower() == "sgd":
        opt = optim.SGD(
            model.parameters(),
            lr=cfg["train"]["lr"],
            momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"],
        )
    else:
        opt = optim.Adam(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )

    criterion = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": []}
    best_acc = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        run_loss = 0.0
        run_acc = 0.0
        steps = 0

        for step, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            run_loss += loss.item()
            run_acc += accuracy(logits.detach(), y)
            steps += 1

            if step % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix(
                    loss=f"{run_loss / steps:.4f}",
                    acc=f"{run_acc / steps:.3f}",
                )

        train_loss = run_loss / max(1, steps)
        train_acc = run_acc / max(1, steps)

        # Eval
        model.eval()
        tot, correct, eval_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                eval_loss_sum += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                tot += x.size(0)
        eval_loss = eval_loss_sum / tot
        eval_acc = correct / tot

        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["eval_loss"].append(eval_loss)
        hist["eval_acc"].append(eval_acc)

        if logger:
            logger.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "eval/loss": eval_loss,
                    "eval/acc": eval_acc,
                },
                step=epoch,
            )

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_path)

        print(
            f"[epoch {epoch}] train_acc={train_acc:.3f} eval_acc={eval_acc:.3f} best={best_acc:.3f}"
        )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    save_plot(hist, out_dir, "LeNet-5")

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
