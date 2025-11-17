from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(
    root: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    tfms = [transforms.ToTensor()]
    if augment:
        tfms.insert(0, transforms.RandomCrop(28, padding=2))
    transform = (
        transforms.compose(tfms)
        if hasattr(transforms, "compose")
        else transforms.Compose(tfms)
    )

    train_ds = datasets.MNIST(str(root), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(
        str(root), train=False, download=True, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
