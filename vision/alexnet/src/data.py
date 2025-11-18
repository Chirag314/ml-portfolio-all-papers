from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_loaders(
    root: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, val, test loaders for CIFAR-10.
    """

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tfms = []
    if augment:
        train_tfms.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_transform = transforms.Compose(train_tfms)

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_full = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_ds = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=test_transform,
    )

    n_train = len(train_full)
    n_val = int(val_split * n_train)
    n_train_final = n_train - n_val
    train_ds, val_ds = random_split(
        train_full,
        [n_train_final, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
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

    return train_loader, val_loader, test_loader
