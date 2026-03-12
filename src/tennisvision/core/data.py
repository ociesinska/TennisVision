from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_preprocess():
     return T.Compose([
          T.Resize((224,224)),
          T.ToTensor(),
          T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
     ])

@dataclass(frozen=True)
class Split:
    idx_train: list[int]
    idx_val: list[int]
    idx_test: list[int]
    class_to_idx: dict[str, int]
    seed: int


def make_split(image_root: str | Path, seed: int = 42, test_size: float = 0.1, val_size: float = 0.1) -> Split:
    """
    Makes a  stratified split (train/val/test) based on ImageFolder.targets.
    val_size and test_size are a share of the whole dataset (np. 0.1 i 0.1).
    """

    base = datasets.ImageFolder(root=str(image_root), transform=None)
    targets = np.array(base.targets)
    idx = np.arange(len(targets))

    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=targets)

    val_from_train = val_size / (1 - test_size)
    targets_train = targets[idx_train]

    idx_train, idx_val = train_test_split(idx_train, test_size=val_from_train, random_state=seed, stratify=targets_train)

    return Split(
        idx_train=idx_train.tolist(),
        idx_val=idx_val.tolist(),
        idx_test=idx_test.tolist(),
        class_to_idx=base.class_to_idx,
        seed=seed,
    )

def build_transforms(
    weights=None,
    img_size: int = 224,
    train_aug: bool = True,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
):
    """
    Val/test ->  weights.transforms()
    Train -> custom augmentation + Normalize aligned with weights.

    Returns (train_tfms, val_tfms)
    """
    # mean / std
    if not mean or not std:
        if weights is not None and hasattr(weights, "meta"):
            mean = tuple(weights.meta.get("mean", IMAGENET_MEAN))
            std = tuple(weights.meta.get("std", IMAGENET_STD))
        else:
            # for custom models trainig fully, statistics for the dataset can be calucalted later
            mean = IMAGENET_MEAN
            std = IMAGENET_STD

    # val / test transforms

    if weights is not None:
        val_tfms = weights.transforms()
    else:
        val_tfms = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    # train transforms
    if not train_aug:
        train_tfms = val_tfms
    else:
        train_tfms = T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    return train_tfms, val_tfms


def build_loaders(
    image_root: str | Path,
    split: Split,
    train_tfms,
    val_tfms,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,  # not available on mps now
):
    """
    Creates 3 ImageFolder with different transforms, but same idx_* (Subset).
    """

    ds_train_all = datasets.ImageFolder(image_root, transform=train_tfms)
    ds_val_all = datasets.ImageFolder(root=str(image_root), transform=val_tfms)
    ds_test_all = datasets.ImageFolder(root=str(image_root), transform=val_tfms)

    train_ds = Subset(ds_train_all, split.idx_train)
    val_ds = Subset(ds_val_all, split.idx_val)
    test_ds = Subset(ds_test_all, split.idx_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, val_loader, test_loader), (train_ds, val_ds, test_ds), ds_train_all.classes
