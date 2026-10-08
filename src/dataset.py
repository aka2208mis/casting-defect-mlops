"""
dataset.py — PyTorch Dataset + DataLoader factory for the casting splits.

Reads the version-controlled split file lists produced by data_prep.build_splits and
applies the train/eval transforms. Loaders are seeded for reproducibility.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep


class CastingDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], train: bool):
        self.items = items
        self.tf = data_prep.get_transforms(train=train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            x = self.tf(im.convert("L"))
        return x, label


def make_loaders(version: str, root: Path):
    g = torch.Generator()
    g.manual_seed(config.RANDOM_SEED)
    loaders = {}
    for name in ("train", "val", "test"):
        items = data_prep.load_split(version, name, root)
        ds = CastingDataset(items, train=(name == "train"))
        loaders[name] = DataLoader(
            ds, batch_size=config.BATCH_SIZE,
            shuffle=(name == "train"),
            num_workers=config.NUM_WORKERS,
            generator=g if name == "train" else None,
        )
    return loaders
