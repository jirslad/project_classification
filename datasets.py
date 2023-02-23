import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple
from pathlib import Path
import csv

SEED = 42
NUM_WORKERS = 0 #os.cpu_count()


def create_dataloaders(dataset_dir: str,
    multilabel: bool,
    split_ratio: List,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS,
    seed=SEED
):
    """Creates train, val and test dataloaders from random split of DTD dataset."""

    dataset = DTDDataset(Path(dataset_dir), transform, multilabel)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, split_ratio, generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader


class DTDDataset(Dataset):
    """DTD dataset for single-label or multi-label multi-class classification"""
    def __init__(self, dataset_path: Path, transform: transforms.Compose=None, multilabel=False):

        with open(dataset_path / "class_names.txt") as f:
            self.class_names = f.read().split(" ")

        with open(dataset_path / "annotations.csv") as f:
            reader = csv.reader(f, delimiter="\t")
            self.annotations = []
            for row in reader:
                self.annotations.append(row)

        self.num_classes = len(self.class_names)
        self.transform = transform

        # parse labels in annotations 
        for anno in self.annotations:
            if multilabel:
                # to multi-hot (torch.Tensor)
                anno[1] = self._str2multihot(anno[1], self.num_classes)
            else:
                # to single class index (int)
                anno[1] = self.class_names.index(Path(anno[0]).parent.stem)
        


    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, labels = self.annotations[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, labels

    def _str2multihot(self, label, num_classes):
        class_idxs = label.split(",")
        label_vector = torch.zeros(num_classes, dtype=torch.float32)
        for idx in class_idxs:
            label_vector[int(idx)] = 1.
        return label_vector
