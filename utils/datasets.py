import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
from typing import List, Tuple
from pathlib import Path
import csv

SEED = 42
NUM_WORKERS = 0 #os.cpu_count()


def create_dataloaders(dataset_dir: str,
    split_ratio: List,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    multilabel: bool=False,
    num_workers: int=NUM_WORKERS,
    seed: int=SEED
):
    """Creates train, val and test dataloaders from random split of DTD dataset.
    
    Args:
        dataset_dir (str): path to data relative to `train.py`. Dataset creation must be incorporated
            into this function first.
        split_ratio (list): typically list of 3 floats summing to 1, which represent train/val/test splits.
        train_transform (torchvision.transforms.Compose): training transforms.
        val_transform (torchvision.transforms.Compose): validation transforms.    
        batch_size (int): number of images in a batch.
        multilabel (bool): flag to use multilabel dataset (if aplicable).
        num_workers (int): number of CPU workers to load images from the dataloaders.
        seed (int): RNG seed.
    """

    # TODO: try faster image loading (https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/3)
    if dataset_dir.name == "dtd":
        train_dataset = DTDDataset(dataset_path=Path(dataset_dir),
                             transform=train_transform,
                             multilabel=multilabel) # TODO: separete transforms for train and val/test
    elif dataset_dir.name == "food-101":
        train_dataset = datasets.Food101(root=Path(dataset_dir).parent,
                                         split="train",
                                         transform=train_transform,
                                         download=False)
        val_dataset = datasets.Food101(root=Path(dataset_dir).parent,
                                       split="test",
                                       transform=test_transform,
                                       download=False)
    elif dataset_dir.name == "pizza_steak_sushi":
        train_dataset = datasets.ImageFolder(root=Path(dataset_dir) / "train",
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(root=Path(dataset_dir) / "test",
                                            transform=test_transform)
    else:
        assert False, "Wrong dataset path, dataset does not exist."

    test_dataset = val_dataset

    if len(split_ratio) == 3:
        train_dataset, val_dataset, test_dataset = random_split(
            train_dataset, split_ratio, generator=torch.Generator().manual_seed(seed)
        )
    elif len(split_ratio) == 2:
        train_dataset, _ = random_split(
            train_dataset, [split_ratio[0], 1 - split_ratio[0]], generator=torch.Generator().manual_seed(seed)
        )
        val_dataset, test_dataset = random_split(
            val_dataset, [split_ratio[1], 1 - split_ratio[1]], generator=torch.Generator().manual_seed(seed)
        )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


class DTDDataset(Dataset):
    """DTD dataset for single-label or multi-label multi-class classification"""
    def __init__(self,
                 dataset_path: Path,
                 transform: transforms.Compose=None,
                 multilabel=False):

        with open(dataset_path / "class_names.txt", encoding="utf-8") as f:
            self.classes = f.read().split(" ")

        with open(dataset_path / "annotations.csv") as f:
            reader = csv.reader(f, delimiter="\t")
            self.annotations = []
            for row in reader:
                self.annotations.append(row)

        self.num_classes = len(self.classes)
        self.transform = transform

        # parse labels in annotations 
        for anno in self.annotations:
            if multilabel:
                # to multi-hot (torch.Tensor)
                anno[1] = self._str2multihot(anno[1], self.num_classes)
            else:
                # to single class index (int)
                anno[1] = self.classes.index(Path(anno[0]).parent.stem)
        
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
        class_idxs = [int(x) for x in class_idxs]
        label_vector = torch.zeros(num_classes, dtype=torch.float32)
        label_vector[class_idxs] = 1.
        return label_vector
