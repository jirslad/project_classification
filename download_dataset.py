from torchvision import datasets, transforms
from pathlib import Path

dataset_path = Path("dataset")

train_dataset = datasets.DTD(root=dataset_path,
                             split="train",
                             transform=transforms.ToTensor(),
                             download=True)

val_dataset = datasets.DTD(root=dataset_path,
                           split="val",
                           transform=transforms.ToTensor(),
                           download=True)

test_dataset = datasets.DTD(root=dataset_path,
                            split="test",
                            transform=transforms.ToTensor(),
                            download=True)