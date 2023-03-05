import torch
from torchvision import transforms
from torchinfo import summary
# from torchmetrics.functional.classification import multilabel_accuracy
import os
from pathlib import Path
import argparse
from typing import List

import datasets
from models import TinyVGG
from engine import train
from utils import multilabel_accuracy

SEED = 42
NUM_WORKERS = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}.")

### ARGUMENTS
multilabel = False
img_size = 128

def main(args):

    ### TRANSFORM
    # TODO: add cropping to square
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    ### DATASET ###
    dataset_path = Path("datasets/dtd/dtd")
    dataset_path = Path("datasets/food-101")
    split_ratio = args.split_ratio
    BATCH_SIZE = args.batch

    train_dataloader, val_dataloader, test_dataloader = datasets.create_dataloaders(
        dataset_dir=dataset_path,
        split_ratio=split_ratio,
        transform=transform,
        multilabel=multilabel,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED
    )

    classes = train_dataloader.dataset.dataset.classes

    print(f"Dataset contains {len(train_dataloader.dataset.dataset)} images of " \
        f"{len(classes)} classes, batch size is {BATCH_SIZE}. \n" \
        f"DataLoaders have {len(train_dataloader)} training batches, {len(val_dataloader)} " \
        f"validation batches and {len(test_dataloader)} testing batches."
    )

    ### MODEL ###
    torch.manual_seed(SEED)
    model_0 = TinyVGG(input_channels=3,
                    hidden_channels=10,
                    output_classes=len(classes)).to(device)

    summary(model_0, input_size=[1, 3, img_size, img_size])

    ### TRAINING ###
    EPOCHS = args.epochs
    accuracy_fn = multilabel_accuracy
    if multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model_0.parameters(),
                            lr=args.lr)
    train(model_0, train_dataloader, val_dataloader, loss_fn, optim,
        EPOCHS, device, accuracy_fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--split-ratio", nargs=3, type=float, help='List of ratios of train, val, test dataset split (e.g. [0.6, 0.2, 0.2])')
    return parser.parse_args()

# arguments for debugging
# arguments = [
#     '--epochs', '5',
#     '--lr', '0.001',
#     '--batch', '32',
#     '--split-ratio', '0.01', '0.01', '0.98'
# ]

if __name__ == "__main__":
    
    args = parse_args()
    main(args)


aa = 1