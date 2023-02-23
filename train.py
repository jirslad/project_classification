import torch
from torchvision import transforms
from torchinfo import summary
# from torchmetrics.functional.classification import multilabel_accuracy
import os
from pathlib import Path

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

def main():

    ### TRANSFORM
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    ### DATASET ###
    dataset_path = Path("datasets/dtd/dtd")
    split_ratio = [0.6, 0.2, 0.2]
    BATCH_SIZE = 32

    train_dataloader, val_dataloader, test_dataloader = datasets.create_dataloaders(
        dataset_dir=dataset_path,
        multilabel=multilabel,
        split_ratio=split_ratio,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED
    )

    class_names = train_dataloader.dataset.dataset.class_names

    print(f"Dataset contains {len(train_dataloader.dataset.dataset)} images of " \
        f"{len(class_names)} classes., batch size is {BATCH_SIZE}. \n" \
        f"DataLoaders have {len(train_dataloader)} training batches, {len(val_dataloader)} " \
        f"validation batches and {len(test_dataloader)} testing batches."
    )

    ### MODEL ###
    torch.manual_seed(SEED)
    model_0 = TinyVGG(input_channels=3,
                    hidden_channels=10,
                    output_classes=len(class_names)).to(device)

    summary(model_0, input_size=[1, 3, 64, 64])

    ### TRAINING ###
    EPOCHS = 5
    accuracy_fn = multilabel_accuracy
    if multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model_0.parameters(),
                            lr=0.01)
    train(model_0, train_dataloader, val_dataloader, loss_fn, optim,
        EPOCHS, device, accuracy_fn)


if __name__ == "__main__":
    main()

aa = 1