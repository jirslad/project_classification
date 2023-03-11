import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchinfo import summary
# from torchmetrics.functional.classification import multilabel_accuracy
import os
from pathlib import Path
import argparse
from typing import List

import datasets
from models import TinyVGG
from engine import train
from utils import multiclass_accuracy, multilabel_accuracy, save_model

SEED = 42
NUM_WORKERS = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}.")

### ARGUMENTS
multilabel = False

def main(args):

    ### TRANSFORM
    # TODO: add cropping to square
    if args.model.lower() == "tinyvgg":
        img_size = 64
        transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
        ])
    elif args.model.lower() == "efficientnet":
        img_size = 224
        # transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        parameters = EfficientNet_B0_Weights.DEFAULT
        transform = parameters.transforms() # auto-transforms

    ### DATASET ###
    # dataset_path = Path("datasets/dtd/dtd")
    # dataset_path = Path("datasets/food-101")
    dataset_path = Path("datasets/pizza_steak_sushi/all")
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
    if args.model.lower() == "tinyvgg":
        model = TinyVGG(input_channels=3,
                        hidden_channels=10,
                        output_classes=len(classes)).to(device)
    elif args.model.lower() == "efficientnet":
        model = efficientnet_b0(parameters=parameters).to(device)
        model.features.requires_grad_=False
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=1280,
                            out_features=len(train_dataloader.dataset.dataset.classes))
        ).to(device)
                               
    summary(model,
            input_size=[1, 3, img_size, img_size],
            col_names=["output_size", "num_params", "trainable", "mult_adds"])

    ### TRAINING ###
    EPOCHS = args.epochs
    accuracy_fn = multiclass_accuracy
    if multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(),
                            lr=args.lr)
    train(model, train_dataloader, val_dataloader, loss_fn, optim,
        EPOCHS, device, accuracy_fn)
    
    ### SAVE MODEL ###
    save_folder = Path("models")
    model_name = "TinyVGG.pt"
    save_model(model, save_folder, model_name)

    print("TRAINING PROCEDURE FINISHED.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--split-ratio", nargs=3, type=float, help='Ratios of train, val, test dataset split (e.g. 0.6 0.2 0.2)')
    parser.add_argument("--model", type=str, default="tinyvgg", choices=["tinyvgg", "efficientnet"])
    return parser.parse_args()

# arguments for debugging
# arguments = [
#     '--epochs', '20',
#     '--lr', '0.001',
#     '--batch', '32',
#     '--split-ratio', '0.8', '0.19', '0.01',
#     '--model', 'TinyVGG'
# ]

if __name__ == "__main__":
    
    args = parse_args()
    main(args)


aa = 1