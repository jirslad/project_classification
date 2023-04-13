import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B2_Weights, ViT_B_16_Weights
from torchinfo import summary
# from torchmetrics.functional.classification import multilabel_accuracy
from pathlib import Path
import os
import argparse
from typing import List

from utils.datasets import create_dataloaders
from utils.models import TinyVGG, create_EfficientNetB0, create_EfficientNetB2, create_ViTB16
from utils.vit import ViT
from utils.engine import train
from utils.utils import multiclass_accuracy, save_model, load_model, create_writer
from utils.plotting import plot_loss_curves, plot_dataset_distribution

SEED = 42
NUM_WORKERS = os.cpu_count() # os.cpu_count() lauches debugged script multiple times
device = "cuda" if torch.cuda.is_available() else "cpu"

### MAIN
def main(args):
    
    ### TRANSFORM
    img_size = 224
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    augmented_resized_transform = transforms.Compose([
            transforms.Resize(img_size+32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.9, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31), # 31
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            imagenet_normalize
    ])
    test_resized_transform = transforms.Compose([
            transforms.Resize(img_size+32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            imagenet_normalize
    ])

    if args.model == "tinyvgg":
        img_size = 64
        train_transform = transforms.Compose([
            transforms.Resize(img_size+32, interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        test_transform = train_transform
    elif "efficientnet" in args.model:
        img_size = 224
        train_transform = augmented_resized_transform
        test_transform = test_resized_transform
        # if args.model == "efficientnetB0":
        #     weights = EfficientNet_B0_Weights.DEFAULT
        # elif args.model == "efficientnetB2":
        #     weights = EfficientNet_B2_Weights.DEFAULT
        # train_transform = weights.transforms() # auto-transforms
        # test_transform = train_transform
    elif args.model == "vit_scratch":
        img_size = 224
        train_transform = augmented_resized_transform
        test_transform = test_resized_transform
    elif args.model == "vitB16":
        img_size = 224
        train_transform = augmented_resized_transform
        test_transform = test_resized_transform
        # weights = ViT_B_16_Weights.IMAGENET1K_V1
        # train_transform = weights.transforms()
        # test_transform = train_transform

    ### DATASET ###
    multilabel = False
    # dataset_path = Path("datasets/dtd/dtd")
    # dataset_path = Path("datasets/food-101")
    # dataset_path = Path("datasets/pizza_steak_sushi/train_test")
    # dataset_path = Path("datasets/pizza_steak_sushi/all")
    dataset_path = Path(args.data_path)
    
    split_ratio = float(args.split_ratio)
    batch_size = args.batch

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset_dir=dataset_path,
        split_ratio=split_ratio,
        train_transform=train_transform,
        test_transform=test_transform,
        multilabel=multilabel,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        seed=SEED
    )

    classes = train_dataloader.dataset.dataset.classes

    # number of instances per class in training dataset
    if args.plot:
        plot_dataset_distribution(train_dataloader, split="Training")
        plot_dataset_distribution(val_dataloader, split="Validation")

    print(f"Dataset path: {dataset_path}.\n" \
          f"Training dataset contains {len(train_dataloader.dataset.dataset)} images of " \
          f"{len(classes)} classes, batch size is {batch_size}. \n" \
          f"DataLoaders have {len(train_dataloader)} training batches, {len(val_dataloader)} " \
          f"validation batches and {len(test_dataloader)} testing batches."
    )

    ### MODEL ###
    torch.manual_seed(SEED)
    if args.model == "tinyvgg":
        model = TinyVGG(input_channels=3,
                        hidden_channels=10,
                        output_classes=len(classes)).to(device)
    elif args.model == "efficientnetB0":
        model = create_EfficientNetB0(output_classes=len(classes),
                                      freeze_features=args.freeze).to(device)
    elif args.model == "efficientnetB2":
        model = create_EfficientNetB2(output_classes=len(classes),
                                      freeze_features=args.freeze).to(device)
    elif args.model == "vit_scratch":
        model = ViT(img_height=img_size,
            img_width=img_size,
            img_channels=3,
            patch_size=16,
            embedding_dimension=768,
            encoder_layers=12//2,
            msa_heads=12,
            embedding_dropout=0.1,
            msa_dropout=0.0,
            mlp_dropout=0.1,
            mlp_units=3072//3,
            out_classes=len(classes)).to(device)
    elif args.model == "vitB16":
        model = create_ViTB16(output_classes=len(classes),
                              freeze_features=args.freeze).to(device)
     
    if args.model_path:
        model, _ = load_model(model, args.model_path, device)

    if args.summary:                 
        summary(model,
                input_size=[1, 3, img_size, img_size],
                col_names=["input_size", "output_size", "num_params", "trainable", "mult_adds"])

    ### EXPERIMENT TRACKING
    if len(split_ratio) == 3:
        data_percent = int(round((split_ratio[0] / sum(split_ratio)) * 100))
    elif len(split_ratio) == 2:
        data_percent = int(round(split_ratio[0] * 100))
    freeze = "_freeze" if args.freeze else ""
    if args.track:
        writer = create_writer(experiment_name=f"data_{data_percent}_percent",
                               model_name=f"{args.model}{freeze}",
                               num_epochs=f"{args.epochs}ep",
                               extra=f"{args.lr:.6f}lr")
    else:
        writer = None
    
    ### TRAINING ###
    # accuracy function
    accuracy_fn = multiclass_accuracy

    # loss function
    if multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer
    weight_decay = 0.00 if args.freeze else 0.00
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=args.lr,
                             weight_decay=weight_decay)
    # optim = torch.optim.SGD(params=model.parameters(),
    #                          lr=args.lr,
    #                          momentum=0.9,
    #                          weight_decay=weight_decay)
    
    # learning rate scheduler
    scheduler = lr_scheduler.ChainedScheduler([
        lr_scheduler.LinearLR(optim, start_factor=0.025, total_iters=max(2, args.epochs//8)),
        lr_scheduler.ExponentialLR(optim, gamma=0.9)
    ])
    # scheduler = None

    # info for saving the model
    model_name = f"{args.model}{freeze}_{data_percent}perc-data_{args.epochs}ep_{args.lr:.6f}lr.pt"
    save_folder = Path("models")

    # training
    torch.manual_seed(SEED)
    print(f"Training on {device}...")
    results = train(model, train_dataloader, val_dataloader, loss_fn, optim,
                    args.epochs, device, accuracy_fn, scheduler, writer=writer,
                    checkpoint_saving=args.checkpoint, model_path=Path(save_folder, model_name))
    
    ### SAVE MODEL ###
    save_model(model, classes, save_folder, model_name, verbose=True)

    ### PLOT RESULTS ###
    if args.plot:
        plot_loss_curves(results)
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--data-path", type=str, required=True, help="Path to train and val dataset.")
    parser.add_argument("--split-ratio", nargs="+", type=float, help="Ratios of train, val, test dataset split (e.g. 0.6 0.2 0.2).")
    parser.add_argument("--model", type=str, default="tinyvgg", choices=["tinyvgg", "efficientnetB0", "efficientnetB2", "vit_scratch", "vitB16"])
    parser.add_argument("--model-path", type=str, help="Path to a .pt or .pth model to continue training on.")
    parser.add_argument("--checkpoint", action="store_true", help="Save model after each epoch")
    parser.add_argument("--summary", action="store_true", help="Show model summary.")
    parser.add_argument("--track", action="store_true", help="Track model experiment.")
    parser.add_argument("--plot", action="store_true", help="Plot training results.")
    parser.add_argument("--freeze", action="store_true", help="Freeze feature extractor.")
    return parser.parse_args()

# arguments for debugging
# arguments = [
#     '--epochs', '20',
#     '--lr', '0.001',
#     '--batch', '32',
#     '--split-ratio', '0.1', '0.1', '0.8',
#     '--model', 'efficientnet'
# ]


if __name__ == "__main__":
    
    args = parse_args()
    main(args)

