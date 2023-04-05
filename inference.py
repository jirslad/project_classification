""" Script for inference. Classifies images from a folder and plots them.
Also plots confusion matrix. """

import torch
from torchvision import transforms
from pathlib import Path
from models import create_EfficientNetB0, create_EfficientNetB2, create_ViTB16
import random
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from utils.utils import load_model
from utils.plotting import plot_confusion_matrix

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    ### GET DATA ###
    test_data_folder = Path(args.imgs_path)
    test_img_paths = list(test_data_folder.glob("*/*.jpg"))

    ### LOAD MODEL ###
    model_path = args.model_path
    if "efficientnetb0" in model_path.lower():
        model = create_EfficientNetB0(output_classes=args.output_classes, freeze_features=False)
    elif "vitb16" in model_path.lower():
        model = create_ViTB16(output_classes=args.output_classes, freeze_features=False)
    else:
        print("Could not detect model type from model path. Using default model type.")
        model = create_EfficientNetB0(output_classes=args.output_classes, freeze_features=False)
    model, class_names = load_model(model, model_path, device)
    model.to(device)

    ### PERFORM PREDICTIONS AND PLOT THEM ###
    model.eval()
    rows, cols = args.rows, args.columns
    num_imgs = rows * cols
    random.seed(SEED)
    img_paths = random.sample(test_img_paths, min(num_imgs, len(test_img_paths)))

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    correct_count = 0
    target_idxs = []
    pred_idxs = []

    plt.figure(figsize=(2*cols,2*rows))

    with torch.inference_mode():
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            img_transformed = transform(img).unsqueeze(dim=0).to(device)
            logits = model(img_transformed)
            probs = logits.softmax(dim=1)
            prob, class_idx = probs.max(dim=1)
            target_idx = class_names.index(img_path.parent.name)
            target_idxs.append(target_idx)
            pred_idx = class_idx.item()
            pred_idxs.append(pred_idx)
            pred_prob = prob.item()

            plt.subplot(rows, cols, i+1)
            plt.imshow(img)
            title = f"{class_names[target_idx]} | {class_names[pred_idx]} {pred_prob:.3}"
            if pred_idx == target_idx:
                correct_count += 1
                plt.title(title, fontsize=10, c="b")
            else:
                plt.title(title, fontsize=10, c="r")
            plt.axis(False)

    accuracy = 100 * correct_count / (len(img_paths))
    print(f"Accuracy: {accuracy:.1f} % on {len(target_idxs)} images.")
    plt.suptitle(f"Label | Prediction & Probability |" +
                f" Overall Accuracy = {accuracy:.1f} %")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

    ### PLOT CONFUSION MATRIX ###
    plot_confusion_matrix(class_names=class_names,
                          pred_idxs=pred_idxs,
                          target_idxs=target_idxs,
                          task="multiclass")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-path", type=str, required=True, help="Path to folder with images.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a '.pt' or '.pth' model with class names.")
    parser.add_argument("--output-classes", type=int, default=3, help="Number of model's output classes.")
    # parser.add_argument("--model-arch", type=str, required=True, help="Model architecture. E.g. 'efficientnetB0'.")
    parser.add_argument("--rows", type=int, default=3, help="Number of images in a row on the plot with predictions")
    parser.add_argument("--columns", type=int, default=4, help="Number of images in a column on the plot with predictions")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    main(args)