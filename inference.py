""" Script for inference. Classifies images from a folder and plots them.
Also plots confusion matrix. """

import torch
from torchvision import transforms
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path
from models import create_EfficientNetB0
import random
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from utils import load_model

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    ### GET DATA ###
    test_data_folder = Path(args.imgs_path)
    test_img_paths = list(test_data_folder.glob("*/*.jpg"))

    ### LOAD MODEL ###
    model_path = args.model_path
    model = create_EfficientNetB0(output_classes=3, freeze_features=False)
    model, class_names = load_model(model, model_path, device)
    model.to(device)

    ### PERFORM PREDICTIONS AND PLOT THEM ###
    model.eval()
    rows, cols = args.rows, args.columns
    num_imgs = rows * cols
    random.seed(SEED)
    img_paths = random.sample(test_img_paths, min(num_imgs, len(test_img_paths)))

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.CenterCrop((224,224)),
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

    plt.suptitle(f"Label | Prediction & Probability \n" +
                f" Overall Accuracy = {100 * correct_count / (len(img_paths)):.1f} %")
    plt.show()

    ### PLOT CONFUSION MATRIX ###
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confmat_tensor = confmat(preds=torch.tensor(pred_idxs),
                            target=torch.tensor(target_idxs))
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(8, 6)
    )
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-path", type=str, required=True, help="Path to folder with images.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a '.pt' or '.pth' model with class names.")
    parser.add_argument("--rows", type=int, default=3, help="Number of images in a row on the plot with predictions")
    parser.add_argument("--columns", type=int, default=4, help="Number of images in a column on the plot with predictions")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    main(args)