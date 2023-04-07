import argparse
from torchvision import datasets
from pathlib import Path
import os
import csv
from typing import List
import shutil


def DTD():
    '''Prepares DTD dataset for multi-label classification.    
    
    Downloads DTD dataset (sources below). Creates 'class_names.txt' with ordered class names.
    Also creates 'annotations.csv' with image paths and image labels separated by tab, one
    image per line. Image paths are strings of class indexes separated by comma (e.g. '0,13,39').
    
    https://www.robots.ox.ac.uk/~vgg/data/dtd/
    https://pytorch.org/vision/main/generated/torchvision.datasets.DTD.html
    '''

    # download and extract the dataset with PyTorch
    datasets.DTD(root=datasets_path, download=True)

    # set dataset specific paths
    dtd_path = datasets_path / Path("dtd/dtd")
    imgs_folder_path = dtd_path / "images"
    labels_folder_path = dtd_path / "labels"

    # find class names
    class_names = []
    for scan in os.scandir(imgs_folder_path):
        if scan.is_dir():
            class_names.append(scan.name)

    # save class names
    with open(dtd_path / "class_names.txt", "w") as f:
        f.write(" ".join(class_names))

    # # read class names
    # with open(dtd_path / "class_names.txt") as f:
    #     class_names = f.read().split(" ")

    # find annotations (image paths and labels)
    img_paths = []
    img_labels = []
    with open(labels_folder_path / "labels_joint_anno.txt", encoding="utf-8") as f:
        text = f.read()
        annotations = text.split(" \n")
    for anno in annotations:
        path, *label_names = anno.split(" ")
        if len(label_names) == 0:
            print("Invalid row detected and ignored.")
            continue
        img_paths.append(imgs_folder_path / path)
        # label_idxs = []
        # for label_name in label_names:
        #     label_idxs.append(str(class_names.index(label_name)))
        label_idxs = list(map(lambda label: str(class_names.index(label)), label_names))
        img_labels.append(",".join(label_idxs))

    # save annotations
    with open(dtd_path / "annotations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(img_paths, img_labels))

    # # read annotations
    # with open(dtd_path / "annotations.csv") as f:
    #     reader = csv.reader(f, delimiter="\t")
    #     annotations_list = []
    #     for row in reader:
    #         annotations_list.append(row)
    
    print("DTD dataset processed successfully.")


def Food101():
    '''Prepares Food101 dataset.
    
    Downloads Food101 dataset (sources below) with default torchvision class.

    https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html
    '''
    datasets.Food101(root=datasets_path, download=True)

    print("Food101 dataset processed successfully.")


def FoodSubset(datasets_path:str,
               target_classes:List,
               split:List=["test", "train"]):
    """ Prepares pizza_steak_sushi subset of Food-101 dataset.
    """
    if not Path(datasets_path / "food-101").exists():
        print("Preparing Food101 dataset first.")
        Food101()

    imgs_folder = Path(datasets_path / "food-101" / "images")
    metadata_folder = Path(datasets_path / "food-101" / "meta")
    dataset_folder = Path(datasets_path / "pizza_steak_sushi")

    if Path(dataset_folder / "train").exists():
        print("Dataset has already been prepared.")
        return
    
    for split in ["test", "train"]:
        with open(metadata_folder / f"{split}.txt", "r") as f:
            img_paths = [line.strip("\n") + ".jpg" for line in f.readlines() if line.split("/")[0] in target_classes]
        for img_path in img_paths:
            class_name = img_path.split("/")[0]
            class_folder = dataset_folder / split / class_name
            if not class_folder.is_dir():
                class_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(imgs_folder / img_path, class_folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=str, help="Path to directory with datasets.")
    parser.add_argument("--dataset", choices=["DTD", "Food101", "FoodSubset"])

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    datasets_path = Path(args.datasets_dir)

    if args.dataset == "DTD":
        DTD()
    elif args.dataset == "Food101":
        Food101()
    elif args.dataset == "FoodSubset":
        FoodSubset(datasets_path, ["pizza", "steak", "sushi"], ["test", "train"])