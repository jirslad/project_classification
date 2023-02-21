from torchvision import datasets
from pathlib import Path
import os
import csv

datasets_path = Path("datasets")


def DTD():
    '''Prepares DTD dataset.    
    
    Downloads DTD dataset (sources bellow). Creates 'class_names.txt' with ordered class names.
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
    with open(labels_folder_path / "labels_joint_anno.txt") as f:
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
    with open(dtd_path / "annotations.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(img_paths, img_labels))

    # # read annotations
    # with open(dtd_path / "annotations.csv") as f:
    #     reader = csv.reader(f, delimiter="\t")
    #     annotations_list = []
    #     for row in reader:
    #         annotations_list.append(row)
    
    print("DTD dataset processed successfully.")


if __name__ == "__main__":
    DTD()

