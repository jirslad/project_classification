# Image Classification with PyTorch
Personal repository designed to experiment with PyTorch models on image classification tasks. The purpose of its creation was to practice knowledge gained from [Daniel's PyTorch course](https://www.learnpytorch.io/).

## Setup
Minimal requirements for a fresh Miniconda environment with python>=3.8:
```
conda install pytorch>=1.12 torchvision>=0.13 -c pytorch
pip install matplotlib torchmetrics torchinfo mlxtend
```

## Curently supported datasets
Food-101:

* single-label classification

* subset of classes

DTD:

* single-label classification

* multi-label classification (WIP)

## Usage

### Training

Script train.py and scripts under /utils can be modified to support more models, datasets, transforms, learning rate policies etc. 

Example usage:
```
python train.py --data-path datasets/food-101
                --split-ratio 0.4 0.4
                --model efficientnetB0 --freeze
                --batch 32 --epochs 10 --lr 3e-4
                --checkpoint-path models --summary --track --plot
```

### Inference
Example usage:
```
python inference.py --imgs-path datasets/pizza_steak_sushi/custom
                    --model-path models/food101v3_efficientnetB0_100perc-data_10ep_0.000300lr.pt
                    --rows 5 --columns 9 --output-classes 101
```

## Experiments with Food-101 dataset

All models use pretrained weights on ImageNet available from torchvision. 

Table below reports validation loss, top-1 accuracy on validation and test sets.
Test set contains in total 45 images of pizza, steak and sushi from Google Images.

| Exp # | Pretrained model | Dataset               | Epochs | LR   | Val loss | Val acc | Test acc | 
|-------|------------------|-----------------------|--------|------|----------|---------|----------|
| 1     | EfficientNetB0   | Food-101 (3 classes)  | 10     | 5e-4 | 0.052    | 0.992   | 0.956    |
| 2     | ViT16            | Food-101 (3 classes)  | 5      | 2e-5 | 0.042    | 0.990   | 0.956    |
| 3     | EfficientNetB0   | Food-101              | 10     | 3e-4 | 0.508    | 0.864   | 0.867    |
| 4     | ViT16            | Food-101              | 5      | 6e-5 | 0.517    | 0.857   | 0.844    |

### Example Colab notebook for experiment #3
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jirslad/project_classification/blob/main/demo_training_notebook.ipynb)

### Test results for experiment #3
![Alt text](/docs/imgs/food3_v3_effB0.png?raw=true "EfficientNetB0 - Food-101 (3 classes)")
