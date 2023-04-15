# Image Classification with PyTorch
Scalable general-purpose framework designed for my personal use.

## Setup
Minimal requirements for a fresh Miniconda environment:
```
conda install pytorch>=1.12 torchvision>=0.13 -c pytorch
pip install matplotlib torchmetrics torchinfo mlxtend
```

## Usage

### Training

Example:
```
python train.py --data-path datasets/pizza_steak_sushi/all
                --split-ratio 0.6 0.2 0.2
                --model efficientnetB0 --freeze
                --batch 32 --epochs 20 --lr 1e-4
                --checkpoint --summary --track --plot
```

### Inference
Example:
```
python inference.py --imgs-path datasets/pizza_steak_sushi/custom
                    --model-path models/efficientnetB0_30ep_0.000100lr_60perc-data.pt
                    --rows 5 --columns 9 --output-classes 3
```

## Experiments with Food-101 dataset

All models use pretrained weights on ImageNet available in torchvision. 

Table below reports validation loss, top 1 accuracy of validation and test sets.
Test set contains in total 45 images of pizza, steak and sushi from Google Images.

| Pretrained model | Dataset               | Epochs | LR   | Val loss | Val acc | Test acc | 
|------------------|-----------------------|--------|------|----------|---------|----------|
| EfficientNetB0   | Food-101 (3 classes)  | 10     | 5e-4 | 0.508    | 0.821   | 0.611    |
| ViT16            | Food-101 (3 classes)  | 5      | 2e-5 | 0.555    | 0.952   | 0.871    |
| EfficientNetB0   | Food-101              | 10     | 5e-4 | 0.508    | 0.854   | 0.867    |
| ViT16            | Food-101              | 10     | 5e-4 | 0.555    | 0.952   | 0.871    |

![Alt text](/docs/imgs/food3_v3_effB0.png?raw=true "EfficientNetB0 - Food-101 (3 classes)")