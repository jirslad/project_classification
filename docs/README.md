# Image Classification with PyTorch
Scalable general purpose framework for my personal use.

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

| Model          | Pretrained (ImageNet) | Data | Val loss | Vall acc |  
|----------------|---------|--------|----------|---|
| ViT16          | yes     | food3   | 0.5  | 95       |
| EfficientNetB0 | yes     | food101 | 0.7  | 82       |