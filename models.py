from torch import nn
from torchvision.models import (efficientnet_b0, EfficientNet_B0_Weights,
                                efficientnet_b2, EfficientNet_B2_Weights,
                                vit_b_16, ViT_B_16_Weights)

class TinyVGG(nn.Module):
    
    def __init__(self, input_channels:int, hidden_channels:int, output_classes:int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 13 * 13, output_classes) # 64->13, 128->29
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def create_EfficientNetB0(output_classes:int, freeze_features:bool):

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=output_classes)
    )

    return model


def create_EfficientNetB2(output_classes:int, freeze_features:bool):

    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=output_classes)
    )

    return model


def create_ViTB16(output_classes:int, freeze_features:bool):

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    if freeze_features:
        for param in model.conv_proj.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False

    model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=output_classes))

    return model