from torch import nn

class TinyVGG(nn.Module):
    
    def __init__(self, input_channels: int, hidden_channels: int, output_classes: int):
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
            nn.Linear(hidden_channels * 29 * 29, output_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

