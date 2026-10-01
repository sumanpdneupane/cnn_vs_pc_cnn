import torch
from torch import nn


class StandardCNN(nn.Module):
    """Non-residual VGG-style CNN for 64x64 RGB images."""

    def block(self, in_channels, out_channels, num_convs=2, dropout=0.0):
        layers = []

        # Each convolution learns progressively richer visual features.
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3,padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        # Reduce spatial dimensions by half after all convolutions.
        layers.append(nn.MaxPool2d(2))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        return nn.Sequential(*layers)

    def __init__(self, num_classes=200, block3_dropout=0.10, block4_dropout=0.15, classifier_dropout=0.30):
        super().__init__()

        # Level 1: Learn edges, colors and simple visual patterns.
        # Input:  3 x 64 x 64
        # Output: 64 x 32 x 32
        # 2 convolution layers.
        self.level1 = self.block(in_channels=3, out_channels=64, num_convs=2)

        # Level 2: Learn textures, corners and local shapes.
        # Input:  64 x 32 x 32
        # Output: 128 x 16 x 16
        # 2 convolution layers.
        self.level2 = self.block(in_channels=64, out_channels=128, num_convs=2)

        # Level 3: Learn object parts and their combinations.
        # Input:  128 x 16 x 16
        # Output: 256 x 8 x 8
        # 3 convolution layers (one extra layer).
        self.level3 = self.block(in_channels=128, out_channels=256, num_convs=3, dropout=block3_dropout)

        # Level 4: Learn complex object-level representations.
        # Input:  256 x 8 x 8
        # Output: 512 x 4 x 4
        # 3 convolution layers (one extra layer).
        self.level4 = self.block(in_channels=256, out_channels=512, num_convs=3, dropout=block4_dropout)

        # Convert spatial feature maps into a 512-dimensional vector.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classify images using the extracted features.
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)

        # 512-dimensional features for classification and t-SNE.
        features = self.pool(x).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits
