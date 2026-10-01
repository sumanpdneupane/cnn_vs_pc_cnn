import torch
from torch import nn


class StandardCNN(nn.Module):
    """Non-residual VGG-style CNN for 64x64 RGB images."""

    def block(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def __init__(self, num_classes=200, block3_dropout=0.10, block4_dropout=0.15, classifier_dropout=0.30):
        super().__init__()

        # Level 1: edges and color; 64x64 -> 32x32.
        self.level1 = self.block(3, 64)

        # Level 2: textures and local shapes; 32x32 -> 16x16.
        self.level2 = self.block(64, 128)

        # Level 3: object parts; 16x16 -> 8x8.
        self.level3 = self.block(128, 256, block3_dropout)

        # Level 4: higher-level semantics; 8x8 -> 4x4.
        self.level4 = self.block(256, 512, block4_dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        features = self.pool(x).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits
