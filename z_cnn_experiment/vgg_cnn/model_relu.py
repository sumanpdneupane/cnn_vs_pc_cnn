import torch
from torch import nn


class StandardCNN(nn.Module):

    @staticmethod
    def block(in_channels, out_channels, dropout=0.0):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        return nn.Sequential(*layers)


    def __init__(
        self,
        num_classes=200,
        block3_dropout=0.0,
        block4_dropout=0.0,
        classifier_dropout=0.5
    ):
        super().__init__()

        # =====================================================
        # Level 1
        # (B, 3, 64, 64) -> (B, 64, 32, 32)
        # =====================================================
        self.level1 = self.block(
            3,
            64
        )

        # =====================================================
        # Level 2
        # (B, 64, 32, 32) -> (B, 128, 16, 16)
        # =====================================================
        self.level2 = self.block(
            64,
            128
        )

        # =====================================================
        # Level 3
        # (B, 128, 16, 16) -> (B, 256, 8, 8)
        # =====================================================
        self.level3 = self.block(
            128,
            256,
            dropout=block3_dropout
        )

        # =====================================================
        # Level 4
        # (B, 256, 8, 8) -> (B, 512, 4, 4)
        # =====================================================
        self.level4 = self.block(
            256,
            512,
            dropout=block4_dropout
        )

        # =====================================================
        # Flatten
        # 512 × 4 × 4 = 8192
        # =====================================================
        self.flatten = nn.Flatten()

        # =====================================================
        # Fully Connected Classifier
        # 8192 -> 512 -> 256 -> 200
        # =====================================================
        self.classifier = nn.Sequential(

            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),

            nn.Linear(256, num_classes),
        )


    def forward(self, x, return_features=False):

        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)

        # (B, 512, 4, 4) -> (B, 8192)
        x = self.flatten(x)

        # 256-dimensional representation:
        # FC1 -> BN -> ReLU -> Dropout
        # FC2 -> BN -> ReLU
        features = self.classifier[:7](x)

        # Dropout -> final classification layer
        logits = self.classifier[7:](features)

        if return_features:
            return logits, features

        return logits