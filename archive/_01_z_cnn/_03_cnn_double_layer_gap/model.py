import torch
from torch import nn


class StandardCNN(nn.Module):

    @staticmethod
    def block(in_channels, out_channels, dropout=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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

        # 64x64 -> 32x32
        self.level1 = self.block(3, 64)

        # 32x32 -> 16x16
        self.level2 = self.block(64, 128)

        # 16x16 -> 8x8
        self.level3 = self.block(128, 256, dropout=block3_dropout)

        # 8x8 -> 4x4
        self.level4 = self.block(256, 512, dropout=block4_dropout)

        # 512x4x4 -> 512x1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 512 -> 256 -> 200
        self.feature_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        features = self.feature_layer(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits