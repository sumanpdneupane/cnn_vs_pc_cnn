import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class StandardCNN(nn.Module):
    def __init__(self, num_classes=200, feature_dropout=0.20):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dropout_rate = feature_dropout

        # Stem: 3 x 64 x 64 -> 64 x 64 x 64
        self.stem = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Level 1: 64 x 64 x 64
        self.level1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )

        # Level 2: 64 x 64 x 64 -> 128 x 32 x 32
        self.level2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )

        # Level 3: 128 x 32 x 32 -> 256 x 16 x 16
        self.level3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )

        # Level 4: 256 x 16 x 16 -> 512 x 8 x 8
        self.level4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )

        # Classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=feature_dropout),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_features=False):
        stem = self.stem(x)
        r1 = self.level1(stem)
        r2 = self.level2(r1)
        r3 = self.level3(r2)
        r4 = self.level4(r3)

        features = self.global_pool(r4)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)

        if return_features:
            return {
                "scores": logits,
                "features": features,
                "stem": stem,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "r4": r4
            }
        return logits