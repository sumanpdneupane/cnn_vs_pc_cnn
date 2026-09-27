import torch
import torch.nn as nn


class StandardCNN(nn.Module):

    # Standard CNN for Tiny ImageNet-style 64x64 RGB images.
    #
    # Architecture:
    #     3   -> 48  -> 48  -> Pool
    #     48  -> 96  -> 96  -> Pool
    #     96  -> 192 -> 192 -> Pool
    #     192 -> 384 -> 384 -> Pool
    #     AdaptiveAvgPool -> Dropout -> Linear
    #
    # Intended use:
    #     - 20-class development: StandardCNN(num_classes=20)
    #     - 200-class final run: StandardCNN(num_classes=200)


    def __init__(
        self,
        num_classes=20,
        feature_dropout=0.30,
        block3_dropout=0.10,
        block4_dropout=0.15
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dropout_rate = feature_dropout
        self.block3_dropout_rate = block3_dropout
        self.block4_dropout_rate = block4_dropout

        # 64x64 -> 32x32
        self.level1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 32x32 -> 16x16
        self.level2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 16x16 -> 8x8
        self.level3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=block3_dropout)
        )

        # 8x8 -> 4x4
        self.level4 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=block4_dropout)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simple NONLINEAR classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=feature_dropout),
            nn.Linear(384, num_classes)
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
        r1 = self.level1(x)
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
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "r4": r4
            }

        return logits
