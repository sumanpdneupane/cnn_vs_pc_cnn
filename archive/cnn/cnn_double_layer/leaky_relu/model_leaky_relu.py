import torch
from torch import nn


class StandardCNN(nn.Module):

    def __init__(self, num_classes=200, classifier_dropout=0.5):
        super().__init__()

        # =====================================================
        # Level 1: Edges and colors
        # Input : (B, 3, 64, 64)
        # Output: (B, 64, 32, 32)
        # =====================================================
        self.level1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.MaxPool2d(kernel_size=2)
        )

        # =====================================================
        # Level 2: Textures and local shapes
        # Input : (B, 64, 32, 32)
        # Output: (B, 128, 16, 16)
        # =====================================================
        self.level2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.MaxPool2d(kernel_size=2)
        )

        # =====================================================
        # Level 3: Object parts
        # Input : (B, 128, 16, 16)
        # Output: (B, 256, 8, 8)
        # =====================================================
        self.level3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.MaxPool2d(kernel_size=2)
        )

        # =====================================================
        # Level 4: High-level semantic features
        # Input : (B, 256, 8, 8)
        # Output: (B, 512, 4, 4)
        # =====================================================
        self.level4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            nn.MaxPool2d(kernel_size=2)
        )

        # =====================================================
        # Flatten
        # (B, 512, 4, 4) -> (B, 8192)
        # =====================================================
        self.flatten = nn.Flatten()

        # =====================================================
        # Classifier
        # =====================================================
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(p=classifier_dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(p=classifier_dropout),

            nn.Linear(256, num_classes)
        )

        # LeakyReLU-aware initialization
        self._initialize_weights()

        # =====================================================
        # Freeze ONLY convolutional BatchNorm beta at zero.
        #
        # Gamma remains learnable.
        # Running mean/variance remain active.
        #
        # Classifier BatchNorm beta remains learnable.
        # =====================================================
        for level in [self.level1, self.level2, self.level3, self.level4]:
            # Each level has BatchNorm2d at indices 1 and 4
            for bn_index in [1, 4]:
                bn = level[bn_index]
                nn.init.zeros_(bn.bias)
                bn.bias.requires_grad_(False)


    def _initialize_weights(self):

        for m in self.modules():
            # Convolution layers
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")

            # Fully connected layers
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # BatchNorm
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x, return_features=False):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)

        # (B, 512, 4, 4) -> (B, 8192)
        x = self.flatten(x)

        # FC1 -> BN -> LeakyReLU -> Dropout
        # FC2 -> BN -> LeakyReLU
        # Returns 256-dimensional features
        features = self.classifier[:7](x)

        # Dropout -> final Linear
        logits = self.classifier[7:](features)

        if return_features:
            return logits, features

        return logits