from torch import nn

class StandardCNN(nn.Module):
    @staticmethod
    def block(in_channels, out_channels, dropout=0.0, num_convs=2, use_maxpool=True):
        layers = []

        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        if use_maxpool: layers.append(nn.MaxPool2d(kernel_size=2))
        if dropout > 0: layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def __init__(self, num_classes=200, block3_dropout=0.05, block4_dropout=0.10, classifier_dropout=0.20):
        super().__init__()

        # Total 10 Convolutions: 2 + 2 + 3 + 3
        # Spatial resolution transitions: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 8x8
        self.level1 = self.block(3, 64, num_convs=2, use_maxpool=True)        # Out: 32x32
        self.level2 = self.block(64, 128, num_convs=2, use_maxpool=True)      # Out: 16x16
        self.level3 = self.block(128, 256, dropout=block3_dropout, num_convs=3, use_maxpool=True)  # Out: 8x8

        # Kept at 8x8 grid to preserve spatial relationships longer
        self.level4 = self.block(256, 512, dropout=block4_dropout, num_convs=3, use_maxpool=False) # Out: 8x8

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Non-linear vector space to sort out the high-level semantic mix of 512 channels
        self.dense_transition = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)

        x = self.global_pool(x)
        x = self.flatten(x)

        # Process through dense bottleneck layer before projecting to classes
        features = self.dense_transition(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits
