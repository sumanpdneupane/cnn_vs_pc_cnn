from torch import nn


class StandardCNN(nn.Module):

    def __init__(self, num_classes=200, classifier_dropout=0.5):
        super().__init__()

        self.level1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.MaxPool2d(2)
        )

        self.level2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.MaxPool2d(2)
        )

        self.level3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.MaxPool2d(2)
        )

        self.level4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(classifier_dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(classifier_dropout),

            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

        # Keep convolutional BN gamma learnable, but fix beta at zero
        for block in [self.level1, self.level2, self.level3, self.level4]:
            bn = block[1]
            nn.init.zeros_(bn.bias)
            bn.bias.requires_grad_(False)

        # Keep both classifier BN gamma learnable,
        # but fix beta at zero
        for bn in [self.classifier[1], self.classifier[5]]:
            nn.init.zeros_(bn.bias)
            bn.bias.requires_grad_(False)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        x = self.flatten(x)

        features = self.classifier[:7](x)
        logits = self.classifier[7:](features)

        if return_features:
            return logits, features

        return logits