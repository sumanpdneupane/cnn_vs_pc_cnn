import torch
import torch.nn as nn


# Locked thesis model: 8 conv + BN + LeakyReLU, max-pool x3, Dropout2d, GAP, dense 512, 200 classes (2.71M params)
class StandardCNN(nn.Module):
    training_method = 'bp'  # default for TrainModel: backprop

    @staticmethod
    def block(channels, pool_after=False, dilations=(1, 1), dropout=0.0):
        layers = []
        assert len(dilations) == len(channels) - 1

        for i, d in enumerate(dilations):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.LeakyReLU(0.05, inplace=True)
            ])

        if pool_after: layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
        return nn.Sequential(*layers)

    def __init__(self, num_classes=200, block3_dropout=0.15, block4_dropout=0.25, classifier_dropout=0.35,
                 level4_dilation=2):
        super().__init__()

        # 64 → 32 → 16 → 8 → 8
        self.level1 = self.block([3, 32, 64], pool_after=True, dilations=(1, 1))
        self.level2 = self.block([64, 64, 128], pool_after=True, dilations=(1, 1))
        self.level3 = self.block([128, 128, 256], pool_after=True, dilations=(1, 1), dropout=block3_dropout)
        self.level4 = self.block([256, 256, 512], pool_after=False, dilations=(1, level4_dilation),
                                 dropout=block4_dropout)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # GAP 512 → 512
        self.dense_transition = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p=classifier_dropout)
        )
        self.classifier = nn.Linear(512, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.05, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.kaiming_normal_(self.dense_transition[0].weight, a=0.05, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, return_features=False):
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)

        features = self.flatten(self.avgpool(x))
        embedding = self.dense_transition(features)
        logits = self.classifier(embedding)

        return (logits, embedding) if return_features else logits
