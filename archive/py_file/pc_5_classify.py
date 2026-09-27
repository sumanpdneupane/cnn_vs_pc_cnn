import torch
import torch.nn as nn
from torch.nn.grad import conv2d_weight


# ============================================================
# HIERARCHICAL RAO-STYLE PC-CNN
# CONVOLUTIONAL ADAPTATION OF RAO & BALLARD
# NO BACKPROPAGATION
# ============================================================

class HierarchicalPCCNN(nn.Module):

    def __init__(self, num_classes, inference_steps=100):
        super().__init__()

        # ====================================================
        # RAO PARAMETERS
        # ====================================================

        self.k1 = 0.5
        self.k2 = 1.0

        self.sigma_bu_sq = 1.0
        self.sigma_td_sq = 10.0

        self.alpha1 = 0.001
        self.alpha2 = 0.05

        self.lambda_u = 0.02

        # ====================================================
        # RAO k2 LEARNING-RATE SCHEDULE
        # ====================================================

        self.k2_decay_interval = 40
        self.k2_decay_factor = 1.015
        self.training_input_count = 0

        # ====================================================
        # NUMERICAL INTEGRATION PARAMETERS
        # ====================================================

        self.inference_steps = inference_steps

        # Euler integration step for Rao Eq. 7
        self.inference_dt = 0.1

        # Euler integration step for Rao Eq. 9
        self.weight_dt = 0.01

        # ====================================================
        # LEVEL 1
        #
        # Bottom-up:
        # Image [3,224,224] -> r1 [32,112,112]
        #
        # Top-down:
        # r1 [32,112,112] -> predicted image [3,224,224]
        # ====================================================

        self.level1 = nn.ModuleDict({
            "conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # LEVEL 2
        #
        # Bottom-up:
        # r1 [32,112,112] -> r2 [128,56,56]
        #
        # Top-down:
        # r2 [128,56,56] -> predicted r1 [32,112,112]
        # ====================================================

        self.level2 = nn.ModuleDict({
            "conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # CLASSIFICATION READOUT
        #
        # r2 [128,56,56]
        #       ↓
        # Adaptive pooling
        #       ↓
        # [128,7,7]
        #       ↓
        # Flatten
        #       ↓
        # 6272 features
        #       ↓
        # Linear classifier
        # ====================================================

        self.classifier_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(128 * 7 * 7, num_classes, bias=True)

        # ====================================================
        # SMALL RANDOM WEIGHT INITIALIZATION
        # ====================================================

        nn.init.normal_(self.level1["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level2["conv"].weight, mean=0.0, std=0.01)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

        # ====================================================
        # TIED BOTTOM-UP / TOP-DOWN WEIGHTS
        # ====================================================

        self.level1["deconv"].weight = self.level1["conv"].weight
        self.level2["deconv"].weight = self.level2["conv"].weight

        # ====================================================
        # NO AUTOGRAD / NO BACKPROPAGATION
        # ====================================================

        self.level1["conv"].weight.requires_grad_(False)
        self.level2["conv"].weight.requires_grad_(False)

        self.classifier.weight.requires_grad_(False)
        self.classifier.bias.requires_grad_(False)

    # ========================================================
    # INITIALIZE LATENT REPRESENTATIONS
    #
    # RANDOM INITIALIZATION
    # ========================================================

    def initialize_states(self, image):
        image_batch = image.unsqueeze(0)

        r1_shape = self.level1["conv"](image_batch).shape
        r2_shape = self.level2["conv"](torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape

        r1 = torch.randn(r1_shape, device=image.device, dtype=image.dtype) * 0.01
        r2 = torch.randn(r2_shape, device=image.device, dtype=image.dtype) * 0.01

        return r1, r2


    # ========================================================
    # TOP-DOWN GENERATIVE PREDICTIONS
    # ========================================================

    def predict_image(self, r1):
        return self.level1["deconv"](r1)

    def predict_r1(self, r2):
        return self.level2["deconv"](r2)

    # ========================================================
    # PREDICTION ERRORS
    #
    # e0 = I - U1 r1
    # e1 = r1 - U2 r2
    # ========================================================

    def prediction_errors(self, image, r1, r2):
        predicted_image = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)

        e0 = image.unsqueeze(0) - predicted_image
        e1 = r1 - predicted_r1

        return e0, e1, predicted_r1

    # ========================================================
    # RAO EQ. 7
    # REPRESENTATION DYNAMICS
    # ========================================================

    def state_updates(self, image, r1, r2):
        e0, e1, predicted_r1 = self.prediction_errors(image, r1, r2)

        # ====================================================
        # LEVEL 1
        # ====================================================

        dr1 = (self.k1 / self.sigma_bu_sq) * self.level1["conv"](e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        # ====================================================
        # LEVEL 2
        # ====================================================

        dr2 = (self.k1 / self.sigma_bu_sq) * self.level2["conv"](e1)
        dr2 -= self.k1 * self.alpha2 * r2

        return dr1, dr2

    # ========================================================
    # PREDICTIVE-CODING INFERENCE
    #
    # FIXED self.inference_dt
    # ========================================================

    @torch.no_grad()
    def infer(self, image):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2 = self.initialize_states(image)

        step = 0
        converged = False

        while True:
            old_r1 = r1.clone()
            old_r2 = r2.clone()

            dr1, dr2 = self.state_updates(image, r1, r2)

            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2

            step += 1

            if not torch.isfinite(r1).all() or not torch.isfinite(r2).all():
                raise FloatingPointError("PC-CNN inference became non-finite.")

            if torch.allclose(r1, old_r1) and torch.allclose(r2, old_r2):
                converged = True
                break

            if self.inference_steps is not None and step >= self.inference_steps:
                break

        return r1, r2, step, converged

    # ========================================================
    # RAO EQ. 9
    # LOCAL GENERATIVE WEIGHT LEARNING
    # ========================================================

    @torch.no_grad()
    def update_weights(self, image, r1, r2):
        e0, e1, _ = self.prediction_errors(image, r1, r2)

        # Convolutional equivalent of local
        # prediction-error × representation
        dW1 = conv2d_weight(e0, self.level1["conv"].weight.shape, r1, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)
        dW2 = conv2d_weight(e1, self.level2["conv"].weight.shape, r2, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)

        # Spatial averaging for convolutional weight sharing
        dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])
        dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])

        # Rao Eq. 9
        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight
        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight

        # Numerical integration
        self.level1["conv"].weight.add_(self.weight_dt * dW1)
        self.level2["conv"].weight.add_(self.weight_dt * dW2)

        if not torch.isfinite(self.level1["conv"].weight).all() or not torch.isfinite(self.level2["conv"].weight).all():
            raise FloatingPointError("PC-CNN weights became non-finite.")

    # ========================================================
    # RAO k2 DECAY
    # ========================================================

    def update_k2(self):
        self.training_input_count += 1

        if self.training_input_count % self.k2_decay_interval == 0:
            self.k2 = self.k2 / self.k2_decay_factor

    # ========================================================
    # RECONSTRUCTION
    # ========================================================

    @torch.no_grad()
    def reconstruct(self, image):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2, step, converged = self.infer(image)

        reconstruction = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        reconstruction_r2 = self.predict_image(predicted_r1)

        return reconstruction.squeeze(0), predicted_r1.squeeze(0), reconstruction_r2.squeeze(0), r2.squeeze(0), step, converged

    # ========================================================
    # EXTRACT r2 FOR ONE IMAGE
    # ========================================================

    @torch.no_grad()
    def extract_r2(self, image):
        _, r2, _, _ = self.infer(image)

        return r2.squeeze(0)

    # ========================================================
    # EXTRACT r2 FOR BATCH
    # ========================================================

    @torch.no_grad()
    def extract_r2_batch(self, images):
        features = []

        for image in images:
            r2 = self.extract_r2(image)
            features.append(r2)

        return torch.stack(features)

    # ========================================================
    # CLASSIFIER FEATURES
    #
    # r2 → 7×7 pool → flatten
    # ========================================================

    @torch.no_grad()
    def classifier_features(self, images):
        r2 = self.extract_r2_batch(images)

        pooled = self.classifier_pool(r2)
        features = pooled.flatten(1)

        return features

    # ========================================================
    # CLASSIFICATION
    #
    # NO RECONSTRUCTION USED HERE
    # NO BACKPROPAGATION
    # ========================================================

    @torch.no_grad()
    def classify(self, images):
        features = self.classifier_features(images)

        logits = self.classifier(features)

        return logits

    # ========================================================
    # LOCAL SUPERVISED CLASSIFIER LEARNING
    #
    # output_error = target - probability
    #
    # dW = output_error^T × features
    #
    # NO AUTOGRAD
    # NO BACKPROPAGATION
    # NO OPTIMIZER
    # ========================================================

    @torch.no_grad()
    def update_classifier(self, images, labels, learning_rate=0.01):
        features = self.classifier_features(images)

        logits = self.classifier(features)
        probabilities = torch.softmax(logits, dim=1)

        # One-hot target
        targets = torch.zeros_like(probabilities)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)

        # Local output error
        output_error = targets - probabilities

        # Local classifier update
        dW = output_error.T @ features
        dW = dW / features.shape[0]

        db = output_error.mean(dim=0)

        self.classifier.weight.add_(learning_rate * dW)
        self.classifier.bias.add_(learning_rate * db)

        # Loss only for reporting
        true_probabilities = probabilities[
            torch.arange(labels.shape[0], device=labels.device),
            labels
        ]

        loss = -torch.log(true_probabilities + 1e-8).mean().item()

        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()

        return loss, correct, labels.shape[0]

    # ========================================================
    # PC-CNN BACKBONE LOCAL LEARNING
    # ========================================================

    @torch.no_grad()
    def forward(self, image):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        # Predictive-coding inference
        r1, r2, step, converged = self.infer(image)

        # Prediction errors
        e0, e1, _ = self.prediction_errors(image, r1, r2)

        error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()

        # Update PC weights ONLY during training
        if self.training:
            # Local PC-CNN weight learning
            self.update_weights(image, r1, r2)

            # Rao k2 schedule
            self.update_k2()

        return {
            "inference_steps": step,
            "converged": converged,
            "error_0": error_0,
            "error_1": error_1,
            "k2": self.k2,
            "training_input_count": self.training_input_count
        }