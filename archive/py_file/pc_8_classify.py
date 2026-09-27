import torch
import torch.nn as nn

from torch.nn.grad import conv2d_weight


# ============================================================
# HIERARCHICAL RAO-STYLE PC-CNN
# CONVOLUTIONAL ADAPTATION OF RAO & BALLARD
# NO BACKPROPAGATION
# ============================================================

class HierarchicalPCCNN(nn.Module):

    def __init__(self, num_classes, inference_steps=100, convergence_tolerance=1e-5):
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
        # NUMERICAL INTEGRATION
        # ====================================================

        self.inference_steps = inference_steps
        self.inference_dt = 0.1
        self.weight_dt = 0.01
        self.convergence_tolerance = convergence_tolerance

        # ====================================================
        # LEVEL 1
        #
        # Tiny ImageNet 64×64:
        #
        # Image [3,64,64] -> r1 [32,32,32]
        # r1 [32,32,32] -> predicted image [3,64,64]
        # ====================================================

        self.level1 = nn.ModuleDict({
            "conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # LEVEL 2
        #
        # r1 [32,32,32] -> r2 [128,16,16]
        # r2 [128,16,16] -> predicted r1 [32,32,32]
        # ====================================================

        self.level2 = nn.ModuleDict({
            "conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # CLASSIFICATION READOUT
        #
        # r2 [128,16,16]
        #       ↓
        # AdaptiveAvgPool2d(4×4)
        #       ↓
        # [128,4,4]
        #       ↓
        # 2048 features
        #       ↓
        # Linear classifier
        # ====================================================

        self.classifier_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(128 * 4 * 4, num_classes, bias=True)

        # ====================================================
        # WEIGHT INITIALIZATION
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
    # ========================================================

    def initialize_states(self, image):
        image_batch = image.unsqueeze(0)

        r1_shape = self.level1["conv"](image_batch).shape
        r2_shape = self.level2["conv"](torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape

        r1 = torch.zeros(r1_shape, device=image.device, dtype=image.dtype)
        r2 = torch.zeros(r2_shape, device=image.device, dtype=image.dtype)

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
    # e0 = I - U1r1
    # e1 = r1 - U2r2
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

        dr1 = (self.k1 / self.sigma_bu_sq) * self.level1["conv"](e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        dr2 = (self.k1 / self.sigma_bu_sq) * self.level2["conv"](e1)
        dr2 -= self.k1 * self.alpha2 * r2

        return dr1, dr2

    # ========================================================
    # PREDICTIVE-CODING INFERENCE
    # ========================================================

    @torch.no_grad()
    def infer(self, image, return_diagnostics=False):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2 = self.initialize_states(image)

        step = 0
        converged = False

        change_r1 = float("inf")
        change_r2 = float("inf")

        while True:
            old_r1 = r1.clone()
            old_r2 = r2.clone()

            dr1, dr2 = self.state_updates(image, r1, r2)

            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2

            step += 1

            if not torch.isfinite(r1).all():
                raise FloatingPointError("PC-CNN r1 inference became non-finite.")

            if not torch.isfinite(r2).all():
                raise FloatingPointError("PC-CNN r2 inference became non-finite.")

            change_r1 = torch.mean(torch.abs(r1 - old_r1)).item()
            change_r2 = torch.mean(torch.abs(r2 - old_r2)).item()

            if change_r1 < self.convergence_tolerance and change_r2 < self.convergence_tolerance:
                converged = True
                break

            if self.inference_steps is not None and step >= self.inference_steps:
                break

        if return_diagnostics:
            return r1, r2, step, converged, change_r1, change_r2

        return r1, r2, step, converged

    # ========================================================
    # RAO EQ. 9
    # LOCAL GENERATIVE WEIGHT LEARNING
    # ========================================================

    @torch.no_grad()
    def update_weights(self, image, r1, r2):
        e0, e1, _ = self.prediction_errors(image, r1, r2)

        dW1 = conv2d_weight(e0, self.level1["conv"].weight.shape, r1, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)
        dW2 = conv2d_weight(e1, self.level2["conv"].weight.shape, r2, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)

        dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])
        dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])

        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight
        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight

        self.level1["conv"].weight.add_(self.weight_dt * dW1)
        self.level2["conv"].weight.add_(self.weight_dt * dW2)

        if not torch.isfinite(self.level1["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-1 weights became non-finite.")

        if not torch.isfinite(self.level2["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-2 weights became non-finite.")

    # ========================================================
    # RAO k2 DECAY
    # ========================================================

    def update_k2(self):
        self.training_input_count += 1

        if self.training_input_count % self.k2_decay_interval == 0:
            self.k2 = self.k2 / self.k2_decay_factor

    # ========================================================
    # IMAGE RECONSTRUCTION
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
    # r2 -> 4×4 pooling -> flatten
    # ========================================================

    @torch.no_grad()
    def classifier_features(self, images):
        r2 = self.extract_r2_batch(images)

        pooled = self.classifier_pool(r2)
        features = pooled.flatten(1)

        return features

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    @torch.no_grad()
    def classify(self, images):
        features = self.classifier_features(images)
        logits = self.classifier(features)

        return logits

    # ========================================================
    # LOCAL SUPERVISED CLASSIFIER LEARNING
    # ========================================================

    @torch.no_grad()
    def update_classifier(self, images, labels, learning_rate=0.01):
        labels = labels.to(device=self.classifier.weight.device, dtype=torch.long)

        features = self.classifier_features(images)

        logits = self.classifier(features)
        probabilities = torch.softmax(logits, dim=1)

        targets = torch.zeros_like(probabilities)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)

        output_error = targets - probabilities

        dW = output_error.T @ features
        dW = dW / features.shape[0]

        db = output_error.mean(dim=0)

        self.classifier.weight.add_(learning_rate * dW)
        self.classifier.bias.add_(learning_rate * db)

        true_probabilities = probabilities[torch.arange(labels.shape[0], device=labels.device), labels]

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

        r1, r2, step, converged, change_r1, change_r2 = self.infer(image, return_diagnostics=True)

        e0, e1, _ = self.prediction_errors(image, r1, r2)

        error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()

        if self.training:
            self.update_weights(image, r1, r2)
            self.update_k2()

        return {
            "inference_steps": step,
            "converged": converged,
            "final_change_r1": change_r1,
            "final_change_r2": change_r2,
            "error_0": error_0,
            "error_1": error_1,
            "k2": self.k2,
            "training_input_count": self.training_input_count
        }