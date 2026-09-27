import torch
import torch.nn as nn
from torch.nn.grad import conv2d_weight


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
        # RAO k2 SCHEDULE
        # ====================================================

        self.k2_decay_interval = 40
        self.k2_decay_factor = 1.015
        self.training_input_count = 0

        # ====================================================
        # INFERENCE
        # ====================================================

        self.inference_steps = inference_steps
        self.inference_dt = 0.1

        # Rao Eq. 9 numerical integration
        self.weight_dt = 0.01

        # ====================================================
        # LEVEL 1
        # Image [3,224,224] <-> r1 [32,112,112]
        # ====================================================

        self.level1 = nn.ModuleDict({
            "conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # LEVEL 2
        # r1 [32,112,112] <-> r2 [128,56,56]
        # ====================================================

        self.level2 = nn.ModuleDict({
            "conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # CLASSIFICATION READOUT
        # ====================================================

        self.classifier_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(128 * 7 * 7, num_classes, bias=True)

        # ====================================================
        # INITIALIZATION
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
        # NO AUTOGRAD
        # ====================================================

        self.level1["conv"].weight.requires_grad_(False)
        self.level2["conv"].weight.requires_grad_(False)

        self.classifier.weight.requires_grad_(False)
        self.classifier.bias.requires_grad_(False)

    # ========================================================
    # RANDOM LATENT INITIALIZATION
    # ========================================================

    def initialize_states(self, image):
        image_batch = image.unsqueeze(0)

        r1_shape = self.level1["conv"](image_batch).shape
        r2_shape = self.level2["conv"](torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape

        r1 = torch.randn(r1_shape, device=image.device, dtype=image.dtype) * 0.01
        r2 = torch.randn(r2_shape, device=image.device, dtype=image.dtype) * 0.01

        return r1, r2

    # ========================================================
    # GENERATIVE PREDICTIONS
    # ========================================================

    def predict_image(self, r1):
        return self.level1["deconv"](r1)

    def predict_r1(self, r2):
        return self.level2["deconv"](r2)

    # ========================================================
    # PREDICTION ERRORS
    # ========================================================

    def prediction_errors(self, image, r1, r2):
        predicted_image = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)

        e0 = image.unsqueeze(0) - predicted_image
        e1 = r1 - predicted_r1

        return e0, e1, predicted_r1

    # ========================================================
    # RAO EQ. 7
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

        if not torch.isfinite(self.level1["conv"].weight).all() or not torch.isfinite(self.level2["conv"].weight).all():
            raise FloatingPointError("PC-CNN weights became non-finite.")

    # ========================================================
    # k2 UPDATE
    # ========================================================

    def update_k2(self):
        self.training_input_count += 1

        if self.training_input_count % self.k2_decay_interval == 0:
            self.k2 = self.k2 / self.k2_decay_factor

    # ========================================================
    # CLASSIFIER FEATURES FROM EXISTING r2
    #
    # IMPORTANT:
    # DOES NOT RUN INFERENCE AGAIN
    # ========================================================

    @torch.no_grad()
    def classifier_features_from_r2(self, r2):
        pooled = self.classifier_pool(r2)
        features = pooled.flatten(1)

        return features

    # ========================================================
    # CLASSIFY FROM EXISTING r2
    # ========================================================

    @torch.no_grad()
    def classify_from_r2(self, r2):
        features = self.classifier_features_from_r2(r2)
        logits = self.classifier(features)

        return logits

    # ========================================================
    # LOCAL CLASSIFIER UPDATE FROM SAME r2
    # ========================================================

    @torch.no_grad()
    def update_classifier_from_r2(self, r2, label, learning_rate=0.01):
        features = self.classifier_features_from_r2(r2)

        logits = self.classifier(features)
        probabilities = torch.softmax(logits, dim=1)

        label = label.view(1)

        targets = torch.zeros_like(probabilities)
        targets.scatter_(1, label.unsqueeze(1), 1.0)

        output_error = targets - probabilities

        dW = output_error.T @ features
        db = output_error.squeeze(0)

        self.classifier.weight.add_(learning_rate * dW)
        self.classifier.bias.add_(learning_rate * db)

        true_probability = probabilities[0, label.item()]
        loss = -torch.log(true_probability + 1e-8).item()

        prediction = logits.argmax(dim=1)
        correct = int(prediction.item() == label.item())

        return loss, correct, prediction.item()

    # ========================================================
    # JOINT LOCAL TRAINING STEP
    #
    # ONE INFERENCE
    #
    # PC backbone:
    #   prediction errors -> Rao Eq. 9
    #
    # classifier:
    #   r2 -> local supervised readout update
    #
    # NO BACKPROPAGATION
    # ========================================================

    @torch.no_grad()
    def train_step(self, image, label, classifier_learning_rate=0.01):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)
        label = label.to(device=image.device, dtype=torch.long)

        # ----------------------------------------------------
        # 1. Predictive-coding inference ONCE
        # ----------------------------------------------------

        r1, r2, step, converged = self.infer(image)

        # ----------------------------------------------------
        # 2. Measure PC prediction errors
        # ----------------------------------------------------

        e0, e1, _ = self.prediction_errors(image, r1, r2)

        error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()

        # ----------------------------------------------------
        # 3. Classification using SAME r2
        # ----------------------------------------------------

        classification_loss, classification_correct, prediction = self.update_classifier_from_r2(
            r2,
            label,
            learning_rate=classifier_learning_rate
        )

        # ----------------------------------------------------
        # 4. PC-CNN local weight update
        # ----------------------------------------------------

        self.update_weights(image, r1, r2)
        self.update_k2()

        return {
            "image_error": error_0,
            "r1_error": error_1,
            "classification_loss": classification_loss,
            "classification_correct": classification_correct,
            "prediction": prediction,
            "label": label.item(),
            "inference_steps": step,
            "converged": converged,
            "k2": self.k2
        }

    # ========================================================
    # NORMAL CLASSIFICATION FOR EVALUATION
    # ========================================================

    @torch.no_grad()
    def classify(self, images):
        logits_list = []

        for image in images:
            _, r2, _, _ = self.infer(image)

            logits = self.classify_from_r2(r2)

            logits_list.append(logits.squeeze(0))

        return torch.stack(logits_list)

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