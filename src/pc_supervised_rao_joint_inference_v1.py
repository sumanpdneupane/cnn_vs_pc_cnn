import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight


class HierarchicalPCCNN(nn.Module):
    """Hierarchical Rao-style predictive-coding CNN with joint supervised inference.

    Training:
        image + label -> one recurrent inference process in which predictive errors
        (e0/e1/e2) and class error act together from the first inference step.
        All parameters are updated manually with local rules; autograd is disabled.

    Validation/Test:
        image only -> predictive-coding inference -> classifier prediction.
        The true label is never passed into model inference.
    """

    def __init__(self, num_classes, input_size=64, inference_steps=50, sigma_class_sq=0.5, class_learning_rate=0.01, lambda_class=0.001, k2_decay_interval=160):
        super().__init__()

        if input_size % 8 != 0: raise ValueError("input_size must be divisible by 8.")
        if sigma_class_sq <= 0: raise ValueError("sigma_class_sq must be greater than 0.")
        if class_learning_rate <= 0: raise ValueError("class_learning_rate must be greater than 0.")
        if lambda_class < 0: raise ValueError("lambda_class must be >= 0.")
        if k2_decay_interval <= 0: raise ValueError("k2_decay_interval must be greater than 0.")

        self.num_classes = num_classes
        self.input_size = input_size
        self.class_learning_rate = class_learning_rate
        self.lambda_class = lambda_class

        self.k1 = 0.5
        self.k2 = 1.0
        self.sigma_bu_sq = 1.0
        self.sigma_td_sq = 10.0
        self.sigma_class_sq = sigma_class_sq
        self.alpha1 = 0.001
        self.alpha2 = 0.05
        self.alpha3 = 0.05
        self.lambda_u = 0.02

        self.k2_decay_interval = k2_decay_interval
        self.k2_decay_factor = 1.015
        self.training_input_count = 0

        self.inference_steps = inference_steps
        self.inference_dt = 0.1
        self.weight_dt = 0.01
        self.convergence_tolerance = 1e-5

        self.r1_size = input_size // 2
        self.r2_size = input_size // 4
        self.r3_size = input_size // 8
        if self.r3_size % 2 != 0: raise ValueError("r3 spatial size must be divisible by 2 for 2x2 average pooling.")

        self.level1 = nn.ModuleDict({
            "conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        })
        self.level2 = nn.ModuleDict({
            "conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        })
        self.level3 = nn.ModuleDict({
            "conv": nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        })

        self.classifier_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.classifier_spatial_size = self.r3_size // 2
        self.classifier_feature_dim = 256 * self.classifier_spatial_size * self.classifier_spatial_size
        self.classifier = nn.Linear(self.classifier_feature_dim, num_classes, bias=True)

        nn.init.normal_(self.level1["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level2["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level3["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

        self.level1["deconv"].weight = self.level1["conv"].weight
        self.level2["deconv"].weight = self.level2["conv"].weight
        self.level3["deconv"].weight = self.level3["conv"].weight

        for parameter in self.parameters(): parameter.requires_grad_(False)

    def activation(self, state): return torch.tanh(state)

    def activation_derivative(self, state):
        activated = torch.tanh(state)
        return 1.0 - activated * activated

    def label_value(self, label):
        value = int(label.item()) if torch.is_tensor(label) else int(label)
        if value < 0 or value >= self.num_classes: raise ValueError(f"Class label {value} is outside the valid range 0-{self.num_classes - 1}.")
        return value

    def validate_image_size(self, image):
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            raise ValueError(f"Expected image size {self.input_size}x{self.input_size}, but received {image.shape[-2]}x{image.shape[-1]}.")

    def initialize_states(self, image):
        self.validate_image_size(image)

        image_batch = image.unsqueeze(0)
        r1_shape = self.level1["conv"](image_batch).shape

        dummy_r1 = torch.zeros(r1_shape, device=image.device, dtype=image.dtype)
        r2_shape = self.level2["conv"](dummy_r1).shape

        dummy_r2 = torch.zeros(r2_shape, device=image.device, dtype=image.dtype)
        r3_shape = self.level3["conv"](dummy_r2).shape

        r1 = torch.zeros(r1_shape, device=image.device, dtype=image.dtype)
        r2 = torch.zeros(r2_shape, device=image.device, dtype=image.dtype)
        r3 = torch.zeros(r3_shape, device=image.device, dtype=image.dtype)
        return r1, r2, r3

    def predict_image(self, r1): return self.level1["deconv"](self.activation(r1))
    def predict_r1(self, r2): return self.level2["deconv"](self.activation(r2))
    def predict_r2(self, r3): return self.level3["deconv"](self.activation(r3))

    def prediction_errors(self, image, r1, r2, r3):
        predicted_image = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        predicted_r2 = self.predict_r2(r3)
        e0 = image.unsqueeze(0) - predicted_image
        e1 = r1 - predicted_r1
        e2 = r2 - predicted_r2
        return e0, e1, e2, predicted_r1, predicted_r2

    def class_features(self, r3): return self.classifier_pool(self.activation(r3)).flatten(1)

    def class_prediction(self, r3):
        features = self.class_features(r3)
        scores = self.classifier(features)
        probabilities = torch.softmax(scores, dim=1)
        return features, scores, probabilities

    def class_prediction_error(self, r3, label):
        features, scores, probabilities = self.class_prediction(r3)
        label_value = self.label_value(label)
        e_class = -probabilities.clone()
        e_class[0, label_value] += 1.0
        return features, scores, probabilities, e_class

    def class_feedback(self, e_class, r3):
        pooled_feedback = e_class @ self.classifier.weight
        pooled_feedback = pooled_feedback.view(1, 256, self.classifier_spatial_size, self.classifier_spatial_size)
        feedback = F.interpolate(pooled_feedback, scale_factor=2, mode="nearest") / 4.0
        return feedback * self.activation_derivative(r3)

    def state_updates(self, image, r1, r2, r3, label=None):
        e0, e1, e2, predicted_r1, predicted_r2 = self.prediction_errors(image, r1, r2, r3)

        dr1 = (self.k1 / self.sigma_bu_sq) * self.activation_derivative(r1) * self.level1["conv"](e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        dr2 = (self.k1 / self.sigma_bu_sq) * self.activation_derivative(r2) * self.level2["conv"](e1)
        dr2 += (self.k1 / self.sigma_td_sq) * (predicted_r2 - r2)
        dr2 -= self.k1 * self.alpha2 * r2

        dr3 = (self.k1 / self.sigma_bu_sq) * self.activation_derivative(r3) * self.level3["conv"](e2)
        dr3 -= self.k1 * self.alpha3 * r3

        if label is not None:
            _, _, _, e_class = self.class_prediction_error(r3, label)
            dr3 += (self.k1 / self.sigma_class_sq) * self.class_feedback(e_class, r3)

        return dr1, dr2, dr3

    @torch.no_grad()
    def infer(self, image, label=None, return_diagnostics=False):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)
        if label is not None: self.label_value(label)
        r1, r2, r3 = self.initialize_states(image)
        step = 0
        converged = False
        change_r1 = change_r2 = change_r3 = float("inf")

        while True:
            old_r1, old_r2, old_r3 = r1.clone(), r2.clone(), r3.clone()
            dr1, dr2, dr3 = self.state_updates(image, r1, r2, r3, label=label)
            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2
            r3 = r3 + self.inference_dt * dr3
            step += 1

            if not torch.isfinite(r1).all() or not torch.isfinite(r2).all() or not torch.isfinite(r3).all():
                mode = "joint supervised" if label is not None else "free"
                raise FloatingPointError(f"PC-CNN {mode} inference became non-finite.")

            change_r1 = torch.mean(torch.abs(r1 - old_r1)).item()
            change_r2 = torch.mean(torch.abs(r2 - old_r2)).item()
            change_r3 = torch.mean(torch.abs(r3 - old_r3)).item()

            if change_r1 < self.convergence_tolerance and change_r2 < self.convergence_tolerance and change_r3 < self.convergence_tolerance:
                converged = True
                break
            if self.inference_steps is not None and step >= self.inference_steps: break

        if return_diagnostics: return r1, r2, r3, step, converged, change_r1, change_r2, change_r3
        return r1, r2, r3, step, converged

    @torch.no_grad()
    def update_weights(self, image, r1, r2, r3, label):
        e0, e1, e2, _, _ = self.prediction_errors(image, r1, r2, r3)
        a1, a2, a3 = self.activation(r1), self.activation(r2), self.activation(r3)

        dW1 = conv2d_weight(e0, self.level1["conv"].weight.shape, a1, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)
        dW2 = conv2d_weight(e1, self.level2["conv"].weight.shape, a2, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)
        dW3 = conv2d_weight(e2, self.level3["conv"].weight.shape, a3, stride=self.level3["conv"].stride, padding=self.level3["conv"].padding)

        dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])
        dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])
        dW3 = dW3 / (r3.shape[-2] * r3.shape[-1])

        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight
        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight
        dW3 = (self.k2 / self.sigma_bu_sq) * dW3 - self.k2 * self.lambda_u * self.level3["conv"].weight

        self.level1["conv"].weight.add_(self.weight_dt * dW1)
        self.level2["conv"].weight.add_(self.weight_dt * dW2)
        self.level3["conv"].weight.add_(self.weight_dt * dW3)

        features, _, _, e_class = self.class_prediction_error(r3, label)
        dWc = e_class.T @ features - self.lambda_class * self.classifier.weight
        dbc = e_class.squeeze(0)
        self.classifier.weight.add_(self.class_learning_rate * dWc)
        self.classifier.bias.add_(self.class_learning_rate * dbc)

        for name, parameter in [("U1", self.level1["conv"].weight), ("U2", self.level2["conv"].weight), ("U3", self.level3["conv"].weight), ("Wc", self.classifier.weight), ("bc", self.classifier.bias)]:
            if not torch.isfinite(parameter).all(): raise FloatingPointError(f"PC-CNN {name} became non-finite.")

    def update_k2(self):
        self.training_input_count += 1
        if self.training_input_count % self.k2_decay_interval == 0: self.k2 = self.k2 / self.k2_decay_factor

    @torch.no_grad()
    def classify(self, image):
        r1, r2, r3, step, converged, change_r1, change_r2, change_r3 = self.infer(image, label=None, return_diagnostics=True)
        features, scores, probabilities = self.class_prediction(r3)
        prediction = scores.argmax(dim=1)
        return {"scores": scores, "probabilities": probabilities, "prediction": prediction, "features": features, "r1": r1, "r2": r2, "r3": r3, "inference_steps": step, "converged": converged, "final_change_r1": change_r1, "final_change_r2": change_r2, "final_change_r3": change_r3}

    @torch.no_grad()
    def classify_batch(self, images):
        results = [self.classify(image) for image in images]
        return {
            "scores": torch.stack([result["scores"].squeeze(0) for result in results]),
            "probabilities": torch.stack([result["probabilities"].squeeze(0) for result in results]),
            "predictions": torch.stack([result["prediction"].squeeze(0) for result in results]),
            "inference_steps": [result["inference_steps"] for result in results],
            "converged": [result["converged"] for result in results],
            "final_change_r1": [result["final_change_r1"] for result in results],
            "final_change_r2": [result["final_change_r2"] for result in results],
            "final_change_r3": [result["final_change_r3"] for result in results],
        }

    @torch.no_grad()
    def reconstruct(self, image):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)
        r1, r2, r3, step, converged = self.infer(image, label=None)
        reconstruction = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        predicted_r2 = self.predict_r2(r3)
        _, scores, probabilities = self.class_prediction(r3)
        predicted_class = scores.argmax(dim=1)
        return {"reconstruction": reconstruction.squeeze(0), "predicted_r1": predicted_r1.squeeze(0), "predicted_r2": predicted_r2.squeeze(0), "r1": r1.squeeze(0), "r2": r2.squeeze(0), "r3": r3.squeeze(0), "probabilities": probabilities.squeeze(0), "predicted_class": predicted_class.item(), "inference_steps": step, "converged": converged}

    @torch.no_grad()
    def forward(self, image, label=None):
        if self.training and label is None: raise ValueError("A class label is required during joint supervised predictive-coding training.")

        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)
        r1, r2, r3, steps, converged, change_r1, change_r2, change_r3 = self.infer(image, label=label, return_diagnostics=True)
        e0, e1, e2, _, _ = self.prediction_errors(image, r1, r2, r3)
        features, scores, probabilities = self.class_prediction(r3)

        error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()
        error_2 = torch.sqrt(torch.mean(e2 ** 2)).item()
        prediction = scores.argmax(dim=1).item()

        class_error = None
        class_loss = None
        true_probability = None
        if label is not None:
            _, _, probabilities, e_class = self.class_prediction_error(r3, label)
            class_error = torch.sqrt(torch.mean(e_class ** 2)).item()
            label_value = self.label_value(label)
            true_probability = probabilities[0, label_value]
            class_loss = -torch.log(true_probability.clamp_min(1e-8)).item()

        if self.training:
            self.update_weights(image, r1, r2, r3, label)
            self.update_k2()

        return {
            "inference_steps": steps,
            "converged": converged,
            "final_change_r1": change_r1,
            "final_change_r2": change_r2,
            "final_change_r3": change_r3,
            "error_0": error_0,
            "error_1": error_1,
            "error_2": error_2,
            "class_error": class_error,
            "class_loss": class_loss,
            "true_probability": true_probability.item() if true_probability is not None else None,
            "prediction": prediction,
            "probabilities": probabilities.squeeze(0),
            "features": features.squeeze(0),
            "r1": r1.squeeze(0),
            "r2": r2.squeeze(0),
            "r3": r3.squeeze(0),
            "k2": self.k2,
            "training_input_count": self.training_input_count,
        }
