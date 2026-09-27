import torch
import torch.nn as nn

from torch.nn.grad import conv2d_weight


# ============================================================
# SUPERVISED HIERARCHICAL RAO-STYLE PC-CNN
#
# Image <-> r1 <-> r2 -> supervised class prediction
#
# Lower hierarchy:
# Rao-style predictive coding
#
# Top level:
# Local supervised prediction-error extension
#
# NO AUTOGRAD
# NO BACKPROPAGATION
# NO OPTIMIZER
# ============================================================

class HierarchicalPCCNN(nn.Module):

    def __init__(self, num_classes, input_size=64, inference_steps=100, convergence_tolerance=1e-5, sigma_class_sq=1.0, class_pool_size=4):
        super().__init__()

        if input_size % 4 != 0:
            raise ValueError("input_size must be divisible by 4.")

        if sigma_class_sq <= 0:
            raise ValueError("sigma_class_sq must be greater than 0.")

        self.num_classes = num_classes
        self.input_size = input_size
        self.class_pool_size = class_pool_size

        # ====================================================
        # RAO PARAMETERS
        # ====================================================

        self.k1 = 0.5
        self.k2 = 1.0

        self.sigma_bu_sq = 1.0
        self.sigma_td_sq = 10.0

        # Supervised top-level prediction-error variance
        self.sigma_class_sq = sigma_class_sq

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
        # NUMERICAL INTEGRATION
        # ====================================================

        self.inference_steps = inference_steps
        self.inference_dt = 0.1
        self.weight_dt = 0.01
        self.convergence_tolerance = convergence_tolerance

        # ====================================================
        # SPATIAL DIMENSIONS
        #
        # 64 -> 32 -> 16
        #
        # 224 -> 112 -> 56
        # ====================================================

        self.r1_size = input_size // 2
        self.r2_size = input_size // 4

        if self.r2_size % class_pool_size != 0:
            raise ValueError("r2 spatial size must be divisible by class_pool_size.")

        self.class_pool_kernel = self.r2_size // class_pool_size

        # ====================================================
        # LEVEL 1
        #
        # Image <-> r1
        # ====================================================

        self.level1 = nn.ModuleDict({"conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)})

        # ====================================================
        # LEVEL 2
        #
        # r1 <-> r2
        # ====================================================

        self.level2 = nn.ModuleDict({"conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False), "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)})

        # ====================================================
        # LEVEL 3
        #
        # r2
        #  ↓
        # 4x4 average pooling
        #  ↓
        # supervised class prediction
        #
        # W3 is locally updated from class prediction error.
        # ====================================================

        self.class_pool = nn.AvgPool2d(kernel_size=self.class_pool_kernel, stride=self.class_pool_kernel)

        self.level3 = nn.ModuleDict({"linear": nn.Linear(128 * class_pool_size * class_pool_size, num_classes, bias=False)})

        # ====================================================
        # WEIGHT INITIALIZATION
        # ====================================================

        nn.init.normal_(self.level1["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level2["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level3["linear"].weight, mean=0.0, std=0.01)

        # ====================================================
        # TIED GENERATIVE / ERROR-CORRECTION WEIGHTS
        # ====================================================

        self.level1["deconv"].weight = self.level1["conv"].weight
        self.level2["deconv"].weight = self.level2["conv"].weight

        # ====================================================
        # NO AUTOGRAD
        # ====================================================

        self.level1["conv"].weight.requires_grad_(False)
        self.level2["conv"].weight.requires_grad_(False)
        self.level3["linear"].weight.requires_grad_(False)

    # ========================================================
    # IMAGE SIZE CHECK
    # ========================================================

    def validate_image_size(self, image):
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            raise ValueError(f"Expected image size {self.input_size}x{self.input_size}, but received {image.shape[-2]}x{image.shape[-1]}.")

    # ========================================================
    # INITIAL REPRESENTATIONS
    # ========================================================

    def initialize_states(self, image):
        self.validate_image_size(image)

        image_batch = image.unsqueeze(0)

        r1_shape = self.level1["conv"](image_batch).shape
        r2_shape = self.level2["conv"](torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape

        r1 = torch.zeros(r1_shape, device=image.device, dtype=image.dtype)
        r2 = torch.zeros(r2_shape, device=image.device, dtype=image.dtype)

        return r1, r2

    # ========================================================
    # VISUAL GENERATIVE PREDICTIONS
    # ========================================================

    def predict_image(self, r1):
        return self.level1["deconv"](r1)

    def predict_r1(self, r2):
        return self.level2["deconv"](r2)

    # ========================================================
    # VISUAL PREDICTION ERRORS
    #
    # e0 = image - U1 r1
    #
    # e1 = r1 - U2 r2
    # ========================================================

    def prediction_errors(self, image, r1, r2):
        predicted_image = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)

        e0 = image.unsqueeze(0) - predicted_image
        e1 = r1 - predicted_r1

        return e0, e1, predicted_r1

    # ========================================================
    # CLASSIFICATION FEATURES
    #
    # r2 -> 4x4 pooling -> flatten
    # ========================================================

    def class_features(self, r2):
        pooled = self.class_pool(r2)
        features = pooled.flatten(1)

        return features

    # ========================================================
    # CLASS PREDICTION
    # ========================================================

    def class_prediction(self, r2):
        features = self.class_features(r2)

        scores = self.level3["linear"](features)
        probabilities = torch.softmax(scores, dim=1)

        return features, scores, probabilities

    # ========================================================
    # CREATE ONE-HOT TARGET
    # ========================================================

    def create_target(self, label, device, dtype):
        label_value = int(label.item()) if torch.is_tensor(label) else int(label)

        if label_value < 0 or label_value >= self.num_classes:
            raise ValueError(f"Class label {label_value} is outside the valid range 0-{self.num_classes - 1}.")

        target = torch.zeros(1, self.num_classes, device=device, dtype=dtype)
        target[0, label_value] = 1.0

        return target

    # ========================================================
    # SUPERVISED TOP-LEVEL PREDICTION ERROR
    #
    # e2 = target - probability
    #
    # This is local to the top layer.
    # ========================================================

    def class_prediction_error(self, r2, label):
        features, scores, probabilities = self.class_prediction(r2)

        target = self.create_target(label, probabilities.device, probabilities.dtype)

        e2 = target - probabilities

        return features, scores, probabilities, target, e2

    # ========================================================
    # TOP-LEVEL ERROR FEEDBACK TO r2
    #
    # e2
    #  ↓
    # W3^T e2
    #  ↓
    # pooled-r2 correction
    #  ↓
    # spatial r2 correction
    #
    # This is the local supervised signal entering the PC
    # hierarchy.
    # ========================================================

    def class_feedback(self, e2, r2):
        pooled_feedback = e2 @ self.level3["linear"].weight

        pooled_feedback = pooled_feedback.view(r2.shape[0], 128, self.class_pool_size, self.class_pool_size)

        feedback = pooled_feedback.repeat_interleave(self.class_pool_kernel, dim=-2)
        feedback = feedback.repeat_interleave(self.class_pool_kernel, dim=-1)

        feedback = feedback / float(self.class_pool_kernel * self.class_pool_kernel)

        return feedback

    # ========================================================
    # RAO-STYLE REPRESENTATION DYNAMICS
    #
    # Training:
    #
    # class error locally corrects r2
    #
    # Testing:
    #
    # no target is available, so visual PC inference runs
    # without supervised feedback.
    # ========================================================

    def state_updates(self, image, r1, r2, label=None):
        e0, e1, predicted_r1 = self.prediction_errors(image, r1, r2)

        # ====================================================
        # r1
        # ====================================================

        dr1 = (self.k1 / self.sigma_bu_sq) * self.level1["conv"](e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        # ====================================================
        # r2
        # ====================================================

        dr2 = (self.k1 / self.sigma_bu_sq) * self.level2["conv"](e1)
        dr2 -= self.k1 * self.alpha2 * r2

        # ====================================================
        # SUPERVISED CLASS ERROR
        #
        # Only available during supervised training.
        # ====================================================

        if label is not None:
            _, _, _, _, e2 = self.class_prediction_error(r2, label)

            feedback = self.class_feedback(e2, r2)

            dr2 += (self.k1 / self.sigma_class_sq) * feedback

        return dr1, dr2

    # ========================================================
    # PREDICTIVE-CODING INFERENCE
    # ========================================================

    @torch.no_grad()
    def infer(self, image, label=None, return_diagnostics=False):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2 = self.initialize_states(image)

        step = 0
        converged = False

        change_r1 = float("inf")
        change_r2 = float("inf")

        while True:
            old_r1 = r1.clone()
            old_r2 = r2.clone()

            dr1, dr2 = self.state_updates(image, r1, r2, label=label)

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
    # LOCAL WEIGHT LEARNING
    #
    # U1:
    # Rao-style visual prediction-error learning
    #
    # U2:
    # Rao-style hierarchical prediction-error learning
    #
    # W3:
    # local supervised output-error learning
    #
    # NO BACKPROPAGATION
    # ========================================================

    @torch.no_grad()
    def update_weights(self, image, r1, r2, label):
        e0, e1, _ = self.prediction_errors(image, r1, r2)

        features, _, _, _, e2 = self.class_prediction_error(r2, label)

        # ====================================================
        # U1
        # ====================================================

        dW1 = conv2d_weight(e0, self.level1["conv"].weight.shape, r1, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)

        dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])

        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight

        # ====================================================
        # U2
        # ====================================================

        dW2 = conv2d_weight(e1, self.level2["conv"].weight.shape, r2, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)

        dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])

        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight

        # ====================================================
        # W3
        #
        # e2^T × local r2 features
        #
        # Correct class receives positive correction.
        #
        # Competing classes receive negative correction.
        #
        # This directly prevents the class-prototype collapse
        # observed in the previous model.
        # ====================================================

        dW3 = e2.T @ features

        dW3 = (self.k2 / self.sigma_class_sq) * dW3 - self.k2 * self.lambda_u * self.level3["linear"].weight

        # ====================================================
        # EULER WEIGHT INTEGRATION
        # ====================================================

        self.level1["conv"].weight.add_(self.weight_dt * dW1)
        self.level2["conv"].weight.add_(self.weight_dt * dW2)
        self.level3["linear"].weight.add_(self.weight_dt * dW3)

        # ====================================================
        # NUMERICAL SAFETY
        # ====================================================

        if not torch.isfinite(self.level1["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-1 weights became non-finite.")

        if not torch.isfinite(self.level2["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-2 weights became non-finite.")

        if not torch.isfinite(self.level3["linear"].weight).all():
            raise FloatingPointError("PC-CNN supervised level weights became non-finite.")

    # ========================================================
    # RAO k2 DECAY
    # ========================================================

    def update_k2(self):
        self.training_input_count += 1

        if self.training_input_count % self.k2_decay_interval == 0:
            self.k2 = self.k2 / self.k2_decay_factor

    # ========================================================
    # CLASSIFY ONE IMAGE
    #
    # Test-time inference:
    #
    # image -> visual PC inference -> r2 -> class prediction
    # ========================================================

    @torch.no_grad()
    def classify(self, image):
        r1, r2, step, converged, change_r1, change_r2 = self.infer(image, label=None, return_diagnostics=True)

        features, scores, probabilities = self.class_prediction(r2)

        prediction = scores.argmax(dim=1)

        return {"scores": scores, "probabilities": probabilities, "prediction": prediction, "features": features, "r1": r1, "r2": r2, "inference_steps": step, "converged": converged, "final_change_r1": change_r1, "final_change_r2": change_r2}

    # ========================================================
    # CLASSIFY BATCH
    # ========================================================

    @torch.no_grad()
    def classify_batch(self, images):
        scores = []
        probabilities = []
        predictions = []

        steps = []
        converged = []

        final_change_r1 = []
        final_change_r2 = []

        for image in images:
            result = self.classify(image)

            scores.append(result["scores"].squeeze(0))
            probabilities.append(result["probabilities"].squeeze(0))
            predictions.append(result["prediction"].squeeze(0))

            steps.append(result["inference_steps"])
            converged.append(result["converged"])

            final_change_r1.append(result["final_change_r1"])
            final_change_r2.append(result["final_change_r2"])

        return {"scores": torch.stack(scores), "probabilities": torch.stack(probabilities), "predictions": torch.stack(predictions), "inference_steps": steps, "converged": converged, "final_change_r1": final_change_r1, "final_change_r2": final_change_r2}

    # ========================================================
    # IMAGE RECONSTRUCTION
    # ========================================================

    @torch.no_grad()
    def reconstruct(self, image):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2, step, converged = self.infer(image, label=None)

        reconstruction = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)

        _, scores, probabilities = self.class_prediction(r2)

        predicted_class = scores.argmax(dim=1)

        return reconstruction.squeeze(0), predicted_r1.squeeze(0), r2.squeeze(0), probabilities.squeeze(0), predicted_class.item(), step, converged

    # ========================================================
    # TRAIN ONE IMAGE
    #
    # Training:
    #
    # 1. Infer r1/r2 using visual + supervised local errors
    # 2. Update U1 locally
    # 3. Update U2 locally
    # 4. Update W3 locally
    #
    # NO loss.backward()
    # NO optimizer
    # ========================================================

    @torch.no_grad()
    def forward(self, image, label=None):
        if self.training and label is None:
            raise ValueError("A class label is required during supervised predictive-coding training.")

        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2, step, converged, change_r1, change_r2 = self.infer(image, label=label, return_diagnostics=True)

        e0, e1, _ = self.prediction_errors(image, r1, r2)

        _, scores, probabilities, target, e2 = self.class_prediction_error(r2, label)

        error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()
        error_2 = torch.sqrt(torch.mean(e2 ** 2)).item()

        true_probability = probabilities[0, int(label.item()) if torch.is_tensor(label) else int(label)]
        class_loss = -torch.log(true_probability + 1e-8).item()

        prediction = scores.argmax(dim=1).item()

        if self.training:
            self.update_weights(image, r1, r2, label)
            self.update_k2()

        return {"inference_steps": step, "converged": converged, "final_change_r1": change_r1, "final_change_r2": change_r2, "error_0": error_0, "error_1": error_1, "error_2": error_2, "class_loss": class_loss, "prediction": prediction, "probabilities": probabilities.squeeze(0), "k2": self.k2, "training_input_count": self.training_input_count}