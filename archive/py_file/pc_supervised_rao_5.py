import torch
import torch.nn as nn

from torch.nn.grad import conv2d_weight


# ============================================================
# SUPERVISED HIERARCHICAL RAO-STYLE PC-CNN
#
# Image <-> r1 <-> r2 -> supervised class prediction
#
# TRAINING:
#
# 1. Free predictive-coding inference
# 2. Classification from free r2
# 3. Local supervised class error
# 4. Short supervised nudging phase
# 5. Local U1/U2/W3 learning
#
# TESTING:
#
# 1. Free predictive-coding inference
# 2. Classification from free r2
#
# NO AUTOGRAD
# NO BACKPROPAGATION
# NO OPTIMIZER
# ============================================================

class HierarchicalPCCNN(nn.Module):

    def __init__(self, num_classes, input_size=64, inference_steps=100, supervised_nudge_steps=5, convergence_tolerance=1e-5, sigma_class_sq=1.0, class_pool_size=4):
        super().__init__()

        if input_size % 4 != 0:
            raise ValueError("input_size must be divisible by 4.")

        if sigma_class_sq <= 0:
            raise ValueError("sigma_class_sq must be greater than 0.")

        if supervised_nudge_steps < 0:
            raise ValueError("supervised_nudge_steps must be >= 0.")

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

        # ====================================================
        # SUPERVISED CLASS-ERROR PRECISION
        # ====================================================

        self.sigma_class_sq = sigma_class_sq

        # ====================================================
        # REPRESENTATION PRIORS
        # ====================================================

        self.alpha1 = 0.001
        self.alpha2 = 0.05

        # ====================================================
        # WEIGHT PRIOR
        # ====================================================

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
        self.supervised_nudge_steps = supervised_nudge_steps

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
        # SUPERVISED LEVEL
        #
        # r2
        # ↓
        # spatial average pooling
        # ↓
        # flattened features
        # ↓
        # local linear class prediction
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
        # TIED VISUAL WEIGHTS
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
    # IMAGE SIZE VALIDATION
    # ========================================================

    def validate_image_size(self, image):
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            raise ValueError(f"Expected image size {self.input_size}x{self.input_size}, but received {image.shape[-2]}x{image.shape[-1]}.")

    # ========================================================
    # INITIALIZE VISUAL REPRESENTATIONS
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
    # TOP-DOWN VISUAL PREDICTIONS
    # ========================================================

    def predict_image(self, r1):
        return self.level1["deconv"](r1)

    def predict_r1(self, r2):
        return self.level2["deconv"](r2)

    # ========================================================
    # VISUAL PREDICTION ERRORS
    #
    # e0 = image - U1r1
    #
    # e1 = r1 - U2r2
    # ========================================================

    def prediction_errors(self, image, r1, r2):
        predicted_image = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)

        e0 = image.unsqueeze(0) - predicted_image
        e1 = r1 - predicted_r1

        return e0, e1, predicted_r1

    # ========================================================
    # CLASSIFICATION FEATURES
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
    # TARGET
    # ========================================================

    def create_target(self, label, device, dtype):
        label_value = int(label.item()) if torch.is_tensor(label) else int(label)

        if label_value < 0 or label_value >= self.num_classes:
            raise ValueError(f"Class label {label_value} is outside the valid range 0-{self.num_classes - 1}.")

        target = torch.zeros(1, self.num_classes, device=device, dtype=dtype)
        target[0, label_value] = 1.0

        return target

    # ========================================================
    # LOCAL SUPERVISED CLASS PREDICTION ERROR
    #
    # e2 = target - softmax(W3 z)
    # ========================================================

    def class_prediction_error(self, r2, label):
        features, scores, probabilities = self.class_prediction(r2)

        target = self.create_target(label, probabilities.device, probabilities.dtype)

        e2 = target - probabilities

        return features, scores, probabilities, target, e2

    # ========================================================
    # CLASS ERROR FEEDBACK TO r2
    #
    # e2 -> W3^T e2 -> pooled-r2 correction
    #                    -> full spatial r2 correction
    # ========================================================

    def class_feedback(self, e2, r2):
        pooled_feedback = e2 @ self.level3["linear"].weight

        pooled_feedback = pooled_feedback.view(r2.shape[0], 128, self.class_pool_size, self.class_pool_size)

        feedback = pooled_feedback.repeat_interleave(self.class_pool_kernel, dim=-2)
        feedback = feedback.repeat_interleave(self.class_pool_kernel, dim=-1)

        feedback = feedback / float(self.class_pool_kernel * self.class_pool_kernel)

        return feedback

    # ========================================================
    # FREE RAO-STYLE STATE DYNAMICS
    #
    # NO LABEL
    # ========================================================

    def free_state_updates(self, image, r1, r2):
        e0, e1, predicted_r1 = self.prediction_errors(image, r1, r2)

        dr1 = (self.k1 / self.sigma_bu_sq) * self.level1["conv"](e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        dr2 = (self.k1 / self.sigma_bu_sq) * self.level2["conv"](e1)
        dr2 -= self.k1 * self.alpha2 * r2

        return dr1, dr2

    # ========================================================
    # SUPERVISED NUDGED STATE DYNAMICS
    #
    # Starts from FREE state.
    #
    # Adds only a short local class-error correction to r2.
    # ========================================================

    def nudged_state_updates(self, image, r1, r2, label):
        e0, e1, predicted_r1 = self.prediction_errors(image, r1, r2)

        dr1 = (self.k1 / self.sigma_bu_sq) * self.level1["conv"](e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        dr2 = (self.k1 / self.sigma_bu_sq) * self.level2["conv"](e1)
        dr2 -= self.k1 * self.alpha2 * r2

        _, _, _, _, e2 = self.class_prediction_error(r2, label)

        feedback = self.class_feedback(e2, r2)

        dr2 += (self.k1 / self.sigma_class_sq) * feedback

        return dr1, dr2

    # ========================================================
    # FREE PREDICTIVE-CODING INFERENCE
    #
    # Used in BOTH training and testing.
    #
    # This removes the train/test representation mismatch.
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

            dr1, dr2 = self.free_state_updates(image, r1, r2)

            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2

            step += 1

            if not torch.isfinite(r1).all():
                raise FloatingPointError("PC-CNN r1 free inference became non-finite.")

            if not torch.isfinite(r2).all():
                raise FloatingPointError("PC-CNN r2 free inference became non-finite.")

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
    # SHORT SUPERVISED NUDGING PHASE
    #
    # Starts from FREE inference state.
    #
    # Does NOT restart states.
    # ========================================================

    @torch.no_grad()
    def supervised_nudge(self, image, r1_free, r2_free, label):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1 = r1_free.clone()
        r2 = r2_free.clone()

        if self.supervised_nudge_steps == 0:
            return r1, r2, 0, 0.0, 0.0

        change_r1 = 0.0
        change_r2 = 0.0

        for step in range(self.supervised_nudge_steps):
            old_r1 = r1.clone()
            old_r2 = r2.clone()

            dr1, dr2 = self.nudged_state_updates(image, r1, r2, label)

            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2

            if not torch.isfinite(r1).all():
                raise FloatingPointError("PC-CNN r1 supervised nudging became non-finite.")

            if not torch.isfinite(r2).all():
                raise FloatingPointError("PC-CNN r2 supervised nudging became non-finite.")

            change_r1 = torch.mean(torch.abs(r1 - old_r1)).item()
            change_r2 = torch.mean(torch.abs(r2 - old_r2)).item()

        return r1, r2, self.supervised_nudge_steps, change_r1, change_r2

    # ========================================================
    # LOCAL WEIGHT LEARNING
    #
    # U1/U2:
    # updated from short NUDGED predictive-coding state
    #
    # W3:
    # updated from FREE r2 representation
    #
    # Therefore W3 learns exactly the type of representation
    # available at test time.
    # ========================================================

    @torch.no_grad()
    def update_weights(self, image, r1_nudged, r2_nudged, r2_free, label):
        # ====================================================
        # U1 / U2 FROM NUDGED PC STATE
        # ====================================================

        e0_nudged, e1_nudged, _ = self.prediction_errors(image, r1_nudged, r2_nudged)

        dW1 = conv2d_weight(e0_nudged, self.level1["conv"].weight.shape, r1_nudged, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)
        dW2 = conv2d_weight(e1_nudged, self.level2["conv"].weight.shape, r2_nudged, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)

        dW1 = dW1 / (r1_nudged.shape[-2] * r1_nudged.shape[-1])
        dW2 = dW2 / (r2_nudged.shape[-2] * r2_nudged.shape[-1])

        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight
        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight

        # ====================================================
        # W3 FROM FREE r2
        #
        # IMPORTANT:
        #
        # No label-conditioned r2 is used here.
        # ====================================================

        features_free, _, _, _, e2_free = self.class_prediction_error(r2_free, label)

        dW3 = e2_free.T @ features_free

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
            raise FloatingPointError("PC-CNN supervised W3 weights became non-finite.")

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
    # TEST:
    #
    # Image -> FREE PC inference -> r2 -> W3
    # ========================================================

    @torch.no_grad()
    def classify(self, image):
        r1, r2, step, converged, change_r1, change_r2 = self.infer(image, return_diagnostics=True)

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
    # RECONSTRUCTION
    #
    # Uses FREE inference only.
    # ========================================================

    @torch.no_grad()
    def reconstruct(self, image):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2, step, converged = self.infer(image)

        reconstruction = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)

        _, scores, probabilities = self.class_prediction(r2)

        predicted_class = scores.argmax(dim=1)

        return reconstruction.squeeze(0), predicted_r1.squeeze(0), r2.squeeze(0), probabilities.squeeze(0), predicted_class.item(), step, converged

    # ========================================================
    # TRAIN ONE IMAGE
    #
    # PHASE 1:
    # FREE predictive-coding inference
    #
    # PHASE 2:
    # Calculate supervised class error from FREE r2
    #
    # PHASE 3:
    # Short supervised nudge from FREE state
    #
    # PHASE 4:
    # Local weight updates
    #
    # W3 uses FREE representation.
    #
    # U1/U2 use lightly NUDGED predictive-coding state.
    #
    # NO loss.backward()
    # NO optimizer
    # ========================================================

    @torch.no_grad()
    def forward(self, image, label=None):
        if self.training and label is None:
            raise ValueError("A class label is required during supervised predictive-coding training.")

        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        # ====================================================
        # PHASE 1 — FREE INFERENCE
        # ====================================================

        r1_free, r2_free, free_steps, free_converged, free_change_r1, free_change_r2 = self.infer(image, return_diagnostics=True)

        # ====================================================
        # FREE VISUAL ERRORS
        # ====================================================

        e0_free, e1_free, _ = self.prediction_errors(image, r1_free, r2_free)

        # ====================================================
        # FREE CLASSIFICATION ERROR
        #
        # This is exactly the representation available at test.
        # ====================================================

        features_free, scores_free, probabilities_free, _, e2_free = self.class_prediction_error(r2_free, label)

        error_0 = torch.sqrt(torch.mean(e0_free ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1_free ** 2)).item()
        error_2 = torch.sqrt(torch.mean(e2_free ** 2)).item()

        label_value = int(label.item()) if torch.is_tensor(label) else int(label)

        true_probability = probabilities_free[0, label_value]

        class_loss = -torch.log(true_probability + 1e-8).item()

        prediction = scores_free.argmax(dim=1).item()

        # ====================================================
        # PHASE 2 — SHORT SUPERVISED NUDGE
        # ====================================================

        r1_nudged, r2_nudged, nudge_steps, nudge_change_r1, nudge_change_r2 = self.supervised_nudge(image, r1_free, r2_free, label)

        # ====================================================
        # NUDGED VISUAL ERRORS
        # ====================================================

        e0_nudged, e1_nudged, _ = self.prediction_errors(image, r1_nudged, r2_nudged)

        nudged_error_0 = torch.sqrt(torch.mean(e0_nudged ** 2)).item()
        nudged_error_1 = torch.sqrt(torch.mean(e1_nudged ** 2)).item()

        # ====================================================
        # LOCAL LEARNING
        # ====================================================

        if self.training:
            self.update_weights(image, r1_nudged, r2_nudged, r2_free, label)
            self.update_k2()

        return {
            "inference_steps": free_steps,
            "converged": free_converged,
            "final_change_r1": free_change_r1,
            "final_change_r2": free_change_r2,

            "nudge_steps": nudge_steps,
            "nudge_change_r1": nudge_change_r1,
            "nudge_change_r2": nudge_change_r2,

            "error_0": error_0,
            "error_1": error_1,
            "error_2": error_2,

            "nudged_error_0": nudged_error_0,
            "nudged_error_1": nudged_error_1,

            "class_loss": class_loss,
            "prediction": prediction,
            "probabilities": probabilities_free.squeeze(0),

            "k2": self.k2,
            "training_input_count": self.training_input_count
        }