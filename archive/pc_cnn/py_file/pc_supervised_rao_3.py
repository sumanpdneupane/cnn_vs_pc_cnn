import torch
import torch.nn as nn

from torch.nn.grad import conv2d_weight


# ============================================================
# SUPERVISED HIERARCHICAL RAO-STYLE PC-CNN
#
# Hierarchy:
#
# Image <-> r1 <-> r2 <-> class state
#
# Supervised top-level predictive-coding extension
#
# Class state during training:
# known one-hot label
#
# Class state during testing:
# inferred on probability simplex
#
# NO AUTOGRAD
# NO BACKPROPAGATION
# NO OPTIMIZER
# ============================================================

class HierarchicalPCCNN(nn.Module):

    def __init__(self, num_classes, input_size=64, inference_steps=100, convergence_tolerance=1e-5, sigma_class_sq=5.0):
        super().__init__()

        if input_size % 4 != 0:
            raise ValueError("input_size must be divisible by 4.")

        if sigma_class_sq <= 0:
            raise ValueError("sigma_class_sq must be greater than 0.")

        self.num_classes = num_classes
        self.input_size = input_size

        # ====================================================
        # RAO PARAMETERS
        # ====================================================

        self.k1 = 0.5
        self.k2 = 1.0

        # Bottom-up visual prediction-error variance
        self.sigma_bu_sq = 1.0

        # Ordinary hierarchical top-down variance
        self.sigma_td_sq = 10.0

        # Separate supervised class-level variance
        #
        # Smaller sigma_class_sq = stronger class precision
        # Current controlled experiment = 5.0
        # ====================================================

        self.sigma_class_sq = sigma_class_sq

        self.alpha1 = 0.001
        self.alpha2 = 0.05
        self.alpha3 = 0.05

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
        # SPATIAL DIMENSIONS
        #
        # Tiny ImageNet:
        #
        # 64 -> 32 -> 16
        #
        # Future 224x224:
        #
        # 224 -> 112 -> 56
        # ====================================================

        self.r1_size = input_size // 2
        self.r2_size = input_size // 4

        # ====================================================
        # LEVEL 1
        #
        # Image -> r1
        # r1 -> predicted image
        # ====================================================

        self.level1 = nn.ModuleDict({"conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)})

        # ====================================================
        # LEVEL 2
        #
        # r1 -> r2
        # r2 -> predicted r1
        # ====================================================

        self.level2 = nn.ModuleDict({"conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False), "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)})

        # ====================================================
        # LEVEL 3
        #
        # For input_size = 64:
        #
        # r2 [128,16,16]
        #       ⇅
        # class state [K,1,1]
        #
        # For input_size = 224:
        #
        # r2 [128,56,56]
        #       ⇅
        # class state [K,1,1]
        # ====================================================

        self.level3 = nn.ModuleDict({"conv": nn.Conv2d(128, num_classes, kernel_size=self.r2_size, stride=1, padding=0, bias=False), "deconv": nn.ConvTranspose2d(num_classes, 128, kernel_size=self.r2_size, stride=1, padding=0, bias=False)})

        # ====================================================
        # WEIGHT INITIALIZATION
        # ====================================================

        nn.init.normal_(self.level1["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level2["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level3["conv"].weight, mean=0.0, std=0.01)

        # ====================================================
        # TIED BOTTOM-UP / TOP-DOWN WEIGHTS
        # ====================================================

        self.level1["deconv"].weight = self.level1["conv"].weight
        self.level2["deconv"].weight = self.level2["conv"].weight
        self.level3["deconv"].weight = self.level3["conv"].weight

        # ====================================================
        # NO AUTOGRAD / NO BACKPROPAGATION
        # ====================================================

        self.level1["conv"].weight.requires_grad_(False)
        self.level2["conv"].weight.requires_grad_(False)
        self.level3["conv"].weight.requires_grad_(False)

    # ========================================================
    # CHECK IMAGE SIZE
    # ========================================================

    def validate_image_size(self, image):
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            raise ValueError(f"Expected image size {self.input_size}x{self.input_size}, but received {image.shape[-2]}x{image.shape[-1]}.")

    # ========================================================
    # CREATE CLAMPED CLASS STATE
    #
    # Example:
    #
    # class 2 of 5
    #
    # [0, 0, 1, 0, 0]
    # ========================================================

    def create_class_state(self, label, device, dtype):
        label_value = int(label.item()) if torch.is_tensor(label) else int(label)

        if label_value < 0 or label_value >= self.num_classes:
            raise ValueError(f"Class label {label_value} is outside the valid range 0-{self.num_classes - 1}.")

        class_state = torch.zeros(1, self.num_classes, 1, 1, device=device, dtype=dtype)
        class_state[0, label_value, 0, 0] = 1.0

        return class_state

    # ========================================================
    # PROJECT CLASS STATE ONTO PROBABILITY SIMPLEX
    #
    # Constraints:
    #
    # c_i >= 0
    #
    # sum(c_i) = 1
    #
    # This introduces competition between class hypotheses.
    # ========================================================

    @torch.no_grad()
    def project_class_state(self, class_state):
        flat = class_state.reshape(class_state.shape[0], -1)

        sorted_values, _ = torch.sort(flat, dim=1, descending=True)

        cumulative = torch.cumsum(sorted_values, dim=1) - 1.0

        indices = torch.arange(1, flat.shape[1] + 1, device=flat.device, dtype=flat.dtype).view(1, -1)

        condition = sorted_values - cumulative / indices > 0

        rho = condition.sum(dim=1) - 1
        rho = rho.clamp_min(0)

        theta = cumulative.gather(1, rho.unsqueeze(1))
        theta = theta / (rho.to(flat.dtype).unsqueeze(1) + 1.0)

        projected = torch.clamp(flat - theta, min=0.0)

        return projected.reshape_as(class_state)

    # ========================================================
    # INITIALIZE REPRESENTATIONS
    #
    # r1 = zero
    #
    # r2 = zero
    #
    # Training:
    #
    # class state = one-hot true label
    #
    # Testing:
    #
    # class state = uniform distribution
    # ========================================================

    def initialize_states(self, image, label=None):
        self.validate_image_size(image)

        image_batch = image.unsqueeze(0)

        r1_shape = self.level1["conv"](image_batch).shape
        r2_shape = self.level2["conv"](torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape

        r1 = torch.zeros(r1_shape, device=image.device, dtype=image.dtype)
        r2 = torch.zeros(r2_shape, device=image.device, dtype=image.dtype)

        if label is None:
            class_state = torch.full((1, self.num_classes, 1, 1), 1.0 / self.num_classes, device=image.device, dtype=image.dtype)
        else:
            class_state = self.create_class_state(label, image.device, image.dtype)

        return r1, r2, class_state

    # ========================================================
    # TOP-DOWN GENERATIVE PREDICTIONS
    # ========================================================

    def predict_image(self, r1):
        return self.level1["deconv"](r1)

    def predict_r1(self, r2):
        return self.level2["deconv"](r2)

    def predict_r2(self, class_state):
        return self.level3["deconv"](class_state)

    # ========================================================
    # PREDICTION ERRORS
    #
    # e0 = image - U1 r1
    #
    # e1 = r1 - U2 r2
    #
    # e2 = r2 - U3 c
    # ========================================================

    def prediction_errors(self, image, r1, r2, class_state):
        predicted_image = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        predicted_r2 = self.predict_r2(class_state)

        e0 = image.unsqueeze(0) - predicted_image
        e1 = r1 - predicted_r1
        e2 = r2 - predicted_r2

        return e0, e1, e2, predicted_r1, predicted_r2

    # ========================================================
    # RAO-STYLE STATE DYNAMICS
    # ========================================================

    def state_updates(self, image, r1, r2, class_state, class_clamped):
        e0, e1, e2, predicted_r1, predicted_r2 = self.prediction_errors(image, r1, r2, class_state)

        # ====================================================
        # LEVEL 1 REPRESENTATION
        #
        # Bottom-up image prediction error
        #
        # +
        #
        # Top-down r2 prediction
        #
        # -
        #
        # Representation prior
        # ====================================================

        dr1 = (self.k1 / self.sigma_bu_sq) * self.level1["conv"](e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        # ====================================================
        # LEVEL 2 REPRESENTATION
        #
        # Bottom-up r1 prediction error
        #
        # +
        #
        # Supervised class-level prediction
        #
        # -
        #
        # Representation prior
        #
        # Class influence uses sigma_class_sq.
        # ====================================================

        dr2 = (self.k1 / self.sigma_bu_sq) * self.level2["conv"](e1)
        dr2 += (self.k1 / self.sigma_class_sq) * (predicted_r2 - r2)
        dr2 -= self.k1 * self.alpha2 * r2

        # ====================================================
        # LEVEL 3 CLASS STATE
        #
        # TRAINING:
        #
        # class state clamped
        #
        # TESTING:
        #
        # class state freely inferred
        #
        # U3 energy normalization prevents unstable dynamics.
        # ====================================================

        if class_clamped:
            dc = torch.zeros_like(class_state)

        else:
            class_drive = self.level3["conv"](e2)

            class_weight_energy = self.level3["conv"].weight.pow(2).flatten(1).sum(dim=1)
            class_weight_energy = class_weight_energy.view(1, self.num_classes, 1, 1)
            class_weight_energy = class_weight_energy.clamp_min(1e-6)

            dc = (self.k1 / self.sigma_class_sq) * (class_drive / class_weight_energy)
            dc -= self.k1 * self.alpha3 * class_state

        return dr1, dr2, dc

    # ========================================================
    # PREDICTIVE-CODING INFERENCE
    #
    # Training:
    #
    # c = one-hot known class
    #
    # Testing:
    #
    # c is updated locally and projected onto simplex
    # after every inference step.
    # ========================================================

    @torch.no_grad()
    def infer(self, image, label=None, return_diagnostics=False):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        class_clamped = label is not None

        r1, r2, class_state = self.initialize_states(image, label)

        step = 0
        converged = False

        change_r1 = float("inf")
        change_r2 = float("inf")
        change_class = 0.0 if class_clamped else float("inf")

        while True:
            old_r1 = r1.clone()
            old_r2 = r2.clone()
            old_class_state = class_state.clone()

            dr1, dr2, dc = self.state_updates(image, r1, r2, class_state, class_clamped)

            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2

            if not class_clamped:
                class_state = class_state + self.inference_dt * dc
                class_state = self.project_class_state(class_state)

            step += 1

            if not torch.isfinite(r1).all():
                raise FloatingPointError("PC-CNN r1 inference became non-finite.")

            if not torch.isfinite(r2).all():
                raise FloatingPointError("PC-CNN r2 inference became non-finite.")

            if not torch.isfinite(class_state).all():
                raise FloatingPointError("PC-CNN class-state inference became non-finite.")

            change_r1 = torch.mean(torch.abs(r1 - old_r1)).item()
            change_r2 = torch.mean(torch.abs(r2 - old_r2)).item()

            if class_clamped:
                change_class = 0.0
            else:
                change_class = torch.mean(torch.abs(class_state - old_class_state)).item()

            representation_converged = change_r1 < self.convergence_tolerance and change_r2 < self.convergence_tolerance
            class_converged = class_clamped or change_class < self.convergence_tolerance

            if representation_converged and class_converged:
                converged = True
                break

            if self.inference_steps is not None and step >= self.inference_steps:
                break

        if return_diagnostics:
            return r1, r2, class_state, step, converged, change_r1, change_r2, change_class

        return r1, r2, class_state, step, converged

    # ========================================================
    # RAO EQ. 9-STYLE LOCAL WEIGHT LEARNING
    #
    # U1 <- e0 x r1
    #
    # U2 <- e1 x r2
    #
    # U3 <- e2 x class state
    # ========================================================

    @torch.no_grad()
    def update_weights(self, image, r1, r2, class_state):
        e0, e1, e2, _, _ = self.prediction_errors(image, r1, r2, class_state)

        # ====================================================
        # LOCAL CORRELATION TERMS
        # ====================================================

        dW1 = conv2d_weight(e0, self.level1["conv"].weight.shape, r1, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)
        dW2 = conv2d_weight(e1, self.level2["conv"].weight.shape, r2, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)
        dW3 = conv2d_weight(e2, self.level3["conv"].weight.shape, class_state, stride=self.level3["conv"].stride, padding=self.level3["conv"].padding)

        # ====================================================
        # SPATIAL AVERAGING
        # ====================================================

        dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])
        dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])

        # U3 class-state output is 1x1, so no spatial averaging.

        # ====================================================
        # LOCAL RAO-STYLE WEIGHT UPDATES
        #
        # U1/U2 use visual prediction-error precision.
        #
        # U3 uses class-level prediction-error precision.
        # ====================================================

        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight
        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight
        dW3 = (self.k2 / self.sigma_class_sq) * dW3 - self.k2 * self.lambda_u * self.level3["conv"].weight

        # ====================================================
        # EULER WEIGHT INTEGRATION
        # ====================================================

        self.level1["conv"].weight.add_(self.weight_dt * dW1)
        self.level2["conv"].weight.add_(self.weight_dt * dW2)
        self.level3["conv"].weight.add_(self.weight_dt * dW3)

        # ====================================================
        # NUMERICAL SAFETY
        # ====================================================

        if not torch.isfinite(self.level1["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-1 weights became non-finite.")

        if not torch.isfinite(self.level2["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-2 weights became non-finite.")

        if not torch.isfinite(self.level3["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-3 weights became non-finite.")

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
    # Class state is already simplex constrained.
    #
    # Therefore:
    #
    # class_state itself is used as the probability
    # distribution.
    #
    # NO additional softmax.
    # ========================================================

    @torch.no_grad()
    def classify(self, image):
        r1, r2, class_state, step, converged, change_r1, change_r2, change_class = self.infer(image, label=None, return_diagnostics=True)

        class_scores = class_state.squeeze(-1).squeeze(-1)

        probabilities = class_scores
        prediction = class_scores.argmax(dim=1)

        probability_sum = probabilities.sum(dim=1)

        if not torch.allclose(probability_sum, torch.ones_like(probability_sum), atol=1e-5):
            raise FloatingPointError("Class-state probabilities do not sum to 1.")

        if (probabilities < -1e-7).any():
            raise FloatingPointError("Class-state probabilities became negative.")

        return {"scores": class_scores, "probabilities": probabilities, "prediction": prediction, "r1": r1, "r2": r2, "class_state": class_state, "inference_steps": step, "converged": converged, "final_change_r1": change_r1, "final_change_r2": change_r2, "final_change_class": change_class}

    # ========================================================
    # CLASSIFY BATCH
    #
    # Predictive inference remains image-by-image.
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
        final_change_class = []

        for image in images:
            result = self.classify(image)

            scores.append(result["scores"].squeeze(0))
            probabilities.append(result["probabilities"].squeeze(0))
            predictions.append(result["prediction"].squeeze(0))

            steps.append(result["inference_steps"])
            converged.append(result["converged"])

            final_change_r1.append(result["final_change_r1"])
            final_change_r2.append(result["final_change_r2"])
            final_change_class.append(result["final_change_class"])

        return {"scores": torch.stack(scores), "probabilities": torch.stack(probabilities), "predictions": torch.stack(predictions), "inference_steps": steps, "converged": converged, "final_change_r1": final_change_r1, "final_change_r2": final_change_r2, "final_change_class": final_change_class}

    # ========================================================
    # IMAGE RECONSTRUCTION
    # ========================================================

    @torch.no_grad()
    def reconstruct(self, image, label=None):
        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2, class_state, step, converged = self.infer(image, label=label)

        reconstruction = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        predicted_r2 = self.predict_r2(class_state)

        class_scores = class_state.squeeze(-1).squeeze(-1)
        predicted_class = class_scores.argmax(dim=1)

        return reconstruction.squeeze(0), predicted_r1.squeeze(0), predicted_r2.squeeze(0), r2.squeeze(0), class_state.squeeze(0), predicted_class.item(), step, converged

    # ========================================================
    # TRAIN ONE IMAGE
    #
    # Correct class state is clamped.
    #
    # e0 trains U1
    #
    # e1 trains U2
    #
    # e2 trains U3
    #
    # NO classification backpropagation
    # ========================================================

    @torch.no_grad()
    def forward(self, image, label=None):
        if self.training and label is None:
            raise ValueError("A class label is required during supervised predictive-coding training.")

        image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)

        r1, r2, class_state, step, converged, change_r1, change_r2, change_class = self.infer(image, label=label, return_diagnostics=True)

        e0, e1, e2, _, _ = self.prediction_errors(image, r1, r2, class_state)

        error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()
        error_2 = torch.sqrt(torch.mean(e2 ** 2)).item()

        if self.training:
            self.update_weights(image, r1, r2, class_state)
            self.update_k2()

        return {"inference_steps": step, "converged": converged, "final_change_r1": change_r1, "final_change_r2": change_r2, "final_change_class": change_class, "error_0": error_0, "error_1": error_1, "error_2": error_2, "k2": self.k2, "training_input_count": self.training_input_count}