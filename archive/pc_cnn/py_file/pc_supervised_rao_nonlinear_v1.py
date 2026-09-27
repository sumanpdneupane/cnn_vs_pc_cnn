import torch
import torch.nn as nn

from torch.nn.grad import conv2d_weight


# ============================================================
# NONLINEAR SUPERVISED HIERARCHICAL RAO-STYLE PC-CNN — V3
#
# Image <-> r1 <-> r2 <-> r3 -> supervised class prediction
#
# TRAINING
# 1. Free predictive-coding inference
# 2. Classification from FREE r3
# 3. Local supervised class error
# 4. Supervised nudge applied only to r3
# 5. Class influence moves through the predictive hierarchy
# 6. U1/U2/U3 updated from nudged local predictive errors
# 7. Classifier updated locally from FREE r3
#
# TESTING
# 1. Free predictive-coding inference only
# 2. Classification from the same FREE r3 representation
#
# NO AUTOGRAD
# NO BACKPROPAGATION
# NO OPTIMIZER
# ============================================================


class HierarchicalPCCNN(nn.Module):

    def __init__(
        self,
        num_classes,
        input_size=64,
        inference_steps=100,
        supervised_nudge_steps=50,
        convergence_tolerance=1e-5,
        sigma_class_sq=1.0,
        class_learning_rate=0.01,
    ):
        super().__init__()

        if input_size % 8 != 0:
            raise ValueError("input_size must be divisible by 8.")

        if sigma_class_sq <= 0:
            raise ValueError("sigma_class_sq must be greater than 0.")

        if supervised_nudge_steps < 0:
            raise ValueError("supervised_nudge_steps must be >= 0.")

        if class_learning_rate <= 0:
            raise ValueError("class_learning_rate must be greater than 0.")

        self.num_classes = num_classes
        self.input_size = input_size
        self.class_learning_rate = class_learning_rate

        # ====================================================
        # RAO PARAMETERS
        # ====================================================

        self.k1 = 0.5
        self.k2 = 1.0

        self.sigma_bu_sq = 1.0
        self.sigma_td_sq = 10.0
        self.sigma_class_sq = sigma_class_sq

        self.alpha1 = 0.001
        self.alpha2 = 0.05
        self.alpha3 = 0.05

        self.lambda_u = 0.02

        # ====================================================
        # RAO k2 SCHEDULE — ONLY FOR VISUAL U1/U2/U3
        # ====================================================

        self.k2_decay_interval = 2500 # 160 # 40
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
        # ====================================================

        self.r1_size = input_size // 2
        self.r2_size = input_size // 4
        self.r3_size = input_size // 8

        # ====================================================
        # LEVEL 1
        # Image <-> r1
        # ====================================================

        self.level1 = nn.ModuleDict({
            "conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        })

        # ====================================================
        # LEVEL 2
        # r1 <-> r2
        # ====================================================

        self.level2 = nn.ModuleDict({
            "conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        })

        # ====================================================
        # LEVEL 3
        # r2 <-> r3
        # ====================================================

        self.level3 = nn.ModuleDict({
            "conv": nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        })

        # ====================================================
        # SUPERVISED OUTPUT
        # FREE r3 -> tanh -> flatten -> Linear -> Softmax
        # ====================================================

        self.classifier = nn.Linear(256 * self.r3_size * self.r3_size, num_classes, bias=True)

        # ====================================================
        # INITIALIZATION
        # ====================================================

        nn.init.normal_(self.level1["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level2["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level3["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

        # ====================================================
        # TIED VISUAL WEIGHTS
        # ====================================================

        self.level1["deconv"].weight = self.level1["conv"].weight
        self.level2["deconv"].weight = self.level2["conv"].weight
        self.level3["deconv"].weight = self.level3["conv"].weight

        # ====================================================
        # NO AUTOGRAD
        # ====================================================

        self.level1["conv"].weight.requires_grad_(False)
        self.level2["conv"].weight.requires_grad_(False)
        self.level3["conv"].weight.requires_grad_(False)
        self.classifier.weight.requires_grad_(False)
        self.classifier.bias.requires_grad_(False)

    # ========================================================
    # NONLINEAR REPRESENTATION
    # phi(r) = tanh(r)
    # ========================================================

    def activation(self, state):
        return torch.tanh(state)

    def activation_derivative(self, state):
        activated = torch.tanh(state)
        return 1.0 - activated * activated

    # ========================================================
    # IMAGE SIZE VALIDATION
    # ========================================================

    def validate_image_size(self, image):
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected image size {self.input_size}x{self.input_size}, "
                f"but received {image.shape[-2]}x{image.shape[-1]}."
            )

    # ========================================================
    # INITIALIZE VISUAL STATES
    # ========================================================

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

    # ========================================================
    # TOP-DOWN VISUAL PREDICTIONS
    # ========================================================

    def predict_image(self, r1):
        return self.level1["deconv"](self.activation(r1))

    def predict_r1(self, r2):
        return self.level2["deconv"](self.activation(r2))

    def predict_r2(self, r3):
        return self.level3["deconv"](self.activation(r3))

    # ========================================================
    # VISUAL PREDICTION ERRORS
    #
    # e0 = image - U1 phi(r1)
    # e1 = r1    - U2 phi(r2)
    # e2 = r2    - U3 phi(r3)
    # ========================================================

    def prediction_errors(self, image, r1, r2, r3):
        predicted_image = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        predicted_r2 = self.predict_r2(r3)

        e0 = image.unsqueeze(0) - predicted_image
        e1 = r1 - predicted_r1
        e2 = r2 - predicted_r2

        return e0, e1, e2, predicted_r1, predicted_r2

    # ========================================================
    # CLASS FEATURES
    # Full nonlinear FREE r3 representation
    # ========================================================

    def class_features(self, r3):
        return self.activation(r3).flatten(1)

    # ========================================================
    # CLASS PREDICTION
    # ========================================================

    def class_prediction(self, r3):
        features = self.class_features(r3)
        scores = self.classifier(features)
        probabilities = torch.softmax(scores, dim=1)

        return features, scores, probabilities

    # ========================================================
    # ONE-HOT TARGET
    # ========================================================

    def create_target(self, label, device, dtype):
        label_value = int(label.item()) if torch.is_tensor(label) else int(label)

        if label_value < 0 or label_value >= self.num_classes:
            raise ValueError(
                f"Class label {label_value} is outside the valid range "
                f"0-{self.num_classes - 1}."
            )

        target = torch.zeros(1, self.num_classes, device=device, dtype=dtype)
        target[0, label_value] = 1.0

        return target

    # ========================================================
    # LOCAL SUPERVISED CLASS ERROR
    #
    # e_class = target - softmax(Wc z + bc)
    # ========================================================

    def class_prediction_error(self, r3, label):
        features, scores, probabilities = self.class_prediction(r3)
        target = self.create_target(label, probabilities.device, probabilities.dtype)
        e_class = target - probabilities

        return features, scores, probabilities, target, e_class

    # ========================================================
    # CLASS ERROR FEEDBACK TO r3
    #
    # feedback = phi'(r3) * reshape(Wc^T e_class)
    #
    # Classification error enters ONLY at r3.
    # Lower states are affected through the predictive hierarchy.
    # ========================================================

    def class_feedback(self, e_class, r3):
        feedback = e_class @ self.classifier.weight
        feedback = feedback.view_as(r3)
        feedback = feedback * self.activation_derivative(r3)

        return feedback

    # ========================================================
    # FREE RAO-STYLE STATE DYNAMICS
    # ========================================================

    def free_state_updates(self, image, r1, r2, r3):
        e0, e1, e2, predicted_r1, predicted_r2 = self.prediction_errors(
            image, r1, r2, r3
        )

        # r1 receives bottom-up e0 correction and top-down r2 prediction
        dr1 = (
            (self.k1 / self.sigma_bu_sq)
            * self.activation_derivative(r1)
            * self.level1["conv"](e0)
        )
        dr1 += (
            (self.k1 / self.sigma_td_sq)
            * (predicted_r1 - r1)
        )
        dr1 -= self.k1 * self.alpha1 * r1

        # r2 receives bottom-up e1 correction and top-down r3 prediction
        dr2 = (
            (self.k1 / self.sigma_bu_sq)
            * self.activation_derivative(r2)
            * self.level2["conv"](e1)
        )
        dr2 += (
            (self.k1 / self.sigma_td_sq)
            * (predicted_r2 - r2)
        )
        dr2 -= self.k1 * self.alpha2 * r2

        # r3 is the highest visual state
        dr3 = (
            (self.k1 / self.sigma_bu_sq)
            * self.activation_derivative(r3)
            * self.level3["conv"](e2)
        )
        dr3 -= self.k1 * self.alpha3 * r3

        return dr1, dr2, dr3

    # ========================================================
    # SUPERVISED NUDGED DYNAMICS
    # ========================================================

    def nudged_state_updates(self, image, r1, r2, r3, label):
        dr1, dr2, dr3 = self.free_state_updates(image, r1, r2, r3)

        _, _, _, _, e_class = self.class_prediction_error(r3, label)
        feedback = self.class_feedback(e_class, r3)

        dr3 += (self.k1 / self.sigma_class_sq) * feedback

        return dr1, dr2, dr3

    # ========================================================
    # FREE INFERENCE
    # Used in training, validation and testing
    # ========================================================

    @torch.no_grad()
    def infer(self, image, return_diagnostics=False):
        image = image.to(
            device=self.level1["conv"].weight.device,
            dtype=self.level1["conv"].weight.dtype,
        )

        r1, r2, r3 = self.initialize_states(image)

        step = 0
        converged = False

        change_r1 = float("inf")
        change_r2 = float("inf")
        change_r3 = float("inf")

        while True:
            old_r1 = r1.clone()
            old_r2 = r2.clone()
            old_r3 = r3.clone()

            dr1, dr2, dr3 = self.free_state_updates(image, r1, r2, r3)

            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2
            r3 = r3 + self.inference_dt * dr3

            step += 1

            if not torch.isfinite(r1).all():
                raise FloatingPointError(
                    "PC-CNN r1 free inference became non-finite."
                )

            if not torch.isfinite(r2).all():
                raise FloatingPointError(
                    "PC-CNN r2 free inference became non-finite."
                )

            if not torch.isfinite(r3).all():
                raise FloatingPointError(
                    "PC-CNN r3 free inference became non-finite."
                )

            change_r1 = torch.mean(torch.abs(r1 - old_r1)).item()
            change_r2 = torch.mean(torch.abs(r2 - old_r2)).item()
            change_r3 = torch.mean(torch.abs(r3 - old_r3)).item()

            if (
                change_r1 < self.convergence_tolerance
                and change_r2 < self.convergence_tolerance
                and change_r3 < self.convergence_tolerance
            ):
                converged = True
                break

            if self.inference_steps is not None and step >= self.inference_steps:
                break

        if return_diagnostics:
            return (
                r1,
                r2,
                r3,
                step,
                converged,
                change_r1,
                change_r2,
                change_r3,
            )

        return r1, r2, r3, step, converged

    # ========================================================
    # SUPERVISED NUDGE
    # Starts from the FREE state
    # ========================================================

    @torch.no_grad()
    def supervised_nudge(
        self,
        image,
        r1_free,
        r2_free,
        r3_free,
        label,
    ):
        image = image.to(
            device=self.level1["conv"].weight.device,
            dtype=self.level1["conv"].weight.dtype,
        )

        r1 = r1_free.clone()
        r2 = r2_free.clone()
        r3 = r3_free.clone()

        if self.supervised_nudge_steps == 0:
            return r1, r2, r3, 0, 0.0, 0.0, 0.0

        change_r1 = 0.0
        change_r2 = 0.0
        change_r3 = 0.0

        for _ in range(self.supervised_nudge_steps):
            old_r1 = r1.clone()
            old_r2 = r2.clone()
            old_r3 = r3.clone()

            dr1, dr2, dr3 = self.nudged_state_updates(
                image, r1, r2, r3, label
            )

            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2
            r3 = r3 + self.inference_dt * dr3

            if not torch.isfinite(r1).all():
                raise FloatingPointError(
                    "PC-CNN r1 supervised nudging became non-finite."
                )

            if not torch.isfinite(r2).all():
                raise FloatingPointError(
                    "PC-CNN r2 supervised nudging became non-finite."
                )

            if not torch.isfinite(r3).all():
                raise FloatingPointError(
                    "PC-CNN r3 supervised nudging became non-finite."
                )

            change_r1 = torch.mean(torch.abs(r1 - old_r1)).item()
            change_r2 = torch.mean(torch.abs(r2 - old_r2)).item()
            change_r3 = torch.mean(torch.abs(r3 - old_r3)).item()

        return (
            r1,
            r2,
            r3,
            self.supervised_nudge_steps,
            change_r1,
            change_r2,
            change_r3,
        )

    # ========================================================
    # LOCAL WEIGHT LEARNING
    #
    # U1/U2/U3:
    # local Rao-style predictive learning from NUDGED states
    #
    # Wc/bc:
    # independent local supervised learning from FREE r3
    #
    # NO GLOBAL BACKPROPAGATION
    # ========================================================

    @torch.no_grad()
    def update_weights(
        self,
        image,
        r1_nudged,
        r2_nudged,
        r3_nudged,
        r3_free,
        label,
    ):
        # ---------------- VISUAL U1 / U2 / U3 ----------------

        e0_nudged, e1_nudged, e2_nudged, _, _ = self.prediction_errors(
            image,
            r1_nudged,
            r2_nudged,
            r3_nudged,
        )

        a1_nudged = self.activation(r1_nudged)
        a2_nudged = self.activation(r2_nudged)
        a3_nudged = self.activation(r3_nudged)

        dW1 = conv2d_weight(
            e0_nudged,
            self.level1["conv"].weight.shape,
            a1_nudged,
            stride=self.level1["conv"].stride,
            padding=self.level1["conv"].padding,
        )

        dW2 = conv2d_weight(
            e1_nudged,
            self.level2["conv"].weight.shape,
            a2_nudged,
            stride=self.level2["conv"].stride,
            padding=self.level2["conv"].padding,
        )

        dW3 = conv2d_weight(
            e2_nudged,
            self.level3["conv"].weight.shape,
            a3_nudged,
            stride=self.level3["conv"].stride,
            padding=self.level3["conv"].padding,
        )

        dW1 = dW1 / (
            r1_nudged.shape[-2] * r1_nudged.shape[-1]
        )

        dW2 = dW2 / (
            r2_nudged.shape[-2] * r2_nudged.shape[-1]
        )

        dW3 = dW3 / (
            r3_nudged.shape[-2] * r3_nudged.shape[-1]
        )

        dW1 = (
            (self.k2 / self.sigma_bu_sq) * dW1
            - self.k2
            * self.lambda_u
            * self.level1["conv"].weight
        )

        dW2 = (
            (self.k2 / self.sigma_bu_sq) * dW2
            - self.k2
            * self.lambda_u
            * self.level2["conv"].weight
        )

        dW3 = (
            (self.k2 / self.sigma_bu_sq) * dW3
            - self.k2
            * self.lambda_u
            * self.level3["conv"].weight
        )

        self.level1["conv"].weight.add_(
            self.weight_dt * dW1
        )

        self.level2["conv"].weight.add_(
            self.weight_dt * dW2
        )

        self.level3["conv"].weight.add_(
            self.weight_dt * dW3
        )

        # ---------------- LOCAL CLASSIFIER ----------------

        features_free, _, _, _, e_class_free = (
            self.class_prediction_error(r3_free, label)
        )

        dWc = e_class_free.T @ features_free
        dbc = e_class_free.squeeze(0)

        self.classifier.weight.add_(
            self.class_learning_rate * dWc
        )

        self.classifier.bias.add_(
            self.class_learning_rate * dbc
        )

        # ---------------- NUMERICAL SAFETY ----------------

        if not torch.isfinite(
            self.level1["conv"].weight
        ).all():
            raise FloatingPointError(
                "PC-CNN level-1 weights became non-finite."
            )

        if not torch.isfinite(
            self.level2["conv"].weight
        ).all():
            raise FloatingPointError(
                "PC-CNN level-2 weights became non-finite."
            )

        if not torch.isfinite(
            self.level3["conv"].weight
        ).all():
            raise FloatingPointError(
                "PC-CNN level-3 weights became non-finite."
            )

        if not torch.isfinite(
            self.classifier.weight
        ).all():
            raise FloatingPointError(
                "PC-CNN classifier weights became non-finite."
            )

        if not torch.isfinite(
            self.classifier.bias
        ).all():
            raise FloatingPointError(
                "PC-CNN classifier bias became non-finite."
            )

    # ========================================================
    # RAO k2 DECAY
    # ========================================================

    def update_k2(self):
        self.training_input_count += 1

        if (
            self.training_input_count
            % self.k2_decay_interval
            == 0
        ):
            self.k2 = (
                self.k2
                / self.k2_decay_factor
            )

    # ========================================================
    # CLASSIFY ONE IMAGE
    # ========================================================

    @torch.no_grad()
    def classify(self, image):
        (
            r1,
            r2,
            r3,
            step,
            converged,
            change_r1,
            change_r2,
            change_r3,
        ) = self.infer(
            image,
            return_diagnostics=True,
        )

        features, scores, probabilities = (
            self.class_prediction(r3)
        )

        prediction = scores.argmax(dim=1)

        return {
            "scores": scores,
            "probabilities": probabilities,
            "prediction": prediction,
            "features": features,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "inference_steps": step,
            "converged": converged,
            "final_change_r1": change_r1,
            "final_change_r2": change_r2,
            "final_change_r3": change_r3,
        }

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
        final_change_r3 = []

        for image in images:
            result = self.classify(image)

            scores.append(
                result["scores"].squeeze(0)
            )

            probabilities.append(
                result["probabilities"].squeeze(0)
            )

            predictions.append(
                result["prediction"].squeeze(0)
            )

            steps.append(
                result["inference_steps"]
            )

            converged.append(
                result["converged"]
            )

            final_change_r1.append(
                result["final_change_r1"]
            )

            final_change_r2.append(
                result["final_change_r2"]
            )

            final_change_r3.append(
                result["final_change_r3"]
            )

        return {
            "scores": torch.stack(scores),
            "probabilities": torch.stack(probabilities),
            "predictions": torch.stack(predictions),
            "inference_steps": steps,
            "converged": converged,
            "final_change_r1": final_change_r1,
            "final_change_r2": final_change_r2,
            "final_change_r3": final_change_r3,
        }

    # ========================================================
    # RECONSTRUCTION
    # ========================================================

    @torch.no_grad()
    def reconstruct(self, image):
        image = image.to(
            device=self.level1["conv"].weight.device,
            dtype=self.level1["conv"].weight.dtype,
        )

        r1, r2, r3, step, converged = self.infer(image)

        reconstruction = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        predicted_r2 = self.predict_r2(r3)

        _, scores, probabilities = self.class_prediction(r3)

        predicted_class = scores.argmax(dim=1)

        return {
            "reconstruction": reconstruction.squeeze(0),
            "predicted_r1": predicted_r1.squeeze(0),
            "predicted_r2": predicted_r2.squeeze(0),
            "r1": r1.squeeze(0),
            "r2": r2.squeeze(0),
            "r3": r3.squeeze(0),
            "probabilities": probabilities.squeeze(0),
            "predicted_class": predicted_class.item(),
            "inference_steps": step,
            "converged": converged,
        }

    # ========================================================
    # TRAIN ONE IMAGE
    # ========================================================

    @torch.no_grad()
    def forward(self, image, label=None):
        if self.training and label is None:
            raise ValueError(
                "A class label is required during supervised "
                "predictive-coding training."
            )

        image = image.to(
            device=self.level1["conv"].weight.device,
            dtype=self.level1["conv"].weight.dtype,
        )

        # ---------------- FREE PHASE ----------------

        (
            r1_free,
            r2_free,
            r3_free,
            free_steps,
            free_converged,
            free_change_r1,
            free_change_r2,
            free_change_r3,
        ) = self.infer(
            image,
            return_diagnostics=True,
        )

        (
            e0_free,
            e1_free,
            e2_free,
            _,
            _,
        ) = self.prediction_errors(
            image,
            r1_free,
            r2_free,
            r3_free,
        )

        (
            _,
            scores_free,
            probabilities_free,
            _,
            e_class_free,
        ) = self.class_prediction_error(
            r3_free,
            label,
        )

        error_0 = torch.sqrt(
            torch.mean(e0_free ** 2)
        ).item()

        error_1 = torch.sqrt(
            torch.mean(e1_free ** 2)
        ).item()

        error_2 = torch.sqrt(
            torch.mean(e2_free ** 2)
        ).item()

        class_error = torch.sqrt(
            torch.mean(e_class_free ** 2)
        ).item()

        label_value = (
            int(label.item())
            if torch.is_tensor(label)
            else int(label)
        )

        true_probability = probabilities_free[
            0, label_value
        ]

        class_loss = -torch.log(
            true_probability + 1e-8
        ).item()

        prediction = scores_free.argmax(
            dim=1
        ).item()

        # ---------------- NUDGED PHASE ----------------

        (
            r1_nudged,
            r2_nudged,
            r3_nudged,
            nudge_steps,
            nudge_change_r1,
            nudge_change_r2,
            nudge_change_r3,
        ) = self.supervised_nudge(
            image,
            r1_free,
            r2_free,
            r3_free,
            label,
        )

        (
            e0_nudged,
            e1_nudged,
            e2_nudged,
            _,
            _,
        ) = self.prediction_errors(
            image,
            r1_nudged,
            r2_nudged,
            r3_nudged,
        )

        nudged_error_0 = torch.sqrt(
            torch.mean(e0_nudged ** 2)
        ).item()

        nudged_error_1 = torch.sqrt(
            torch.mean(e1_nudged ** 2)
        ).item()

        nudged_error_2 = torch.sqrt(
            torch.mean(e2_nudged ** 2)
        ).item()

        # ---------------- LOCAL LEARNING ----------------

        if self.training:
            self.update_weights(
                image,
                r1_nudged,
                r2_nudged,
                r3_nudged,
                r3_free,
                label,
            )

            self.update_k2()

        return {
            "inference_steps": free_steps,
            "converged": free_converged,
            "final_change_r1": free_change_r1,
            "final_change_r2": free_change_r2,
            "final_change_r3": free_change_r3,
            "nudge_steps": nudge_steps,
            "nudge_change_r1": nudge_change_r1,
            "nudge_change_r2": nudge_change_r2,
            "nudge_change_r3": nudge_change_r3,
            "error_0": error_0,
            "error_1": error_1,
            "error_2": error_2,
            "class_error": class_error,
            "nudged_error_0": nudged_error_0,
            "nudged_error_1": nudged_error_1,
            "nudged_error_2": nudged_error_2,
            "class_loss": class_loss,
            "prediction": prediction,
            "probabilities": probabilities_free.squeeze(0),
            "k2": self.k2,
            "training_input_count": self.training_input_count,
        }
