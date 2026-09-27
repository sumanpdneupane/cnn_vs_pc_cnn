import torch
import torch.nn as nn
from torch.nn.grad import conv2d_weight


# ============================================================
# HIERARCHICAL RAO-STYLE PC-CNN
# CONVOLUTIONAL ADAPTATION OF RAO & BALLARD
# ============================================================

class HierarchicalPCCNN(nn.Module):

    def __init__(self, inference_steps=1000):
        super().__init__()

        # ====================================================
        # RAO PARAMETERS
        # ====================================================

        self.k1 = 0.5
        self.k2 = 1.0

        self.sigma_bu_sq = 1.0
        self.sigma_td_sq = 10.0

        self.alpha1 = 0.001 # 1.0
        self.alpha2 = 0.01 #0.05

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

        # Maximum predictive-coding inference steps
        self.inference_steps = inference_steps

        # Euler integration step for Rao Eq. 7
        self.inference_dt = 0.2 #0.1->50 #0.005

        # Euler integration step for Rao Eq. 9
        self.weight_dt = 0.01

        # ====================================================
        # BOTTOM-UP PREDICTION-ERROR CORRECTION PATHWAYS
        # ====================================================

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False)

        # ====================================================
        # TOP-DOWN GENERATIVE PREDICTION PATHWAYS
        # ====================================================

        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

        # ====================================================
        # SMALL RANDOM WEIGHT INITIALIZATION
        # ====================================================

        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

        # ====================================================
        # TIED BOTTOM-UP / TOP-DOWN WEIGHTS
        #
        # U   -> top-down prediction
        # U^T -> bottom-up prediction-error correction
        # ====================================================

        self.deconv1.weight = self.conv1.weight
        self.deconv2.weight = self.conv2.weight

        # ====================================================
        # NO AUTOGRAD / BACKPROPAGATION
        # ====================================================

        self.conv1.weight.requires_grad_(False)
        self.conv2.weight.requires_grad_(False)

    # ========================================================
    # INITIALIZE LATENT REPRESENTATIONS
    # ========================================================

    # def initialize_states(self, image):
    #     image_batch = image.unsqueeze(0)
    #
    #     r1_shape = self.conv1(image_batch).shape
    #     r2_shape = self.conv2(torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape
    #
    #     r1 = torch.randn(r1_shape, device=image.device, dtype=image.dtype) * 0.01
    #     r2 = torch.randn(r2_shape, device=image.device, dtype=image.dtype) * 0.01
    #
    #     return r1, r2

    def initialize_states(self, image):
        image_batch = image.unsqueeze(0)

        r1 = self.conv1(image_batch)
        r2 = self.conv2(r1)

        return r1, r2

    # ========================================================
    # TOP-DOWN GENERATIVE PREDICTIONS
    # ========================================================

    def predict_image(self, r1):
        return self.deconv1(r1)

    def predict_r1(self, r2):
        return self.deconv2(r2)

    # ========================================================
    # PREDICTION ERRORS
    #
    # e0 = I  - U1 r1
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

        # ----------------------------------------------------
        # LEVEL 1
        #
        # dr1/dt =
        # (k1 / sigma^2) U1^T e0
        # +
        # (k1 / sigma_td^2) (U2 r2 - r1)
        # -
        # k1 alpha1 r1
        # ----------------------------------------------------

        dr1 = (self.k1 / self.sigma_bu_sq) * self.conv1(e0)
        dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
        dr1 -= self.k1 * self.alpha1 * r1

        # ----------------------------------------------------
        # LEVEL 2
        #
        # Highest level: no r3 top-down prediction
        #
        # dr2/dt =
        # (k1 / sigma^2) U2^T e1
        # -
        # k1 alpha2 r2
        # ----------------------------------------------------

        dr2 = (self.k1 / self.sigma_bu_sq) * self.conv2(e1)
        dr2 -= self.k1 * self.alpha2 * r2

        return dr1, dr2

    # ========================================================
    # PREDICTIVE-CODING INFERENCE
    # EULER INTEGRATION OF RAO EQ. 7
    # ========================================================

    @torch.no_grad()
    def infer(self, image):
        image = image.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)

        r1, r2 = self.initialize_states(image)

        step = 0
        converged = False

        while True:
            old_r1 = r1.clone()
            old_r2 = r2.clone()

            dr1, dr2 = self.state_updates(image, r1, r2)

            # Euler integration of Rao Eq. 7
            r1 = r1 + self.inference_dt * dr1
            r2 = r2 + self.inference_dt * dr2

            step += 1

            # Numerical stability check
            if not torch.isfinite(r1).all() or not torch.isfinite(r2).all():
                raise FloatingPointError("PC-CNN inference became non-finite.")

            # Automatic numerical convergence
            if torch.allclose(r1, old_r1) and torch.allclose(r2, old_r2):
                converged = True
                break

            # Maximum inference-step cap
            if self.inference_steps is not None and step >= self.inference_steps:
                break

        return r1, r2, step, converged

    @torch.no_grad()
    def infer(self, image):
        image = image.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)

        r1, r2 = self.initialize_states(image)

        step = 0
        converged = False

        while True:
            old_r1 = r1.clone()
            old_r2 = r2.clone()

            dr1, dr2 = self.state_updates(image, r1, r2)

            if step < 25:
                dt = 0.6
            elif step < 75:
                dt = 0.4
            else:
                dt = 0.2

            r1 = r1 + dt * dr1
            r2 = r2 + dt * dr2

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

        # ----------------------------------------------------
        # Convolutional equivalent of Rao Eq. 9
        #
        # dU/dt =
        # (k2 / sigma^2) e r^T
        # -
        # k2 lambda U
        #
        # Spatial averaging for convolutional weight sharing
        # ----------------------------------------------------

        # Convolutional equivalent of local prediction-error × representation
        dW1 = conv2d_weight(e0, self.conv1.weight.shape, r1, stride=self.conv1.stride, padding=self.conv1.padding)
        dW2 = conv2d_weight(e1, self.conv2.weight.shape, r2, stride=self.conv2.stride, padding=self.conv2.padding)

        # Average accumulated contributions caused by convolutional weight sharing
        dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])
        dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])

        # Rao Eq. 9
        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.conv1.weight
        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.conv2.weight

        # Numerical integration
        self.conv1.weight.add_(self.weight_dt * dW1)
        self.conv2.weight.add_(self.weight_dt * dW2)

        if not torch.isfinite(self.conv1.weight).all() or not torch.isfinite(self.conv2.weight).all():
            raise FloatingPointError("PC-CNN weights became non-finite.")

    # ========================================================
    # RAO k2 DECAY
    # DIVIDE k2 BY 1.015 AFTER EVERY 40 TRAINING INPUTS
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
        image = image.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)

        r1, r2, step, converged = self.infer(image)

        reconstruction = self.predict_image(r1)
        predicted_r1 = self.predict_r1(r2)
        reconstruction_r2 = self.predict_image(predicted_r1)

        return reconstruction.squeeze(0), predicted_r1.squeeze(0),reconstruction_r2.squeeze(0), r2.squeeze(0), step, converged


    # ========================================================
    # FORWARD
    # TRAIN ONE IMAGE USING LOCAL PREDICTIVE-CODING LEARNING
    # ========================================================

    @torch.no_grad()
    def forward(self, image):
        image = image.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)

        # 1. Iterative predictive-coding inference
        r1, r2, step, converged = self.infer(image)

        # 2. Prediction errors at final inferred states
        e0, e1, _ = self.prediction_errors(image, r1, r2)

        error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
        error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()

        # 3. Rao Eq. 9 local weight learning
        self.update_weights(image, r1, r2)

        # 4. Rao k2 learning-rate schedule
        self.update_k2()

        return {
            "inference_steps": step,
            "converged": converged,
            "error_0": error_0,
            "error_1": error_1,
            "k2": self.k2,
            "training_input_count": self.training_input_count,
        }