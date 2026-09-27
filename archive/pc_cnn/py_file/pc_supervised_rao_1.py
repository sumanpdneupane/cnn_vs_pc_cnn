# import torch
# import torch.nn as nn
#
# from torch.nn.grad import conv2d_weight
#
#
# # ============================================================
# # SUPERVISED HIERARCHICAL RAO-STYLE PC-CNN
# #
# # Convolutional adaptation / extension of Rao & Ballard
# #
# # Hierarchy:
# #
# # Image <-> r1 <-> r2 <-> class state
# #
# # NO AUTOGRAD
# # NO BACKPROPAGATION
# # NO OPTIMIZER
# # ============================================================
#
# class HierarchicalPCCNN(nn.Module):
#
#     def __init__(self, num_classes, input_size=64, inference_steps=100, convergence_tolerance=1e-5):
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.input_size = input_size
#
#         # ====================================================
#         # RAO PARAMETERS
#         # ====================================================
#
#         self.k1 = 0.5
#         self.k2 = 1.0
#
#         self.sigma_bu_sq = 1.0
#         self.sigma_td_sq = 10.0
#
#         self.alpha1 = 0.001
#         self.alpha2 = 0.05
#         self.alpha3 = 0.05
#
#         self.lambda_u = 0.02
#
#         # ====================================================
#         # RAO k2 LEARNING-RATE SCHEDULE
#         # ====================================================
#
#         self.k2_decay_interval = 40
#         self.k2_decay_factor = 1.015
#         self.training_input_count = 0
#
#         # ====================================================
#         # NUMERICAL INTEGRATION
#         # ====================================================
#
#         self.inference_steps = inference_steps
#         self.inference_dt = 0.1
#         self.weight_dt = 0.01
#         self.convergence_tolerance = convergence_tolerance
#
#         # ====================================================
#         # SPATIAL DIMENSIONS
#         #
#         # stride=2 convolution:
#         #
#         # 64 -> 32 -> 16
#         # 32 -> 16 -> 8
#         # 224 -> 112 -> 56
#         # ====================================================
#
#         self.r1_size = (input_size + 1) // 2
#         self.r2_size = (self.r1_size + 1) // 2
#
#         # ====================================================
#         # LEVEL 1
#         #
#         # Image -> r1
#         # r1 -> predicted image
#         # ====================================================
#
#         self.level1 = nn.ModuleDict({
#             "conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         })
#
#         # ====================================================
#         # LEVEL 2
#         #
#         # r1 -> r2
#         # r2 -> predicted r1
#         # ====================================================
#
#         self.level2 = nn.ModuleDict({
#             "conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
#             "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         })
#
#         # ====================================================
#         # LEVEL 3
#         #
#         # r2 -> class state
#         #
#         # For input_size=64:
#         #
#         # r2 = [128,16,16]
#         #
#         # Bottom-up:
#         # [128,16,16] -> [num_classes,1,1]
#         #
#         # Top-down:
#         # [num_classes,1,1] -> [128,16,16]
#         # ====================================================
#
#         self.level3 = nn.ModuleDict({
#             "conv": nn.Conv2d(128, num_classes, kernel_size=self.r2_size, stride=1, padding=0, bias=False),
#             "deconv": nn.ConvTranspose2d(num_classes, 128, kernel_size=self.r2_size, stride=1, padding=0, bias=False)
#         })
#
#         # ====================================================
#         # WEIGHT INITIALIZATION
#         # ====================================================
#
#         nn.init.normal_(self.level1["conv"].weight, mean=0.0, std=0.01)
#         nn.init.normal_(self.level2["conv"].weight, mean=0.0, std=0.01)
#         nn.init.normal_(self.level3["conv"].weight, mean=0.0, std=0.01)
#
#         # ====================================================
#         # TIED BOTTOM-UP / TOP-DOWN WEIGHTS
#         # ====================================================
#
#         self.level1["deconv"].weight = self.level1["conv"].weight
#         self.level2["deconv"].weight = self.level2["conv"].weight
#         self.level3["deconv"].weight = self.level3["conv"].weight
#
#         # ====================================================
#         # NO AUTOGRAD / NO BACKPROPAGATION
#         # ====================================================
#
#         self.level1["conv"].weight.requires_grad_(False)
#         self.level2["conv"].weight.requires_grad_(False)
#         self.level3["conv"].weight.requires_grad_(False)
#
#     # ========================================================
#     # CHECK IMAGE SIZE
#     # ========================================================
#
#     def validate_image_size(self, image):
#         if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
#             raise ValueError(f"Expected image size {self.input_size}x{self.input_size}, but received {image.shape[-2]}x{image.shape[-1]}.")
#
#     # ========================================================
#     # CREATE CLAMPED CLASS STATE
#     #
#     # Example for class 2 of 5:
#     #
#     # [0, 0, 1, 0, 0]
#     # ========================================================
#
#     def create_class_state(self, label, device, dtype):
#         class_state = torch.zeros(1, self.num_classes, 1, 1, device=device, dtype=dtype)
#         class_state[0, int(label), 0, 0] = 1.0
#
#         return class_state
#
#     # ========================================================
#     # INITIALIZE LATENT STATES
#     #
#     # r1 = 0
#     # r2 = 0
#     #
#     # Training:
#     # class state = known one-hot label
#     #
#     # Testing:
#     # class state = zero and inferred
#     # ========================================================
#
#     def initialize_states(self, image, label=None):
#         self.validate_image_size(image)
#
#         image_batch = image.unsqueeze(0)
#
#         r1_shape = self.level1["conv"](image_batch).shape
#         r2_shape = self.level2["conv"](torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape
#
#         r1 = torch.zeros(r1_shape, device=image.device, dtype=image.dtype)
#         r2 = torch.zeros(r2_shape, device=image.device, dtype=image.dtype)
#
#         if label is None:
#             class_state = torch.zeros(1, self.num_classes, 1, 1, device=image.device, dtype=image.dtype)
#         else:
#             class_state = self.create_class_state(label, image.device, image.dtype)
#
#         return r1, r2, class_state
#
#     # ========================================================
#     # TOP-DOWN GENERATIVE PREDICTIONS
#     # ========================================================
#
#     def predict_image(self, r1):
#         return self.level1["deconv"](r1)
#
#     def predict_r1(self, r2):
#         return self.level2["deconv"](r2)
#
#     def predict_r2(self, class_state):
#         return self.level3["deconv"](class_state)
#
#     # ========================================================
#     # HIERARCHICAL PREDICTION ERRORS
#     #
#     # e0 = image - U1 r1
#     #
#     # e1 = r1 - U2 r2
#     #
#     # e2 = r2 - U3 class_state
#     # ========================================================
#
#     def prediction_errors(self, image, r1, r2, class_state):
#         predicted_image = self.predict_image(r1)
#         predicted_r1 = self.predict_r1(r2)
#         predicted_r2 = self.predict_r2(class_state)
#
#         e0 = image.unsqueeze(0) - predicted_image
#         e1 = r1 - predicted_r1
#         e2 = r2 - predicted_r2
#
#         return e0, e1, e2, predicted_r1, predicted_r2
#
#     # ========================================================
#     # RAO-STYLE REPRESENTATION DYNAMICS
#     # ========================================================
#
#     def state_updates(self, image, r1, r2, class_state, class_clamped):
#         e0, e1, e2, predicted_r1, predicted_r2 = self.prediction_errors(image, r1, r2, class_state)
#
#         # ====================================================
#         # LEVEL 1
#         #
#         # bottom-up e0
#         # + top-down prediction from r2
#         # - prior
#         # ====================================================
#
#         dr1 = (self.k1 / self.sigma_bu_sq) * self.level1["conv"](e0)
#         dr1 += (self.k1 / self.sigma_td_sq) * (predicted_r1 - r1)
#         dr1 -= self.k1 * self.alpha1 * r1
#
#         # ====================================================
#         # LEVEL 2
#         #
#         # bottom-up e1
#         # + top-down class prediction
#         # - prior
#         # ====================================================
#
#         dr2 = (self.k1 / self.sigma_bu_sq) * self.level2["conv"](e1)
#         dr2 += (self.k1 / self.sigma_td_sq) * (predicted_r2 - r2)
#         dr2 -= self.k1 * self.alpha2 * r2
#
#         # ====================================================
#         # LEVEL 3 / CLASS STATE
#         #
#         # During training:
#         # class state is clamped -> dc = 0
#         #
#         # During testing:
#         # class state is inferred from e2
#         # ====================================================
#
#         if class_clamped:
#             dc = torch.zeros_like(class_state)
#         else:
#             dc = (self.k1 / self.sigma_bu_sq) * self.level3["conv"](e2)
#             dc -= self.k1 * self.alpha3 * class_state
#
#         return dr1, dr2, dc
#
#     # ========================================================
#     # PREDICTIVE-CODING INFERENCE
#     #
#     # Training:
#     # label provided -> class state clamped
#     #
#     # Testing:
#     # label=None -> class state inferred
#     # ========================================================
#
#     @torch.no_grad()
#     def infer(self, image, label=None, return_diagnostics=False):
#         image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)
#
#         class_clamped = label is not None
#
#         r1, r2, class_state = self.initialize_states(image, label)
#
#         step = 0
#         converged = False
#
#         change_r1 = float("inf")
#         change_r2 = float("inf")
#         change_class = 0.0 if class_clamped else float("inf")
#
#         while True:
#             old_r1 = r1.clone()
#             old_r2 = r2.clone()
#             old_class_state = class_state.clone()
#
#             dr1, dr2, dc = self.state_updates(image, r1, r2, class_state, class_clamped)
#
#             r1 = r1 + self.inference_dt * dr1
#             r2 = r2 + self.inference_dt * dr2
#
#             if not class_clamped:
#                 class_state = class_state + self.inference_dt * dc
#
#             step += 1
#
#             if not torch.isfinite(r1).all():
#                 raise FloatingPointError("PC-CNN r1 inference became non-finite.")
#
#             if not torch.isfinite(r2).all():
#                 raise FloatingPointError("PC-CNN r2 inference became non-finite.")
#
#             if not torch.isfinite(class_state).all():
#                 raise FloatingPointError("PC-CNN class-state inference became non-finite.")
#
#             change_r1 = torch.mean(torch.abs(r1 - old_r1)).item()
#             change_r2 = torch.mean(torch.abs(r2 - old_r2)).item()
#
#             if class_clamped:
#                 change_class = 0.0
#             else:
#                 change_class = torch.mean(torch.abs(class_state - old_class_state)).item()
#
#             representation_converged = change_r1 < self.convergence_tolerance and change_r2 < self.convergence_tolerance
#             class_converged = class_clamped or change_class < self.convergence_tolerance
#
#             if representation_converged and class_converged:
#                 converged = True
#                 break
#
#             if self.inference_steps is not None and step >= self.inference_steps:
#                 break
#
#         if return_diagnostics:
#             return r1, r2, class_state, step, converged, change_r1, change_r2, change_class
#
#         return r1, r2, class_state, step, converged
#
#     # ========================================================
#     # RAO EQ. 9-STYLE LOCAL GENERATIVE WEIGHT LEARNING
#     #
#     # U1 <- e0 × r1
#     # U2 <- e1 × r2
#     # U3 <- e2 × class_state
#     # ========================================================
#
#     @torch.no_grad()
#     def update_weights(self, image, r1, r2, class_state):
#         e0, e1, e2, _, _ = self.prediction_errors(image, r1, r2, class_state)
#
#         # ====================================================
#         # LOCAL WEIGHT CORRELATIONS
#         # ====================================================
#
#         dW1 = conv2d_weight(e0, self.level1["conv"].weight.shape, r1, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)
#         dW2 = conv2d_weight(e1, self.level2["conv"].weight.shape, r2, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)
#         dW3 = conv2d_weight(e2, self.level3["conv"].weight.shape, class_state, stride=self.level3["conv"].stride, padding=self.level3["conv"].padding)
#
#         # ====================================================
#         # SPATIAL AVERAGING FOR SHARED CONVOLUTIONAL FILTERS
#         # ====================================================
#
#         dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])
#         dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])
#
#         # U3 has a 1×1 class representation output, so no
#         # spatial averaging is required.
#         # ====================================================
#
#         # ====================================================
#         # RAO EQ. 9-STYLE UPDATE + WEIGHT DECAY
#         # ====================================================
#
#         dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight
#         dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight
#         dW3 = (self.k2 / self.sigma_bu_sq) * dW3 - self.k2 * self.lambda_u * self.level3["conv"].weight
#
#         # ====================================================
#         # EULER WEIGHT INTEGRATION
#         # ====================================================
#
#         self.level1["conv"].weight.add_(self.weight_dt * dW1)
#         self.level2["conv"].weight.add_(self.weight_dt * dW2)
#         self.level3["conv"].weight.add_(self.weight_dt * dW3)
#
#         # ====================================================
#         # NUMERICAL SAFETY
#         # ====================================================
#
#         if not torch.isfinite(self.level1["conv"].weight).all():
#             raise FloatingPointError("PC-CNN level-1 weights became non-finite.")
#
#         if not torch.isfinite(self.level2["conv"].weight).all():
#             raise FloatingPointError("PC-CNN level-2 weights became non-finite.")
#
#         if not torch.isfinite(self.level3["conv"].weight).all():
#             raise FloatingPointError("PC-CNN level-3 weights became non-finite.")
#
#     # ========================================================
#     # RAO k2 DECAY
#     # ========================================================
#
#     def update_k2(self):
#         self.training_input_count += 1
#
#         if self.training_input_count % self.k2_decay_interval == 0:
#             self.k2 = self.k2 / self.k2_decay_factor
#
#     # ========================================================
#     # CLASSIFICATION FOR ONE IMAGE
#     #
#     # No label is supplied.
#     #
#     # Class state is inferred.
#     # ========================================================
#
#     @torch.no_grad()
#     def classify(self, image):
#         r1, r2, class_state, step, converged, change_r1, change_r2, change_class = self.infer(image, label=None, return_diagnostics=True)
#
#         class_scores = class_state.squeeze(-1).squeeze(-1)
#         probabilities = torch.softmax(class_scores, dim=1)
#         prediction = class_scores.argmax(dim=1)
#
#         return {
#             "scores": class_scores,
#             "probabilities": probabilities,
#             "prediction": prediction,
#             "r1": r1,
#             "r2": r2,
#             "class_state": class_state,
#             "inference_steps": step,
#             "converged": converged,
#             "final_change_r1": change_r1,
#             "final_change_r2": change_r2,
#             "final_change_class": change_class
#         }
#
#     # ========================================================
#     # CLASSIFICATION FOR BATCH
#     #
#     # Inference remains image-by-image.
#     # ========================================================
#
#     @torch.no_grad()
#     def classify_batch(self, images):
#         scores = []
#         probabilities = []
#         predictions = []
#         steps = []
#         converged = []
#
#         for image in images:
#             result = self.classify(image)
#
#             scores.append(result["scores"].squeeze(0))
#             probabilities.append(result["probabilities"].squeeze(0))
#             predictions.append(result["prediction"].squeeze(0))
#
#             steps.append(result["inference_steps"])
#             converged.append(result["converged"])
#
#         return {
#             "scores": torch.stack(scores),
#             "probabilities": torch.stack(probabilities),
#             "predictions": torch.stack(predictions),
#             "inference_steps": steps,
#             "converged": converged
#         }
#
#     # ========================================================
#     # IMAGE RECONSTRUCTION
#     #
#     # Default:
#     # class state inferred from image.
#     # ========================================================
#
#     @torch.no_grad()
#     def reconstruct(self, image, label=None):
#         image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)
#
#         r1, r2, class_state, step, converged = self.infer(image, label=label)
#
#         reconstruction = self.predict_image(r1)
#         predicted_r1 = self.predict_r1(r2)
#         predicted_r2 = self.predict_r2(class_state)
#
#         class_scores = class_state.squeeze(-1).squeeze(-1)
#         predicted_class = class_scores.argmax(dim=1)
#
#         return reconstruction.squeeze(0), predicted_r1.squeeze(0), predicted_r2.squeeze(0), r2.squeeze(0), class_state.squeeze(0), predicted_class.item(), step, converged
#
#     # ========================================================
#     # TRAINING FOR ONE IMAGE
#     #
#     # Correct class is CLAMPED.
#     #
#     # Learning occurs through local prediction errors.
#     # ========================================================
#
#     @torch.no_grad()
#     def forward(self, image, label=None):
#         if self.training and label is None:
#             raise ValueError("A class label is required while training the supervised hierarchical PC-CNN.")
#
#         image = image.to(device=self.level1["conv"].weight.device, dtype=self.level1["conv"].weight.dtype)
#
#         r1, r2, class_state, step, converged, change_r1, change_r2, change_class = self.infer(image, label=label, return_diagnostics=True)
#
#         e0, e1, e2, _, _ = self.prediction_errors(image, r1, r2, class_state)
#
#         error_0 = torch.sqrt(torch.mean(e0 ** 2)).item()
#         error_1 = torch.sqrt(torch.mean(e1 ** 2)).item()
#         error_2 = torch.sqrt(torch.mean(e2 ** 2)).item()
#
#         if self.training:
#             self.update_weights(image, r1, r2, class_state)
#             self.update_k2()
#
#         return {
#             "inference_steps": step,
#             "converged": converged,
#             "final_change_r1": change_r1,
#             "final_change_r2": change_r2,
#             "final_change_class": change_class,
#             "error_0": error_0,
#             "error_1": error_1,
#             "error_2": error_2,
#             "k2": self.k2,
#             "training_input_count": self.training_input_count
#         }

import torch
import torch.nn as nn

from torch.nn.grad import conv2d_weight


# ============================================================
# SUPERVISED HIERARCHICAL RAO-STYLE PC-CNN
#
# Image <-> r1 <-> r2 <-> class state
#
# NO AUTOGRAD
# NO BACKPROPAGATION
# NO OPTIMIZER
# ============================================================

class HierarchicalPCCNN(nn.Module):

    def __init__(self, num_classes, input_size=64, inference_steps=100, convergence_tolerance=1e-5):
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # ====================================================
        # RAO PARAMETERS
        # ====================================================

        self.k1 = 0.5
        self.k2 = 1.0

        self.sigma_bu_sq = 1.0
        self.sigma_td_sq = 10.0

        self.alpha1 = 0.001
        self.alpha2 = 0.05
        self.alpha3 = 0.05

        self.lambda_u = 0.02

        # ====================================================
        # k2 LEARNING-RATE SCHEDULE
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
        # SPATIAL SIZES
        # ====================================================

        self.r1_size = (input_size + 1) // 2
        self.r2_size = (self.r1_size + 1) // 2

        # ====================================================
        # LEVEL 1
        #
        # 64x64 -> 32x32
        # ====================================================

        self.level1 = nn.ModuleDict({
            "conv": nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # LEVEL 2
        #
        # 32x32 -> 16x16
        # ====================================================

        self.level2 = nn.ModuleDict({
            "conv": nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            "deconv": nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        })

        # ====================================================
        # LEVEL 3
        #
        # r2 [128,16,16] <-> class state [K,1,1]
        # ====================================================

        self.level3 = nn.ModuleDict({
            "conv": nn.Conv2d(128, num_classes, kernel_size=self.r2_size, stride=1, padding=0, bias=False),
            "deconv": nn.ConvTranspose2d(num_classes, 128, kernel_size=self.r2_size, stride=1, padding=0, bias=False)
        })

        # ====================================================
        # INITIALIZATION
        # ====================================================

        nn.init.normal_(self.level1["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level2["conv"].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.level3["conv"].weight, mean=0.0, std=0.01)

        # ====================================================
        # TIED WEIGHTS
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

    # ========================================================
    # IMAGE SIZE CHECK
    # ========================================================

    def validate_image_size(self, image):
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            raise ValueError(f"Expected image size {self.input_size}x{self.input_size}, but received {image.shape[-2]}x{image.shape[-1]}.")

    # ========================================================
    # CLASS STATE
    # ========================================================

    def create_class_state(self, label, device, dtype):
        class_state = torch.zeros(1, self.num_classes, 1, 1, device=device, dtype=dtype)
        class_state[0, int(label), 0, 0] = 1.0

        return class_state

    # ========================================================
    # INITIAL STATES
    # ========================================================

    def initialize_states(self, image, label=None):
        self.validate_image_size(image)

        image_batch = image.unsqueeze(0)

        r1_shape = self.level1["conv"](image_batch).shape
        r2_shape = self.level2["conv"](torch.zeros(r1_shape, device=image.device, dtype=image.dtype)).shape

        r1 = torch.zeros(r1_shape, device=image.device, dtype=image.dtype)
        r2 = torch.zeros(r2_shape, device=image.device, dtype=image.dtype)

        if label is None:
            class_state = torch.zeros(1, self.num_classes, 1, 1, device=image.device, dtype=image.dtype)
        else:
            class_state = self.create_class_state(label, image.device, image.dtype)

        return r1, r2, class_state

    # ========================================================
    # GENERATIVE PREDICTIONS
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
    # e1 = r1 - U2 r2
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
    # REPRESENTATION DYNAMICS
    # ========================================================

    def state_updates(self, image, r1, r2, class_state, class_clamped):
        e0, e1, e2, predicted_r1, predicted_r2 = self.prediction_errors(image, r1, r2, class_state)

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
        dr2 += (self.k1 / self.sigma_td_sq) * (predicted_r2 - r2)
        dr2 -= self.k1 * self.alpha2 * r2

        # ====================================================
        # LEVEL 3
        #
        # Training:
        # class state is clamped.
        #
        # Testing:
        # class state inferred through normalized local
        # prediction-error correction.
        # ====================================================

        if class_clamped:
            dc = torch.zeros_like(class_state)

        else:
            class_drive = self.level3["conv"](e2)

            class_weight_energy = self.level3["conv"].weight.pow(2).flatten(1).sum(dim=1)
            class_weight_energy = class_weight_energy.view(1, self.num_classes, 1, 1)
            class_weight_energy = class_weight_energy.clamp_min(1e-6)

            dc = (self.k1 / self.sigma_bu_sq) * (class_drive / class_weight_energy)
            dc -= self.k1 * self.alpha3 * class_state

        return dr1, dr2, dc

    # ========================================================
    # PREDICTIVE-CODING INFERENCE
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
    # LOCAL GENERATIVE WEIGHT LEARNING
    # ========================================================

    @torch.no_grad()
    def update_weights(self, image, r1, r2, class_state):
        e0, e1, e2, _, _ = self.prediction_errors(image, r1, r2, class_state)

        dW1 = conv2d_weight(e0, self.level1["conv"].weight.shape, r1, stride=self.level1["conv"].stride, padding=self.level1["conv"].padding)
        dW2 = conv2d_weight(e1, self.level2["conv"].weight.shape, r2, stride=self.level2["conv"].stride, padding=self.level2["conv"].padding)
        dW3 = conv2d_weight(e2, self.level3["conv"].weight.shape, class_state, stride=self.level3["conv"].stride, padding=self.level3["conv"].padding)

        dW1 = dW1 / (r1.shape[-2] * r1.shape[-1])
        dW2 = dW2 / (r2.shape[-2] * r2.shape[-1])

        dW1 = (self.k2 / self.sigma_bu_sq) * dW1 - self.k2 * self.lambda_u * self.level1["conv"].weight
        dW2 = (self.k2 / self.sigma_bu_sq) * dW2 - self.k2 * self.lambda_u * self.level2["conv"].weight
        dW3 = (self.k2 / self.sigma_bu_sq) * dW3 - self.k2 * self.lambda_u * self.level3["conv"].weight

        self.level1["conv"].weight.add_(self.weight_dt * dW1)
        self.level2["conv"].weight.add_(self.weight_dt * dW2)
        self.level3["conv"].weight.add_(self.weight_dt * dW3)

        if not torch.isfinite(self.level1["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-1 weights became non-finite.")

        if not torch.isfinite(self.level2["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-2 weights became non-finite.")

        if not torch.isfinite(self.level3["conv"].weight).all():
            raise FloatingPointError("PC-CNN level-3 weights became non-finite.")

    # ========================================================
    # k2 DECAY
    # ========================================================

    def update_k2(self):
        self.training_input_count += 1

        if self.training_input_count % self.k2_decay_interval == 0:
            self.k2 = self.k2 / self.k2_decay_factor

    # ========================================================
    # CLASSIFY ONE IMAGE
    # ========================================================

    @torch.no_grad()
    def classify(self, image):
        r1, r2, class_state, step, converged, change_r1, change_r2, change_class = self.infer(image, label=None, return_diagnostics=True)

        class_scores = class_state.squeeze(-1).squeeze(-1)

        probabilities = torch.softmax(class_scores, dim=1)
        prediction = class_scores.argmax(dim=1)

        return {
            "scores": class_scores,
            "probabilities": probabilities,
            "prediction": prediction,
            "r1": r1,
            "r2": r2,
            "class_state": class_state,
            "inference_steps": step,
            "converged": converged,
            "final_change_r1": change_r1,
            "final_change_r2": change_r2,
            "final_change_class": change_class
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

        for image in images:
            result = self.classify(image)

            scores.append(result["scores"].squeeze(0))
            probabilities.append(result["probabilities"].squeeze(0))
            predictions.append(result["prediction"].squeeze(0))

            steps.append(result["inference_steps"])
            converged.append(result["converged"])

        return {
            "scores": torch.stack(scores),
            "probabilities": torch.stack(probabilities),
            "predictions": torch.stack(predictions),
            "inference_steps": steps,
            "converged": converged
        }

    # ========================================================
    # RECONSTRUCTION
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

        return {
            "inference_steps": step,
            "converged": converged,
            "final_change_r1": change_r1,
            "final_change_r2": change_r2,
            "final_change_class": change_class,
            "error_0": error_0,
            "error_1": error_1,
            "error_2": error_2,
            "k2": self.k2,
            "training_input_count": self.training_input_count
        }