import os

import torch


# Hidden Layers: ReLU
# Output Layer: Softmax: Multi-Class Classification: probability distribution
# Input Image → Conv Layer → ReLU → Pooling → Conv Layer → ReLU → Flatten → Fully Connected → Softmax / Sigmoid (Output).


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

class ActivationFunction:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearActivation(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


class ReLUActivation(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).to(x.dtype)


class Softmax(ActivationFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.max(x, dim=-1, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self.forward(x)
        diagonal = torch.diag_embed(probabilities)
        outer_product = probabilities.unsqueeze(-1) * probabilities.unsqueeze(-2)
        return diagonal - outer_product

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def save_pccnn_model(model, file_path, epochs=None, history=None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "alpha1": model.alpha1,
        "alpha2": model.alpha2,
        "lambda_u": model.lambda_u,
        "inference_steps": model.inference_steps,
        "representation_update_rate": model.representation_update_rate,
        "weight_learning_rate": model.weight_learning_rate,
        "epochs": epochs,
        "history": history,
    }, file_path)

    print(f"Model saved: {file_path}")

def load_pccnn_model(model, file_path, device):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return model, 0

    checkpoint = torch.load(file_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.alpha1 = checkpoint["alpha1"]
    model.alpha2 = checkpoint["alpha2"]
    model.lambda_u = checkpoint["lambda_u"]
    model.inference_steps = checkpoint["inference_steps"]
    model.representation_update_rate = checkpoint["representation_update_rate"]
    model.weight_learning_rate = checkpoint["weight_learning_rate"]

    print(f"Model loaded: {file_path}")

    return model, checkpoint
