import numpy as np
import torch


class GaussianNoise:
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def __call__(self, tensor):
        if self.sigma == 0.0:
            return tensor
        noise = torch.randn_like(tensor) * self.sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)


class SaltPepperNoise:
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, tensor):
        if self.noise_level == 0.0:
            return tensor
        noisy = tensor.clone()

        # noise_level = total fraction of corrupted pixels
        # Half become pepper (0), half become salt (1)
        rand = torch.rand(1, tensor.size(1), tensor.size(2), device=tensor.device)
        pepper_mask = rand < (self.noise_level / 2)
        salt_mask = rand > (1.0 - self.noise_level / 2)
        noisy[:, pepper_mask.squeeze(0)] = 0.0
        noisy[:, salt_mask.squeeze(0)] = 1.0
        return noisy
