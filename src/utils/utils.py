import os
import random

import numpy as np
import torch

# ImageNet means and stds
# means = (0.485, 0.456, 0.406)
# stds = (0.229, 0.224, 0.225)

# TinyImageNet means and stds
means = [0.4802, 0.4481, 0.3975]
stds = [0.2764, 0.2689, 0.2816]


# Seed Python, NumPy and PyTorch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Seeded torch.Generator (e.g. DataLoader shuffle order)
def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device():
    # CUDA if available, else Apple MPS, else CPU.
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# Print how many CUDA / MPS devices are available, or the CPU cores
def print_device_info(device):
    if device.type == 'cuda':
        print(f'Number of CUDA devices: {torch.cuda.device_count()} ({torch.cuda.get_device_name(device)})')
    elif device.type == 'mps':
        print(f'Number of MPS devices: {torch.mps.device_count()}')
    else:
        print(f'Running on CPU ({os.cpu_count()} cores)')


# Total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def convert_to_rgb(image):
    return image.convert("RGB")
