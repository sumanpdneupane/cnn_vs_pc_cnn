import numpy as np
import torch

class MixUpAugumentation:
    def __init__(self, images, labels, alpha=0.2):
        self.alpha = alpha
        self.images = images
        self.labels = labels

    def __call__(self):
        if self.alpha <= 0:
            return self.images, self.labels, self.labels, 1.0, 'Clean'
        batch_size = self.images.size(0)
        device = self.images.device
        perm = torch.randperm(batch_size, device=device)
        lam = float(np.random.beta(self.alpha, self.alpha))
        mixed_images = lam * self.images + (1.0 - lam) * self.images[perm]
        labels_a = self.labels
        labels_b = self.labels[perm]
        return (mixed_images, labels_a, labels_b, lam, 'MixUp')


class CutMixAugumentation:
    def __init__(self, images, labels, alpha=1.0):
        self.alpha = alpha
        self.images = images
        self.labels = labels

    def __call__(self):
        if self.alpha <= 0:
            return self.images, self.labels, self.labels, 1.0, 'Clean'
        batch_size, _, height, width = self.images.shape
        device = self.images.device
        perm = torch.randperm(batch_size, device=device)
        lam = float(np.random.beta(self.alpha, self.alpha))
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = int(np.random.randint(0, width))
        cy = int(np.random.randint(0, height))
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, height)
        mixed_images = self.images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = self.images[perm, :, y1:y2, x1:x2]
        patch_area = (x2 - x1) * (y2 - y1)
        lam_adjusted = 1.0 - patch_area / float(width * height)
        labels_a = self.labels
        labels_b = self.labels[perm]
        return (mixed_images, labels_a, labels_b, lam_adjusted, 'CutMix')
