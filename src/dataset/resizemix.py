import torch
from PIL import Image
from torch.utils.data import Dataset


# ResizeMix: paste a shrunk random image (tau ~ U(0.1, 0.8)) on mix_prob of images; label weight 1 - tau^2
class ResizeMixDataset(Dataset):
    def __init__(self, subset, after_mix_transform, scale=(0.1, 0.8), mix_prob=1.0):
        self.subset = subset
        self.after_mix_transform = after_mix_transform
        self.scale = scale
        self.mix_prob = mix_prob

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label_a = self.subset[index]
        if torch.rand(1).item() >= self.mix_prob:
            return self.after_mix_transform(image), label_a, label_a, torch.tensor(1.0, dtype=torch.float32)
        other, label_b = self.subset[int(torch.randint(len(self.subset), (1,)))]
        low, high = self.scale
        tau = low + (high - low) * torch.rand(1).item()
        width, height = image.size
        patch_w, patch_h = max(1, round(width * tau)), max(1, round(height * tau))
        left = int(torch.randint(width - patch_w + 1, (1,)))
        top = int(torch.randint(height - patch_h + 1, (1,)))
        mixed = image.copy()
        mixed.paste(other.resize((patch_w, patch_h), Image.BILINEAR), (left, top))
        lam = 1.0 - (patch_w * patch_h) / (width * height)
        return self.after_mix_transform(mixed), label_a, label_b, torch.tensor(lam, dtype=torch.float32)
