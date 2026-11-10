import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class BoxMixDataset(Dataset):
    """39-style mixing (each image mixed with probability mix_prob, RandAugment after mixing), made box-aware with the
    Tiny ImageNet train boxes (training only).

    Before mixing, image and object mask get the same RandomCrop 64 (padding 4, reflect) and horizontal flip as 39.
    mode='resizemix': the other image, shrunk to tau of the size (tau ~ U(scale)), is pasted on the less
      object-covering of `candidates` random spots. Label weight of A = vis_a / (vis_a + p), with p = pasted area
      share (the shrunk image is all there, at p of the area) and vis_a = share of A's object still visible. Same rule
      as the cutmix mode; for an object filling the whole image it is the paper's 1 - tau^2, for a hidden one 0.
    mode='cutmix': cut size as CutMix (lambda ~ Beta(alpha, alpha), side = 64 * sqrt(1 - lambda), random centre,
      clipped); destination = less object-covering of `candidates` spots in A; the piece of B is cut around a random
      pixel of B's object; weight of A = vis_a / (vis_a + vis_b).
    Both fall back to the pixel-area weight when a mask is empty. Returns (image, label_a, label_b, lam) like
    ResizeMixDataset, so 39's training loop is unchanged. Lives in a module so DataLoader workers can unpickle it
    on macOS (spawn). Uses only torch random numbers, so every worker draws its own.
    """

    def __init__(self, subset, base, train_root, after_mix_transform, mode, mix_prob=0.5, scale=(0.1, 0.8),
                 cutmix_alpha=1.0, candidates=2):
        assert mode in ('resizemix', 'cutmix'), mode
        self.subset, self.base, self.after_mix_transform, self.mode = subset, base, after_mix_transform, mode
        self.mix_prob, self.scale, self.candidates = mix_prob, scale, candidates
        self.beta = torch.distributions.Beta(torch.tensor(float(cutmix_alpha)), torch.tensor(float(cutmix_alpha)))
        self.boxes = {}
        for wnid in base.classes:
            with open(train_root / wnid / f'{wnid}_boxes.txt') as f:
                for line in f:
                    name, *box = line.split()
                    self.boxes[name] = tuple(int(v) for v in box)  # x0, y0, x1, y1, inclusive

    def __len__(self):
        return len(self.subset)

    def load(self, index):
        """Image and object mask after 39's before-mix transform (crop with reflect padding, flip)."""
        base_index, label = self.subset.samples[index]
        path = self.base.samples[base_index][0]
        image = Image.open(path).convert('RGB')
        x0, y0, x1, y1 = self.boxes[path.name]
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        mask[y0:y1 + 1, x0:x1 + 1] = 255
        mask = Image.fromarray(mask)
        image, mask = F.pad(image, 4, padding_mode='reflect'), F.pad(mask, 4, padding_mode='reflect')
        top, left, h, w = transforms.RandomCrop.get_params(image, (64, 64))
        image, mask = F.crop(image, top, left, h, w), F.crop(mask, top, left, h, w)
        if torch.rand(1) < 0.5:
            image, mask = F.hflip(image), F.hflip(mask)
        return image, np.array(mask) > 127, label

    def least_covering_spot(self, mask, w, h, centred):
        """Top-left of the w x h rectangle (from `candidates` random tries) covering the fewest object pixels."""
        width, height = mask.shape[1], mask.shape[0]
        best = None
        for _ in range(self.candidates):
            if centred:  # CutMix: random centre, clipped to the image
                cx, cy = int(torch.randint(width, (1,))), int(torch.randint(height, (1,)))
                x1, x2 = max(cx - w // 2, 0), min(cx + w // 2, width)
                y1, y2 = max(cy - h // 2, 0), min(cy + h // 2, height)
            else:  # ResizeMix: fully inside the image
                x1, y1 = int(torch.randint(width - w + 1, (1,))), int(torch.randint(height - h + 1, (1,)))
                x2, y2 = x1 + w, y1 + h
            covered = int(mask[y1:y2, x1:x2].sum())
            if best is None or covered < best[0]:
                best = (covered, x1, y1, x2, y2)
        return best

    def __getitem__(self, index):
        image, mask_a, label_a = self.load(index)
        if torch.rand(1).item() >= self.mix_prob:
            return self.after_mix_transform(image), label_a, label_a, torch.tensor(1.0, dtype=torch.float32)
        other, mask_b, label_b = self.load(int(torch.randint(len(self.subset), (1,))))
        width, height = image.size
        area_a, area_b = int(mask_a.sum()), int(mask_b.sum())
        mixed = image.copy()
        if self.mode == 'resizemix':
            low, high = self.scale
            tau = low + (high - low) * torch.rand(1).item()
            pw, ph = max(1, round(width * tau)), max(1, round(height * tau))
            covered, x1, y1, _, _ = self.least_covering_spot(mask_a, pw, ph, centred=False)
            mixed.paste(other.resize((pw, ph), Image.BILINEAR), (x1, y1))
            pasted = (pw * ph) / (width * height)
            if area_a == 0:
                lam = 1.0 - pasted
            else:
                vis_a = (area_a - covered) / area_a
                lam = vis_a / (vis_a + pasted)
        else:
            cut = float(torch.sqrt(1.0 - self.beta.sample()))
            cw, ch = int(width * cut), int(height * cut)
            covered, x1, y1, x2, y2 = self.least_covering_spot(mask_a, cw, ch, centred=True)
            w, h = x2 - x1, y2 - y1
            if w == 0 or h == 0:
                return self.after_mix_transform(image), label_a, label_b, torch.tensor(1.0, dtype=torch.float32)
            ys, xs = np.nonzero(mask_b)
            if len(ys):
                k = int(torch.randint(len(ys), (1,)))
                sx = min(max(int(xs[k]) - w // 2, 0), width - w)
                sy = min(max(int(ys[k]) - h // 2, 0), height - h)
            else:
                sx, sy = x1, y1
            mixed.paste(other.crop((sx, sy, sx + w, sy + h)), (x1, y1))
            pasted = (w * h) / (width * height)
            if area_a == 0 or area_b == 0:
                lam = 1.0 - pasted
            else:
                vis_a = (area_a - covered) / area_a
                vis_b = int(mask_b[sy:sy + h, sx:sx + w].sum()) / area_b
                lam = vis_a / (vis_a + vis_b) if vis_a + vis_b > 0 else 1.0 - pasted
        return self.after_mix_transform(mixed), label_a, label_b, torch.tensor(lam, dtype=torch.float32)
