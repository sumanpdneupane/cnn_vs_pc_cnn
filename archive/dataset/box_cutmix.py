import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.autoaugment import _apply_op

GEOMETRIC_OPS = ('ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate')


class BoxMaskTrainDataset(Dataset):
    """Training images with an object mask built from the Tiny ImageNet train box (train boxes only).

    The image gets exactly 33's train transform: RandomCrop 64 (padding 4, reflect), horizontal flip 0.5,
    RandAugment(num_ops, magnitude), ToTensor, Normalize. The same random calls are made in the same order, so with
    the same RNG state the image is identical to 33's. The mask goes through the same crop, flip and geometric
    RandAugment ops (shear, translate, rotate; nearest, fill 0), so it marks where the object is in the final image.
    Returns (image, label, mask) with mask a bool tensor [64, 64]. Lives in a module so DataLoader workers can
    unpickle it on macOS (spawn).
    """

    def __init__(self, subset, base, train_root, mean, std, num_ops=2, magnitude=5):
        self.subset, self.base = subset, base
        self.randaugment = transforms.RandAugment(num_ops=num_ops, magnitude=magnitude,
                                                  interpolation=InterpolationMode.BILINEAR)
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.boxes = {}
        for wnid in base.classes:
            with open(train_root / wnid / f'{wnid}_boxes.txt') as f:
                for line in f:
                    name, *box = line.split()
                    self.boxes[name] = tuple(int(v) for v in box)  # x0, y0, x1, y1, inclusive

    def __len__(self):
        return len(self.subset)

    def object_mask(self, path, size):
        x0, y0, x1, y1 = self.boxes[path.name]
        mask = Image.new('L', size, 0)
        ImageDraw.Draw(mask).rectangle((x0, y0, x1, y1), fill=255)
        return mask

    def randaugment_pair(self, image, mask):
        ra = self.randaugment  # same sampling as torchvision RandAugment.forward
        _, height, width = F.get_dimensions(image)
        op_meta = ra._augmentation_space(ra.num_magnitude_bins, (height, width))
        for _ in range(ra.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[ra.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            image = _apply_op(image, op_name, magnitude, interpolation=ra.interpolation, fill=ra.fill)
            if op_name in GEOMETRIC_OPS:
                mask = _apply_op(mask, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        return image, mask

    def __getitem__(self, index):
        base_index, label = self.subset.samples[index]
        path = self.base.samples[base_index][0]
        image = Image.open(path).convert('RGB')
        mask = self.object_mask(path, image.size)
        image, mask = F.pad(image, 4, padding_mode='reflect'), F.pad(mask, 4, padding_mode='reflect')
        top, left, h, w = transforms.RandomCrop.get_params(image, (64, 64))
        image, mask = F.crop(image, top, left, h, w), F.crop(mask, top, left, h, w)
        if torch.rand(1) < 0.5:
            image, mask = F.hflip(image), F.hflip(mask)
        image, mask = self.randaugment_pair(image, mask)
        image = self.normalize(F.to_tensor(image))
        return image, label, torch.from_numpy(np.array(mask) > 127)


def box_cutmix(images, labels, masks, alpha=1.0, candidates=2, info=None):
    """CutMix with labels from the objects, using train-box masks.

    Cut size and clipping as in 33's CutMix (lambda ~ Beta(alpha, alpha), side = size * sqrt(1 - lambda), random
    centre, clipped to the image), but per image:
      - the destination is the less object-covering of `candidates` random rectangles in image A;
      - the pasted piece of image B is the same-sized rectangle placed around a random pixel of B's object;
      - label weights come from the objects: vis_a = share of A's object still visible, vis_b = share of B's object
        pasted, weight of A = vis_a / (vis_a + vis_b). Falls back to the pixel-area weight when a mask is empty.
    Works on CPU tensors (call before moving the batch to the device). Returns images, labels_a, labels_b,
    per-image lam (weight of labels_a), 'BoxCutMix'. If `info` is a dict, it gets perm, vis_a and vis_b (for checks).
    """
    batch_size, _, height, width = images.shape
    perm = torch.randperm(batch_size)
    mixed = images.clone()
    lam = torch.empty(batch_size)
    mask_area = masks.flatten(1).sum(1)
    vis = torch.full((2, batch_size), float('nan'))
    for i in range(batch_size):
        j = int(perm[i])
        cut = np.sqrt(1.0 - float(np.random.beta(alpha, alpha)))
        cut_w, cut_h = int(width * cut), int(height * cut)
        best = None
        for _ in range(candidates):
            cx, cy = int(torch.randint(width, (1,))), int(torch.randint(height, (1,)))
            x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
            y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
            covered = int(masks[i, y1:y2, x1:x2].sum())
            if best is None or covered < best[0]:
                best = (covered, x1, x2, y1, y2)
        covered, x1, x2, y1, y2 = best
        w, h = x2 - x1, y2 - y1
        if w == 0 or h == 0:
            lam[i] = 1.0
            continue
        ys, xs = torch.nonzero(masks[j], as_tuple=True)
        if len(ys):
            k = int(torch.randint(len(ys), (1,)))
            sx = min(max(int(xs[k]) - w // 2, 0), width - w)
            sy = min(max(int(ys[k]) - h // 2, 0), height - h)
        else:
            sx, sy = x1, y1
        mixed[i, :, y1:y2, x1:x2] = images[j, :, sy:sy + h, sx:sx + w]
        area_a, area_b = int(mask_area[i]), int(mask_area[j])
        if area_a == 0 or area_b == 0:
            lam[i] = 1.0 - (w * h) / float(width * height)
            continue
        vis_a = (area_a - covered) / area_a
        vis_b = int(masks[j, sy:sy + h, sx:sx + w].sum()) / area_b
        vis[0, i], vis[1, i] = vis_a, vis_b
        lam[i] = vis_a / (vis_a + vis_b) if vis_a + vis_b > 0 else 1.0 - (w * h) / float(width * height)
    if info is not None:
        info.update(perm=perm, vis_a=vis[0], vis_b=vis[1])
    return mixed, labels, labels[perm], lam, 'BoxCutMix'
