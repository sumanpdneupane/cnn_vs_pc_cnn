import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image

TINY_ROOT = Path("./data/tiny_imagenet/tiny-imagenet-200-450train-100val")

means = [0.4803270399570465, 0.44804883003234863, 0.3975021243095398]
stds = [0.27645865082740784, 0.26883983612060547, 0.28148162364959717]


def show_image(images, labels, num_show, class_name, name="", has_norm=True):
    MEAN_T = torch.tensor(means).view(3, 1, 1)
    STD_T = torch.tensor(stds).view(3, 1, 1)

    def denormalize(image):
        image = image.cpu()
        if has_norm:
            image = image * STD_T + MEAN_T
        return image.clamp(0, 1)

    plt.figure(figsize=(16, 10))

    for i in range(min(num_show, len(images))):
        image = denormalize(images[i])
        image = image.permute(1, 2, 0).numpy()

        plt.subplot(4, 5, i + 1)
        plt.imshow(image)
        plt.title(class_name[int(labels[i])], fontsize=9)
        plt.axis("off")

    plt.suptitle(name, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def get_tiny_imagenet_class_names(dataset, tiny_root=TINY_ROOT):
    tiny_root = Path(tiny_root)
    words_path = tiny_root / "words.txt"

    if not words_path.exists():
        raise FileNotFoundError(f"words.txt not found: {words_path}")

    wnid_to_name = {}
    with open(words_path, "r") as f:
        for line in f:
            wnid, names = line.strip().split("\t", 1)
            wnid_to_name[wnid] = names.split(",")[0].strip()

    base_dataset = dataset
    while hasattr(base_dataset, "dataset"):
        base_dataset = base_dataset.dataset
    if not hasattr(base_dataset, "classes"):
        raise AttributeError("Base dataset does not contain a 'classes' attribute.")

    class_wnids = list(base_dataset.classes)
    missing_wnids = [wnid for wnid in class_wnids if wnid not in wnid_to_name]
    if missing_wnids:
        raise ValueError(f"Missing class names for WordNet IDs: {missing_wnids[:10]}")
    return [wnid_to_name[wnid] for wnid in class_wnids]


def convert_to_rgb(image):
    return image.convert("RGB")


from torchvision import transforms

# train_transform = transforms.Compose([
#     transforms.Lambda(convert_to_rgb),
#
#     # 1. Geometry & Spatial Heavy Augmentation
#     transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.3),  # Use if orientation doesn't matter (e.g., satellite/cells)
#     transforms.RandomRotation(degrees=(-30, 30)),
#
#     # 2. Automated Auto-Augment (Tuned Up)
#     # Increased magnitude (max is 30) and ops for heavier variety
#     transforms.RandAugment(num_ops=4, magnitude=15),
#
#     # 3. Explicit Color Distortion
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#
#     # 4. Mandatory Conversions
#     transforms.ToTensor(),
#     transforms.Normalize(mean=means, std=stds),
#
#     # 5. Pixel-Level Masking (Must be after ToTensor)
#     transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), value='random'),
# ])

train_transform = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

test_transform = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

class RandomRGBShift:
    """Apply a small, independent random shift to each RGB channel."""

    def __init__(self, shift=10):
        self.shift = shift

    def __call__(self, image):
        image = np.asarray(image, dtype=np.int16)

        shifts = np.array(
            [random.randint(-self.shift, self.shift) for _ in range(3)],
            dtype=np.int16
        )

        image = np.clip(image + shifts, 0, 255).astype(np.uint8)
        return Image.fromarray(image, mode="RGB")


class TinyImageNetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")
        with open(self.root / "wnids.txt", "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {
            wnid: class_id
            for class_id, wnid in enumerate(self.classes)
        }

        self.samples = []

        if split in ["train", "val"]:
            for wnid in self.classes:
                images_dir = self.root / split / wnid / "images"
                if not images_dir.exists():
                    raise FileNotFoundError(f"Missing folder: {images_dir}")
                class_id = self.class_to_idx[wnid]
                for image_path in sorted(images_dir.glob("*.JPEG")):
                    self.samples.append((image_path, class_id))
        else:
            images_dir = self.root / "test" / "images"
            if not images_dir.exists():
                raise FileNotFoundError(f"Missing folder: {images_dir}")
            for image_path in sorted(images_dir.glob("*.JPEG")):
                self.samples.append((image_path, -1))

        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class TinyImageNetLoader():
    def __init__(self, root=TINY_ROOT, train_transform=train_transform, test_transform=test_transform, seed=42):
        self.root = Path(root)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed

    def load(self):
        train_dataset = TinyImageNetDataset(root=self.root, split="train", transform=self.train_transform)
        val_dataset = TinyImageNetDataset(root=self.root, split="val", transform=self.test_transform)
        test_dataset = TinyImageNetDataset(root=self.root, split="test", transform=self.test_transform)

        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError("Train and validation class mappings are different.")

        print("Tiny ImageNet loaded")
        print("Classes:", len(train_dataset.classes))
        print("Train images:", len(train_dataset))
        print("Validation images:", len(val_dataset))
        print("Test images:", len(test_dataset))

        return train_dataset, val_dataset, test_dataset, "tiny_imagenet"
