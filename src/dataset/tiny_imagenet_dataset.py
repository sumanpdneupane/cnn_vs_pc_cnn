from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from tinyimagenet import TinyImageNet
from src.dataset._loader import ImageNetLoader

TINY_ROOT = Path("./data/tiny_imagenet")

means = [
    0.48022549245655144,
    0.4480585117404561,
    0.397521028501032
]
stds = [
    0.2661645123876411,
    0.2582361193999296,
    0.2718131688966652
]

def show_image(images, labels, num_show, class_name, name=''):
    MEAN_T = torch.tensor(means).view(3, 1, 1)
    STD_T = torch.tensor(stds).view(3, 1, 1)

    def denormalize(image):
        image = image.cpu()
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

    plt.suptitle(f'{name}', fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def get_tiny_imagenet_class_names(dataset, tiny_root):
    tiny_root = Path(tiny_root)
    words_path = tiny_root / 'tiny-imagenet-200'/ "words.txt"
    wnid_to_name = {}
    with open(words_path, "r") as f:
        for line in f:
            wnid, names = line.strip().split("\t", 1)
            wnid_to_name[wnid] = names.split(",")[0].strip()
    base_dataset = dataset
    while hasattr(base_dataset, "dataset"): base_dataset = base_dataset.dataset
    if not hasattr(base_dataset, "classes"): raise AttributeError("Base dataset does not contain a 'classes' attribute.")
    class_wnids = list(base_dataset.classes)
    missing_wnids = [wnid for wnid in class_wnids if wnid not in wnid_to_name]
    if missing_wnids: raise ValueError(f"Missing class names for WordNet IDs: {missing_wnids[:10]}")
    return [wnid_to_name[wnid] for wnid in class_wnids]

def identity_target(target): return target

def convert_to_rgb(image):
    return image.convert("RGB")


## 78.00
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


class TinyImageNetLoader(ImageNetLoader):
    def __init__(
            self,
            root=TINY_ROOT,
            train_transform=train_transform,
            test_transform=test_transform,
            seed=42,
    ):
        self.root = Path(root)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed

    def load(self):
        train_dataset = TinyImageNet(root=self.root, split="train", transform=self.train_transform, target_transform=identity_target)
        val_dataset = TinyImageNet(root=self.root, split="val", transform=self.test_transform, target_transform=identity_target)
        test_dataset = TinyImageNet(root=self.root, split="test", transform=self.test_transform, target_transform=identity_target)

        self._remove_zip()
        return (train_dataset, val_dataset, test_dataset, "tiny_imagenet")

    def _remove_zip(self):
        zip_path = self.root / "tiny-imagenet-200.zip"
        if zip_path.exists():
            zip_path.unlink()
            print(f"Deleted ZIP file: {zip_path}")

# tiny_imagenet = TinyImageNetLoader(
#     seed=42
# ).load()

# cleaned_tiny_imagenet = model_ready_dataset.ModelReadyDatasetLoader(
#     TinyImageNetLoader(
#         seed=42
#     )
# ).load()
