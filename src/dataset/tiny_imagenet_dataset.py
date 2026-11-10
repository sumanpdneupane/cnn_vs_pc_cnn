from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# Class indices used for training and evaluation (the first num_classes)
def get_selected_classes(num_classes):
    return list(range(num_classes))


def get_tiny_imagenet_class_names(dataset, tiny_root=None):
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


# Names of the selected classes, in label order
def get_selected_class_names(dataset, tiny_root, num_classes):
    names = get_tiny_imagenet_class_names(dataset, tiny_root)
    return [names[i] for i in get_selected_classes(num_classes)]


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
    def __init__(self, root=None, train_transform=None, test_transform=None, seed=42):
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

        return train_dataset, val_dataset, test_dataset, "tiny_imagenet"
