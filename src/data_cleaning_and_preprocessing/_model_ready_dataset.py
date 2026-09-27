import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F
from tqdm import tqdm

IMAGE_SIZE = 224

SPLIT_FILES = {
    "Train": "train.json",
    "Validation": "validation.json",
    "Test": "test.json",
}


# ============================================================
# File Helpers
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Image Preprocessing
# ============================================================

def prepare_image(image, mean, std, image_size=IMAGE_SIZE):
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = F.to_tensor(image)

    elif torch.is_tensor(image):
        if image.ndim != 3:
            raise ValueError(
                f"Expected image tensor [C, H, W], got {tuple(image.shape)}"
            )

        if not image.is_floating_point():
            image = image.float() / 255.0
        else:
            image = image.float()

    else:
        image = F.to_tensor(image)

    if image.shape[0] != 3:
        raise ValueError(
            f"Expected 3 RGB channels, got {image.shape[0]}"
        )

    if image.shape[1:] != (image_size, image_size):
        image = F.resize(
            image,
            [image_size, image_size],
            antialias=True,
        )

    # normalize = Normalize(mean=mean, std=std)
    return  image  #normalize(image)  #


# ============================================================
# Class Metadata
# ============================================================

def get_base_dataset(dataset):
    base_dataset = getattr(dataset, "base_dataset", None)
    return base_dataset if base_dataset is not None else dataset


def get_source_classes(source_datasets):
    for dataset in source_datasets.values():
        dataset = get_base_dataset(dataset)
        classes = getattr(dataset, "classes", None)

        if classes is not None:
            return list(classes)

    return []


def get_source_class_name(source_datasets, index, class_id):
    for dataset in source_datasets.values():
        dataset = get_base_dataset(dataset)

        # ImageNet-21K / ImageNet-21K-P / custom mappings
        class_to_name = getattr(dataset, "class_to_name", None)

        if class_to_name:
            class_name = class_to_name.get(class_id)

            if class_name is not None and str(class_name) != str(class_id):
                return class_name

        # Tiny ImageNet
        idx_to_words = getattr(dataset, "idx_to_words", None)

        if idx_to_words is not None:
            if isinstance(idx_to_words, dict):
                if index in idx_to_words:
                    return idx_to_words[index]

                if str(index) in idx_to_words:
                    return idx_to_words[str(index)]

                if class_id in idx_to_words:
                    return idx_to_words[class_id]

            elif isinstance(idx_to_words, (list, tuple)) and index < len(idx_to_words):
                return idx_to_words[index]

    return None


def build_class_metadata(source_datasets, class_mapping):
    source_classes = get_source_classes(source_datasets)
    mappings = sorted(class_mapping, key=lambda item: int(item["new_target"]))

    classes = []
    class_to_idx = {}
    class_to_name = {}
    idx_to_name = {}

    for item in mappings:
        old_target = int(item["old_target"])
        new_target = int(item["new_target"])

        class_id = (
            source_classes[old_target]
            if old_target < len(source_classes)
            else str(old_target)
        )

        # First priority: source dataset human-readable name
        class_name = get_source_class_name(
            source_datasets,
            old_target,
            class_id,
        )

        # Second priority: saved cleaning metadata
        if class_name is None:
            mapped_name = item.get("class_name")

            if mapped_name is not None and str(mapped_name) != str(class_id):
                class_name = mapped_name

        # Final fallback
        if class_name is None:
            class_name = class_id

        classes.append(class_id)
        class_to_idx[class_id] = new_target
        class_to_name[class_id] = class_name
        idx_to_name[new_target] = class_name

    return classes, class_to_idx, class_to_name, idx_to_name


# ============================================================
# Model-Ready Dataset
# ============================================================

class ModelReadyDataset(Dataset):
    def __init__(
            self,
            source_datasets,
            records,
            class_mapping,
            mean,
            std,
            dataset_name,
            split_name,
            image_size=IMAGE_SIZE,
    ):
        self.source_datasets = source_datasets
        self.records = records

        self.dataset_name = dataset_name
        self.split_name = split_name

        self.mean = list(mean)
        self.std = list(std)
        self.image_size = image_size

        self.base_dataset = None
        self.indices = None
        self.class_mapping = class_mapping

        (
            self.classes,
            self.class_to_idx,
            self.class_to_name,
            self.idx_to_name,
        ) = build_class_metadata(
            source_datasets,
            class_mapping,
        )

        # Compatibility with existing Tiny ImageNet EDA functions
        self.idx_to_words = self.idx_to_name

        self.num_classes = len(self.classes)

        record_targets = [
            record.get("target")
            for record in records
        ]

        if record_targets and all(
                target is not None
                for target in record_targets
        ):
            self.targets = [
                int(target)
                for target in record_targets
            ]
        else:
            self.targets = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        source_split = record["source_split"]
        source_index = int(record["source_index"])

        source_dataset = self.source_datasets[source_split]

        if source_index >= len(source_dataset):
            raise IndexError(
                f"{self.dataset_name} {self.split_name}: "
                f"source index {source_index} exceeds "
                f"{source_split} size {len(source_dataset)}."
            )

        item = source_dataset[source_index]

        image = (
            item[0]
            if isinstance(item, (tuple, list))
            else item
        )

        image = prepare_image(
            image,
            self.mean,
            self.std,
            self.image_size,
        )

        target = record.get("target")

        if target is None:
            target = -1

        if isinstance(item, tuple):
            return image, int(target), *item[2:]

        if isinstance(item, list):
            return [image, int(target), *item[2:]]

        return image, int(target)

    def get_class_name(self, target):
        return self.idx_to_name[int(target)]


# ============================================================
# Final Dataset Loader
# ============================================================

class ModelReadyDatasetLoader:
    def __init__(
            self,
            source_loader,
            root=None,
            image_size=IMAGE_SIZE,
    ):
        self.source_loader = source_loader

        self.root = (
            Path(root)
            if root is not None
            else Path(source_loader.root)
        )

        self.image_size = image_size

    def load(self):
        source_tuple = self.source_loader.load()

        (
            train_source,
            val_source,
            test_source,
            dataset_name,
        ) = source_tuple

        source_datasets = {
            "Train": train_source,
            "Validation": val_source,
            "Test": test_source,
        }

        clean_root = self.root / "splits" / "clean"
        cleaning_root = self.root / "cleaning"
        preprocessing_root = self.root / "preprocessing"

        records = {
            split_name: load_json(clean_root / filename)
            for split_name, filename in SPLIT_FILES.items()
        }

        class_mapping = load_json(
            cleaning_root / "class_mapping.json"
        )

        normalization = load_json(
            preprocessing_root / "normalization.json"
        )

        mean = normalization["mean"]
        std = normalization["std"]

        self.mean = list(mean)
        self.std = list(std)

        train_dataset = ModelReadyDataset(
            source_datasets,
            records["Train"],
            class_mapping,
            mean,
            std,
            dataset_name,
            "Train",
            self.image_size,
        )

        val_dataset = ModelReadyDataset(
            source_datasets,
            records["Validation"],
            class_mapping,
            mean,
            std,
            dataset_name,
            "Validation",
            self.image_size,
        )

        test_dataset = ModelReadyDataset(
            source_datasets,
            records["Test"],
            class_mapping,
            mean,
            std,
            dataset_name,
            "Test",
            self.image_size,
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            dataset_name,
        )


# ============================================================
# Final Verification
# ============================================================

def verify_model_ready_datasets(
        dataset_list,
        max_images=None,
        seed=42,
):
    rows = []

    for train_dataset, val_dataset, test_dataset, dataset_name in tqdm(
            dataset_list,
            desc="Model-ready datasets",
    ):
        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in splits:
            total_images = len(dataset)

            if max_images is None or max_images >= total_images:
                indices = list(range(total_images))
            else:
                rng = random.Random(
                    f"{seed}:{dataset_name}:{split_name}"
                )

                indices = sorted(
                    rng.sample(
                        range(total_images),
                        max_images,
                    )
                )

            invalid_shape = 0
            invalid_channels = 0
            invalid_values = 0
            invalid_targets = 0

            labelled = dataset.targets is not None

            for index in tqdm(
                    indices,
                    desc=f"{dataset_name} {split_name}",
                    leave=False,
            ):
                item = dataset[index]
                image = item[0]
                target = item[1]

                if not torch.is_tensor(image):
                    invalid_shape += 1
                    continue

                if tuple(image.shape) != (
                        3,
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                ):
                    invalid_shape += 1

                if image.shape[0] != 3:
                    invalid_channels += 1

                if not torch.isfinite(image).all():
                    invalid_values += 1

                if labelled:
                    if target < 0 or target >= dataset.num_classes:
                        invalid_targets += 1

            valid = (
                    invalid_shape == 0
                    and invalid_channels == 0
                    and invalid_values == 0
                    and invalid_targets == 0
            )

            rows.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": total_images,
                "Verified Images": len(indices),
                "Classes": dataset.num_classes if labelled else "N/A",
                "Expected Shape": f"3 × {IMAGE_SIZE} × {IMAGE_SIZE}",
                "Invalid Shape": invalid_shape,
                "Invalid Channels": invalid_channels,
                "Invalid Values": invalid_values,
                "Invalid Targets": invalid_targets if labelled else "N/A",
                "Status": "Passed" if valid else "Failed",
            })

    return pd.DataFrame(rows)
