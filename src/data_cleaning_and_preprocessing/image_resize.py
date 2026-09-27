from PIL import Image

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from tqdm import tqdm


IMAGE_SIZE = 224


# ============================================================
# Image Preprocessing
# ============================================================

def resize_image(image):
    if isinstance(image, Image.Image):
        image = F.to_tensor(image)

    elif not torch.is_tensor(image):
        image = F.to_tensor(image)

    image = F.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        antialias=True,
    )

    return image


# ============================================================
# Resized Dataset View
# ============================================================

class ResizedDatasetView(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.targets = getattr(dataset, "targets", None)
        self.classes = getattr(dataset, "classes", None)
        self.class_to_idx = getattr(dataset, "class_to_idx", None)
        self.class_to_name = getattr(dataset, "class_to_name", None)
        self.class_mapping = getattr(dataset, "class_mapping", None)

        self.base_dataset = dataset
        self.indices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        if isinstance(item, tuple):
            image = resize_image(item[0])
            return (image, *item[1:])

        if isinstance(item, list):
            image = resize_image(item[0])
            return [image, *item[1:]]

        return resize_image(item)


# ============================================================
# Resize Dataset Views
# ============================================================

def resize_datasets(dataset_list):
    resized_datasets = []

    for train_dataset, val_dataset, test_dataset, dataset_name in tqdm(
        dataset_list,
        desc="Preparing resized datasets",
    ):
        resized_datasets.append((
            ResizedDatasetView(train_dataset),
            ResizedDatasetView(val_dataset),
            ResizedDatasetView(test_dataset),
            dataset_name,
        ))

    return resized_datasets


# ============================================================
# Verification
# ============================================================

def verify_image_sizes(dataset_list):
    rows = []

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in tqdm(
            splits,
            desc=f"{dataset_name}: verifying sizes",
        ):
            invalid_images = 0
            channel_counts = {}
            shape_counts = {}

            for index in tqdm(
                range(len(dataset)),
                desc=split_name,
                leave=False,
            ):
                item = dataset[index]
                image = item[0] if isinstance(item, (tuple, list)) else item

                shape = tuple(image.shape)
                channels = int(image.shape[0])

                shape_counts[shape] = shape_counts.get(shape, 0) + 1
                channel_counts[channels] = channel_counts.get(channels, 0) + 1

                if shape != (3, IMAGE_SIZE, IMAGE_SIZE):
                    invalid_images += 1

            rows.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": len(dataset),
                "Expected Shape": f"3 × {IMAGE_SIZE} × {IMAGE_SIZE}",
                "Unique Shapes": len(shape_counts),
                "3-Channel Images": channel_counts.get(3, 0),
                "Invalid Images": invalid_images,
                "Status": "Passed" if invalid_images == 0 else "Failed",
            })

    return pd.DataFrame(rows)