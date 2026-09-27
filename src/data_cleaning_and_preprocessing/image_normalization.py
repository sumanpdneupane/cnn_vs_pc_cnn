import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from tqdm import tqdm


# ============================================================
# Normalized Dataset View
# ============================================================

class NormalizedDatasetView(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.normalize = Normalize(mean=mean, std=std)

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
            return (self.normalize(item[0]), *item[1:])

        if isinstance(item, list):
            return [self.normalize(item[0]), *item[1:]]

        return self.normalize(item)


# ============================================================
# Normalize All Datasets
# ============================================================

def normalize_datasets(dataset_list, normalization_stats):
    normalized_datasets = []

    for train_dataset, val_dataset, test_dataset, dataset_name in tqdm(
        dataset_list,
        desc="Applying normalization",
    ):
        stats = normalization_stats[dataset_name]

        mean = stats["mean"]
        std = stats["std"]

        normalized_datasets.append((
            NormalizedDatasetView(train_dataset, mean, std),
            NormalizedDatasetView(val_dataset, mean, std),
            NormalizedDatasetView(test_dataset, mean, std),
            dataset_name,
        ))

    return normalized_datasets


# ============================================================
# Normalization Verification
# ============================================================

def compute_normalized_stats(
    dataset,
    dataset_name,
    split_name,
    batch_size=64,
    num_workers=0,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)

    pixel_count = 0
    invalid_images = 0

    for batch in tqdm(
        loader,
        desc=f"{dataset_name} - {split_name}",
        leave=False,
    ):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            invalid_images += images.shape[0]
            continue

        finite_mask = torch.isfinite(images).flatten(1).all(dim=1)
        invalid_images += int((~finite_mask).sum())

        images = images.to(torch.float64)

        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))

        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0))

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "invalid_images": invalid_images,
    }


def verify_normalization(
    dataset_list,
    batch_size=64,
    num_workers=0,
):
    rows = []

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        for split_name, dataset in tqdm(
            [
                ("Train", train_dataset),
                ("Validation", val_dataset),
                ("Test", test_dataset),
            ],
            desc=f"{dataset_name}: verifying normalization",
        ):
            result = compute_normalized_stats(
                dataset,
                dataset_name,
                split_name,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            mean = result["mean"]
            std = result["std"]

            train_passed = (
                all(abs(value) < 1e-5 for value in mean)
                and all(abs(value - 1) < 1e-5 for value in std)
            )

            status = (
                "Passed"
                if result["invalid_images"] == 0
                and (split_name != "Train" or train_passed)
                else "Failed"
            )

            rows.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Images": len(dataset),
                "Mean R": mean[0],
                "Mean G": mean[1],
                "Mean B": mean[2],
                "Std R": std[0],
                "Std G": std[1],
                "Std B": std[2],
                "Invalid Images": result["invalid_images"],
                "Status": status,
            })

    return pd.DataFrame(rows)