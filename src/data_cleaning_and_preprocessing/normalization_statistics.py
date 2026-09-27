import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# Train Mean and Standard Deviation
# ============================================================

def compute_mean_std(dataset, dataset_name, batch_size=64, num_workers=0):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)

    pixel_count = 0
    min_value = float("inf")
    max_value = float("-inf")

    for batch in tqdm(
        loader,
        desc=f"{dataset_name}: Train mean/std",
    ):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        if images.ndim != 4:
            raise ValueError(
                f"{dataset_name}: expected [N, C, H, W], got {tuple(images.shape)}"
            )

        if images.shape[1] != 3:
            raise ValueError(
                f"{dataset_name}: expected 3 channels, got {images.shape[1]}"
            )

        images = images.to(torch.float64)

        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))

        pixels_per_channel = images.shape[0] * images.shape[2] * images.shape[3]
        pixel_count += pixels_per_channel

        min_value = min(min_value, images.min().item())
        max_value = max(max_value, images.max().item())

    mean = channel_sum / pixel_count

    variance = (
        channel_squared_sum / pixel_count
        - mean ** 2
    )

    std = torch.sqrt(
        torch.clamp(variance, min=0)
    )

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pixel_count_per_channel": pixel_count,
        "min_pixel_value": min_value,
        "max_pixel_value": max_value,
    }


# ============================================================
# Calculate Statistics for All Datasets
# ============================================================

def compute_normalization_statistics(
    dataset_list,
    batch_size=64,
    num_workers=0,
):
    statistics = {}
    rows = []

    for train_dataset, _, _, dataset_name in tqdm(
        dataset_list,
        desc="Computing Train normalization statistics",
    ):
        result = compute_mean_std(
            train_dataset,
            dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        result["train_images"] = len(train_dataset)
        result["channels"] = 3
        result["image_size"] = [224, 224]

        statistics[dataset_name] = result

        rows.append({
            "Dataset": dataset_name,
            "Train Images": len(train_dataset),
            "Mean R": result["mean"][0],
            "Mean G": result["mean"][1],
            "Mean B": result["mean"][2],
            "Std R": result["std"][0],
            "Std G": result["std"][1],
            "Std B": result["std"][2],
            "Min Pixel": result["min_pixel_value"],
            "Max Pixel": result["max_pixel_value"],
        })

    return statistics, pd.DataFrame(rows)


# ============================================================
# Save Statistics
# ============================================================

def save_normalization_statistics(statistics, dataset_roots):
    for dataset_name, result in statistics.items():
        output_dir = (
            Path(dataset_roots[dataset_name])
            / "preprocessing"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "normalization.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
            )

        print(
            f"{dataset_name}: saved {output_path}"
        )