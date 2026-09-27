import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_base_dataset(dataset):
    base_dataset = getattr(dataset, "base_dataset", None)
    return base_dataset if base_dataset is not None else dataset


def get_sample_indices(dataset, max_images=5000, seed=42):
    total_images = len(dataset)

    if max_images is None or total_images <= max_images:
        return list(range(total_images))

    rng = random.Random(seed)
    return sorted(rng.sample(range(total_images), max_images))


def get_original_image_path(dataset, index):
    base_dataset = get_base_dataset(dataset)

    indices = getattr(dataset, "indices", None)
    base_index = int(indices[index]) if indices is not None else index

    samples = getattr(base_dataset, "samples", None)

    if samples is not None and base_index < len(samples):
        sample = samples[base_index]
        return sample[0] if isinstance(sample, (tuple, list)) else sample

    return None


def get_rgb_array(dataset, index):
    image_path = get_original_image_path(dataset, index)

    if image_path is not None:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return np.asarray(image, dtype=np.float64) / 255.0

    item = dataset[index]
    image = item[0] if isinstance(item, (tuple, list)) else item

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.float64) / 255.0

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    elif image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    elif image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image.astype(np.float64)

    if image.max() > 1:
        image /= 255.0

    return image


def rgb_mean_std_analysis(dataset_list, max_images=5000, seed=42):
    summaries = []

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in tqdm(splits):
            sample_seed = f"{seed}:{dataset_name}:{split_name}"
            indices = get_sample_indices(dataset, max_images, sample_seed)

            channel_sum = np.zeros(3, dtype=np.float64)
            channel_squared_sum = np.zeros(3, dtype=np.float64)

            total_pixels = 0
            analyzed = 0
            skipped = 0

            for index in indices:
                try:
                    image = get_rgb_array(dataset, index)

                    pixels = image.reshape(-1, 3)

                    channel_sum += pixels.sum(axis=0)
                    channel_squared_sum += np.square(pixels).sum(axis=0)

                    total_pixels += len(pixels)
                    analyzed += 1

                except Exception:
                    skipped += 1

            if total_pixels == 0:
                continue

            mean = channel_sum / total_pixels

            variance = (
                channel_squared_sum / total_pixels
                - np.square(mean)
            )

            std = np.sqrt(np.maximum(variance, 0))

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": len(dataset),
                "Analyzed Images": analyzed,
                "Skipped Images": skipped,
                "Mean R": round(mean[0], 4),
                "Mean G": round(mean[1], 4),
                "Mean B": round(mean[2], 4),
                "Std R": round(std[0], 4),
                "Std G": round(std[1], 4),
                "Std B": round(std[2], 4),
            })

    return pd.DataFrame(summaries)

def plot_rgb_mean_std(rgb_stats):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K Winter21",
        "imagenet21k_p": "ImageNet-21K-P Winter21",
    }

    split_names = ["Train", "Validation", "Test"]

    split_colors = {
        "Train": "tab:green",
        "Validation": "tab:orange",
        "Test": "tab:blue",
    }

    channels = ["R", "G", "B"]
    x = np.arange(len(channels))
    width = 0.35

    for dataset_name in rgb_stats["Dataset"].unique():
        dataset_data = rgb_stats[
            rgb_stats["Dataset"] == dataset_name
        ]

        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            sharey=True,
        )

        for ax, split_name in zip(axes, split_names):
            row = dataset_data[
                dataset_data["Split"] == split_name
            ]

            if row.empty:
                ax.set_title(split_name)
                continue

            row = row.iloc[0]

            means = [
                row["Mean R"],
                row["Mean G"],
                row["Mean B"],
            ]

            stds = [
                row["Std R"],
                row["Std G"],
                row["Std B"],
            ]

            mean_bars = ax.bar(
                x - width / 2,
                means,
                width,
                color=split_colors[split_name],
                label="Mean",
            )

            std_bars = ax.bar(
                x + width / 2,
                stds,
                width,
                color=split_colors[split_name],
                alpha=0.45,
                label="Standard Deviation",
            )

            ax.set_title(
                f"{split_name}\n"
                f"n = {int(row['Analyzed Images']):,}"
            )

            ax.set_xticks(x)
            ax.set_xticklabels(channels)

            ax.set_xlabel("RGB Channel")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend()

            for bars in [mean_bars, std_bars]:
                for bar in bars:
                    value = bar.get_height()

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.015,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        axes[0].set_ylabel("Normalized Pixel Value")

        fig.suptitle(
            f"{dataset_title} - RGB Mean and Standard Deviation Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()