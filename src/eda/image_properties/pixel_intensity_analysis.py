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
            return np.asarray(image, dtype=np.float32) / 255.0

    item = dataset[index]
    image = item[0] if isinstance(item, (tuple, list)) else item

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0

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

    image = image.astype(np.float32)

    if image.max() > 1:
        image /= 255.0

    return image


def percentile_from_histogram(histogram, bin_edges, percentile):
    cumulative = np.cumsum(histogram)
    target = cumulative[-1] * percentile / 100

    index = np.searchsorted(cumulative, target)
    index = min(index, len(bin_edges) - 2)

    return (bin_edges[index] + bin_edges[index + 1]) / 2


def pixel_intensity_analysis(
    dataset_list,
    max_images=5000,
    seed=42,
    bins=256,
):
    summaries = []
    intensity_details = {}

    bin_edges = np.linspace(0, 1, bins + 1)

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        dataset_details = {}

        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in tqdm(splits):
            sample_seed = f"{seed}:{dataset_name}:{split_name}"
            indices = get_sample_indices(dataset, max_images, sample_seed)

            histogram = np.zeros(bins, dtype=np.int64)

            intensity_sum = 0.0
            intensity_squared_sum = 0.0
            total_pixels = 0

            min_intensity = 1.0
            max_intensity = 0.0

            analyzed = 0
            skipped = 0

            for index in indices:
                try:
                    image = get_rgb_array(dataset, index)

                    intensity = (
                        0.299 * image[:, :, 0]
                        + 0.587 * image[:, :, 1]
                        + 0.114 * image[:, :, 2]
                    )

                    values = intensity.ravel()

                    hist, _ = np.histogram(
                        values,
                        bins=bin_edges,
                    )

                    histogram += hist

                    intensity_sum += values.sum(dtype=np.float64)
                    intensity_squared_sum += np.square(values, dtype=np.float64).sum()

                    total_pixels += values.size

                    min_intensity = min(
                        min_intensity,
                        float(values.min()),
                    )

                    max_intensity = max(
                        max_intensity,
                        float(values.max()),
                    )

                    analyzed += 1

                except Exception:
                    skipped += 1

            if total_pixels == 0:
                continue

            mean = intensity_sum / total_pixels

            variance = (
                intensity_squared_sum / total_pixels
                - mean ** 2
            )

            std = np.sqrt(max(variance, 0))

            median = percentile_from_histogram(
                histogram,
                bin_edges,
                50,
            )

            p05 = percentile_from_histogram(
                histogram,
                bin_edges,
                5,
            )

            p25 = percentile_from_histogram(
                histogram,
                bin_edges,
                25,
            )

            p75 = percentile_from_histogram(
                histogram,
                bin_edges,
                75,
            )

            p95 = percentile_from_histogram(
                histogram,
                bin_edges,
                95,
            )

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": len(dataset),
                "Analyzed Images": analyzed,
                "Skipped Images": skipped,
                "Total Pixels": total_pixels,
                "Min Intensity": round(min_intensity, 4),
                "Max Intensity": round(max_intensity, 4),
                "Mean Intensity": round(mean, 4),
                "Median Intensity": round(median, 4),
                "Std Intensity": round(std, 4),
                "5th Percentile": round(p05, 4),
                "25th Percentile": round(p25, 4),
                "75th Percentile": round(p75, 4),
                "95th Percentile": round(p95, 4),
            })

            dataset_details[split_name] = {
                "Histogram": histogram,
                "Bin Edges": bin_edges,
                "Analyzed Images": analyzed,
            }

        intensity_details[dataset_name] = dataset_details

    return pd.DataFrame(summaries), intensity_details

def plot_pixel_intensity_distribution(intensity_details):
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

    for dataset_name, splits in intensity_details.items():
        dataset_title = dataset_titles.get(dataset_name, dataset_name)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            sharex=True,
            sharey=True,
        )

        for ax, split_name in zip(axes, split_names):
            details = splits.get(split_name)

            if details is None:
                ax.set_title(split_name)
                continue

            histogram = details["Histogram"]
            bin_edges = details["Bin Edges"]

            centers = (
                bin_edges[:-1]
                + bin_edges[1:]
            ) / 2

            percentages = (
                histogram
                / histogram.sum()
                * 100
            )

            mean = np.average(
                centers,
                weights=histogram,
            )

            ax.plot(
                centers,
                percentages,
                color=split_colors[split_name],
                linewidth=1.8,
            )

            ax.fill_between(
                centers,
                percentages,
                color=split_colors[split_name],
                alpha=0.2,
            )

            ax.axvline(
                mean,
                color=split_colors[split_name],
                linestyle="--",
                linewidth=1.5,
                label=f"Mean = {mean:.3f}",
            )

            ax.set_title(
                f"{split_name}\n"
                f"n = {details['Analyzed Images']:,}"
            )

            ax.set_xlabel("Normalized Pixel Intensity")
            ax.set_xlim(0, 1)
            ax.grid(
                linestyle="--",
                alpha=0.3,
            )

            ax.legend()

        axes[0].set_ylabel("Pixels per Bin (%)")

        fig.suptitle(
            f"{dataset_title} - Pixel Intensity Distribution Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()