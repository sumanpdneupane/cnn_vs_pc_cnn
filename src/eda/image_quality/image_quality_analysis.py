import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


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


def get_luminance(image):
    return (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )


def calculate_sharpness(luminance):
    if luminance.shape[0] < 3 or luminance.shape[1] < 3:
        return np.nan

    center = luminance[1:-1, 1:-1]

    laplacian = (
        luminance[:-2, 1:-1]
        + luminance[2:, 1:-1]
        + luminance[1:-1, :-2]
        + luminance[1:-1, 2:]
        - 4 * center
    )

    return float(np.var(laplacian))


def image_quality_analysis(dataset_list, max_images=5000, seed=42):
    summaries = []
    quality_details = {}

    for train_dataset, val_dataset, test_dataset, dataset_name in tqdm(dataset_list):
        dataset_details = {}

        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in splits:
            sample_seed = f"{seed}:{dataset_name}:{split_name}"
            indices = get_sample_indices(dataset, max_images, sample_seed)

            records = []
            skipped = 0

            for index in tqdm(
                indices,
                desc=split_name,
                leave=False,
            ):
                try:
                    image = get_rgb_array(dataset, index)
                    luminance = get_luminance(image)

                    brightness = float(np.mean(luminance))
                    contrast = float(np.std(luminance))
                    sharpness = calculate_sharpness(luminance)

                    records.append({
                        "Image Index": index,
                        "Brightness": brightness,
                        "Contrast": contrast,
                        "Sharpness": sharpness,
                    })

                except Exception:
                    skipped += 1

            quality = pd.DataFrame(records)
            quality = quality.dropna()

            if quality.empty:
                dataset_details[split_name] = quality
                continue

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": len(dataset),
                "Analyzed Images": len(quality),
                "Skipped Images": skipped,

                "Mean Brightness": round(quality["Brightness"].mean(), 4),
                "Median Brightness": round(quality["Brightness"].median(), 4),
                "Std Brightness": round(quality["Brightness"].std(), 4),

                "Mean Contrast": round(quality["Contrast"].mean(), 4),
                "Median Contrast": round(quality["Contrast"].median(), 4),
                "Std Contrast": round(quality["Contrast"].std(), 4),

                "Mean Sharpness": round(quality["Sharpness"].mean(), 6),
                "Median Sharpness": round(quality["Sharpness"].median(), 6),
                "Std Sharpness": round(quality["Sharpness"].std(), 6),
            })

            dataset_details[split_name] = quality

        quality_details[dataset_name] = dataset_details

    return pd.DataFrame(summaries), quality_details

def plot_quality_metric_distribution(
    quality_details,
    metric,
    bins=40,
):
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

    for dataset_name, splits in quality_details.items():
        dataset_title = dataset_titles.get(dataset_name, dataset_name)

        all_values = []

        for split_name in split_names:
            data = splits.get(split_name)

            if data is not None and not data.empty:
                all_values.extend(data[metric].dropna().tolist())

        if not all_values:
            continue

        min_value = min(all_values)
        max_value = max(all_values)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            sharex=True,
            sharey=True,
        )

        for ax, split_name in zip(axes, split_names):
            data = splits.get(split_name)

            if data is None or data.empty:
                ax.set_title(split_name)
                continue

            values = data[metric].dropna()

            ax.hist(
                values,
                bins=bins,
                range=(min_value, max_value),
                color=split_colors[split_name],
                alpha=0.75,
                edgecolor="white",
            )

            mean = values.mean()
            median = values.median()

            ax.axvline(
                mean,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label=f"Mean = {mean:.3f}",
            )

            ax.axvline(
                median,
                color="black",
                linestyle=":",
                linewidth=1.4,
                label=f"Median = {median:.3f}",
            )

            ax.set_title(
                f"{split_name}\n"
                f"n = {len(values):,}"
            )

            ax.set_xlabel(metric)
            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.3,
            )

            ax.legend()

        axes[0].set_ylabel("Number of Images")

        fig.suptitle(
            f"{dataset_title} - {metric} Distribution Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()