import random
from collections import Counter

import matplotlib.pyplot as plt
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

        if isinstance(sample, (tuple, list)):
            return sample[0]

        return sample

    return None


# def get_mode_and_channels(dataset, index):
#     image_path = get_original_image_path(dataset, index)
#
#     if image_path is not None:
#         with Image.open(image_path) as image:
#             return image.mode, len(image.getbands())
#
#     item = dataset[index]
#     image = item[0] if isinstance(item, (tuple, list)) else item
#
#     if isinstance(image, Image.Image):
#         return image.mode, len(image.getbands())
#
#     if hasattr(image, "shape"):
#         if len(image.shape) == 3:
#             channels = int(image.shape[0])
#
#             mode = {
#                 1: "L",
#                 3: "RGB",
#                 4: "RGBA",
#             }.get(channels, f"{channels}-Channel")
#
#             return mode, channels
#
#         if len(image.shape) == 2:
#             return "L", 1
#
#     raise ValueError("Could not determine image mode/channels.")

def get_mode_and_channels(dataset, index):
    item = dataset[index]
    image = item[0] if isinstance(item, (tuple, list)) else item

    if isinstance(image, Image.Image):
        return image.mode, len(image.getbands())

    if hasattr(image, "shape"):
        if len(image.shape) == 3:
            channels = int(image.shape[0])

            mode = {
                1: "L",
                3: "RGB",
                4: "RGBA",
            }.get(channels, f"{channels}-Channel")

            return mode, channels

        if len(image.shape) == 2:
            return "L", 1

    raise ValueError("Could not determine image mode/channels.")

def image_mode_channel_analysis(
    dataset_list,
    max_images=5000,
    seed=42,
):
    summaries = []
    mode_details = {}

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        dataset_details = {}

        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in tqdm(splits):
            sample_seed = f"{seed}:{dataset_name}:{split_name}"

            indices = get_sample_indices(
                dataset,
                max_images=max_images,
                seed=sample_seed,
            )

            mode_counts = Counter()
            channel_counts = Counter()
            skipped = 0

            for index in indices:
                try:
                    mode, channels = get_mode_and_channels(dataset, index)

                    mode_counts[mode] += 1
                    channel_counts[channels] += 1

                except Exception:
                    skipped += 1

            analyzed = sum(mode_counts.values())

            rgb_count = mode_counts.get("RGB", 0)
            grayscale_count = mode_counts.get("L", 0)
            rgba_count = mode_counts.get("RGBA", 0)

            other_count = (
                analyzed
                - rgb_count
                - grayscale_count
                - rgba_count
            )

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": len(dataset),
                "Analyzed Images": analyzed,
                "Skipped Images": skipped,
                "Unique Modes": len(mode_counts),
                "Most Common Mode": (
                    mode_counts.most_common(1)[0][0]
                    if mode_counts
                    else "N/A"
                ),
                "RGB (%)": round(rgb_count / analyzed * 100, 2) if analyzed else 0,
                "Grayscale (%)": round(grayscale_count / analyzed * 100, 2) if analyzed else 0,
                "RGBA (%)": round(rgba_count / analyzed * 100, 2) if analyzed else 0,
                "Other (%)": round(other_count / analyzed * 100, 2) if analyzed else 0,
            })

            dataset_details[split_name] = {
                "Mode Counts": mode_counts,
                "Channel Counts": channel_counts,
            }

        mode_details[dataset_name] = dataset_details

    return pd.DataFrame(summaries), mode_details

def plot_image_modes(mode_details):
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

    for dataset_name, splits in mode_details.items():
        dataset_title = dataset_titles.get(dataset_name, dataset_name)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5),
            sharey=False,
        )

        for ax, split_name in zip(axes, split_names):
            details = splits.get(split_name)

            if details is None:
                ax.set_title(split_name)
                continue

            counts = details["Mode Counts"]
            modes = list(counts.keys())
            values = list(counts.values())

            display_modes = [
                "Grayscale" if mode == "L" else mode
                for mode in modes
            ]

            bars = ax.bar(
                display_modes,
                values,
                color=split_colors[split_name],
            )

            ax.set_title(
                f"{split_name}\nn = {sum(values):,}"
            )

            ax.set_xlabel("Image Mode")
            ax.set_ylabel("Number of Images")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            max_height = max(values) if values else 1
            ax.set_ylim(0, max_height * 1.15)

            for bar in bars:
                height = bar.get_height()

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max_height * 0.015,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig.suptitle(
            f"{dataset_title} - Image Mode Distribution Across Splits",
            fontsize=16,
            y=0.98,
        )

        fig.subplots_adjust(
            top=0.80,
            bottom=0.15,
            left=0.06,
            right=0.98,
            wspace=0.25,
        )

        plt.show()