import hashlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


# ============================================================
# Dataset Helpers
# ============================================================

def get_base_dataset(dataset):
    base_dataset = getattr(dataset, "base_dataset", None)
    return base_dataset if base_dataset is not None else dataset


def get_sample_indices(dataset, max_images=None, seed=42):
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


def get_target(dataset, index):
    targets = getattr(dataset, "targets", None)

    if targets is None:
        return None

    try:
        return int(targets[index])
    except Exception:
        return None


# ============================================================
# Image Conversion and Exact Hashing
# ============================================================

def image_to_uint8_rgb(dataset, index):
    image_path = get_original_image_path(dataset, index)

    if image_path is not None:
        with Image.open(image_path) as image:
            return np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

    item = dataset[index]
    image = item[0] if isinstance(item, (tuple, list)) else item

    if isinstance(image, Image.Image):
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    if image.ndim == 2:
        image = np.stack(
            [image, image, image],
            axis=-1,
        )

    elif image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.transpose(
            image,
            (1, 2, 0),
        )

    if image.shape[-1] == 1:
        image = np.repeat(
            image,
            3,
            axis=-1,
        )

    elif image.shape[-1] == 4:
        image = image[:, :, :3]

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1:
            image = image * 255

    return np.clip(
        np.rint(image),
        0,
        255,
    ).astype(np.uint8)


def calculate_image_hash(dataset, index):
    image = image_to_uint8_rgb(
        dataset,
        index,
    )

    height, width = image.shape[:2]

    hasher = hashlib.sha256()

    hasher.update(
        f"{width}x{height}".encode()
    )

    hasher.update(
        image.tobytes()
    )

    return hasher.hexdigest()


# ============================================================
# Duplicate and Leakage Analysis
# ============================================================

def duplicate_leakage_analysis(
    dataset_list,
    max_images=None,
    seed=42,
):
    summaries = []
    duplicate_details = {}

    split_names = [
        "Train",
        "Validation",
        "Test",
    ]

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        records = []
        failed = 0

        for split_name, dataset in tqdm(
            splits,
            desc=f"{dataset_name} splits",
        ):
            sample_seed = (
                f"{seed}:{dataset_name}:{split_name}"
            )

            indices = get_sample_indices(
                dataset,
                max_images=max_images,
                seed=sample_seed,
            )

            for index in tqdm(
                indices,
                desc=split_name,
                leave=False,
            ):
                try:
                    image_hash = calculate_image_hash(
                        dataset,
                        index,
                    )

                    image_path = get_original_image_path(
                        dataset,
                        index,
                    )

                    records.append({
                        "Split": split_name,
                        "Image Index": index,
                        "Target": get_target(
                            dataset,
                            index,
                        ),
                        "Image Path": (
                            str(image_path)
                            if image_path is not None
                            else None
                        ),
                        "Hash": image_hash,
                    })

                except Exception:
                    failed += 1

        data = pd.DataFrame(records)

        dataset_details = {
            "All Images": data,
        }

        summary = {
            "Dataset": dataset_name,
            "Analyzed Images": len(data),
            "Hash Failures": failed,
        }

        # ====================================================
        # Within-Split Duplicate Detection
        # ====================================================

        for split_name in split_names:
            split_data = data[
                data["Split"] == split_name
            ]

            hash_counts = (
                split_data["Hash"]
                .value_counts()
            )

            duplicate_hashes = hash_counts[
                hash_counts > 1
            ]

            duplicate_groups = len(
                duplicate_hashes
            )

            duplicate_images = int(
                (duplicate_hashes - 1).sum()
            )

            involved_images = int(
                duplicate_hashes.sum()
            )

            summary[
                f"{split_name} Duplicate Groups"
            ] = duplicate_groups

            summary[
                f"{split_name} Duplicate Images"
            ] = duplicate_images

            summary[
                f"{split_name} Duplicate (%)"
            ] = round(
                duplicate_images
                / len(split_data)
                * 100,
                4,
            ) if len(split_data) else 0

            dataset_details[
                f"{split_name} Duplicates"
            ] = (
                split_data[
                    split_data["Hash"].isin(
                        duplicate_hashes.index
                    )
                ]
                .sort_values(
                    [
                        "Hash",
                        "Image Index",
                    ]
                )
                .reset_index(drop=True)
            )

            dataset_details[
                f"{split_name} Duplicate Involved Images"
            ] = involved_images

        # ====================================================
        # Cross-Split Leakage Detection
        # ====================================================

        split_hashes = {
            split_name: set(
                data[
                    data["Split"] == split_name
                ]["Hash"]
            )
            for split_name in split_names
        }

        train_val = (
            split_hashes["Train"]
            & split_hashes["Validation"]
        )

        train_test = (
            split_hashes["Train"]
            & split_hashes["Test"]
        )

        val_test = (
            split_hashes["Validation"]
            & split_hashes["Test"]
        )

        all_cross_split_hashes = (
            train_val
            | train_test
            | val_test
        )

        summary[
            "Train ↔ Validation Leakage"
        ] = len(train_val)

        summary[
            "Train ↔ Test Leakage"
        ] = len(train_test)

        summary[
            "Validation ↔ Test Leakage"
        ] = len(val_test)

        summary[
            "Unique Leakage Groups"
        ] = len(all_cross_split_hashes)

        leakage_rows = (
            data[
                data["Hash"].isin(
                    all_cross_split_hashes
                )
            ]
            .sort_values(
                [
                    "Hash",
                    "Split",
                    "Image Index",
                ]
            )
            .reset_index(drop=True)
        )

        dataset_details[
            "Cross-Split Leakage"
        ] = leakage_rows

        summary[
            "Leakage Status"
        ] = (
            "Passed"
            if len(all_cross_split_hashes) == 0
            else "Leakage Detected"
        )

        summaries.append(summary)

        duplicate_details[
            dataset_name
        ] = dataset_details

    return (
        pd.DataFrame(summaries),
        duplicate_details,
    )


# ============================================================
# Duplicate Label Consistency Analysis
# ============================================================

def analyze_duplicate_label_consistency(
    duplicate_details,
):
    rows = []

    for dataset_name, details in duplicate_details.items():
        leakage = details.get(
            "Cross-Split Leakage"
        )

        if leakage is None or leakage.empty:
            continue

        for image_hash, group in leakage.groupby(
            "Hash"
        ):
            targets = (
                group["Target"]
                .dropna()
                .astype(int)
                .unique()
            )

            targets = sorted(targets)

            rows.append({
                "Dataset": dataset_name,
                "Hash": image_hash,
                "Occurrences": len(group),
                "Splits": ", ".join(
                    sorted(
                        group["Split"].unique()
                    )
                ),
                "Unique Targets": len(targets),
                "Targets": (
                    ", ".join(
                        map(str, targets)
                    )
                    if targets
                    else "N/A"
                ),
                "Label Consistency": (
                    "N/A"
                    if len(targets) == 0
                    else (
                        "Consistent"
                        if len(targets) == 1
                        else "Conflict"
                    )
                ),
            })

    return pd.DataFrame(rows)


def duplicate_label_consistency_summary(
    duplicate_label_details,
):
    if duplicate_label_details.empty:
        return pd.DataFrame()

    summary = (
        duplicate_label_details
        .groupby(
            [
                "Dataset",
                "Label Consistency",
            ]
        )
        .size()
        .reset_index(
            name="Leakage Groups"
        )
    )

    return summary


# ============================================================
# Duplicate Visualization
# ============================================================

def plot_duplicate_images(
    duplicate_stats,
):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K Winter21",
        "imagenet21k_p": "ImageNet-21K-P Winter21",
    }

    split_names = [
        "Train",
        "Validation",
        "Test",
    ]

    split_colors = {
        "Train": "tab:green",
        "Validation": "tab:orange",
        "Test": "tab:blue",
    }

    for _, row in duplicate_stats.iterrows():
        dataset_name = row["Dataset"]

        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        percentages = [
            row[
                f"{split_name} Duplicate (%)"
            ]
            for split_name in split_names
        ]

        counts = [
            int(
                row[
                    f"{split_name} Duplicate Images"
                ]
            )
            for split_name in split_names
        ]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            sharey=True,
        )

        max_percentage = max(
            max(percentages),
            0.1,
        )

        for (
            ax,
            split_name,
            percentage,
            count,
        ) in zip(
            axes,
            split_names,
            percentages,
            counts,
        ):
            bar = ax.bar(
                ["Duplicates"],
                [percentage],
                color=split_colors[
                    split_name
                ],
                width=0.5,
            )

            ax.set_title(
                f"{split_name}\n"
                f"Duplicates = {count:,}"
            )

            ax.set_xlabel(
                "Duplicate Check"
            )

            ax.set_ylim(
                0,
                max_percentage * 1.25,
            )

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.3,
            )

            ax.text(
                bar[0].get_x()
                + bar[0].get_width() / 2,
                percentage
                + max_percentage * 0.03,
                f"{percentage:.4f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        axes[0].set_ylabel(
            "Duplicate Images (%)"
        )

        fig.suptitle(
            f"{dataset_title} - Exact Duplicate Images Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# Cross-Split Leakage Visualization
# ============================================================

def plot_data_leakage(
    duplicate_stats,
):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K Winter21",
        "imagenet21k_p": "ImageNet-21K-P Winter21",
    }

    comparisons = [
        "Train ↔ Validation",
        "Train ↔ Test",
        "Validation ↔ Test",
    ]

    columns = [
        "Train ↔ Validation Leakage",
        "Train ↔ Test Leakage",
        "Validation ↔ Test Leakage",
    ]

    for _, row in duplicate_stats.iterrows():
        dataset_name = row["Dataset"]

        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        values = [
            int(row[column])
            for column in columns
        ]

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        bars = ax.bar(
            comparisons,
            values,
        )

        ax.set_title(
            f"{dataset_title} - Cross-Split Data Leakage"
        )

        ax.set_xlabel(
            "Split Comparison"
        )

        ax.set_ylabel(
            "Shared Exact Images"
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

        max_value = max(
            max(values),
            1,
        )

        ax.set_ylim(
            0,
            max_value * 1.2,
        )

        for bar, value in zip(
            bars,
            values,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value
                + max_value * 0.02,
                f"{value:,}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()


# ============================================================
# Label Consistency Visualization
# ============================================================

def plot_label_consistency(
    duplicate_label_details,
):
    if duplicate_label_details.empty:
        print(
            "No cross-split leakage found."
        )
        return

    datasets = (
        duplicate_label_details[
            "Dataset"
        ]
        .unique()
    )

    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K Winter21",
        "imagenet21k_p": "ImageNet-21K-P Winter21",
    }

    categories = [
        "Consistent",
        "Conflict",
        "N/A",
    ]

    for dataset_name in datasets:
        data = duplicate_label_details[
            duplicate_label_details[
                "Dataset"
            ] == dataset_name
        ]

        values = [
            int(
                (
                    data[
                        "Label Consistency"
                    ] == category
                ).sum()
            )
            for category in categories
        ]

        if sum(values) == 0:
            continue

        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        bars = plt.bar(
            categories,
            values,
        )

        plt.title(
            f"{dataset_title} - Duplicate Label Consistency"
        )

        plt.xlabel(
            "Label Consistency"
        )

        plt.ylabel(
            "Leakage Groups"
        )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

        max_value = max(
            max(values),
            1,
        )

        plt.ylim(
            0,
            max_value * 1.2,
        )

        for bar, value in zip(
            bars,
            values,
        ):
            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value
                + max_value * 0.02,
                f"{value:,}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()
