import random
import re
from collections import defaultdict
from itertools import combinations

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


def get_targets(dataset):
    targets = getattr(dataset, "targets", None)

    if targets is None:
        return None

    return [int(target) for target in targets]


def is_labelled_split(dataset_name, split_name, dataset):
    if dataset_name == "tiny_imagenet" and split_name == "Test":
        return False

    targets = get_targets(dataset)

    return targets is not None and len(targets) == len(dataset)


# ============================================================
# Human-Readable Class Names
# ============================================================

def format_class_name(name):
    if isinstance(name, (tuple, list)):
        name = name[0]

    name = str(name)

    name = re.sub(r"\\+n", "\n", name)

    return (
        name
        .replace("_", " ")
        .replace(",", ",\n")
        .strip()
    )


def get_class_name(dataset, class_id):
    base_dataset = get_base_dataset(dataset)

    classes = getattr(base_dataset, "classes", None)
    class_to_name = getattr(base_dataset, "class_to_name", None)
    idx_to_words = getattr(base_dataset, "idx_to_words", None)

    class_value = class_id

    if classes is not None:
        try:
            class_value = classes[class_id]
        except Exception:
            pass

    if class_to_name is not None:
        possible_keys = [
            class_id,
            str(class_id),
            class_value,
            str(class_value),
        ]

        for key in possible_keys:
            try:
                if key in class_to_name:
                    return format_class_name(
                        class_to_name[key]
                    )
            except Exception:
                continue

    if idx_to_words is not None:
        possible_keys = [
            class_id,
            str(class_id),
            class_value,
            str(class_value),
        ]

        for key in possible_keys:
            try:
                if key in idx_to_words:
                    return format_class_name(
                        idx_to_words[key]
                    )
            except Exception:
                continue

    return format_class_name(class_value)


# ============================================================
# Image Conversion
# ============================================================

def get_rgb_image(dataset, index):
    item = dataset[index]
    image = item[0] if isinstance(item, (tuple, list)) else item

    if isinstance(image, Image.Image):
        return image.convert("RGB")

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

    image = np.clip(
        np.rint(image),
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(image)


# ============================================================
# Feature Representation
# ============================================================

def image_to_feature(dataset, index, feature_size=32):
    image = get_rgb_image(dataset, index)

    image = image.resize(
        (feature_size, feature_size),
        Image.Resampling.BILINEAR,
    )

    feature = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    return feature.reshape(-1)


# ============================================================
# Class Sampling
# ============================================================

def get_class_indices(dataset):
    targets = get_targets(dataset)

    if targets is None:
        return {}

    class_indices = defaultdict(list)

    for index, target in enumerate(targets):
        class_indices[target].append(index)

    return dict(class_indices)


def select_classes(
    class_indices,
    max_classes=None,
    seed=42,
):
    class_ids = sorted(class_indices.keys())

    if max_classes is None or len(class_ids) <= max_classes:
        return class_ids

    rng = random.Random(seed)

    return sorted(
        rng.sample(
            class_ids,
            max_classes,
        )
    )


def sample_class_indices(
    indices,
    max_images_per_class=50,
    seed=42,
):
    if (
        max_images_per_class is None
        or len(indices) <= max_images_per_class
    ):
        return list(indices)

    rng = random.Random(seed)

    return sorted(
        rng.sample(
            indices,
            max_images_per_class,
        )
    )


# ============================================================
# Main Inter-Class Variation Analysis
# ============================================================

def inter_class_variation_analysis(
    dataset_list,
    max_classes=200,
    max_images_per_class=50,
    min_images_per_class=2,
    feature_size=32,
    seed=42,
):
    summaries = []
    pair_details = []

    for (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_name,
    ) in dataset_list:

        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in tqdm(
            splits,
            desc=f"{dataset_name} splits",
        ):
            # ------------------------------------------------
            # Unlabelled Split
            # ------------------------------------------------

            if not is_labelled_split(
                dataset_name,
                split_name,
                dataset,
            ):
                summaries.append({
                    "Dataset": dataset_name,
                    "Split": split_name,
                    "Label Status": "Unlabelled",
                    "Classes Available": "N/A",
                    "Classes Analyzed": "N/A",
                    "Images Analyzed": "N/A",
                    "Class Pairs": "N/A",
                    "Mean Inter-Class Distance": "N/A",
                    "Median Inter-Class Distance": "N/A",
                    "Std Inter-Class Distance": "N/A",
                    "Min Inter-Class Distance": "N/A",
                    "Max Inter-Class Distance": "N/A",
                    "Status": "N/A",
                })

                continue

            # ------------------------------------------------
            # Class → Image Index Mapping
            # ------------------------------------------------

            class_indices = get_class_indices(dataset)

            eligible_classes = {
                class_id: indices
                for class_id, indices in class_indices.items()
                if len(indices) >= min_images_per_class
            }

            selected_classes = select_classes(
                eligible_classes,
                max_classes=max_classes,
                seed=f"{seed}:{dataset_name}:{split_name}",
            )

            centroids = {}
            analyzed_counts = {}

            images_analyzed = 0

            # ------------------------------------------------
            # Build Class Centroids
            # ------------------------------------------------

            for class_id in tqdm(
                selected_classes,
                desc=split_name,
                leave=False,
            ):
                indices = sample_class_indices(
                    eligible_classes[class_id],
                    max_images_per_class=max_images_per_class,
                    seed=(
                        f"{seed}:"
                        f"{dataset_name}:"
                        f"{split_name}:"
                        f"{class_id}"
                    ),
                )

                features = []

                for index in indices:
                    try:
                        feature = image_to_feature(
                            dataset,
                            index,
                            feature_size=feature_size,
                        )

                        features.append(feature)

                    except Exception:
                        continue

                if len(features) < min_images_per_class:
                    continue

                features = np.asarray(
                    features,
                    dtype=np.float32,
                )

                centroids[class_id] = features.mean(
                    axis=0
                )

                analyzed_counts[class_id] = len(
                    features
                )

                images_analyzed += len(
                    features
                )

            class_ids = sorted(
                centroids.keys()
            )

            # ------------------------------------------------
            # Pairwise Class-Centroid Distances
            # ------------------------------------------------

            distances = []

            class_pairs = list(
                combinations(
                    class_ids,
                    2,
                )
            )

            for class_a, class_b in tqdm(
                class_pairs,
                desc=f"{split_name} pairs",
                leave=False,
            ):
                centroid_a = centroids[class_a]
                centroid_b = centroids[class_b]

                distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                centroid_a
                                - centroid_b
                            ) ** 2
                        )
                    )
                )

                distances.append(
                    distance
                )

                pair_details.append({
                    "Dataset": dataset_name,
                    "Split": split_name,
                    "Class A ID": class_a,
                    "Class A Name": get_class_name(
                        dataset,
                        class_a,
                    ),
                    "Class A Images": analyzed_counts[
                        class_a
                    ],
                    "Class B ID": class_b,
                    "Class B Name": get_class_name(
                        dataset,
                        class_b,
                    ),
                    "Class B Images": analyzed_counts[
                        class_b
                    ],
                    "Inter-Class Distance": round(
                        distance,
                        6,
                    ),
                })

            if not distances:
                continue

            distances = np.asarray(
                distances,
                dtype=np.float32,
            )

            # ------------------------------------------------
            # Split-Level Summary
            # ------------------------------------------------

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Label Status": "Labelled",
                "Classes Available": len(
                    class_indices
                ),
                "Classes Analyzed": len(
                    class_ids
                ),
                "Images Analyzed": images_analyzed,
                "Class Pairs": len(
                    distances
                ),
                "Mean Inter-Class Distance": round(
                    float(
                        distances.mean()
                    ),
                    6,
                ),
                "Median Inter-Class Distance": round(
                    float(
                        np.median(
                            distances
                        )
                    ),
                    6,
                ),
                "Std Inter-Class Distance": round(
                    float(
                        distances.std()
                    ),
                    6,
                ),
                "Min Inter-Class Distance": round(
                    float(
                        distances.min()
                    ),
                    6,
                ),
                "Max Inter-Class Distance": round(
                    float(
                        distances.max()
                    ),
                    6,
                ),
                "Status": "Analyzed",
            })

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(pair_details),
    )


# ============================================================
# Distribution Visualization
# ============================================================

def plot_inter_class_distribution(
    inter_class_details,
    bins=40,
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

    datasets = (
        inter_class_details[
            "Dataset"
        ]
        .unique()
    )

    for dataset_name in datasets:
        dataset_data = inter_class_details[
            inter_class_details[
                "Dataset"
            ] == dataset_name
        ]

        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        all_values = dataset_data[
            "Inter-Class Distance"
        ].values

        min_value = all_values.min()
        max_value = all_values.max()

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            sharex=True,
            sharey=True,
        )

        for ax, split_name in zip(
            axes,
            split_names,
        ):
            split_data = dataset_data[
                dataset_data[
                    "Split"
                ] == split_name
            ]

            if split_data.empty:
                ax.set_title(
                    f"{split_name}\n"
                    f"Unlabelled / N/A"
                )

                ax.text(
                    0.5,
                    0.5,
                    "No labelled class data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                continue

            values = split_data[
                "Inter-Class Distance"
            ]

            mean = values.mean()
            median = values.median()

            ax.hist(
                values,
                bins=bins,
                range=(
                    min_value,
                    max_value,
                ),
                color=split_colors[
                    split_name
                ],
                alpha=0.75,
                edgecolor="white",
            )

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
                f"Pairs = {len(values):,}"
            )

            ax.set_xlabel(
                "Inter-Class Distance"
            )

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.3,
            )

            ax.legend()

        axes[0].set_ylabel(
            "Number of Class Pairs"
        )

        fig.suptitle(
            f"{dataset_title} - "
            f"Inter-Class Distance Distribution Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# Closest Class Pairs
# ============================================================

def plot_closest_class_pairs(
    inter_class_details,
    top_n=10,
    min_plot_images=10,
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

    datasets = (
        inter_class_details[
            "Dataset"
        ]
        .unique()
    )

    for dataset_name in datasets:
        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(24, 8),
        )

        for ax, split_name in zip(
            axes,
            split_names,
        ):
            data = inter_class_details[
                (
                    inter_class_details[
                        "Dataset"
                    ] == dataset_name
                )
                & (
                    inter_class_details[
                        "Split"
                    ] == split_name
                )
            ]

            # --------------------------------------------
            # Reliability Filter
            # --------------------------------------------

            data = data[
                (
                    data["Class A Images"]
                    >= min_plot_images
                )
                & (
                    data["Class B Images"]
                    >= min_plot_images
                )
            ]

            if data.empty:
                ax.set_title(
                    f"{split_name}\nN/A"
                )

                ax.axis("off")
                continue

            closest = (
                data
                .nsmallest(
                    top_n,
                    "Inter-Class Distance",
                )
                .copy()
            )

            closest["Class Pair"] = (
                closest["Class A Name"]
                + "\n↔\n"
                + closest["Class B Name"]
            )

            closest = closest.sort_values(
                "Inter-Class Distance",
                ascending=False,
            )

            bars = ax.barh(
                closest["Class Pair"],
                closest[
                    "Inter-Class Distance"
                ],
                color=split_colors[
                    split_name
                ],
            )

            ax.set_title(
                f"{split_name} - "
                f"{top_n} Closest Class Pairs"
            )

            ax.set_xlabel(
                "Inter-Class Distance"
            )

            ax.tick_params(
                axis="y",
                labelsize=7,
            )

            ax.grid(
                axis="x",
                linestyle="--",
                alpha=0.3,
            )

            max_value = closest[
                "Inter-Class Distance"
            ].max()

            ax.set_xlim(
                0,
                max_value * 1.15,
            )

            for bar, value in zip(
                bars,
                closest[
                    "Inter-Class Distance"
                ],
            ):
                ax.text(
                    value
                    + max_value * 0.01,
                    bar.get_y()
                    + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center",
                    fontsize=8,
                )

        fig.suptitle(
            f"{dataset_title} - "
            f"Most Visually Similar Class Pairs",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# Farthest Class Pairs
# ============================================================

def plot_farthest_class_pairs(
    inter_class_details,
    top_n=10,
    min_plot_images=10,
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

    datasets = (
        inter_class_details[
            "Dataset"
        ]
        .unique()
    )

    for dataset_name in datasets:
        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(24, 8),
        )

        for ax, split_name in zip(
            axes,
            split_names,
        ):
            data = inter_class_details[
                (
                    inter_class_details[
                        "Dataset"
                    ] == dataset_name
                )
                & (
                    inter_class_details[
                        "Split"
                    ] == split_name
                )
            ]

            data = data[
                (
                    data["Class A Images"]
                    >= min_plot_images
                )
                & (
                    data["Class B Images"]
                    >= min_plot_images
                )
            ]

            if data.empty:
                ax.set_title(
                    f"{split_name}\nN/A"
                )

                ax.axis("off")
                continue

            farthest = (
                data
                .nlargest(
                    top_n,
                    "Inter-Class Distance",
                )
                .copy()
            )

            farthest["Class Pair"] = (
                farthest["Class A Name"]
                + "\n↔\n"
                + farthest["Class B Name"]
            )

            farthest = farthest.sort_values(
                "Inter-Class Distance",
                ascending=True,
            )

            bars = ax.barh(
                farthest["Class Pair"],
                farthest[
                    "Inter-Class Distance"
                ],
                color=split_colors[
                    split_name
                ],
            )

            ax.set_title(
                f"{split_name} - "
                f"{top_n} Farthest Class Pairs"
            )

            ax.set_xlabel(
                "Inter-Class Distance"
            )

            ax.tick_params(
                axis="y",
                labelsize=7,
            )

            ax.grid(
                axis="x",
                linestyle="--",
                alpha=0.3,
            )

            max_value = farthest[
                "Inter-Class Distance"
            ].max()

            ax.set_xlim(
                0,
                max_value * 1.12,
            )

            for bar, value in zip(
                bars,
                farthest[
                    "Inter-Class Distance"
                ],
            ):
                ax.text(
                    value
                    + max_value * 0.01,
                    bar.get_y()
                    + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center",
                    fontsize=8,
                )

        fig.suptitle(
            f"{dataset_title} - "
            f"Most Visually Separated Class Pairs",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()