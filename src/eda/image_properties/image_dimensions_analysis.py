import random

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def get_image_dimensions(image):
    if hasattr(image, "size") and not hasattr(image, "shape"):
        width, height = image.size
        return int(width), int(height)

    if hasattr(image, "shape"):
        shape = image.shape

        if len(shape) == 3:
            if shape[0] in {1, 3, 4}:
                return int(shape[2]), int(shape[1])

            return int(shape[1]), int(shape[0])

        if len(shape) == 2:
            return int(shape[1]), int(shape[0])

    raise ValueError("Could not determine image dimensions.")


def get_sample_indices(dataset, max_images, seed):
    total_images = len(dataset)

    if max_images is None or total_images <= max_images:
        return list(range(total_images))

    rng = random.Random(seed)
    return sorted(rng.sample(range(total_images), max_images))


def image_dimensions_analysis(dataset_list, max_images=5000, seed=42):
    summaries = []
    dimension_details = {}

    for train_dataset, val_dataset, test_dataset, dataset_name in tqdm(dataset_list):
        dataset_dimensions = {}

        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in splits:
            sample_seed = f"{seed}:{dataset_name}:{split_name}"
            print(sample_seed)

            indices = get_sample_indices(dataset, max_images=max_images, seed=sample_seed)

            records = []
            skipped = 0

            for index in indices:
                try:
                    item = dataset[index]
                    image = item[0] if isinstance(item, (tuple, list)) else item

                    width, height = get_image_dimensions(image)

                    records.append({
                        "Image Index": index,
                        "Width": width,
                        "Height": height,
                    })

                except Exception:
                    skipped += 1

            dimensions = pd.DataFrame(records)

            if dimensions.empty:
                summaries.append({
                    "Dataset": dataset_name,
                    "Split": split_name,
                    "Total Images": len(dataset),
                    "Analyzed Images": 0,
                    "Skipped Images": skipped,
                    "Min Width": "N/A",
                    "Max Width": "N/A",
                    "Mean Width": "N/A",
                    "Median Width": "N/A",
                    "Min Height": "N/A",
                    "Max Height": "N/A",
                    "Mean Height": "N/A",
                    "Median Height": "N/A",
                    "Unique Resolutions": "N/A",
                })

                dataset_dimensions[split_name] = dimensions
                continue

            resolution_count = (
                dimensions[["Width", "Height"]]
                .drop_duplicates()
                .shape[0]
            )

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": len(dataset),
                "Analyzed Images": len(dimensions),
                "Skipped Images": skipped,
                "Min Width": dimensions["Width"].min(),
                "Max Width": dimensions["Width"].max(),
                "Mean Width": round(dimensions["Width"].mean(), 2),
                "Median Width": round(dimensions["Width"].median(), 2),
                "Min Height": dimensions["Height"].min(),
                "Max Height": dimensions["Height"].max(),
                "Mean Height": round(dimensions["Height"].mean(), 2),
                "Median Height": round(dimensions["Height"].median(), 2),
                "Unique Resolutions": resolution_count,
            })

            dataset_dimensions[split_name] = dimensions

        dimension_details[dataset_name] = dataset_dimensions

    return pd.DataFrame(summaries), dimension_details

def plot_image_dimensions(dimension_details):
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

    for dataset_name, splits in dimension_details.items():
        dataset_title = dataset_titles.get(dataset_name, dataset_name)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5),
            sharex=True,
            sharey=True,
        )

        all_widths = []
        all_heights = []

        for split_name in split_names:
            dimensions = splits.get(split_name)

            if dimensions is None or dimensions.empty:
                continue

            all_widths.extend(dimensions["Width"].tolist())
            all_heights.extend(dimensions["Height"].tolist())

        min_width = min(all_widths)
        max_width = max(all_widths)
        min_height = min(all_heights)
        max_height = max(all_heights)

        width_margin = max((max_width - min_width) * 0.05, 5)
        height_margin = max((max_height - min_height) * 0.05, 5)

        for ax, split_name in zip(axes, split_names):
            dimensions = splits.get(split_name)

            if dimensions is None or dimensions.empty:
                ax.set_title(split_name)

                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                continue

            ax.scatter(
                dimensions["Width"],
                dimensions["Height"],
                color=split_colors[split_name],
                alpha=0.3,
                s=14,
                edgecolors="none",
            )

            ax.set_title(f"{split_name}\n n = {len(dimensions):,}")
            ax.set_xlabel("Image Width (pixels)")
            ax.grid(linestyle="--", alpha=0.3)
            ax.set_xlim(min_width - width_margin, max_width + width_margin)
            ax.set_ylim(min_height - height_margin, max_height + height_margin)
        axes[0].set_ylabel("Image Height (pixels)")

        fig.suptitle(
            f"{dataset_title} - Image Dimensions Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()