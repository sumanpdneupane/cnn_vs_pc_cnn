import random

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


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


def check_image_integrity(dataset, index):
    image_path = get_original_image_path(dataset, index)

    try:
        if image_path is not None:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image.load()

                width, height = image.size

                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"Invalid dimensions: {width} x {height}"
                    )

            return {
                "Valid": True,
                "Image Path": str(image_path),
                "Error Type": None,
                "Error Message": None,
            }

        item = dataset[index]
        image = item[0] if isinstance(item, (tuple, list)) else item

        if isinstance(image, Image.Image):
            image.load()
            width, height = image.size

        elif hasattr(image, "shape"):
            shape = image.shape

            if len(shape) < 2:
                raise ValueError(
                    f"Invalid image shape: {shape}"
                )

            if len(shape) == 3 and shape[0] in {1, 3, 4}:
                height, width = shape[1], shape[2]

            else:
                height, width = shape[0], shape[1]

        else:
            raise ValueError(
                "Unsupported image representation."
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid dimensions: {width} x {height}"
            )

        return {
            "Valid": True,
            "Image Path": None,
            "Error Type": None,
            "Error Message": None,
        }

    except Exception as error:
        return {
            "Valid": False,
            "Image Path": (
                str(image_path)
                if image_path is not None
                else None
            ),
            "Error Type": type(error).__name__,
            "Error Message": str(error),
        }

def corrupted_image_analysis(
    dataset_list,
    max_images=None,
    seed=42,
):
    summaries = []
    corrupted_details = {}

    for train_dataset, val_dataset, test_dataset, dataset_name in tqdm(dataset_list):
        dataset_details = {}

        splits = [
            ("Train", train_dataset),
            ("Validation", val_dataset),
            ("Test", test_dataset),
        ]

        for split_name, dataset in splits:
            sample_seed = f"{seed}:{dataset_name}:{split_name}"

            indices = get_sample_indices(
                dataset,
                max_images=max_images,
                seed=sample_seed,
            )

            corrupted = []

            for index in tqdm(
                indices,
                desc=split_name,
                leave=False,
            ):
                result = check_image_integrity(
                    dataset,
                    index,
                )

                if not result["Valid"]:
                    corrupted.append({
                        "Image Index": index,
                        "Image Path": result["Image Path"],
                        "Error Type": result["Error Type"],
                        "Error Message": result["Error Message"],
                    })

            corrupted_df = pd.DataFrame(corrupted)

            analyzed = len(indices)
            corrupted_count = len(corrupted_df)
            valid_count = analyzed - corrupted_count

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Total Images": len(dataset),
                "Analyzed Images": analyzed,
                "Valid Images": valid_count,
                "Corrupted Images": corrupted_count,
                "Corrupted (%)": round(
                    corrupted_count / analyzed * 100,
                    4,
                ) if analyzed else 0,
                "Integrity Status": (
                    "Passed"
                    if corrupted_count == 0
                    else "Issues Found"
                ),
            })

            dataset_details[split_name] = corrupted_df

        corrupted_details[dataset_name] = dataset_details

    return (
        pd.DataFrame(summaries),
        corrupted_details,
    )

def plot_corrupted_images(corruption_stats):
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

    for dataset_name in corruption_stats["Dataset"].unique():
        dataset_data = corruption_stats[
            corruption_stats["Dataset"] == dataset_name
        ]

        dataset_title = dataset_titles.get(
            dataset_name,
            dataset_name,
        )

        percentages = dataset_data[
            "Corrupted (%)"
        ].tolist()

        max_percentage = max(
            max(percentages),
            0.1,
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            sharey=True,
        )

        for ax, split_name in zip(
            axes,
            split_names,
        ):
            row = dataset_data[
                dataset_data["Split"] == split_name
            ]

            if row.empty:
                ax.set_title(split_name)
                continue

            row = row.iloc[0]

            percentage = row["Corrupted (%)"]
            corrupted = int(row["Corrupted Images"])
            analyzed = int(row["Analyzed Images"])

            bar = ax.bar(
                ["Corrupted"],
                [percentage],
                color=split_colors[split_name],
                width=0.5,
            )

            ax.set_title(
                f"{split_name}\n"
                f"n = {analyzed:,}"
            )

            ax.set_xlabel("Integrity Check")
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
                f"{corrupted:,}\n"
                f"({percentage:.4f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        axes[0].set_ylabel(
            "Corrupted Images (%)"
        )

        fig.suptitle(
            f"{dataset_title} - Corrupted Image Detection Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()