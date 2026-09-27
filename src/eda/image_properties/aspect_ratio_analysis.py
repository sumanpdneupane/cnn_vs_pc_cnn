import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def aspect_ratio_analysis(dimension_details):
    summaries = []
    aspect_ratio_details = {}

    for dataset_name, splits in dimension_details.items():
        dataset_ratios = {}

        for split_name, dimensions in tqdm(splits.items()):
            if dimensions is None or dimensions.empty:
                dataset_ratios[split_name] = None
                continue

            data = dimensions.copy()
            data["Aspect Ratio"] = data["Width"] / data["Height"]

            data["Orientation"] = np.select(
                [
                    data["Aspect Ratio"] < 0.95,
                    data["Aspect Ratio"] > 1.05,
                ],
                [
                    "Portrait",
                    "Landscape",
                ],
                default="Near Square",
            )

            total = len(data)

            portrait = (data["Orientation"] == "Portrait").sum()
            square = (data["Orientation"] == "Near Square").sum()
            landscape = (data["Orientation"] == "Landscape").sum()

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Analyzed Images": total,
                "Min Aspect Ratio": round(data["Aspect Ratio"].min(), 3),
                "Max Aspect Ratio": round(data["Aspect Ratio"].max(), 3),
                "Mean Aspect Ratio": round(data["Aspect Ratio"].mean(), 3),
                "Median Aspect Ratio": round(data["Aspect Ratio"].median(), 3),
                "Std Aspect Ratio": round(data["Aspect Ratio"].std(), 3),
                "Portrait (%)": round(portrait / total * 100, 2),
                "Near Square (%)": round(square / total * 100, 2),
                "Landscape (%)": round(landscape / total * 100, 2),
            })

            dataset_ratios[split_name] = data

        aspect_ratio_details[dataset_name] = dataset_ratios

    return pd.DataFrame(summaries), aspect_ratio_details

def plot_aspect_ratio_distribution(aspect_ratio_details, bins=40):
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

    for dataset_name, splits in aspect_ratio_details.items():
        dataset_title = dataset_titles.get(dataset_name, dataset_name)

        all_ratios = []

        for split_name in split_names:
            data = splits.get(split_name)

            if data is not None and not data.empty:
                all_ratios.extend(data["Aspect Ratio"].tolist())

        if not all_ratios:
            continue

        min_ratio = min(all_ratios)
        max_ratio = max(all_ratios)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 4.5),
            sharex=True,
            sharey=True,
        )

        for ax, split_name in zip(axes, split_names):
            data = splits.get(split_name)

            if data is None or data.empty:
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

            ratios = data["Aspect Ratio"]

            ax.hist(
                ratios,
                bins=bins,
                range=(min_ratio, max_ratio),
                color=split_colors[split_name],
                alpha=0.75,
                edgecolor="white",
            )

            ax.axvline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Square = 1.0",
            )

            ax.set_title(
                f"{split_name}\n"
                f"n = {len(data):,}"
            )

            ax.set_xlabel("Aspect Ratio (Width / Height)")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend()

        axes[0].set_ylabel("Number of Images")

        fig.suptitle(
            f"{dataset_title} - Aspect Ratio Distribution Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()