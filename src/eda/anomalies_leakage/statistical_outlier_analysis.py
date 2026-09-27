import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_iqr_bounds(values, multiplier=1.5):
    values = pd.Series(values).dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return lower, upper


def statistical_outlier_analysis(
    dimension_details,
    quality_details,
    iqr_multiplier=1.5,
):
    summaries = []
    outlier_details = {}
    threshold_details = {}

    metrics = [
        "Width",
        "Height",
        "Aspect Ratio",
        "Brightness",
        "Contrast",
        "Sharpness",
    ]

    split_names = ["Train", "Validation", "Test"]

    for dataset_name, dimension_splits in dimension_details.items():
        dataset_outliers = {}
        dataset_thresholds = {}

        for split_name in tqdm(
            split_names,
            desc=f"{dataset_name} splits",
        ):
            dimensions = dimension_splits.get(split_name)
            quality = quality_details.get(dataset_name, {}).get(split_name)

            if (
                dimensions is None
                or dimensions.empty
                or quality is None
                or quality.empty
            ):
                continue

            dimensions = dimensions.copy()
            quality = quality.copy()

            dimensions["Aspect Ratio"] = (
                dimensions["Width"]
                / dimensions["Height"]
            )

            data = dimensions[
                [
                    "Image Index",
                    "Width",
                    "Height",
                    "Aspect Ratio",
                ]
            ].merge(
                quality[
                    [
                        "Image Index",
                        "Brightness",
                        "Contrast",
                        "Sharpness",
                    ]
                ],
                on="Image Index",
                how="inner",
            )

            split_thresholds = {}

            for metric in metrics:
                lower, upper = get_iqr_bounds(
                    data[metric],
                    multiplier=iqr_multiplier,
                )

                split_thresholds[metric] = {
                    "Lower": lower,
                    "Upper": upper,
                }

                data[f"{metric} Outlier"] = (
                    (data[metric] < lower)
                    | (data[metric] > upper)
                )

            outlier_columns = [
                f"{metric} Outlier"
                for metric in metrics
            ]

            data["Outlier Count"] = (
                data[outlier_columns]
                .sum(axis=1)
            )

            data["Is Outlier"] = (
                data["Outlier Count"] > 0
            )

            data["Outlier Metrics"] = data.apply(
                lambda row: ", ".join(
                    metric
                    for metric in metrics
                    if row[f"{metric} Outlier"]
                ),
                axis=1,
            )

            total = len(data)
            outlier_images = int(data["Is Outlier"].sum())

            summary = {
                "Dataset": dataset_name,
                "Split": split_name,
                "Analyzed Images": total,
                "Outlier Images": outlier_images,
                "Outlier (%)": round(
                    outlier_images / total * 100,
                    2,
                ),
            }

            for metric in metrics:
                count = int(
                    data[f"{metric} Outlier"].sum()
                )

                summary[f"{metric} Outliers"] = count

                summary[f"{metric} Outlier (%)"] = round(
                    count / total * 100,
                    2,
                )

            summaries.append(summary)

            dataset_outliers[split_name] = data
            dataset_thresholds[split_name] = split_thresholds

        outlier_details[dataset_name] = dataset_outliers
        threshold_details[dataset_name] = dataset_thresholds

    return (
        pd.DataFrame(summaries),
        outlier_details,
        threshold_details,
    )

def plot_statistical_outliers(outlier_details):
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

    metrics = [
        "Width",
        "Height",
        "Aspect Ratio",
        "Brightness",
        "Contrast",
        "Sharpness",
    ]

    display_metrics = [
        "Width",
        "Height",
        "Aspect\nRatio",
        "Brightness",
        "Contrast",
        "Sharpness",
    ]

    for dataset_name, splits in outlier_details.items():
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

        for ax, split_name in zip(
            axes,
            split_names,
        ):
            data = splits.get(split_name)

            if data is None or data.empty:
                ax.set_title(split_name)
                continue

            percentages = [
                data[f"{metric} Outlier"].mean() * 100
                for metric in metrics
            ]

            bars = ax.bar(
                display_metrics,
                percentages,
                color=split_colors[split_name],
            )

            total_outliers = (
                data["Is Outlier"].sum()
            )

            total_percentage = (
                data["Is Outlier"].mean()
                * 100
            )

            ax.set_title(
                f"{split_name}\n"
                f"Any outlier = "
                f"{total_outliers:,} "
                f"({total_percentage:.2f}%)"
            )

            ax.set_xlabel("Outlier Metric")

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.3,
            )

            max_value = max(percentages) if percentages else 1
            max_value = max(max_value, 1)

            ax.set_ylim(
                0,
                max_value * 1.20,
            )

            for bar, value in zip(
                bars,
                percentages,
            ):
                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height()
                    + max_value * 0.02,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        axes[0].set_ylabel("Outlier Images (%)")

        fig.suptitle(
            f"{dataset_title} - Statistical Outliers Across Splits",
            fontsize=16,
        )

        plt.tight_layout()
        plt.show()