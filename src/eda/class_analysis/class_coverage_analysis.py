from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_class_set(dataset):
    targets = getattr(dataset, "targets", None)

    if targets is None:
        raise AttributeError("Could not determine dataset targets.")
    return set(int(target) for target in targets)


def class_coverage_analysis(dataset_list):
    summaries = []
    coverage_details = {}

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        train_classes = get_class_set(train_dataset)
        val_classes = get_class_set(val_dataset)

        test_unlabelled = dataset_name == "tiny_imagenet"

        if test_unlabelled:
            test_classes = None
        else:
            test_classes = get_class_set(test_dataset)

        missing_in_val = train_classes - val_classes
        extra_in_val = val_classes - train_classes

        val_coverage = (
            len(train_classes & val_classes) / len(train_classes) * 100
            if train_classes else 0
        )

        if test_classes is not None:
            missing_in_test = train_classes - test_classes
            extra_in_test = test_classes - train_classes

            test_coverage = (
                len(train_classes & test_classes) / len(train_classes) * 100
                if train_classes else 0
            )

            common_classes = train_classes & val_classes & test_classes
            all_split_coverage = len(common_classes) / len(train_classes) * 100

            coverage_status = (
                "Complete"
                if not missing_in_val
                and not missing_in_test
                and not extra_in_val
                and not extra_in_test
                else "Incomplete"
            )

        else:
            missing_in_test = None
            extra_in_test = None
            test_coverage = None
            common_classes = train_classes & val_classes
            all_split_coverage = len(common_classes) / len(train_classes) * 100

            coverage_status = (
                "Complete for Labelled Splits"
                if not missing_in_val and not extra_in_val
                else "Incomplete"
            )

        summaries.append({
            "Dataset": dataset_name,
            "Train Classes": len(train_classes),
            "Validation Classes": len(val_classes),
            "Test Classes": "N/A" if test_classes is None else len(test_classes),
            "Validation Coverage (%)": round(val_coverage, 2),
            "Test Coverage (%)": "N/A" if test_coverage is None else round(test_coverage, 2),
            "All Labelled Splits Coverage (%)": round(all_split_coverage, 2),
            "Missing in Validation": len(missing_in_val),
            "Missing in Test": "N/A" if missing_in_test is None else len(missing_in_test),
            "Extra in Validation": len(extra_in_val),
            "Extra in Test": "N/A" if extra_in_test is None else len(extra_in_test),
            "Coverage Status": coverage_status,
        })

        coverage_details[dataset_name] = {
            "Train Classes": train_classes,
            "Validation Classes": val_classes,
            "Test Classes": test_classes,
            "Missing in Validation": missing_in_val,
            "Missing in Test": missing_in_test,
            "Extra in Validation": extra_in_val,
            "Extra in Test": extra_in_test,
            "Common Classes": common_classes,
        }

    return pd.DataFrame(summaries), coverage_details

def plot_class_coverage(coverage_summary):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K Winter21",
        "imagenet21k_p": "ImageNet-21K-P Winter21",
    }

    plot_data = coverage_summary.copy()

    plot_data["Dataset Name"] = (
        plot_data["Dataset"]
        .map(dataset_titles)
        .fillna(plot_data["Dataset"])
    )

    x = np.arange(len(plot_data))
    width = 0.25

    train_values = plot_data["Train Classes"].astype(float)
    val_values = plot_data["Validation Classes"].astype(float)
    test_values = pd.to_numeric(plot_data["Test Classes"], errors="coerce")

    plt.figure(figsize=(12, 5))

    train_bars = plt.bar(x - width, train_values, width, label="Train")
    val_bars = plt.bar(x, val_values, width, label="Validation")
    test_bars = plt.bar(x + width, test_values.fillna(0), width, label="Test")

    plt.title("Class Coverage Across Dataset Splits")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Classes")

    plt.xticks(x, plot_data["Dataset Name"], rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    for bars in [train_bars, val_bars]:
        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for index, bar in enumerate(test_bars):
        if pd.isna(test_values.iloc[index]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                0,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()