from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_base_dataset(dataset):
    base_dataset = getattr(dataset, "base_dataset", None)
    return base_dataset if base_dataset is not None else dataset


def get_class_counts(dataset):
    targets = getattr(dataset, "targets", None)

    if targets is None:
        raise AttributeError("Could not determine dataset targets.")
    return Counter(int(target) for target in targets)


def get_class_name(dataset, class_index, class_name_mapping=None):
    base_dataset = get_base_dataset(dataset)
    class_index = int(class_index)

    classes = getattr(base_dataset, "classes", None)
    class_to_name = getattr(base_dataset, "class_to_name", None)
    idx_to_words = getattr(base_dataset, "idx_to_words", None)

    class_id = None

    if classes is not None and class_index < len(classes):
        class_id = classes[class_index]

    # Dataset-provided mapping:
    # Full ImageNet and ImageNet-21K-P
    if class_to_name is not None and class_id in class_to_name:
        return str(class_to_name[class_id])

    # Optional external mapping
    if class_name_mapping is not None:
        if class_id in class_name_mapping:
            return str(class_name_mapping[class_id])

        if class_index in class_name_mapping:
            return str(class_name_mapping[class_index])

        if str(class_index) in class_name_mapping:
            return str(class_name_mapping[str(class_index)])

    # Tiny ImageNet mapping
    if idx_to_words is not None:
        if isinstance(idx_to_words, dict):
            if class_index in idx_to_words:
                name = idx_to_words[class_index]

            elif str(class_index) in idx_to_words:
                name = idx_to_words[str(class_index)]

            elif class_id in idx_to_words:
                name = idx_to_words[class_id]

            else:
                name = None

        elif isinstance(idx_to_words, (list, tuple)) and class_index < len(idx_to_words):
            name = idx_to_words[class_index]

        else:
            name = None

        if name is not None:
            if isinstance(name, (list, tuple)):
                return ", ".join(map(str, name))

            return str(name)

    if class_id is not None:
        return str(class_id)
    return str(class_index)


def class_distribution_analysis(dataset_list, class_name_mappings=None):
    summaries = []
    distributions = {}

    class_name_mappings = class_name_mappings or {}

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        dataset_distributions = {}
        mapping = class_name_mappings.get(dataset_name)

        splits = [
            ("Train", train_dataset, "Labelled"),
            ("Validation", val_dataset, "Labelled"),
            ("Test", test_dataset, "Unlabelled" if dataset_name == "tiny_imagenet" else "Labelled"),
        ]

        for split_name, dataset, label_status in splits:
            if label_status == "Unlabelled":
                summaries.append({
                    "Dataset": dataset_name,
                    "Split": split_name,
                    "Label Status": label_status,
                    "Total Images": len(dataset),
                    "Classes": "N/A",
                    "Min Images/Class": "N/A",
                    "Max Images/Class": "N/A",
                    "Mean Images/Class": "N/A",
                    "Median Images/Class": "N/A",
                    "Std Images/Class": "N/A",
                    "Imbalance Ratio": "N/A",
                    "Coefficient of Variation": "N/A",
                    "Distribution Status": "N/A",
                })

                dataset_distributions[split_name] = None
                continue

            counts = get_class_counts(dataset)
            values = np.array(list(counts.values()))

            min_count = values.min()
            max_count = values.max()
            mean_count = values.mean()
            median_count = np.median(values)
            std_count = values.std()

            imbalance_ratio = max_count / min_count if min_count > 0 else np.inf
            coefficient_of_variation = std_count / mean_count if mean_count > 0 else 0

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Label Status": label_status,
                "Total Images": len(dataset),
                "Classes": len(counts),
                "Min Images/Class": min_count,
                "Max Images/Class": max_count,
                "Mean Images/Class": round(mean_count, 2),
                "Median Images/Class": round(median_count, 2),
                "Std Images/Class": round(std_count, 2),
                "Imbalance Ratio": round(imbalance_ratio, 3),
                "Coefficient of Variation": round(coefficient_of_variation, 3),
                "Distribution Status": "Balanced" if min_count == max_count else "Imbalanced",
            })

            base_dataset = get_base_dataset(dataset)
            classes = getattr(base_dataset, "classes", None)

            class_indices = list(counts.keys())

            class_ids = [
                classes[int(index)]
                if classes is not None and int(index) < len(classes)
                else str(index)
                for index in class_indices
            ]

            class_names = [
                get_class_name(
                    dataset,
                    index,
                    class_name_mapping=mapping,
                )
                for index in class_indices
            ]

            dataset_distributions[split_name] = (
                pd.DataFrame({
                    "Class Index": class_indices,
                    "Class ID": class_ids,
                    "Class Name": class_names,
                    "Image Count": list(counts.values()),
                })
                .sort_values(
                    ["Image Count", "Class Index"],
                    ascending=[False, True],
                )
                .reset_index(drop=True)
            )

        distributions[dataset_name] = dataset_distributions
    return pd.DataFrame(summaries), distributions


def plot_class_distribution(distributions, sample_n=10):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K Winter21",
        "imagenet21k_p": "ImageNet-21K-P Winter21",
    }

    for dataset_name, splits in distributions.items():
        dataset_title = dataset_titles.get(dataset_name, dataset_name)

        train_distribution = splits.get("Train")

        if train_distribution is None:
            continue

        # Select classes ONCE from train split
        if dataset_name == "tiny_imagenet":
            selected_class_ids = train_distribution.head(sample_n)["Class ID"].tolist()
        else:
            selected_class_ids = (
                train_distribution
                .sort_values("Image Count", ascending=False)
                .head(sample_n)["Class ID"]
                .tolist()
            )

        for split_name, distribution in splits.items():
            if distribution is None:
                continue

            plot_data = distribution[
                distribution["Class ID"].isin(selected_class_ids)
            ].copy()

            # Keep exactly the same class order across all splits
            plot_data["Class Order"] = plot_data["Class ID"].map(
                {class_id: i for i, class_id in enumerate(selected_class_ids)}
            )

            plot_data = (
                plot_data
                .sort_values("Class Order")
                .reset_index(drop=True)
            )

            plot_data["Display Name"] = (
                plot_data["Class Name"]
                .astype(str)
                .str.replace("_", " ", regex=False)
                .str.replace(",", "\n", regex=False)
                .str.strip()
            )

            title = (
                f"{dataset_title} - {split_name} Class Distribution "
                f"({sample_n} Selected Classes)"
            )

            plt.figure(figsize=(16, 5))

            bars = plt.bar(plot_data["Display Name"], plot_data["Image Count"])

            plt.title(title)
            plt.xlabel("Class")
            plt.ylabel("Number of Images")
            plt.xticks(rotation=0, ha="center", fontsize=9)
            plt.grid(axis="y", linestyle="--", alpha=0.3)

            max_height = plot_data["Image Count"].max()
            plt.ylim(0, max_height * 1.15)

            for bar in bars:
                height = bar.get_height()

                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max_height * 0.01,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            plt.tight_layout()
            plt.show()
