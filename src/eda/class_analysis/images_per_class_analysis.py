from collections import Counter

import numpy as np
import pandas as pd


def get_class_counts(dataset):
    targets = getattr(dataset, "targets", None)

    if targets is None:
        raise AttributeError("Could not determine dataset targets.")
    return Counter(int(target) for target in targets)


def images_per_class_analysis(dataset_list):
    summaries = []
    class_counts = {}

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        dataset_counts = {}

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
                })

                dataset_counts[split_name] = None
                continue

            counts = get_class_counts(dataset)
            values = np.array(list(counts.values()))

            summaries.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Label Status": label_status,
                "Total Images": len(dataset),
                "Classes": len(counts),
                "Min Images/Class": values.min(),
                "Max Images/Class": values.max(),
                "Mean Images/Class": round(values.mean(), 2),
                "Median Images/Class": round(np.median(values), 2),
                "Std Images/Class": round(values.std(), 2),
            })

            dataset_counts[split_name] = counts

        class_counts[dataset_name] = dataset_counts

    return pd.DataFrame(summaries), class_counts