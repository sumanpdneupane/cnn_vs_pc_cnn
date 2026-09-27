import pandas as pd

def get_num_classes(dataset):
    if hasattr(dataset, "classes"):
        return len(dataset.classes)

    if hasattr(dataset, "idx_to_words"):
        return len(dataset.idx_to_words)

    if hasattr(dataset, "base_dataset") and hasattr(dataset.base_dataset, "classes"):
        return len(dataset.base_dataset.classes)
    raise AttributeError("Could not determine number of classes.")


def dataset_overview(dataset_list):
    summaries = []

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        summaries.append({
            "Dataset": dataset_name,
            "Train Images": len(train_dataset),
            "Validation Images": len(val_dataset),
            "Test Images": len(test_dataset),
            "Total Images": len(train_dataset) + len(val_dataset) + len(test_dataset),
            "Number of Classes": get_num_classes(train_dataset)
        })

    return pd.DataFrame(summaries)