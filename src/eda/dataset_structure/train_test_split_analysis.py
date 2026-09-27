import pandas as pd


def dataset_split_analysis(dataset_list):
    results = []

    for train_dataset, val_dataset, test_dataset, dataset_name in dataset_list:
        train_count = len(train_dataset)
        val_count = len(val_dataset)
        test_count = len(test_dataset)
        total = train_count + val_count + test_count

        results.append({
            "Dataset": dataset_name,
            "Train Images": train_count,
            "Train %": round(train_count / total * 100, 2),
            "Validation Images": val_count,
            "Validation %": round(val_count / total * 100, 2),
            "Test Images": test_count,
            "Test %": round(test_count / total * 100, 2),
            "Total Images": total
        })

    return pd.DataFrame(results)