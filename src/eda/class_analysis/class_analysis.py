import pandas as pd


def get_base_dataset(dataset):
    base_dataset = getattr(dataset, "base_dataset", None)
    return base_dataset if base_dataset is not None else dataset


def class_synset_analysis(dataset_list):
    summaries = []
    mappings = {}

    for train_dataset, _, _, dataset_name in dataset_list:
        base_dataset = get_base_dataset(train_dataset)

        classes = getattr(base_dataset, "classes", None)
        idx_to_words = getattr(base_dataset, "idx_to_words", None)
        class_to_name = getattr(base_dataset, "class_to_name", None)

        if classes is not None:
            num_classes = len(classes)
        elif idx_to_words is not None:
            num_classes = len(idx_to_words)
        else:
            raise AttributeError(
                f"Could not determine classes for {dataset_name}."
            )

        summaries.append({
            "Dataset": dataset_name,
            "Number of Classes": num_classes
        })

        rows = []

        if classes is not None:
            for index, class_id in enumerate(classes):
                class_name = None

                # Full ImageNet + ImageNet-21K-P
                if class_to_name is not None:
                    class_name = class_to_name.get(class_id)

                # Tiny ImageNet
                if class_name is None and idx_to_words is not None:
                    if isinstance(idx_to_words, dict):
                        if index in idx_to_words:
                            class_name = idx_to_words[index]
                        elif str(index) in idx_to_words:
                            class_name = idx_to_words[str(index)]
                        elif class_id in idx_to_words:
                            class_name = idx_to_words[class_id]

                    elif isinstance(idx_to_words, (list, tuple)) and index < len(idx_to_words):
                        class_name = idx_to_words[index]

                if class_name is None:
                    class_name = class_id

                rows.append({
                    "Class Index": index,
                    "Class / Synset ID": class_id,
                    "Class Name": class_name
                })

        mappings[dataset_name] = pd.DataFrame(rows)

    return pd.DataFrame(summaries), mappings