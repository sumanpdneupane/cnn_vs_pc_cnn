# import random
#
# from torch.utils.data import Dataset
# from collections import defaultdict
#
# class BalancedClassSubset(Dataset):
#     def __init__(self, dataset, selected_classes, images_per_class, seed=42):
#         self.dataset = dataset
#         self.selected_classes = list(selected_classes)
#         self.images_per_class = images_per_class
#
#         self.original_to_new = {
#             original_label: new_label
#             for new_label, original_label in enumerate(self.selected_classes)
#         }
#
#         class_indices = defaultdict(list)
#
#         for index in range(len(dataset)):
#             label = self.get_label(dataset, index)
#             if label in self.original_to_new:
#                 class_indices[label].append(index)
#
#         rng = random.Random(seed)
#         self.samples = []
#
#         for original_label in self.selected_classes:
#             available_indices = class_indices[original_label]
#
#             if len(available_indices) < images_per_class:
#                 raise ValueError(
#                     f"Class {original_label} has only {len(available_indices)} images, "
#                     f"but {images_per_class} were requested."
#                 )
#
#             selected_indices = rng.sample(available_indices, images_per_class)
#             new_label = self.original_to_new[original_label]
#
#             for index in selected_indices:
#                 self.samples.append((index, new_label))
#
#         rng.shuffle(self.samples)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         dataset_index, new_label = self.samples[index]
#         image, _ = self.dataset[dataset_index]
#         return image, new_label
#
#     def get_label(self, dataset, index):
#         _, label = dataset[index]
#         return int(label)


import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Subset


class BalancedClassSubset(Dataset):
    def __init__(self, dataset, selected_classes, images_per_class, seed=42):
        self.dataset = dataset
        self.selected_classes = list(selected_classes)
        self.images_per_class = images_per_class

        self.original_to_new = {
            original_label: new_label
            for new_label, original_label in enumerate(self.selected_classes)
        }

        labels = self._get_labels(dataset)

        class_indices = defaultdict(list)

        for index, label in enumerate(labels):
            label = int(label)

            if label in self.original_to_new:
                class_indices[label].append(index)

        rng = random.Random(seed)
        self.samples = []

        for original_label in self.selected_classes:
            available_indices = class_indices[original_label]

            if len(available_indices) < images_per_class:
                raise ValueError(
                    f"Class {original_label} has only {len(available_indices)} images, "
                    f"but {images_per_class} were requested."
                )

            selected_indices = rng.sample(available_indices, images_per_class)
            new_label = self.original_to_new[original_label]

            for index in selected_indices:
                self.samples.append((index, new_label))

        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        dataset_index, new_label = self.samples[index]
        image, _ = self.dataset[dataset_index]
        return image, new_label

    @staticmethod
    def _get_labels(dataset):
        if hasattr(dataset, "targets"):
            return dataset.targets

        if hasattr(dataset, "labels"):
            return dataset.labels

        if hasattr(dataset, "samples"):
            return [label for _, label in dataset.samples]

        if hasattr(dataset, "imgs"):
            return [label for _, label in dataset.imgs]

        if isinstance(dataset, Subset):
            parent_labels = BalancedClassSubset._get_labels(dataset.dataset)
            return [parent_labels[i] for i in dataset.indices]

        raise AttributeError(
            "Could not find label metadata in dataset. "
            "Expected one of: targets, labels, samples, imgs."
        )