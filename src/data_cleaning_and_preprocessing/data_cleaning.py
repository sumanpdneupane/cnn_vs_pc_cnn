import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


SPLIT_NAMES = ["Train", "Validation", "Test"]

MANIFEST_COLUMNS = [
    "Dataset",
    "Split",
    "Image Index",
    "Target",
    "Hash",
    "Action",
    "Reason",
]


# ============================================================
# Dataset Helpers
# ============================================================

def get_base_dataset(dataset):
    base_dataset = getattr(dataset, "base_dataset", None)
    return base_dataset if base_dataset is not None else dataset


def is_labelled_split(dataset_name, split_name):
    return not (dataset_name == "tiny_imagenet" and split_name == "Test")


def get_class_name(dataset, target):
    base_dataset = get_base_dataset(dataset)

    classes = getattr(base_dataset, "classes", None)

    if classes is not None and 0 <= int(target) < len(classes):
        name = classes[int(target)]
        return str(name[0] if isinstance(name, (tuple, list)) else name)

    class_to_name = getattr(base_dataset, "class_to_name", None)

    if class_to_name is not None:
        for key in [target, int(target), str(target)]:
            if key in class_to_name:
                return str(class_to_name[key])

    return str(target)


def empty_manifest():
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


# ============================================================
# Split Priority
# ============================================================

def get_split_priority(dataset_name):
    if dataset_name == "tiny_imagenet":
        return ["Validation", "Train", "Test"]

    return ["Test", "Validation", "Train"]


# ============================================================
# Duplicate, Leakage and Label-Conflict Cleaning
# ============================================================

def resolve_duplicates_and_leakage(dataset_name, duplicate_details):
    data = duplicate_details[dataset_name]["All Images"].copy()

    required_columns = {
        "Split",
        "Image Index",
        "Hash",
        "Target",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing columns "
            f"{sorted(missing_columns)}"
        )

    priority = get_split_priority(dataset_name)

    removed_keys = {}
    conflict_rows = []

    grouped = data.groupby("Hash")

    for image_hash, group in tqdm(
        grouped,
        total=data["Hash"].nunique(),
        desc=f"{dataset_name}: cleaning exact-content groups",
    ):
        if len(group) == 1:
            continue

        labelled_targets = []

        for _, row in group.iterrows():
            split_name = row["Split"]

            if not is_labelled_split(dataset_name, split_name):
                continue

            target = row["Target"]

            if pd.notna(target):
                labelled_targets.append(int(target))

        unique_targets = sorted(set(labelled_targets))

        # ----------------------------------------------------
        # Contradictory labels for identical image content
        # ----------------------------------------------------

        if len(unique_targets) > 1:
            for _, row in group.iterrows():
                split_name = row["Split"]
                image_index = int(row["Image Index"])

                removed_keys[
                    (split_name, image_index)
                ] = "Label Conflict"

                conflict_rows.append({
                    "Dataset": dataset_name,
                    "Hash": image_hash,
                    "Split": split_name,
                    "Image Index": image_index,
                    "Target": (
                        int(row["Target"])
                        if pd.notna(row["Target"])
                        else None
                    ),
                    "Labelled Targets": unique_targets,
                })

            continue

        # ----------------------------------------------------
        # Same exact content with consistent label
        # ----------------------------------------------------

        available_splits = set(group["Split"])

        keeper_split = next(
            split_name
            for split_name in priority
            if split_name in available_splits
        )

        keeper_rows = (
            group[group["Split"] == keeper_split]
            .sort_values("Image Index")
        )

        keeper_index = int(
            keeper_rows.iloc[0]["Image Index"]
        )

        for _, row in group.iterrows():
            split_name = row["Split"]
            image_index = int(row["Image Index"])

            if split_name == keeper_split and image_index == keeper_index:
                continue

            if split_name == keeper_split:
                reason = "Within-Split Duplicate"
            else:
                reason = "Cross-Split Leakage"

            removed_keys[
                (split_name, image_index)
            ] = reason

    # ========================================================
    # Build Clean Records and Manifest
    # ========================================================

    clean_records = []
    manifest_rows = []

    for _, row in data.iterrows():
        split_name = row["Split"]
        image_index = int(row["Image Index"])

        labelled = is_labelled_split(
            dataset_name,
            split_name,
        )

        target = (
            int(row["Target"])
            if labelled and pd.notna(row["Target"])
            else None
        )

        record = {
            "source_split": split_name,
            "source_index": image_index,
            "split": split_name,
            "target": target,
            "hash": row["Hash"],
        }

        key = (
            split_name,
            image_index,
        )

        if key in removed_keys:
            manifest_rows.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Image Index": image_index,
                "Target": target,
                "Hash": row["Hash"],
                "Action": "Removed",
                "Reason": removed_keys[key],
            })

            continue

        clean_records.append(record)

    manifest = pd.DataFrame(
        manifest_rows,
        columns=MANIFEST_COLUMNS,
    )

    conflicts = pd.DataFrame(
        conflict_rows,
        columns=[
            "Dataset",
            "Hash",
            "Split",
            "Image Index",
            "Target",
            "Labelled Targets",
        ],
    )

    return clean_records, manifest, conflicts


# ============================================================
# ImageNet-21K Rare-Class Handling and Stratified Splitting
# ============================================================

def rebuild_imagenet21k_splits(
    records,
    train_dataset,
    min_images_per_class=3,
    val_ratio=1 / 12,
    test_ratio=1 / 12,
    seed=42,
):
    class_records = {}

    for record in records:
        target = record["target"]

        if target is None:
            continue

        class_records.setdefault(
            int(target),
            [],
        ).append(record)

    eligible_targets = sorted([
        target
        for target, samples in class_records.items()
        if len(samples) >= min_images_per_class
    ])

    excluded_targets = sorted(
        set(class_records) - set(eligible_targets)
    )

    # ========================================================
    # Rare-Class Manifest
    # ========================================================

    rare_manifest_rows = []

    for target in excluded_targets:
        for record in class_records[target]:
            rare_manifest_rows.append({
                "Dataset": "imagenet21k",
                "Split": record["source_split"],
                "Image Index": int(record["source_index"]),
                "Target": int(target),
                "Hash": record["hash"],
                "Action": "Removed",
                "Reason": "Rare-Class Exclusion",
            })

    rare_manifest = pd.DataFrame(
        rare_manifest_rows,
        columns=MANIFEST_COLUMNS,
    )

    # ========================================================
    # Contiguous Class Mapping
    # ========================================================

    target_map = {
        old_target: new_target
        for new_target, old_target in enumerate(eligible_targets)
    }

    class_mapping = []

    for old_target in eligible_targets:
        class_mapping.append({
            "old_target": int(old_target),
            "new_target": int(target_map[old_target]),
            "class_name": get_class_name(
                train_dataset,
                old_target,
            ),
            "total_images": len(
                class_records[old_target]
            ),
        })

    # ========================================================
    # Stratified Split Construction
    # ========================================================

    rebuilt_records = []

    for old_target in tqdm(
        eligible_targets,
        desc="imagenet21k: rebuilding stratified splits",
    ):
        samples = [
            sample.copy()
            for sample in class_records[old_target]
        ]

        rng = random.Random(
            f"{seed}:imagenet21k:{old_target}"
        )

        rng.shuffle(samples)

        total = len(samples)

        n_val = max(
            1,
            round(total * val_ratio),
        )

        n_test = max(
            1,
            round(total * test_ratio),
        )

        if n_val + n_test >= total:
            n_val = 1
            n_test = 1

        n_train = total - n_val - n_test

        if n_train < 1:
            raise ValueError(
                f"Class {old_target} cannot be split safely."
            )

        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]

        for split_name, split_samples in [
            ("Train", train_samples),
            ("Validation", val_samples),
            ("Test", test_samples),
        ]:
            for sample in split_samples:
                sample["split"] = split_name
                sample["original_target"] = int(old_target)
                sample["target"] = int(target_map[old_target])

                rebuilt_records.append(sample)

    return (
        rebuilt_records,
        class_mapping,
        excluded_targets,
        rare_manifest,
    )


# ============================================================
# Class Mapping
# ============================================================

def build_identity_class_mapping(records, train_dataset):
    targets = sorted({
        int(record["target"])
        for record in records
        if record["target"] is not None
    })

    return [
        {
            "old_target": target,
            "new_target": target,
            "class_name": get_class_name(
                train_dataset,
                target,
            ),
        }
        for target in targets
    ]


# ============================================================
# Clean Dataset View
# ============================================================

class CleanDatasetView(Dataset):
    def __init__(
        self,
        split_datasets,
        records,
        class_mapping=None,
    ):
        self.split_datasets = split_datasets
        self.records = records
        self.base_dataset = None
        self.indices = None
        self.class_mapping = class_mapping or []

        targets = [
            record["target"]
            for record in records
        ]

        self.targets = (
            [int(target) for target in targets]
            if targets
            and all(target is not None for target in targets)
            else None
        )

        if self.class_mapping:
            mapping = sorted(
                self.class_mapping,
                key=lambda item: item["new_target"],
            )

            self.classes = [
                item["class_name"]
                for item in mapping
            ]

            self.class_to_idx = {
                item["class_name"]: int(item["new_target"])
                for item in mapping
            }

            self.class_to_name = {
                int(item["new_target"]): item["class_name"]
                for item in mapping
            }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        source_dataset = self.split_datasets[
            record["source_split"]
        ]

        item = source_dataset[
            int(record["source_index"])
        ]

        target = record["target"]

        if target is None:
            return item

        if isinstance(item, tuple):
            return (
                item[0],
                int(target),
                *item[2:],
            )

        if isinstance(item, list):
            return [
                item[0],
                int(target),
                *item[2:],
            ]

        return item, int(target)


# ============================================================
# Build Clean Dataset Tuple
# ============================================================

def build_clean_dataset_tuple(
    dataset_tuple,
    records,
    class_mapping,
):
    (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_name,
    ) = dataset_tuple

    split_datasets = {
        "Train": train_dataset,
        "Validation": val_dataset,
        "Test": test_dataset,
    }

    train_records = [
        record
        for record in records
        if record["split"] == "Train"
    ]

    val_records = [
        record
        for record in records
        if record["split"] == "Validation"
    ]

    test_records = [
        record
        for record in records
        if record["split"] == "Test"
    ]

    clean_train = CleanDatasetView(
        split_datasets,
        train_records,
        class_mapping,
    )

    clean_val = CleanDatasetView(
        split_datasets,
        val_records,
        class_mapping,
    )

    clean_test = CleanDatasetView(
        split_datasets,
        test_records,
        class_mapping,
    )

    return (
        clean_train,
        clean_val,
        clean_test,
        dataset_name,
    )


# ============================================================
# Duplicate and Leakage Verification
# ============================================================

def verify_clean_records(dataset_name, records):
    split_records = {
        split_name: [
            record
            for record in records
            if record["split"] == split_name
        ]
        for split_name in SPLIT_NAMES
    }

    split_hashes = {
        split_name: [
            record["hash"]
            for record in split_records[split_name]
        ]
        for split_name in SPLIT_NAMES
    }

    duplicate_groups = {}

    for split_name in SPLIT_NAMES:
        counts = Counter(
            split_hashes[split_name]
        )

        duplicate_groups[split_name] = sum(
            count > 1
            for count in counts.values()
        )

    train_hashes = set(
        split_hashes["Train"]
    )

    val_hashes = set(
        split_hashes["Validation"]
    )

    test_hashes = set(
        split_hashes["Test"]
    )

    train_val_leakage = len(
        train_hashes & val_hashes
    )

    train_test_leakage = len(
        train_hashes & test_hashes
    )

    val_test_leakage = len(
        val_hashes & test_hashes
    )

    verification_rows = []

    for split_name in SPLIT_NAMES:
        current_records = split_records[
            split_name
        ]

        labelled = is_labelled_split(
            dataset_name,
            split_name,
        )

        targets = [
            record["target"]
            for record in current_records
            if record["target"] is not None
        ]

        verification_rows.append({
            "Dataset": dataset_name,
            "Split": split_name,
            "Images": len(current_records),
            "Classes": (
                len(set(targets))
                if labelled
                else "N/A"
            ),
            "Duplicate Groups": duplicate_groups[
                split_name
            ],
        })

    verification = pd.DataFrame(
        verification_rows
    )

    passed = (
        all(
            count == 0
            for count in duplicate_groups.values()
        )
        and train_val_leakage == 0
        and train_test_leakage == 0
        and val_test_leakage == 0
    )

    leakage_verification = pd.DataFrame([{
        "Dataset": dataset_name,
        "Train ↔ Validation": train_val_leakage,
        "Train ↔ Test": train_test_leakage,
        "Validation ↔ Test": val_test_leakage,
        "Status": "Passed" if passed else "Issues Remain",
    }])

    return verification, leakage_verification


# ============================================================
# Cleaning Summary
# ============================================================

def build_cleaning_summary(
    original_dataset_tuple,
    clean_dataset_tuple,
):
    (
        original_train,
        original_val,
        original_test,
        dataset_name,
    ) = original_dataset_tuple

    (
        clean_train,
        clean_val,
        clean_test,
        _,
    ) = clean_dataset_tuple

    original_splits = {
        "Train": original_train,
        "Validation": original_val,
        "Test": original_test,
    }

    clean_splits = {
        "Train": clean_train,
        "Validation": clean_val,
        "Test": clean_test,
    }

    rows = []

    for split_name in SPLIT_NAMES:
        original_count = len(
            original_splits[split_name]
        )

        clean_count = len(
            clean_splits[split_name]
        )

        rows.append({
            "Dataset": dataset_name,
            "Split": split_name,
            "Original Images": original_count,
            "Clean Images": clean_count,
            "Net Change": clean_count - original_count,
        })

    return pd.DataFrame(rows)


# ============================================================
# Removal Reason Summary
# ============================================================

def build_removal_reason_summary(manifest):
    if manifest.empty:
        return pd.DataFrame(
            columns=[
                "Dataset",
                "Reason",
                "Removed Images",
            ]
        )

    return (
        manifest
        .groupby(
            ["Dataset", "Reason"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "Removed Images"
            }
        )
    )


# ============================================================
# Dataset Audit Summary
# ============================================================

def build_audit_summary(
    original_dataset_tuple,
    clean_dataset_tuple,
    manifest,
):
    original_total = sum(
        len(dataset)
        for dataset
        in original_dataset_tuple[:3]
    )

    clean_total = sum(
        len(dataset)
        for dataset
        in clean_dataset_tuple[:3]
    )

    total_removed = (
        original_total - clean_total
    )

    manifest_entries = len(manifest)

    return pd.DataFrame([{
        "Dataset": original_dataset_tuple[3],
        "Original Images": original_total,
        "Clean Images": clean_total,
        "Removed Images": total_removed,
        "Manifest Entries": manifest_entries,
        "Audit Status": (
            "Passed"
            if total_removed == manifest_entries
            else "Mismatch"
        ),
    }])


# ============================================================
# Save Helpers
# ============================================================

def save_json(path, data):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def save_clean_files(
    root,
    records,
    manifest,
    conflicts,
    class_mapping,
    excluded_classes,
    verification,
    leakage_verification,
    cleaning_summary,
    removal_reason_summary,
    audit_summary,
):
    root = Path(root)

    clean_split_dir = (
        root
        / "splits"
        / "clean"
    )

    cleaning_dir = (
        root
        / "cleaning"
    )

    clean_split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaning_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, filename in [
        ("Train", "train.json"),
        ("Validation", "validation.json"),
        ("Test", "test.json"),
    ]:
        split_records = [
            record
            for record in records
            if record["split"] == split_name
        ]

        save_json(
            clean_split_dir / filename,
            split_records,
        )

    save_json(
        cleaning_dir / "class_mapping.json",
        class_mapping,
    )

    save_json(
        cleaning_dir / "excluded_classes.json",
        excluded_classes,
    )

    manifest.to_csv(
        cleaning_dir / "cleaning_manifest.csv",
        index=False,
    )

    conflicts.to_csv(
        cleaning_dir / "label_conflicts.csv",
        index=False,
    )

    verification.to_csv(
        cleaning_dir / "verification.csv",
        index=False,
    )

    leakage_verification.to_csv(
        cleaning_dir / "leakage_verification.csv",
        index=False,
    )

    cleaning_summary.to_csv(
        cleaning_dir / "cleaning_summary.csv",
        index=False,
    )

    removal_reason_summary.to_csv(
        cleaning_dir / "removal_reason_summary.csv",
        index=False,
    )

    audit_summary.to_csv(
        cleaning_dir / "audit_summary.csv",
        index=False,
    )


# ============================================================
# Complete Cleaning Pipeline
# ============================================================

def clean_datasets(
    dataset_list,
    duplicate_details,
    dataset_roots,
    imagenet21k_min_images_per_class=3,
    seed=42,
):
    clean_dataset_list = []

    cleaning_summaries = []
    removal_reason_summaries = []
    audit_summaries = []
    verification_summaries = []
    leakage_summaries = []

    manifests = {}
    conflicts = {}
    class_mappings = {}
    excluded_classes = {}

    for dataset_tuple in tqdm(
        dataset_list,
        desc="Cleaning datasets",
    ):
        (
            train_dataset,
            _,
            _,
            dataset_name,
        ) = dataset_tuple

        (
            records,
            manifest,
            conflict_df,
        ) = resolve_duplicates_and_leakage(
            dataset_name,
            duplicate_details,
        )

        excluded_targets = []

        # ====================================================
        # ImageNet-21K Coverage Handling
        # ====================================================

        if dataset_name == "imagenet21k":
            (
                records,
                class_mapping,
                excluded_targets,
                rare_manifest,
            ) = rebuild_imagenet21k_splits(
                records,
                train_dataset,
                min_images_per_class=(
                    imagenet21k_min_images_per_class
                ),
                seed=seed,
            )

            manifest = pd.concat(
                [
                    manifest,
                    rare_manifest,
                ],
                ignore_index=True,
            )

        else:
            class_mapping = (
                build_identity_class_mapping(
                    records,
                    train_dataset,
                )
            )

        # ====================================================
        # Build Clean Dataset
        # ====================================================

        clean_dataset_tuple = (
            build_clean_dataset_tuple(
                dataset_tuple,
                records,
                class_mapping,
            )
        )

        # ====================================================
        # Verification
        # ====================================================

        (
            verification,
            leakage_verification,
        ) = verify_clean_records(
            dataset_name,
            records,
        )

        cleaning_summary = (
            build_cleaning_summary(
                dataset_tuple,
                clean_dataset_tuple,
            )
        )

        removal_reason_summary = (
            build_removal_reason_summary(
                manifest
            )
        )

        audit_summary = (
            build_audit_summary(
                dataset_tuple,
                clean_dataset_tuple,
                manifest,
            )
        )

        # ====================================================
        # Save Results
        # ====================================================

        save_clean_files(
            dataset_roots[dataset_name],
            records,
            manifest,
            conflict_df,
            class_mapping,
            excluded_targets,
            verification,
            leakage_verification,
            cleaning_summary,
            removal_reason_summary,
            audit_summary,
        )

        clean_dataset_list.append(
            clean_dataset_tuple
        )

        cleaning_summaries.append(
            cleaning_summary
        )

        removal_reason_summaries.append(
            removal_reason_summary
        )

        audit_summaries.append(
            audit_summary
        )

        verification_summaries.append(
            verification
        )

        leakage_summaries.append(
            leakage_verification
        )

        manifests[
            dataset_name
        ] = manifest

        conflicts[
            dataset_name
        ] = conflict_df

        class_mappings[
            dataset_name
        ] = class_mapping

        excluded_classes[
            dataset_name
        ] = excluded_targets

    return {
        "datasets": clean_dataset_list,

        "cleaning_summary": pd.concat(
            cleaning_summaries,
            ignore_index=True,
        ),

        "removal_reason_summary": pd.concat(
            removal_reason_summaries,
            ignore_index=True,
        ),

        "audit_summary": pd.concat(
            audit_summaries,
            ignore_index=True,
        ),

        "verification": pd.concat(
            verification_summaries,
            ignore_index=True,
        ),

        "leakage_verification": pd.concat(
            leakage_summaries,
            ignore_index=True,
        ),

        "manifests": manifests,
        "conflicts": conflicts,
        "class_mappings": class_mappings,
        "excluded_classes": excluded_classes,
    }