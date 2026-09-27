import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# Helpers
# ============================================================

def to_numeric(value):
    return pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]


def get_row(dataframe, dataset_name, split_name):
    if dataframe is None or dataframe.empty:
        return None

    rows = dataframe[
        (dataframe["Dataset"] == dataset_name)
        & (dataframe["Split"] == split_name)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_value(row, column):
    if row is None or column not in row:
        return np.nan

    return to_numeric(row[column])


def get_duplicate_percentage(
    duplicate_stats,
    dataset_name,
    split_name,
):
    if duplicate_stats is None or duplicate_stats.empty:
        return np.nan

    rows = duplicate_stats[
        duplicate_stats["Dataset"] == dataset_name
    ]

    if rows.empty:
        return np.nan

    column = f"{split_name} Duplicate (%)"

    if column not in rows.columns:
        return np.nan

    return to_numeric(
        rows.iloc[0][column]
    )


def get_dataset_leakage_information(
    duplicate_stats,
    dataset_name,
):
    if duplicate_stats is None or duplicate_stats.empty:
        return np.nan, "N/A"

    rows = duplicate_stats[
        duplicate_stats["Dataset"] == dataset_name
    ]

    if rows.empty:
        return np.nan, "N/A"

    row = rows.iloc[0]

    leakage_groups = (
        to_numeric(
            row["Unique Leakage Groups"]
        )
        if "Unique Leakage Groups" in row
        else np.nan
    )

    leakage_status = (
        row["Leakage Status"]
        if "Leakage Status" in row
        else "N/A"
    )

    return leakage_groups, leakage_status


# ============================================================
# Resolution Diversity
# ============================================================

def calculate_resolution_diversity(
    dimension_row,
):
    unique_resolutions = get_value(
        dimension_row,
        "Unique Resolutions",
    )

    analyzed_images = get_value(
        dimension_row,
        "Analyzed Images",
    )

    if (
        pd.isna(unique_resolutions)
        or pd.isna(analyzed_images)
        or analyzed_images <= 0
    ):
        return np.nan

    return (
        unique_resolutions
        / analyzed_images
        * 100
    )


# ============================================================
# Main Dataset Difficulty Analysis
# ============================================================

def dataset_difficulty_analysis(
    class_distribution_stats,
    image_dimension_stats,
    intra_class_stats,
    inter_class_stats,
    duplicate_stats=None,
):
    rows = []

    split_names = [
        "Train",
        "Validation",
        "Test",
    ]

    dataset_names = list(
        dict.fromkeys(
            intra_class_stats[
                "Dataset"
            ].tolist()
        )
    )

    for dataset_name in tqdm(
        dataset_names,
        desc="Datasets",
    ):
        dataset_leakage_groups, dataset_leakage_status = (
            get_dataset_leakage_information(
                duplicate_stats,
                dataset_name,
            )
        )

        splits = [
            "Train",
            "Validation",
            "Test",
        ]

        for split_name in tqdm(
            splits,
            desc=f"{dataset_name} splits",
            leave=False,
        ):
            class_row = get_row(
                class_distribution_stats,
                dataset_name,
                split_name,
            )

            dimension_row = get_row(
                image_dimension_stats,
                dataset_name,
                split_name,
            )

            intra_row = get_row(
                intra_class_stats,
                dataset_name,
                split_name,
            )

            inter_row = get_row(
                inter_class_stats,
                dataset_name,
                split_name,
            )

            # ------------------------------------------------
            # General Dataset Properties
            # ------------------------------------------------

            total_images = get_value(
                class_row,
                "Total Images",
            )

            if pd.isna(total_images):
                total_images = get_value(
                    dimension_row,
                    "Total Images",
                )

            classes = get_value(
                class_row,
                "Classes",
            )

            imbalance_ratio = get_value(
                class_row,
                "Imbalance Ratio",
            )

            class_cv = get_value(
                class_row,
                "Coefficient of Variation",
            )

            unique_resolutions = get_value(
                dimension_row,
                "Unique Resolutions",
            )

            resolution_diversity = (
                calculate_resolution_diversity(
                    dimension_row
                )
            )

            duplicate_percentage = (
                get_duplicate_percentage(
                    duplicate_stats,
                    dataset_name,
                    split_name,
                )
            )

            # ------------------------------------------------
            # Label Status
            # ------------------------------------------------

            label_status = (
                intra_row["Label Status"]
                if intra_row is not None
                and "Label Status" in intra_row
                else "N/A"
            )

            # ------------------------------------------------
            # Unlabelled Split
            #
            # Keep image-level properties such as resolution,
            # duplicates and leakage.
            # Class-based difficulty remains N/A.
            # ------------------------------------------------

            if (
                intra_row is None
                or inter_row is None
                or label_status != "Labelled"
            ):
                rows.append({
                    "Dataset": dataset_name,
                    "Split": split_name,
                    "Label Status": label_status,
                    "Total Images": total_images,
                    "Classes": classes,
                    "Imbalance Ratio": imbalance_ratio,
                    "Class CV": class_cv,
                    "Unique Resolutions": unique_resolutions,
                    "Resolution Diversity (%)": (
                        round(
                            resolution_diversity,
                            4,
                        )
                        if pd.notna(
                            resolution_diversity
                        )
                        else np.nan
                    ),
                    "Mean Intra-Class Variation": np.nan,
                    "Mean Inter-Class Distance": np.nan,
                    "Low-Level Structural Difficulty Ratio": np.nan,
                    "Relative Structural Rank": pd.NA,
                    "Duplicate (%)": (
                        round(
                            duplicate_percentage,
                            4,
                        )
                        if pd.notna(
                            duplicate_percentage
                        )
                        else np.nan
                    ),
                    "Dataset Leakage Groups": dataset_leakage_groups,
                    "Dataset Leakage Status": dataset_leakage_status,
                    "Status": "N/A",
                })

                continue

            # ------------------------------------------------
            # Intra-Class Variation
            # ------------------------------------------------

            intra = to_numeric(
                intra_row[
                    "Mean Intra-Class Variation"
                ]
            )

            # ------------------------------------------------
            # Inter-Class Separation
            # ------------------------------------------------

            inter = to_numeric(
                inter_row[
                    "Mean Inter-Class Distance"
                ]
            )

            # ------------------------------------------------
            # Low-Level Structural Difficulty Ratio
            #
            # Intra-Class Variation
            # ---------------------
            # Inter-Class Distance
            #
            # Higher = structurally more difficult
            # under the standardized low-level RGB
            # representation.
            # ------------------------------------------------

            if (
                pd.notna(intra)
                and pd.notna(inter)
                and inter > 0
            ):
                structural_ratio = (
                    intra / inter
                )
            else:
                structural_ratio = np.nan

            rows.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "Label Status": "Labelled",
                "Total Images": total_images,
                "Classes": classes,
                "Imbalance Ratio": imbalance_ratio,
                "Class CV": class_cv,
                "Unique Resolutions": unique_resolutions,
                "Resolution Diversity (%)": (
                    round(
                        resolution_diversity,
                        4,
                    )
                    if pd.notna(
                        resolution_diversity
                    )
                    else np.nan
                ),
                "Mean Intra-Class Variation": round(
                    intra,
                    6,
                ),
                "Mean Inter-Class Distance": round(
                    inter,
                    6,
                ),
                "Low-Level Structural Difficulty Ratio": round(
                    structural_ratio,
                    6,
                ),
                "Relative Structural Rank": pd.NA,
                "Duplicate (%)": (
                    round(
                        duplicate_percentage,
                        4,
                    )
                    if pd.notna(
                        duplicate_percentage
                    )
                    else np.nan
                ),
                "Dataset Leakage Groups": dataset_leakage_groups,
                "Dataset Leakage Status": dataset_leakage_status,
                "Status": "Analyzed",
            })

    difficulty_stats = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Relative Structural Ranking
    #
    # Rank 1 = highest low-level structural difficulty.
    # Ranking is performed independently for each split.
    # --------------------------------------------------------

    difficulty_stats[
        "Relative Structural Rank"
    ] = pd.Series(
        pd.NA,
        index=difficulty_stats.index,
        dtype="Int64",
    )

    for split_name in tqdm(
        split_names,
        desc="Structural ranking",
    ):
        mask = (
            (
                difficulty_stats[
                    "Split"
                ] == split_name
            )
            & (
                difficulty_stats[
                    "Label Status"
                ] == "Labelled"
            )
            & (
                difficulty_stats[
                    "Low-Level Structural Difficulty Ratio"
                ].notna()
            )
        )

        difficulty_stats.loc[
            mask,
            "Relative Structural Rank",
        ] = (
            difficulty_stats.loc[
                mask,
                "Low-Level Structural Difficulty Ratio",
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .astype("Int64")
        )

    return difficulty_stats


# ============================================================
# Training-Set Comparison
# ============================================================

def get_train_difficulty_summary(
    difficulty_stats,
):
    columns = [
        "Dataset",
        "Total Images",
        "Classes",
        "Imbalance Ratio",
        "Class CV",
        "Unique Resolutions",
        "Resolution Diversity (%)",
        "Mean Intra-Class Variation",
        "Mean Inter-Class Distance",
        "Low-Level Structural Difficulty Ratio",
        "Relative Structural Rank",
        "Duplicate (%)",
        "Dataset Leakage Groups",
        "Dataset Leakage Status",
    ]

    train_summary = (
        difficulty_stats[
            (
                difficulty_stats[
                    "Split"
                ] == "Train"
            )
            & (
                difficulty_stats[
                    "Label Status"
                ] == "Labelled"
            )
        ][columns]
        .sort_values(
            "Low-Level Structural Difficulty Ratio",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return train_summary


# ============================================================
# Structural Difficulty Visualization
# ============================================================

def plot_structural_difficulty(
    difficulty_stats,
):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K",
        "imagenet21k_p": "ImageNet-21K-P",
    }

    split_names = [
        "Train",
        "Validation",
        "Test",
    ]

    split_colors = {
        "Train": "tab:green",
        "Validation": "tab:orange",
        "Test": "tab:blue",
    }

    valid_values = difficulty_stats[
        "Low-Level Structural Difficulty Ratio"
    ].dropna()

    max_value = (
        valid_values.max()
        if not valid_values.empty
        else 1
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharey=True,
    )

    for ax, split_name in tqdm(
        zip(
            axes,
            split_names,
        ),
        total=len(split_names),
        desc="Structural difficulty plots",
    ):
        data = difficulty_stats[
            (
                difficulty_stats[
                    "Split"
                ] == split_name
            )
            & (
                difficulty_stats[
                    "Low-Level Structural Difficulty Ratio"
                ].notna()
            )
        ].copy()

        if data.empty:
            ax.set_title(
                f"{split_name}\nN/A"
            )

            ax.text(
                0.5,
                0.5,
                "No labelled class data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            continue

        data["Dataset Label"] = (
            data["Dataset"]
            .map(dataset_titles)
            .fillna(
                data["Dataset"]
            )
        )

        bars = ax.bar(
            data["Dataset Label"],
            data[
                "Low-Level Structural Difficulty Ratio"
            ],
            color=split_colors[
                split_name
            ],
        )

        ax.set_title(
            split_name
        )

        ax.set_xlabel(
            "Dataset"
        )

        ax.set_ylim(
            0,
            max_value * 1.18,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

        for bar, value in zip(
            bars,
            data[
                "Low-Level Structural Difficulty Ratio"
            ],
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value
                + max_value * 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    axes[0].set_ylabel(
        "Low-Level Structural Difficulty Ratio"
    )

    fig.suptitle(
        "Dataset Low-Level Structural Difficulty Comparison Across Splits",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Difficulty Map
# ============================================================

def plot_difficulty_map(
    difficulty_stats,
):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K",
        "imagenet21k_p": "ImageNet-21K-P",
    }

    split_names = [
        "Train",
        "Validation",
        "Test",
    ]

    split_colors = {
        "Train": "tab:green",
        "Validation": "tab:orange",
        "Test": "tab:blue",
    }

    valid = difficulty_stats[
        difficulty_stats[
            "Low-Level Structural Difficulty Ratio"
        ].notna()
    ]

    if valid.empty:
        return

    min_x = valid[
        "Mean Inter-Class Distance"
    ].min()

    max_x = valid[
        "Mean Inter-Class Distance"
    ].max()

    min_y = valid[
        "Mean Intra-Class Variation"
    ].min()

    max_y = valid[
        "Mean Intra-Class Variation"
    ].max()

    x_margin = max(
        (max_x - min_x) * 0.15,
        0.01,
    )

    y_margin = max(
        (max_y - min_y) * 0.15,
        0.005,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharex=True,
        sharey=True,
    )

    for ax, split_name in tqdm(
        zip(
            axes,
            split_names,
        ),
        total=len(split_names),
        desc="Difficulty maps",
    ):
        data = difficulty_stats[
            (
                difficulty_stats[
                    "Split"
                ] == split_name
            )
            & (
                difficulty_stats[
                    "Low-Level Structural Difficulty Ratio"
                ].notna()
            )
        ]

        if data.empty:
            ax.set_title(
                f"{split_name}\nN/A"
            )

            ax.text(
                0.5,
                0.5,
                "No labelled class data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            continue

        for _, row in data.iterrows():
            x = row[
                "Mean Inter-Class Distance"
            ]

            y = row[
                "Mean Intra-Class Variation"
            ]

            label = dataset_titles.get(
                row["Dataset"],
                row["Dataset"],
            )

            ax.scatter(
                x,
                y,
                s=100,
                color=split_colors[
                    split_name
                ],
            )

            ax.annotate(
                label,
                (
                    x,
                    y,
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_title(
            split_name
        )

        ax.set_xlabel(
            "Mean Inter-Class Distance"
        )

        ax.set_xlim(
            min_x - x_margin,
            max_x + x_margin,
        )

        ax.set_ylim(
            min_y - y_margin,
            max_y + y_margin,
        )

        ax.grid(
            linestyle="--",
            alpha=0.3,
        )

    axes[0].set_ylabel(
        "Mean Intra-Class Variation"
    )

    fig.suptitle(
        "Dataset Difficulty Map: "
        "Intra-Class Variation vs Inter-Class Separation",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Class Imbalance Visualization
# ============================================================

def plot_class_imbalance_difficulty(
    difficulty_stats,
):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K",
        "imagenet21k_p": "ImageNet-21K-P",
    }

    split_names = [
        "Train",
        "Validation",
        "Test",
    ]

    split_colors = {
        "Train": "tab:green",
        "Validation": "tab:orange",
        "Test": "tab:blue",
    }

    valid_values = difficulty_stats[
        "Class CV"
    ].dropna()

    max_value = (
        valid_values.max()
        if not valid_values.empty
        else 1
    )

    max_value = max(
        max_value,
        0.1,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharey=True,
    )

    for ax, split_name in tqdm(
        zip(
            axes,
            split_names,
        ),
        total=len(split_names),
        desc="Class imbalance plots",
    ):
        data = difficulty_stats[
            (
                difficulty_stats[
                    "Split"
                ] == split_name
            )
            & (
                difficulty_stats[
                    "Class CV"
                ].notna()
            )
        ].copy()

        if data.empty:
            ax.set_title(
                f"{split_name}\nN/A"
            )

            ax.text(
                0.5,
                0.5,
                "No labelled class data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            continue

        data["Dataset Label"] = (
            data["Dataset"]
            .map(dataset_titles)
            .fillna(
                data["Dataset"]
            )
        )

        bars = ax.bar(
            data["Dataset Label"],
            data["Class CV"],
            color=split_colors[
                split_name
            ],
        )

        ax.set_title(
            split_name
        )

        ax.set_xlabel(
            "Dataset"
        )

        ax.set_ylim(
            0,
            max_value * 1.18,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

        for bar, value in zip(
            bars,
            data["Class CV"],
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value
                + max_value * 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    axes[0].set_ylabel(
        "Class Distribution Coefficient of Variation"
    )

    fig.suptitle(
        "Dataset Class Imbalance Comparison Across Splits",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Resolution Diversity Visualization
# ============================================================

def plot_resolution_diversity(
    difficulty_stats,
):
    dataset_titles = {
        "tiny_imagenet": "Tiny ImageNet",
        "imagenet21k": "ImageNet-21K",
        "imagenet21k_p": "ImageNet-21K-P",
    }

    split_names = [
        "Train",
        "Validation",
        "Test",
    ]

    split_colors = {
        "Train": "tab:green",
        "Validation": "tab:orange",
        "Test": "tab:blue",
    }

    valid_values = difficulty_stats[
        "Resolution Diversity (%)"
    ].dropna()

    max_value = (
        valid_values.max()
        if not valid_values.empty
        else 1
    )

    max_value = max(
        max_value,
        1,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharey=True,
    )

    for ax, split_name in tqdm(
        zip(
            axes,
            split_names,
        ),
        total=len(split_names),
        desc="Resolution diversity plots",
    ):
        data = difficulty_stats[
            (
                difficulty_stats[
                    "Split"
                ] == split_name
            )
            & (
                difficulty_stats[
                    "Resolution Diversity (%)"
                ].notna()
            )
        ].copy()

        if data.empty:
            ax.set_title(
                f"{split_name}\nN/A"
            )
            continue

        data["Dataset Label"] = (
            data["Dataset"]
            .map(dataset_titles)
            .fillna(
                data["Dataset"]
            )
        )

        bars = ax.bar(
            data["Dataset Label"],
            data[
                "Resolution Diversity (%)"
            ],
            color=split_colors[
                split_name
            ],
        )

        ax.set_title(
            split_name
        )

        ax.set_xlabel(
            "Dataset"
        )

        ax.set_ylim(
            0,
            max_value * 1.18,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

        for bar, value in zip(
            bars,
            data[
                "Resolution Diversity (%)"
            ],
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value
                + max_value * 0.02,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    axes[0].set_ylabel(
        "Unique Resolutions / Analyzed Images (%)"
    )

    fig.suptitle(
        "Dataset Resolution Diversity Across Splits",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()