"""Exploratory Data Analysis (EDA) module for MP217 Ration Shop Distribution project.

Contains reusable, beginner-friendly functions for loading datasets, computing descriptive
statistics, analyzing categorical distributions, generating visualizations, and comparing datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set non-interactive backend for headless environments
matplotlib.use("Agg")


def load_raw_datasets(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> Dict[str, pd.DataFrame]:
    """Load all three raw CSV datasets without any modification.

    Args:
        train_path: Path to train.csv.
        later_path: Path to later.csv.
        holdout_path: Path to holdout_features.csv.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping dataset names to loaded DataFrames.
    """
    datasets = {
        "train": pd.read_csv(train_path),
        "later": pd.read_csv(later_path),
        "holdout": pd.read_csv(holdout_path),
    }
    return datasets


def get_dataset_overview(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, object]:
    """Generate high-level metadata and shape overview for a dataset.

    Args:
        df: Input DataFrame.
        dataset_name: Label for the dataset.

    Returns:
        Dict[str, object]: Overview dictionary.
    """
    date_col = "date" if "date" in df.columns else None
    min_date = df[date_col].min() if date_col else None
    max_date = df[date_col].max() if date_col else None

    return {
        "name": dataset_name,
        "rows": len(df),
        "columns": list(df.columns),
        "column_count": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_range": (min_date, max_date),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def get_numerical_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute descriptive statistics for numerical columns.

    Args:
        df: Input DataFrame.
        columns: Optional list of numerical column names.

    Returns:
        pd.DataFrame: Table with count, mean, std, min, median (50%), and max.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = df[columns].describe().T
    stats["median"] = df[columns].median()
    # Reorder columns logically
    cols_order = ["count", "mean", "std", "min", "median", "max"]
    available_cols = [c for c in cols_order if c in stats.columns]
    return stats[available_cols]


def get_categorical_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute frequency counts and percentage share for categorical columns.

    Args:
        df: Input DataFrame.
        columns: Optional list of categorical column names.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of frequency tables per categorical feature.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()
        if "record_id" in columns:
            columns.remove("record_id")
        if "date" in columns:
            columns.remove("date")

    summary = {}
    for col in columns:
        counts = df[col].value_counts(dropna=False)
        pcts = df[col].value_counts(dropna=False, normalize=True) * 100
        summary[col] = pd.DataFrame({"count": counts, "percentage": pcts.round(2)})
    return summary


def analyze_relationships(
    train_df: pd.DataFrame,
    target_col: str = "grain_kg",
) -> Dict[str, object]:
    """Analyze correlations and subgroup relationships against the regression target.

    Args:
        train_df: Raw training DataFrame.
        target_col: Target column name.

    Returns:
        Dict[str, object]: Correlation series and subgroup summary DataFrames.
    """
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = train_df[num_cols].corr()
    target_corrs = corr_matrix[target_col].sort_values(ascending=False)

    block_stats = train_df.groupby("block")[target_col].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    area_stats = train_df.groupby("shop_area")[target_col].agg(
        ["count", "mean", "median", "std"]
    )

    return {
        "correlation_with_target": target_corrs,
        "full_correlation_matrix": corr_matrix,
        "target_by_block": block_stats,
        "target_by_shop_area": area_stats,
    }


def compare_train_later(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare numerical summary statistics between train.csv and later.csv.

    Args:
        train_df: Training DataFrame.
        later_df: Production monitoring DataFrame.

    Returns:
        pd.DataFrame: Side-by-side comparison of mean, std, median, min, max.
    """
    num_cols = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
        "grain_kg",
    ]

    records = []
    for col in num_cols:
        records.append({
            "feature": col,
            "train_mean": round(train_df[col].mean(), 2),
            "later_mean": round(later_df[col].mean(), 2),
            "train_median": round(train_df[col].median(), 2),
            "later_median": round(later_df[col].median(), 2),
            "train_std": round(train_df[col].std(), 2),
            "later_std": round(later_df[col].std(), 2),
            "train_min": round(train_df[col].min(), 2),
            "later_min": round(later_df[col].min(), 2),
            "train_max": round(train_df[col].max(), 2),
            "later_max": round(later_df[col].max(), 2),
        })

    return pd.DataFrame(records).set_index("feature")


def generate_eda_visualizations(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    output_dir: Union[str, Path] = "reports/figures",
) -> List[str]:
    """Generate and save the 6 required EDA figures.

    Args:
        train_df: Raw training DataFrame.
        later_df: Raw production monitoring DataFrame.
        output_dir: Directory where figures will be stored.

    Returns:
        List[str]: List of paths to the generated image files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    num_features = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
    ]

    # 1. Histograms of numerical features (train.csv)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_features):
        axes[i].hist(train_df[col], bins=25, color="#1f77b4", edgecolor="black", alpha=0.75)
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    f1 = out_path / "01_numerical_histograms.png"
    plt.savefig(f1, dpi=150)
    plt.close()
    generated_files.append(str(f1))

    # 2. Boxplots of numerical features (train.csv)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_features):
        axes[i].boxplot(train_df[col], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                        medianprops=dict(color="red", linewidth=2))
        axes[i].set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Value")
    plt.tight_layout()
    f2 = out_path / "02_numerical_boxplots.png"
    plt.savefig(f2, dpi=150)
    plt.close()
    generated_files.append(str(f2))

    # 3. Distribution of the target variable (grain_kg in train.csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(train_df["grain_kg"], bins=40, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax1.set_title("Full Target Distribution (grain_kg)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("grain_kg")
    ax1.set_ylabel("Frequency")
    ax1.axvline(2000, color="red", linestyle="--", label="Typo boundary (> 2000 kg)")
    ax1.legend()

    # Detail view under 2500 kg
    normal_target = train_df[train_df["grain_kg"] <= 2000]["grain_kg"]
    ax2.hist(normal_target, bins=30, color="#98df8a", edgecolor="black", alpha=0.75)
    ax2.set_title("Target Distribution (Zoomed in <= 2000 kg)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("grain_kg")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    f3 = out_path / "03_target_distribution.png"
    plt.savefig(f3, dpi=150)
    plt.close()
    generated_files.append(str(f3))

    # 4. Correlation Heatmap for numerical columns in train.csv
    cols_for_corr = num_features + ["grain_kg"]
    corr = train_df[cols_for_corr].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(cols_for_corr)))
    ax.set_yticks(range(len(cols_for_corr)))
    ax.set_xticklabels(cols_for_corr, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(cols_for_corr, fontsize=9)
    for i in range(len(cols_for_corr)):
        for j in range(len(cols_for_corr)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Correlation Heatmap (train.csv)", pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()
    f4 = out_path / "04_correlation_heatmap.png"
    plt.savefig(f4, dpi=150)
    plt.close()
    generated_files.append(str(f4))

    # 5. Distribution of Categorical Features (block and top shop_area)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    block_counts = train_df["block"].value_counts()
    ax1.bar(block_counts.index, block_counts.values, color="#ff7f0e", edgecolor="black", alpha=0.8)
    ax1.set_title("Frequency of block in train.csv", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count")
    for idx, val in enumerate(block_counts.values):
        ax1.text(idx, val + 10, str(val), ha="center", fontsize=9)

    shop_counts = train_df["shop_area"].value_counts().head(7)
    ax2.barh(shop_counts.index, shop_counts.values, color="#17becf", edgecolor="black", alpha=0.8)
    ax2.set_title("Top Categories/Variants of shop_area (train.csv)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Count")
    for idx, val in enumerate(shop_counts.values):
        ax2.text(val + 5, idx, str(val), va="center", fontsize=9)
    plt.tight_layout()
    f5 = out_path / "05_categorical_distribution.png"
    plt.savefig(f5, dpi=150)
    plt.close()
    generated_files.append(str(f5))

    # 6. Comparison of Numerical Feature Distributions (train.csv vs later.csv)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_features):
        axes[i].hist(train_df[col], bins=25, alpha=0.5, label="train.csv", color="#1f77b4", density=True)
        axes[i].hist(later_df[col], bins=25, alpha=0.5, label="later.csv", color="#d62728", density=True)
        axes[i].set_title(f"train vs later: {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    f6 = out_path / "06_train_vs_later_comparison.png"
    plt.savefig(f6, dpi=150)
    plt.close()
    generated_files.append(str(f6))

    return generated_files
