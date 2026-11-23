"""Data cleaning module for MP217 Ration Shop Distribution project.

Implements reusable, deterministic functions to resolve the four confirmed data-quality problems:
1. Exact duplicate rows
2. Unit mix-up in Block East stock
3. 10x target typos in grain_kg
4. String spelling variants in shop_area

All functions operate immutably on copies of input DataFrames without mutating original data.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

# Confirmed typo record IDs from Part 04 detection
CONFIRMED_TYPO_RECORD_IDS = [
    "R00019",
    "R00202",
    "R00207",
    "R00212",
    "R00268",
    "R00462",
    "R00632",
    "R00651",
    "R00731",
]

# Canonical category set for shop_area
CANONICAL_SHOP_AREAS = {"tribal", "rural", "urban"}


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with duplicate rows removed and re-indexed.
    """
    cleaned_df = df.copy(deep=True)
    cleaned_df = cleaned_df.drop_duplicates(keep="first").reset_index(drop=True)
    return cleaned_df


def fix_unit_mixup(
    df: pd.DataFrame,
    column: str = "stock_kg_morning",
    group_col: str = "block",
    target_group: str = "Block East",
    factor: float = 1000.0,
) -> pd.DataFrame:
    """Correct stock_kg_morning recorded in grams for Block East by dividing by factor.

    Leaves records in other development blocks strictly unchanged.

    Args:
        df: Input DataFrame.
        column: Numerical column affected by unit mixup.
        group_col: Categorical column identifying the affected administrative division.
        target_group: Group value where unit was misreported.
        factor: Metric conversion factor (1000.0 for grams to kg).

    Returns:
        pd.DataFrame: A new DataFrame with standardized units in kilograms.
    """
    cleaned_df = df.copy(deep=True)
    mask = cleaned_df[group_col] == target_group
    cleaned_df.loc[mask, column] = cleaned_df.loc[mask, column] / factor
    return cleaned_df


def fix_target_typos(
    df: pd.DataFrame,
    target_col: str = "grain_kg",
    typo_ids: Optional[List[str]] = None,
    factor: float = 10.0,
) -> pd.DataFrame:
    """Correct target values inflated by an extra zero (10x too large) by dividing by factor.

    Identifies the confirmed typo records by their unique record_id (or value threshold if record_id absent).
    Leaves all other target records strictly unchanged.

    Args:
        df: Input DataFrame.
        target_col: Target regression column.
        typo_ids: List of unique record_ids confirmed to contain typing errors.
        factor: Multiplier to divide by (10.0 for extra zero).

    Returns:
        pd.DataFrame: A new DataFrame with corrected target values.
    """
    if typo_ids is None:
        typo_ids = CONFIRMED_TYPO_RECORD_IDS

    cleaned_df = df.copy(deep=True)

    if "record_id" in cleaned_df.columns:
        mask = cleaned_df["record_id"].isin(typo_ids)
    else:
        # Fallback if record_id is omitted: threshold based on distribution void (> 2000 kg)
        mask = cleaned_df[target_col] > 2000.0

    cleaned_df.loc[mask, target_col] = cleaned_df.loc[mask, target_col] / factor
    return cleaned_df


def normalize_spelling(
    df: pd.DataFrame,
    column: str = "shop_area",
) -> pd.DataFrame:
    """Normalize messy text categories into the three canonical levels: tribal, rural, urban.

    Handles leading/trailing whitespaces, casing discrepancies, and abbreviations (e.g. 'tri.', 'urb.', 'rur.').

    Args:
        df: Input DataFrame.
        column: Categorical column to clean.

    Returns:
        pd.DataFrame: A new DataFrame with canonical string categories.
    """
    cleaned_df = df.copy(deep=True)

    def _canonicalize(val: Any) -> str:
        if pd.isna(val):
            return val
        s = str(val).strip().lower().rstrip(".")
        if "trib" in s or s == "tri":
            return "tribal"
        elif "rur" in s:
            return "rural"
        elif "urb" in s:
            return "urban"
        return s

    cleaned_df[column] = cleaned_df[column].apply(_canonicalize)
    return cleaned_df


def validate_cleaned_data(
    df: pd.DataFrame,
    is_train: bool = True,
) -> Dict[str, Any]:
    """Verify that cleaned dataset meets all project quality criteria.

    Checks:
    - 0 duplicate rows remaining.
    - 0 missing values across all columns.
    - All shop_area values are strictly within {'tribal', 'rural', 'urban'}.
    - Block East stock_kg_morning values are normalized to standard kilogram range (< 5000 kg).
    - If training data: grain_kg values are in normal range (< 2500 kg) with 0 records > 2000 kg.
    - All expected columns are preserved.

    Args:
        df: Cleaned DataFrame.
        is_train: True if validating train.csv, False for later.csv.

    Returns:
        Dict[str, Any]: Validation report with boolean pass/fail status per check.

    Raises:
        ValueError: If any validation rule fails.
    """
    checks = {}

    # 1. Duplicate check
    dup_count = int(df.duplicated().sum())
    checks["no_duplicates"] = dup_count == 0

    # 2. Missing values check
    null_count = int(df.isnull().sum().sum())
    checks["no_missing_values"] = null_count == 0

    # 3. Categorical levels check
    observed_areas = set(df["shop_area"].unique())
    checks["canonical_shop_areas"] = observed_areas.issubset(CANONICAL_SHOP_AREAS)

    # 4. Block East stock range check (< 5000 kg)
    block_east_stock = df[df["block"] == "Block East"]["stock_kg_morning"]
    checks["block_east_in_kg"] = bool((block_east_stock < 5000.0).all()) if len(block_east_stock) > 0 else True

    # 5. Target typo check (for train data)
    if is_train and "grain_kg" in df.columns:
        typos_remaining = int((df["grain_kg"] > 2000.0).sum())
        checks["target_typos_resolved"] = typos_remaining == 0
    else:
        checks["target_typos_resolved"] = True

    # Check overall validity
    all_passed = all(checks.values())
    if not all_passed:
        failed_checks = [k for k, v in checks.items() if not v]
        raise ValueError(f"Cleaned dataset validation failed on: {failed_checks}")

    return {
        "is_valid": all_passed,
        "checks": checks,
        "row_count": len(df),
        "column_count": df.shape[1],
        "columns": list(df.columns),
    }


def clean_train_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Execute the complete data cleaning pipeline for train.csv.

    Sequential steps:
    1. Remove exact duplicate rows (1027 -> 1000 rows).
    2. Convert Block East stock_kg_morning from grams to kilograms (/ 1000).
    3. Correct the 9 grain_kg target typos (/ 10).
    4. Normalize shop_area into canonical categories.
    5. Validate all quality constraints.

    Args:
        train_df: Raw train DataFrame.

    Returns:
        pd.DataFrame: Cleaned and validated training DataFrame.
    """
    df = remove_duplicates(train_df)
    df = fix_unit_mixup(df)
    df = fix_target_typos(df)
    df = normalize_spelling(df)
    validate_cleaned_data(df, is_train=True)
    return df


def clean_later_data(later_df: pd.DataFrame) -> pd.DataFrame:
    """Execute cleaning and validation for later.csv.

    Applies categorical normalization and validation while preserving all already-correct numerical readings.

    Args:
        later_df: Raw production monitoring DataFrame.

    Returns:
        pd.DataFrame: Cleaned and validated later DataFrame.
    """
    df = remove_duplicates(later_df)
    df = normalize_spelling(df)
    validate_cleaned_data(df, is_train=False)
    return df
