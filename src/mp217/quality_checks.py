"""Quality checks module for MP217 Ration Shop Distribution project.

Implements programmatic detection of the seven data-quality problems outlined in
STUDENT_HANDBOOK.md and PROBLEM.md without modifying the underlying datasets.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def detect_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str = "grain_kg",
) -> Dict[str, Any]:
    """Detect whether any column in train.csv represents target leakage.

    A leaking column is present in training data but omitted from holdout features,
    and exhibits near-perfect correlation with the target variable.

    Args:
        train_df: Training DataFrame.
        holdout_df: Holdout features DataFrame.
        target_column: Name of the regression target column.

    Returns:
        Dict[str, Any]: Detection result with keys 'present', 'column', and 'evidence'.
    """
    train_cols = set(train_df.columns)
    holdout_cols = set(holdout_df.columns)

    # Columns in train that are missing from holdout, excluding the target itself
    candidate_cols = list(train_cols - holdout_cols - {target_column})

    leaking_col = None
    present = False
    evidence = {}

    if candidate_cols:
        for col in candidate_cols:
            if np.issubdtype(train_df[col].dtype, np.number):
                corr = train_df[col].corr(train_df[target_column])
                evidence[col] = {"correlation_with_target": float(corr)}
                if abs(corr) > 0.95:
                    leaking_col = col
                    present = True
            else:
                evidence[col] = {"type": str(train_df[col].dtype)}

    return {
        "present": present,
        "column": leaking_col,
        "candidate_columns": candidate_cols,
        "evidence": evidence,
    }


def detect_unit_mixup(
    train_df: pd.DataFrame,
    column: str = "stock_kg_morning",
    group_column: str = "block",
    expected_factors: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Detect if one group recorded a numeric column in an alternate unit (e.g. grams vs kg).

    Compares central tendencies across groups. If one group is scaled by ~1000x,
    flags the group and the exact metric conversion factor.

    Args:
        train_df: Training DataFrame.
        column: Column to inspect.
        group_column: Categorical grouping column.
        expected_factors: Standard metric unit conversion factors to test against.

    Returns:
        Dict[str, Any]: Detection result with keys 'present', 'column', 'group_value', 'factor', 'evidence'.
    """
    if expected_factors is None:
        expected_factors = [10, 12, 60, 100, 1000]

    group_stats = train_df.groupby(group_column)[column].agg(["count", "mean", "median", "min", "max"])

    present = False
    affected_group = None
    detected_factor = None
    evidence = {}

    medians = group_stats["median"]
    overall_median = medians.median()

    for group_val, row in group_stats.iterrows():
        ratio = row["median"] / overall_median if overall_median > 0 else 0
        evidence[str(group_val)] = {
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "ratio_to_baseline": float(ratio),
        }

        # Check against standard conversion factors with 20% tolerance
        for factor in expected_factors:
            if 0.8 * factor <= ratio <= 1.2 * factor:
                present = True
                affected_group = str(group_val)
                detected_factor = factor
                break

    return {
        "present": present,
        "column": column if present else None,
        "group_value": affected_group,
        "factor": detected_factor,
        "group_statistics": evidence,
    }


def detect_sentinel_values(
    train_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    candidate_sentinels: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Detect if numeric columns contain sentinel values (-1, -999, 9999, etc.) replacing missing data.

    Args:
        train_df: Training DataFrame.
        columns: Numerical columns to test.
        candidate_sentinels: List of sentinel constants.

    Returns:
        Dict[str, Any]: Detection result with keys 'present', 'column', 'value', 'count', 'evidence'.
    """
    if candidate_sentinels is None:
        candidate_sentinels = [-1, -99, -999, -9999, 999, 9999, 99999]

    if columns is None:
        columns = train_df.select_dtypes(include=[np.number]).columns.tolist()

    present = False
    flagged_col = None
    flagged_val = None
    flagged_count = None
    evidence = {}

    for col in columns:
        counts = train_df[col].value_counts()
        for s in candidate_sentinels:
            if s in counts.index:
                cnt = int(counts[s])
                # If sentinel occurs more than 3 times and is anomalous relative to column range
                min_normal = train_df[train_df[col] != s][col].min()
                if s < 0 and min_normal >= 0 and cnt >= 3:
                    present = True
                    flagged_col = col
                    flagged_val = s
                    flagged_count = cnt
                    break
        evidence[col] = {"min": float(train_df[col].min()), "max": float(train_df[col].max())}

    return {
        "present": present,
        "column": flagged_col,
        "value": flagged_val,
        "count": flagged_count,
        "evidence": evidence,
    }


def detect_duplicate_rows(
    train_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Detect exact duplicate records in train.csv.

    Args:
        train_df: Training DataFrame.

    Returns:
        Dict[str, Any]: Detection result with keys 'present', 'count', 'total_rows', 'unique_records'.
    """
    total_rows = len(train_df)
    dup_mask = train_df.duplicated()
    dup_count = int(dup_mask.sum())
    unique_count = total_rows - dup_count

    return {
        "present": dup_count > 0,
        "count": dup_count,
        "total_rows": total_rows,
        "unique_records": unique_count,
    }


def detect_stuck_sensor(
    train_df: pd.DataFrame,
    date_column: str = "date",
    columns: Optional[List[str]] = None,
    min_consecutive_days: int = 14,
) -> Dict[str, Any]:
    """Detect whether any device/sensor reported an identical reading for an extended run of dates.

    Args:
        train_df: Training DataFrame.
        date_column: Date column name.
        columns: Numerical columns to test.
        min_consecutive_days: Minimum days of continuous flat readings to classify as stuck.

    Returns:
        Dict[str, Any]: Detection result with keys 'present', 'column', 'start_date', 'end_date', 'evidence'.
    """
    if columns is None:
        columns = train_df.select_dtypes(include=[np.number]).columns.tolist()

    df_sorted = train_df.sort_values(date_column).copy()
    present = False
    flagged_col = None
    start_date = None
    end_date = None
    evidence = {}

    for col in columns:
        # Check whether grouped by date, variance is zero across a long run of days
        daily_stats = df_sorted.groupby(date_column)[col].agg(["min", "max", "count"])
        # Dates where min == max (all shops had exact same value)
        zero_var_dates = daily_stats[daily_stats["min"] == daily_stats["max"]]

        # Check consecutive dates with zero variance
        if len(zero_var_dates) >= min_consecutive_days:
            # Verify if identical value persisted across consecutive dates
            present = True
            flagged_col = col
            start_date = str(zero_var_dates.index.min())
            end_date = str(zero_var_dates.index.max())
            break

        evidence[col] = {
            "dates_with_identical_readings": int(len(zero_var_dates)),
        }

    return {
        "present": present,
        "column": flagged_col,
        "start_date": start_date,
        "end_date": end_date,
        "evidence": evidence,
    }


def detect_target_typos(
    train_df: pd.DataFrame,
    target_column: str = "grain_kg",
    gap_threshold: float = 3000.0,
) -> Dict[str, Any]:
    """Detect target values typed with an extra zero (10 times too large).

    Identifies an isolated cluster of extreme targets separated by a substantial empty gap.

    Args:
        train_df: Training DataFrame.
        target_column: Regression target column.
        gap_threshold: Minimum gap size in kg between normal and typo distributions.

    Returns:
        Dict[str, Any]: Detection result with keys 'present', 'count', 'record_ids', 'typo_indices'.
    """
    targets = train_df[target_column].sort_values()
    diffs = targets.diff()

    # Find large gaps in sorted targets
    large_gaps = diffs[diffs > gap_threshold]

    present = False
    typo_count = 0
    record_ids = []
    typo_indices = []

    if not large_gaps.empty:
        cutoff = float(targets.loc[large_gaps.index[0]])
        typo_mask = train_df[target_column] >= cutoff
        typo_rows = train_df[typo_mask]

        # Verify that dividing by 10 brings values into the normal range
        normal_max = float(train_df[~typo_mask][target_column].max())
        corrected_values = typo_rows[target_column] / 10.0
        if (corrected_values <= normal_max * 1.5).all():
            present = True
            typo_count = int(len(typo_rows))
            record_ids = typo_rows["record_id"].tolist() if "record_id" in typo_rows.columns else []
            typo_indices = typo_rows.index.tolist()

    return {
        "present": present,
        "count": typo_count if present else None,
        "record_ids": record_ids,
        "typo_indices": typo_indices,
    }


def detect_spelling_variants(
    train_df: pd.DataFrame,
    column: str = "shop_area",
) -> Dict[str, Any]:
    """Detect messy string variants and determine the true canonical category levels.

    Args:
        train_df: Training DataFrame.
        column: Categorical column to inspect.

    Returns:
        Dict[str, Any]: Detection result with keys 'present', 'column', 'true_levels', 'variants_count', 'mapping'.
    """
    raw_levels = train_df[column].dropna().unique().tolist()

    # Normalization mapping rule
    mapping = {}
    for val in raw_levels:
        cleaned = str(val).strip().lower().rstrip(".")
        if "trib" in cleaned or cleaned == "tri":
            canonical = "tribal"
        elif "rur" in cleaned:
            canonical = "rural"
        elif "urb" in cleaned:
            canonical = "urban"
        else:
            canonical = cleaned
        mapping[val] = canonical

    true_categories = sorted(list(set(mapping.values())))
    true_levels = len(true_categories)
    has_variants = len(raw_levels) > true_levels

    return {
        "present": has_variants,
        "column": column if has_variants else None,
        "true_levels": true_levels,
        "total_variants": len(raw_levels),
        "canonical_categories": true_categories,
        "variant_mapping": mapping,
    }


def run_all_quality_checks(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Run programmatic detection for all seven data quality issues.

    Args:
        train_df: Training DataFrame.
        holdout_df: Holdout features DataFrame.

    Returns:
        Dict[str, Any]: Structured results for all 7 issues formatted for findings.yaml.
    """
    leaking = detect_leaking_column(train_df, holdout_df)
    unit = detect_unit_mixup(train_df)
    sentinel = detect_sentinel_values(train_df)
    duplicates = detect_duplicate_rows(train_df)
    sensor = detect_stuck_sensor(train_df)
    typos = detect_target_typos(train_df)
    spelling = detect_spelling_variants(train_df)

    findings_data = {
        "leaking_column": {
            "present": leaking["present"],
            "column": leaking["column"],
        },
        "unit_mixup": {
            "present": unit["present"],
            "column": unit["column"],
            "group_value": unit["group_value"],
            "factor": unit["factor"],
        },
        "sentinel_value": {
            "present": sentinel["present"],
            "column": sentinel["column"],
            "value": sentinel["value"],
            "count": sentinel["count"],
        },
        "duplicate_rows": {
            "present": duplicates["present"],
            "count": duplicates["count"],
        },
        "stuck_sensor": {
            "present": sensor["present"],
            "column": sensor["column"],
            "start_date": sensor["start_date"],
            "end_date": sensor["end_date"],
        },
        "target_typos": {
            "present": typos["present"],
            "count": typos["count"],
        },
        "spelling_variants": {
            "present": spelling["present"],
            "column": spelling["column"],
            "true_levels": spelling["true_levels"],
        },
    }

    return {
        "findings_data": findings_data,
        "detailed_results": {
            "leaking_column": leaking,
            "unit_mixup": unit,
            "sentinel_value": sentinel,
            "duplicate_rows": duplicates,
            "stuck_sensor": sensor,
            "target_typos": typos,
            "spelling_variants": spelling,
        },
    }
