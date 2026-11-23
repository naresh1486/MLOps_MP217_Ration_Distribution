"""Drift monitoring and model retraining utilities for MP217 (Part 14).

Implements:
- Population Stability Index (PSI) computation for numeric features.
- Performance evaluation across demographic/geographic sub-groups (block).
- Date-based chronological 70/30 data partitioning for monitoring datasets.
- Client promotion rule evaluation for champion/challenger comparison.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) between expected (baseline) and actual distributions.

    Args:
        expected: Baseline array/series (e.g. train_clean data).
        actual: New monitoring array/series (e.g. later_clean data).
        num_bins: Number of quantiles/bins to partition (default 10).
        epsilon: Small constant to avoid log(0) or division by zero.

    Returns:
        float: Calculated PSI value.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # 10 deciles/quantiles calculated from the baseline distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Apply epsilon smoothing for zero-count bins
    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    # Normalize after smoothing
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


def compute_feature_psis(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    features: List[str],
    num_bins: int = 10,
) -> Tuple[Dict[str, float], str]:
    """Calculate PSI for multiple features and identify the most drifted feature.

    Args:
        train_df: Cleaned baseline training DataFrame.
        later_df: Cleaned monitoring later DataFrame.
        features: List of numeric feature column names.
        num_bins: Number of bins.

    Returns:
        Tuple[Dict[str, float], str]: Dict mapping feature name to PSI, and the name of the most drifted feature.
    """
    psi_dict: Dict[str, float] = {}
    for col in features:
        psi_dict[col] = round(calculate_psi(train_df[col], later_df[col], num_bins=num_bins), 4)

    most_drifted = max(psi_dict, key=psi_dict.get)
    return psi_dict, most_drifted


def evaluate_mae_by_group(
    model: Any,
    df: pd.DataFrame,
    features: List[str],
    target: str = "grain_kg",
    group_col: str = "block",
) -> Tuple[Dict[str, float], str]:
    """Compute MAE for each distinct group within a column and find the worst-served group.

    Args:
        model: Trained scikit-learn model or pipeline.
        df: DataFrame containing features, target, and group column.
        features: List of feature names used for inference.
        target: Target column name.
        group_col: Column to group by (e.g., 'block').

    Returns:
        Tuple[Dict[str, float], str]: Dict of MAE per group, and the name of the worst-served group.
    """
    group_maes: Dict[str, float] = {}
    for group_val, sub_df in df.groupby(group_col):
        X_sub = sub_df[features]
        y_sub = sub_df[target]
        preds = model.predict(X_sub)
        group_maes[str(group_val)] = round(float(mean_absolute_error(y_sub, preds)), 4)

    worst_group = max(group_maes, key=group_maes.get)
    return group_maes, worst_group


def split_later_by_date(
    later_df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically sort and partition later data into training and evaluation sets.

    Args:
        later_df: Later monitoring DataFrame.
        date_col: Date column name to sort by.
        train_ratio: Fraction allocated to retraining (default 0.7).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (first 70% train set, remaining 30% check set).
    """
    sorted_df = later_df.sort_values(by=date_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_ratio)
    train_slice = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    check_slice = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_slice, check_slice


def evaluate_promotion_rule(
    mae_v1: float,
    mae_v2: float,
    required_improvement_pct: float = 5.0,
) -> Tuple[str, float]:
    """Check whether candidate model v2 satisfies the client's promotion rule.

    Rule: Promote v2 only if its MAE on the check dataset is at least 5% lower than v1's.

    Args:
        mae_v1: Champion v1 model MAE on check dataset.
        mae_v2: Retrained v2 model MAE on check dataset.
        required_improvement_pct: Minimum percentage reduction required (default 5.0).

    Returns:
        Tuple[str, float]: Decision string ('promote' or 'keep') and improvement percentage.
    """
    improvement_pct = ((mae_v1 - mae_v2) / mae_v1) * 100.0
    decision = "promote" if improvement_pct >= required_improvement_pct else "keep"
    return decision, round(improvement_pct, 2)
