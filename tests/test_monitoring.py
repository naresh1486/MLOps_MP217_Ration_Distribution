"""Automated tests for drift monitoring, PSI, and model retraining (Part 14).

Verifies:
- PSI computation returns 0 for identical distributions and detects distribution shift
- Date-based 70/30 chronological splitting preserves ordering and partitioning
- Group-level MAE calculation and worst-served group identification
- Promotion rule logic (5% improvement threshold)
- MLflow model registry champion alias resolution to Version 2
"""

import numpy as np
import pandas as pd
import pytest
import mlflow
from mlflow.tracking import MlflowClient

from mp217.monitoring import (
    calculate_psi,
    compute_feature_psis,
    evaluate_mae_by_group,
    evaluate_promotion_rule,
    split_later_by_date,
)


def test_psi_identical_distributions():
    """Identical distributions should have a PSI of approximately zero."""
    data = np.random.normal(loc=100, scale=15, size=1000)
    psi = calculate_psi(data, data, num_bins=10)
    assert psi >= 0.0
    assert psi < 0.01


def test_psi_detects_significant_shift():
    """A substantial mean shift should trigger a large PSI (> 0.25)."""
    baseline = np.random.normal(loc=40, scale=10, size=1000)
    shifted = np.random.normal(loc=80, scale=10, size=500)
    psi = calculate_psi(baseline, shifted, num_bins=10)
    assert psi > 0.25


def test_split_later_by_date_chronology():
    """Chronological split should preserve date ordering without leakage."""
    dates = pd.date_range(start="2026-07-01", periods=100, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "val": range(100),
    })
    # Shuffle dataframe to test sorting
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_slice, check_slice = split_later_by_date(df_shuffled, date_col="date", train_ratio=0.7)

    assert len(train_slice) == 70
    assert len(check_slice) == 30
    assert train_slice["date"].max() <= check_slice["date"].min()


def test_evaluate_mae_by_group():
    """Group MAE should correctly compute error per category and identify worst-served group."""
    class MockModel:
        def predict(self, X):
            return np.array([100.0] * len(X))

    df = pd.DataFrame({
        "feat1": [1, 2, 3, 4],
        "block": ["Block North", "Block North", "Block South", "Block South"],
        "grain_kg": [105.0, 95.0, 150.0, 170.0],  # Error North: 5, Error South: 60
    })

    group_maes, worst_group = evaluate_mae_by_group(
        MockModel(), df, features=["feat1"], target="grain_kg", group_col="block"
    )

    assert group_maes["Block North"] == 5.0
    assert group_maes["Block South"] == 60.0
    assert worst_group == "Block South"


def test_promotion_rule_threshold():
    """Promotion rule should promote when improvement >= 5.0%, and keep otherwise."""
    # 10% improvement -> promote
    decision_promote, pct_promote = evaluate_promotion_rule(mae_v1=200.0, mae_v2=180.0, required_improvement_pct=5.0)
    assert decision_promote == "promote"
    assert pct_promote == 10.0

    # 3% improvement -> keep
    decision_keep, pct_keep = evaluate_promotion_rule(mae_v1=200.0, mae_v2=194.0, required_improvement_pct=5.0)
    assert decision_keep == "keep"
    assert pct_keep == 3.0

    # Exact boundary 5% -> promote
    decision_boundary, _ = evaluate_promotion_rule(mae_v1=100.0, mae_v2=95.0, required_improvement_pct=5.0)
    assert decision_boundary == "promote"


def test_mlflow_champion_alias_is_version_2():
    """Verify that in the MLflow model registry, the champion alias points to Version 2."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    model = client.get_registered_model("mp217_ration_model")

    assert "champion" in model.aliases
    assert model.aliases["champion"] == 2
