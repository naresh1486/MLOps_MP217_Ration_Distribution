"""Unit tests for the quality_checks module (Part 04).

Verifies programmatic detection of all seven data-quality problems on both
actual dataset files and synthetic test cases without mutating input data.
"""

import copy
import pandas as pd
import pytest

from mp217 import quality_checks


@pytest.fixture
def actual_datasets():
    """Load actual datasets for verification."""
    train_df = pd.read_csv("data/raw/train.csv")
    holdout_df = pd.read_csv("data/raw/holdout_features.csv")
    return train_df, holdout_df


def test_detect_leaking_column(actual_datasets):
    """Test leaking column detection on actual data and synthetic scenario."""
    train_df, holdout_df = actual_datasets

    # Actual dataset test: Should be absent
    result = quality_checks.detect_leaking_column(train_df, holdout_df)
    assert result["present"] is False
    assert result["column"] is None

    # Synthetic test with artificial leakage column
    syn_train = train_df.copy()
    syn_train["secret_leak"] = syn_train["grain_kg"] * 1.0001
    syn_result = quality_checks.detect_leaking_column(syn_train, holdout_df)
    assert syn_result["present"] is True
    assert syn_result["column"] == "secret_leak"


def test_detect_unit_mixup(actual_datasets):
    """Test unit mixup detection on stock_kg_morning in Block East."""
    train_df, _ = actual_datasets

    # Actual dataset test
    result = quality_checks.detect_unit_mixup(train_df)
    assert result["present"] is True
    assert result["column"] == "stock_kg_morning"
    assert result["group_value"] == "Block East"
    assert result["factor"] == 1000

    # Synthetic test with uniform units
    syn_train = train_df.copy()
    syn_train.loc[syn_train["block"] == "Block East", "stock_kg_morning"] /= 1000.0
    syn_result = quality_checks.detect_unit_mixup(syn_train)
    assert syn_result["present"] is False


def test_detect_sentinel_values(actual_datasets):
    """Test sentinel value detection."""
    train_df, _ = actual_datasets

    # Actual dataset test: Should be absent
    result = quality_checks.detect_sentinel_values(train_df)
    assert result["present"] is False
    assert result["column"] is None
    assert result["value"] is None

    # Synthetic test with -999 sentinel in working_hours
    syn_train = train_df.copy()
    syn_train.loc[:10, "working_hours"] = -999.0
    syn_result = quality_checks.detect_sentinel_values(syn_train)
    assert syn_result["present"] is True
    assert syn_result["column"] == "working_hours"
    assert syn_result["value"] == -999.0


def test_detect_duplicate_rows(actual_datasets):
    """Test duplicate record detection count."""
    train_df, _ = actual_datasets

    result = quality_checks.detect_duplicate_rows(train_df)
    assert result["present"] is True
    assert result["count"] == 27
    assert result["total_rows"] == 1027
    assert result["unique_records"] == 1000

    # Synthetic test with unique data
    syn_train = train_df.drop_duplicates()
    syn_result = quality_checks.detect_duplicate_rows(syn_train)
    assert syn_result["present"] is False
    assert syn_result["count"] == 0


def test_detect_stuck_sensor(actual_datasets):
    """Test stuck sensor detection."""
    train_df, _ = actual_datasets

    # Actual dataset test: Should be absent
    result = quality_checks.detect_stuck_sensor(train_df)
    assert result["present"] is False
    assert result["column"] is None

    # Synthetic test with flat sensor across consecutive dates
    syn_train = train_df.copy()
    dates = syn_train["date"].unique()[:20]
    syn_train.loc[syn_train["date"].isin(dates), "temp_c"] = 25.0
    syn_result = quality_checks.detect_stuck_sensor(syn_train, min_consecutive_days=14)
    assert syn_result["present"] is True
    assert syn_result["column"] == "temp_c"


def test_detect_target_typos(actual_datasets):
    """Test detection of 10x target typos."""
    train_df, _ = actual_datasets

    result = quality_checks.detect_target_typos(train_df)
    assert result["present"] is True
    assert result["count"] == 9
    assert len(result["record_ids"]) == 9
    expected_ids = {"R00019", "R00202", "R00207", "R00212", "R00268", "R00462", "R00632", "R00651", "R00731"}
    assert set(result["record_ids"]) == expected_ids

    # Synthetic test with clean targets
    syn_train = train_df.copy()
    syn_train.loc[syn_train["grain_kg"] > 2000, "grain_kg"] /= 10.0
    syn_result = quality_checks.detect_target_typos(syn_train)
    assert syn_result["present"] is False


def test_detect_spelling_variants(actual_datasets):
    """Test spelling variant identification and mapping."""
    train_df, _ = actual_datasets

    result = quality_checks.detect_spelling_variants(train_df)
    assert result["present"] is True
    assert result["column"] == "shop_area"
    assert result["true_levels"] == 3
    assert result["total_variants"] == 17
    assert set(result["canonical_categories"]) == {"tribal", "rural", "urban"}

    # Verify all variants are covered
    for raw_val, canonical in result["variant_mapping"].items():
        assert canonical in {"tribal", "rural", "urban"}


def test_quality_checks_do_not_modify_inputs(actual_datasets):
    """Verify that quality check functions do NOT mutate input DataFrames."""
    train_df, holdout_df = actual_datasets

    train_copy = train_df.copy(deep=True)
    holdout_copy = holdout_df.copy(deep=True)

    _ = quality_checks.run_all_quality_checks(train_df, holdout_df)

    pd.testing.assert_frame_equal(train_df, train_copy)
    pd.testing.assert_frame_equal(holdout_df, holdout_copy)
