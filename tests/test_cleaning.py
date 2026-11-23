"""Unit tests for the data cleaning module (Part 05).

Verifies that the cleaning transformations resolve all four confirmed data quality issues,
preserve unaffected values, maintain original DataFrame immutability, and output valid clean datasets.
"""

import pandas as pd
import pytest

from mp217 import cleaning


@pytest.fixture
def raw_train():
    """Load raw training dataset."""
    return pd.read_csv("data/raw/train.csv")


@pytest.fixture
def raw_later():
    """Load raw production monitoring dataset."""
    return pd.read_csv("data/raw/later.csv")


def test_remove_duplicates(raw_train):
    """Verify duplicate removal drops exactly 27 duplicate rows and retains 1000 rows."""
    dedup = cleaning.remove_duplicates(raw_train)
    assert len(raw_train) == 1027
    assert len(dedup) == 1000
    assert dedup.duplicated().sum() == 0


def test_fix_unit_mixup(raw_train):
    """Verify unit mixup converts Block East stock from grams to kg while preserving other blocks."""
    fixed = cleaning.fix_unit_mixup(raw_train)

    # Block East should now be in kilograms (< 5000 kg)
    block_east_raw = raw_train[raw_train["block"] == "Block East"]["stock_kg_morning"]
    block_east_fixed = fixed[fixed["block"] == "Block East"]["stock_kg_morning"]

    assert block_east_raw.max() == 2997000.0
    assert block_east_fixed.max() == 2997.0
    pd.testing.assert_series_equal(block_east_fixed, block_east_raw / 1000.0)

    # Block North and South should remain strictly identical
    other_blocks_raw = raw_train[raw_train["block"] != "Block East"]["stock_kg_morning"]
    other_blocks_fixed = fixed[fixed["block"] != "Block East"]["stock_kg_morning"]
    pd.testing.assert_series_equal(other_blocks_fixed, other_blocks_raw)


def test_fix_target_typos(raw_train):
    """Verify target typos for the 9 confirmed records are divided by 10 while preserving others."""
    fixed = cleaning.fix_target_typos(raw_train)

    typo_ids = cleaning.CONFIRMED_TYPO_RECORD_IDS
    assert len(typo_ids) == 9

    # Verify typo records are divided by 10
    typo_raw = raw_train[raw_train["record_id"].isin(typo_ids)]["grain_kg"]
    typo_fixed = fixed[fixed["record_id"].isin(typo_ids)]["grain_kg"]
    pd.testing.assert_series_equal(typo_fixed, typo_raw / 10.0)

    # Max of typo records after correction should be within normal range
    assert typo_fixed.max() == 1485.8

    # Unaffected records should remain strictly identical
    normal_raw = raw_train[~raw_train["record_id"].isin(typo_ids)]["grain_kg"]
    normal_fixed = fixed[~fixed["record_id"].isin(typo_ids)]["grain_kg"]
    pd.testing.assert_series_equal(normal_fixed, normal_raw)


def test_normalize_spelling(raw_train):
    """Verify that all 17 shop_area variants are mapped to canonical tribal, rural, urban."""
    normalized = cleaning.normalize_spelling(raw_train)

    canonical = {"tribal", "rural", "urban"}
    observed = set(normalized["shop_area"].unique())
    assert observed == canonical
    assert len(observed) == 3


def test_preservation_of_unaffected_values(raw_train):
    """Verify that records not affected by typos or unit mixup maintain identical numerical values."""
    cleaned = cleaning.clean_train_data(raw_train)

    # Find a record in Block North that had no typos and no duplicates
    unaffected_id = "R00001"
    raw_row = raw_train[raw_train["record_id"] == unaffected_id].iloc[0]
    clean_row = cleaned[cleaned["record_id"] == unaffected_id].iloc[0]

    for col in ["cards_attached", "working_hours", "biometric_failures", "temp_c", "stock_kg_morning", "days_to_month_end", "grain_kg"]:
        assert clean_row[col] == pytest.approx(raw_row[col])


def test_clean_train_data_shape_and_validation(raw_train):
    """Verify clean_train_data end-to-end output characteristics."""
    cleaned = cleaning.clean_train_data(raw_train)

    assert cleaned.shape == (1000, 11)
    assert list(cleaned.columns) == list(raw_train.columns)
    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0
    assert cleaned["grain_kg"].max() <= 2000.0
    assert cleaned[cleaned["block"] == "Block East"]["stock_kg_morning"].max() <= 3000.0


def test_clean_later_data(raw_later):
    """Verify clean_later_data preserves already-correct numerical readings."""
    cleaned = cleaning.clean_later_data(raw_later)

    assert cleaned.shape == (300, 11)
    assert cleaned.isnull().sum().sum() == 0
    assert set(cleaned["shop_area"].unique()).issubset({"tribal", "rural", "urban"})

    # Numerical features should be identical to raw
    num_cols = ["cards_attached", "working_hours", "biometric_failures", "temp_c", "stock_kg_morning", "days_to_month_end", "grain_kg"]
    pd.testing.assert_frame_equal(cleaned[num_cols], raw_later[num_cols])


def test_immutability_of_input_dataframe(raw_train):
    """Verify that calling clean_train_data does not mutate the original raw DataFrame."""
    train_copy = raw_train.copy(deep=True)
    _ = cleaning.clean_train_data(raw_train)
    pd.testing.assert_frame_equal(raw_train, train_copy)
