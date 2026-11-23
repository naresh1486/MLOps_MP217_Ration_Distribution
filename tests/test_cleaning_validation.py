"""Additional unit tests for data cleaning validation guardrails (Part 09).

Tests negative cases and error handling for validate_cleaned_data:
- Detection of residual duplicates
- Detection of missing values
- Detection of unmapped categorical levels
- Detection of uncorrected unit mixups in Block East
- Detection of uncorrected target typos
"""

import pandas as pd
import pytest

from mp217 import cleaning


@pytest.fixture
def clean_train_sample():
    """Load sample from clean training dataset."""
    df = pd.read_csv("data/processed/train_clean.csv").head(50)
    return df


def test_validation_catches_duplicate_rows(clean_train_sample):
    """Verify that validate_cleaned_data raises ValueError if duplicates are present."""
    corrupted_df = pd.concat([clean_train_sample, clean_train_sample.iloc[[0]]], ignore_index=True)
    assert corrupted_df.duplicated().sum() == 1

    with pytest.raises(ValueError, match="no_duplicates"):
        cleaning.validate_cleaned_data(corrupted_df, is_train=True)


def test_validation_catches_missing_values(clean_train_sample):
    """Verify that validate_cleaned_data raises ValueError if NaN values are introduced."""
    corrupted_df = clean_train_sample.copy()
    corrupted_df.loc[0, "working_hours"] = None

    with pytest.raises(ValueError, match="no_missing_values"):
        cleaning.validate_cleaned_data(corrupted_df, is_train=True)


def test_validation_catches_invalid_shop_area(clean_train_sample):
    """Verify that validate_cleaned_data raises ValueError on invalid/unmapped categories."""
    corrupted_df = clean_train_sample.copy()
    corrupted_df.loc[0, "shop_area"] = "coastal"

    with pytest.raises(ValueError, match="canonical_shop_areas"):
        cleaning.validate_cleaned_data(corrupted_df, is_train=True)


def test_validation_catches_unconverted_block_east(clean_train_sample):
    """Verify that validate_cleaned_data raises ValueError if Block East stock exceeds 5000 kg."""
    corrupted_df = clean_train_sample.copy()
    corrupted_df.loc[corrupted_df["block"] == "Block East", "stock_kg_morning"] = 1400000.0

    with pytest.raises(ValueError, match="block_east_in_kg"):
        cleaning.validate_cleaned_data(corrupted_df, is_train=True)


def test_validation_catches_unresolved_target_typos(clean_train_sample):
    """Verify that validate_cleaned_data raises ValueError if target typos (> 2000 kg) remain."""
    corrupted_df = clean_train_sample.copy()
    corrupted_df.loc[0, "grain_kg"] = 12000.0

    with pytest.raises(ValueError, match="target_typos_resolved"):
        cleaning.validate_cleaned_data(corrupted_df, is_train=True)
