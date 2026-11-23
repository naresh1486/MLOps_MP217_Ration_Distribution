"""Unit tests for configuration loading and generic data utilities (Part 09).

Verifies:
- Accurate parsing of params.yaml
- Graceful error handling for missing and non-file paths
- Correct row sampling with nrows in load_data
"""

from pathlib import Path
import pytest

from mp217 import config, data_utils


def test_config_reads_expected_structure():
    """Verify load_config accurately parses all expected sections from params.yaml."""
    cfg = config.load_config("params.yaml")

    assert "paths" in cfg
    assert "target_column" in cfg
    assert "split" in cfg
    assert "model" in cfg
    assert "mlflow" in cfg

    assert cfg["target_column"] == "grain_kg"
    assert cfg["split"]["random_seed"] == 42
    assert cfg["split"]["test_size"] == 0.2
    assert cfg["mlflow"]["tracking_uri"] == "sqlite:///mlflow.db"


def test_config_raises_on_missing_file():
    """Verify load_config raises FileNotFoundError with helpful message for missing file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config.load_config("missing_params.yaml")


def test_data_utils_load_data_sample():
    """Verify data_utils.load_data correctly respects nrows parameter."""
    df_sample = data_utils.load_data("data/raw/train.csv", nrows=15)
    assert len(df_sample) == 15
    assert df_sample.shape[1] == 11


def test_data_utils_raises_on_directory_path():
    """Verify data_utils.load_data raises ValueError if path points to a directory."""
    with pytest.raises(ValueError, match="Path does not point to a valid file"):
        data_utils.load_data("data/raw")
