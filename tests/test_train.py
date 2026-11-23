"""Unit tests for the model training and MLflow module (Part 07).

Verifies feature/target extraction, reproducible splitting, pipeline construction,
model predictions, and saved model persistence.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from mp217 import train


@pytest.fixture
def clean_train_data():
    """Load clean training dataset."""
    return pd.read_csv("data/processed/train_clean.csv")


def test_compute_file_sha256():
    """Verify SHA-256 computation against expected raw training dataset hash."""
    expected_sha = "d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a"
    actual_sha = train.compute_file_sha256("data/raw/train.csv")
    assert actual_sha == expected_sha


def test_prepare_features_and_target(clean_train_data):
    """Verify that feature preparation extracts exactly the 8 operational features and target."""
    X, y = train.prepare_features_and_target(clean_train_data)

    expected_features = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
        "shop_area",
        "block",
    ]

    assert list(X.columns) == expected_features
    assert "record_id" not in X.columns
    assert "date" not in X.columns
    assert len(X) == 1000
    assert len(y) == 1000
    assert y.name == "grain_kg"


def test_reproducible_split(clean_train_data):
    """Verify that split_data produces deterministic train and validation subsets."""
    X, y = train.prepare_features_and_target(clean_train_data)

    X_train1, X_val1, y_train1, y_val1 = train.split_data(X, y, test_size=0.2, random_state=42)
    X_train2, X_val2, y_train2, y_val2 = train.split_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train1) == 800
    assert len(X_val1) == 200
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_val1, X_val2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_val1, y_val2)


def test_build_pipeline_and_prediction(clean_train_data):
    """Verify that pipeline trains and predicts valid continuous values."""
    X, y = train.prepare_features_and_target(clean_train_data)
    X_train, X_val, y_train, y_val = train.split_data(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    pipe = train.build_pipeline(reg)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    assert len(preds) == len(X_val)
    assert isinstance(preds, np.ndarray)
    assert (preds > 0).all()


def test_saved_model_loading_and_prediction():
    """Verify that models/model.joblib exists, loads via joblib, and yields valid predictions."""
    model_path = Path("models/model.joblib")
    assert model_path.is_file(), "Saved model file models/model.joblib does not exist."

    loaded_model = joblib.load(model_path)

    # Test single-record payload matching the 8 operational features
    sample_input = pd.DataFrame([{
        "cards_attached": 1200,
        "working_hours": 7.5,
        "biometric_failures": 25,
        "temp_c": 32.0,
        "stock_kg_morning": 1500.0,
        "days_to_month_end": 10,
        "shop_area": "tribal",
        "block": "Block North",
    }])

    pred = loaded_model.predict(sample_input)
    assert len(pred) == 1
    assert 200.0 < float(pred[0]) < 2000.0
