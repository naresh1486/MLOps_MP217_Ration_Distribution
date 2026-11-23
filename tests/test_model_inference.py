"""Additional unit tests for model loading, prediction robustness, and MLflow integration (Part 09).

Verifies:
- Model loading via MLflow registry champion alias
- Batch vs single prediction consistency
- Robustness to unseen categorical levels
- Error handling on missing features
- Realistic physical prediction output range
"""

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def champion_model():
    """Load serialized champion model pipeline."""
    return joblib.load("models/model.joblib")


@pytest.fixture
def valid_sample_payload():
    """Standard valid payload containing all 8 operational features."""
    return pd.DataFrame([{
        "cards_attached": 1150,
        "working_hours": 7.0,
        "biometric_failures": 35,
        "temp_c": 29.5,
        "stock_kg_morning": 1600.0,
        "days_to_month_end": 12,
        "shop_area": "tribal",
        "block": "Block North",
    }])


def test_mlflow_model_loading_via_champion_alias(valid_sample_payload):
    """Verify loading the champion model directly through the MLflow model registry URI."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/mp217_ration_model@champion"

    model = mlflow.sklearn.load_model(model_uri)
    assert model is not None

    preds = model.predict(valid_sample_payload)
    assert len(preds) == 1
    assert 200.0 < float(preds[0]) < 2000.0


def test_model_prediction_batch_invariance(champion_model, valid_sample_payload):
    """Verify that predicting a single row yields identical result to the first row of a batch."""
    batch_payload = pd.concat([valid_sample_payload, valid_sample_payload.copy()], ignore_index=True)

    single_pred = champion_model.predict(valid_sample_payload)
    batch_preds = champion_model.predict(batch_payload)

    assert len(single_pred) == 1
    assert len(batch_preds) == 2
    assert float(single_pred[0]) == pytest.approx(float(batch_preds[0]))
    assert float(batch_preds[0]) == pytest.approx(float(batch_preds[1]))


def test_model_handles_unknown_categorical_values(champion_model):
    """Verify pipeline handles unseen categorical values gracefully without crashing."""
    payload_with_unseen_category = pd.DataFrame([{
        "cards_attached": 1000,
        "working_hours": 6.5,
        "biometric_failures": 40,
        "temp_c": 26.0,
        "stock_kg_morning": 1500.0,
        "days_to_month_end": 15,
        "shop_area": "unknown_category",
        "block": "Unknown_Block",
    }])

    pred = champion_model.predict(payload_with_unseen_category)
    assert len(pred) == 1
    assert isinstance(pred, np.ndarray)
    assert 100.0 < float(pred[0]) < 2500.0


def test_model_missing_feature_raises_error(champion_model):
    """Verify that attempting prediction without a required feature column raises an error."""
    incomplete_payload = pd.DataFrame([{
        "working_hours": 7.0,
        "biometric_failures": 35,
        "temp_c": 29.5,
        "stock_kg_morning": 1600.0,
        "days_to_month_end": 12,
        "shop_area": "tribal",
        "block": "Block North",
        # Missing 'cards_attached'
    }])

    with pytest.raises((KeyError, ValueError)):
        champion_model.predict(incomplete_payload)


def test_model_output_range_and_datatype(champion_model, valid_sample_payload):
    """Verify that predictions are continuous positive floats within reasonable grain limits."""
    pred = champion_model.predict(valid_sample_payload)
    val = float(pred[0])

    assert isinstance(pred, np.ndarray)
    assert np.issubdtype(pred.dtype, np.floating)
    assert 0.0 < val < 3000.0
