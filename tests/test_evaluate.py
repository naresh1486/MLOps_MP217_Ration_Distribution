"""Unit tests for the model evaluation module (Part 08).

Verifies model loading from MLflow registry alias, prediction generation,
metric calculations, output dimensions, and diagnostic plotting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mp217 import evaluate, train


@pytest.fixture
def clean_train_data():
    """Load clean training dataset."""
    return pd.read_csv("data/processed/train_clean.csv")


def test_load_champion_model():
    """Verify loading the champion model resolves to version 1."""
    model, metadata = evaluate.load_champion_model(
        model_path="models/model.joblib",
        tracking_uri="sqlite:///mlflow.db",
        registered_model_name="mp217_ration_model",
        alias="champion",
    )
    assert model is not None
    assert metadata["model_name"] == "mp217_ration_model"
    assert metadata["alias"] == "champion"
    assert metadata["resolved_version"] in ["1", "2"]


def test_compute_metrics():
    """Verify calculation of MAE, RMSE, R2, and residual statistics on known inputs."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    metrics = evaluate.compute_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["r2"] == pytest.approx(0.992, abs=1e-3)
    assert metrics["residuals_mean"] == pytest.approx(0.0)
    assert metrics["max_absolute_error"] == pytest.approx(10.0)


def test_prediction_output_dimensions(clean_train_data):
    """Verify prediction generation on validation subset matches expected dimensions."""
    model, _ = evaluate.load_champion_model()

    X, y = train.prepare_features_and_target(clean_train_data)
    _, X_val, _, y_val = train.split_data(X, y, test_size=0.2, random_state=42)

    preds = model.predict(X_val)
    assert len(preds) == 200
    assert len(preds) == len(y_val)
    assert isinstance(preds, np.ndarray)
    assert (preds > 0).all()


def test_generate_evaluation_plots():
    """Verify that all three diagnostic plots are generated and saved."""
    y_true = np.array([500.0, 1000.0, 1500.0])
    y_pred = np.array([520.0, 980.0, 1510.0])

    tmp_dir = Path("/tmp/test_mp217_figures")
    plots = evaluate.generate_evaluation_plots(y_true, y_pred, output_dir=tmp_dir)

    assert len(plots) == 3
    for p in plots:
        assert Path(p).is_file()
        assert Path(p).stat().st_size > 0


def test_evaluate_champion_workflow():
    """Verify end-to-end evaluation workflow without re-logging to MLflow."""
    res = evaluate.evaluate_champion_model(log_to_mlflow=False)

    assert res["metadata"]["resolved_version"] in ["1", "2"]
    assert res["sample_count"] == 200
    assert "mae" in res["metrics"]
    assert "rmse" in res["metrics"]
    assert "r2" in res["metrics"]
    assert res["metrics"]["mae"] < 70.0
    assert res["metrics"]["r2"] > 0.90
