"""Model evaluation and performance analysis module for MP217.

Loads the registered v1 champion model, calculates regression metrics (MAE, RMSE, R²),
analyzes residuals and prediction distributions, generates diagnostic visualizations,
and logs evaluation runs to MLflow without altering model aliases.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from mp217 import config as cfg_module
from mp217 import train

# Set headless backend for matplotlib
matplotlib.use("Agg")


def load_champion_model(
    model_path: Union[str, Path] = "models/model.joblib",
    tracking_uri: str = "sqlite:///mlflow.db",
    registered_model_name: str = "mp217_ration_model",
    alias: str = "champion",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the persisted champion model and verify its MLflow registry alias resolution.

    Args:
        model_path: Path to serialized model pipeline.
        tracking_uri: SQLite MLflow tracking URI.
        registered_model_name: Name of model in registry.
        alias: Model alias to verify.

    Returns:
        Tuple[Pipeline, Dict[str, Any]]: Loaded pipeline and alias verification metadata.

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If alias does not resolve to Version 1.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Champion model file not found at: {path}")

    # Verify alias through MlflowClient
    client = MlflowClient(tracking_uri=tracking_uri)
    alias_info = client.get_model_version_by_alias(registered_model_name, alias)

    if str(alias_info.version) != "1":
        raise ValueError(
            f"Expected alias '{alias}' to resolve to version 1, but got version {alias_info.version}"
        )

    pipeline = joblib.load(path)

    metadata = {
        "model_name": registered_model_name,
        "alias": alias,
        "resolved_version": str(alias_info.version),
        "source_run_id": alias_info.run_id,
        "model_file": str(path),
    }

    return pipeline, metadata


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """Compute regression metrics and residual statistics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dict[str, float]: Calculated metrics: MAE, RMSE, R2, residual mean/std, max abs error.
    """
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()

    residuals = y_true_arr - y_pred_arr
    abs_errors = np.abs(residuals)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "residuals_mean": float(np.mean(residuals)),
        "residuals_std": float(np.std(residuals)),
        "max_absolute_error": float(np.max(abs_errors)),
    }


def generate_evaluation_plots(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    output_dir: Union[str, Path] = "reports/figures",
) -> List[str]:
    """Generate diagnostic evaluation figures: actual vs predicted, residual distribution, prediction error plot.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        output_dir: Output directory for plots.

    Returns:
        List[str]: File paths of generated plots.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    residuals = y_true_arr - y_pred_arr

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Actual vs Predicted Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true_arr, y_pred_arr, color="#1f77b4", alpha=0.7, edgecolors="none", label="Validation Points")
    min_val = min(y_true_arr.min(), y_pred_arr.min()) - 50
    max_val = max(y_true_arr.max(), y_pred_arr.max()) + 50
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal Line (y = x)")
    ax.set_title("Actual vs Predicted grain_kg (Validation Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual grain_kg", fontsize=10)
    ax.set_ylabel("Predicted grain_kg", fontsize=10)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend()
    plt.tight_layout()
    f1 = out_path / "07_actual_vs_predicted.png"
    plt.savefig(f1, dpi=150)
    plt.close()
    generated_plots.append(str(f1))

    # 2. Residual Distribution Histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=25, color="#2ca02c", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Line")
    ax.axvline(np.mean(residuals), color="orange", linestyle="-.", linewidth=2, label=f"Mean Residual ({np.mean(residuals):.2f} kg)")
    ax.set_title("Distribution of Residuals (Actual - Predicted)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual (kg)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend()
    plt.tight_layout()
    f2 = out_path / "08_residual_distribution.png"
    plt.savefig(f2, dpi=150)
    plt.close()
    generated_plots.append(str(f2))

    # 3. Prediction Error Plot (Residuals vs Predicted)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred_arr, residuals, color="#9467bd", alpha=0.7, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Line")
    ax.set_title("Residuals vs Predicted Values (Homoscedasticity Check)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted grain_kg", fontsize=10)
    ax.set_ylabel("Residual (Actual - Predicted in kg)", fontsize=10)
    ax.legend()
    plt.tight_layout()
    f3 = out_path / "09_prediction_error_plot.png"
    plt.savefig(f3, dpi=150)
    plt.close()
    generated_plots.append(str(f3))

    return generated_plots


def evaluate_champion_model(
    config_path: Optional[str] = None,
    log_to_mlflow: bool = True,
    figures_dir: str = "reports/figures",
) -> Dict[str, Any]:
    """Execute evaluation workflow: load model, predict, compute metrics, plot, log to MLflow.

    Args:
        config_path: Optional path to params.yaml.
        log_to_mlflow: Whether to log evaluation metrics into a new MLflow run.
        figures_dir: Directory where diagnostic plots will be stored.

    Returns:
        Dict[str, Any]: Complete evaluation summary.
    """
    config = cfg_module.load_config(config_path)

    clean_train_path = config["paths"].get("clean_train_data", "data/processed/train_clean.csv")
    raw_train_path = config["paths"]["train_data"]
    model_path = config["paths"].get("model_path", "models/model.joblib")
    target_col = config.get("target_column", "grain_kg")
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    random_seed = int(config.get("split", {}).get("random_seed", 42))

    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = config.get("mlflow", {}).get("experiment_name", "mp217_ration_distribution")
    registered_model_name = config.get("mlflow", {}).get("registered_model_name", "mp217_ration_model")

    # 1. Load model and verify champion alias resolves to v1
    model, metadata = load_champion_model(
        model_path=model_path,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
        alias="champion",
    )

    # 2. Prepare validation set
    df_clean = pd.read_csv(clean_train_path)
    X, y = train.prepare_features_and_target(df_clean, target_column=target_col)
    _, X_val, _, y_val = train.split_data(X, y, test_size=test_size, random_state=random_seed)

    # 3. Generate predictions & metrics
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)

    # 4. Generate plots
    plot_files = generate_evaluation_plots(y_val, y_pred, output_dir=figures_dir)

    # 5. Log evaluation metrics to MLflow in a new run without overwriting or changing aliases
    eval_run_id = None
    if log_to_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        original_sha = train.compute_file_sha256(raw_train_path)

        with mlflow.start_run(run_name="v1_champion_validation_evaluation") as run:
            eval_run_id = run.info.run_id
            mlflow.log_param("data_sha256", original_sha)
            mlflow.log_param("evaluated_model", registered_model_name)
            mlflow.log_param("evaluated_version", metadata["resolved_version"])
            mlflow.log_param("model_alias", "champion")
            mlflow.log_param("source_training_run_id", metadata["source_run_id"])
            mlflow.log_param("evaluation_subset", "validation_split_0.2")

            # Log metrics (must contain 'mae')
            mlflow.log_metric("val_mae", metrics["mae"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("val_rmse", metrics["rmse"])
            mlflow.log_metric("val_r2", metrics["r2"])
            mlflow.log_metric("residuals_mean", metrics["residuals_mean"])
            mlflow.log_metric("max_absolute_error", metrics["max_absolute_error"])

            for p in plot_files:
                mlflow.log_artifact(p, artifact_path="evaluation_plots")

    return {
        "metadata": metadata,
        "sample_count": len(X_val),
        "metrics": metrics,
        "plot_files": plot_files,
        "eval_run_id": eval_run_id,
    }
