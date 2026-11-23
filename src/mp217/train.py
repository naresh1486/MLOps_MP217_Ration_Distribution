"""Model training, evaluation, and MLflow experiment tracking module for MP217.

Handles data preparation, scikit-learn pipeline construction, MLflow experiment tracking
to SQLite backend, model comparison, MLflow Model Registry versioning, and champion model persistence.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mp217 import config as cfg_module

DEFAULT_NUMERIC_FEATURES = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "shop_area",
    "block",
]


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file's binary content.

    Args:
        file_path: Path to the target file.

    Returns:
        str: Hexadecimal SHA-256 hash digest.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = "grain_kg",
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract model feature matrix X and target series y.

    Excludes non-feature columns such as record_id and date.

    Args:
        df: Input cleaned DataFrame.
        target_column: Name of the target variable.
        numeric_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X (8 columns) and target y.
    """
    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    feature_cols = numeric_features + categorical_features

    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"Required feature '{col}' missing from DataFrame.")

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' missing from DataFrame.")

    X = df[feature_cols].copy()
    y = df[target_column].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into reproducible train and validation subsets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of samples to allocate to validation set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple: X_train, X_val, y_train, y_val.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_pipeline(
    regressor: BaseEstimator,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Pipeline:
    """Construct an end-to-end scikit-learn pipeline with preprocessing and regression.

    Args:
        regressor: Initialized scikit-learn regressor.
        numeric_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Pipeline: Composed scikit-learn pipeline.
    """
    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def get_candidate_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Provide a dictionary of candidate regression models to evaluate.

    Args:
        random_state: Random seed for regressor initialization.

    Returns:
        Dict[str, BaseEstimator]: Mapping of model names to instantiated regressors.
    """
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state,
        ),
    }


def train_and_track_experiments(
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete training workflow: track candidate runs, compare, register champion.

    1. Loads configuration from params.yaml.
    2. Computes original train.csv SHA256.
    3. Prepares features/target and performs train/val split.
    4. Evaluates and logs >= 3 model runs to SQLite MLflow.
    5. Identifies best model by validation MAE.
    6. Registers best model as version 1 in MLflow Model Registry.
    7. Assigns '@champion' alias to version 1.
    8. Serializes and saves champion pipeline to models/model.joblib.

    Args:
        config_path: Path to params.yaml. If None, default discovery is used.

    Returns:
        Dict[str, Any]: Summary of experiment runs, champion model, and verification details.
    """
    config = cfg_module.load_config(config_path)

    # 1. Configuration parameters
    raw_train_path = config["paths"]["train_data"]
    clean_train_path = config["paths"].get("clean_train_data", "data/processed/train_clean.csv")
    target_col = config.get("target_column", "grain_kg")
    model_save_path = Path(config["paths"].get("model_path", "models/model.joblib"))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    test_size = float(config.get("split", {}).get("test_size", 0.2))
    random_seed = int(config.get("split", {}).get("random_seed", 42))

    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = config.get("mlflow", {}).get("experiment_name", "mp217_ration_distribution")
    registered_model_name = config.get("mlflow", {}).get("registered_model_name", "mp217_ration_model")

    # 2. Compute SHA256 of original raw train.csv
    original_sha256 = compute_file_sha256(raw_train_path)

    # 3. Load clean training data and split
    df_clean = pd.read_csv(clean_train_path)
    X, y = prepare_features_and_target(df_clean, target_column=target_col)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=test_size, random_state=random_seed)

    # 4. Setup MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate_models = get_candidate_models(random_state=random_seed)
    run_results = []

    best_mae = float("inf")
    best_run_id = None
    best_model_name = None
    best_pipeline = None

    for model_name, regressor in candidate_models.items():
        with mlflow.start_run(run_name=f"v1_candidate_{model_name}") as run:
            run_id = run.info.run_id

            # Fit pipeline
            pipeline = build_pipeline(regressor)
            pipeline.fit(X_train, y_train)

            # Evaluate MAE
            val_preds = pipeline.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, val_preds))

            # Log parameters
            mlflow.log_param("data_sha256", original_sha256)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_seed", random_seed)
            mlflow.log_param("test_size", test_size)
            mlflow.log_params(regressor.get_params())

            # Log metrics (metric name MUST contain 'mae')
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("mae", val_mae)

            # Log model artifact with cloudpickle serialization for MLflow 3 compatibility
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            run_results.append({
                "model_name": model_name,
                "run_id": run_id,
                "val_mae": val_mae,
                "data_sha256": original_sha256,
            })

            if val_mae < best_mae:
                best_mae = val_mae
                best_run_id = run_id
                best_model_name = model_name
                best_pipeline = pipeline

    # 5. Register best model as version 1 and set alias champion
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri, registered_model_name)
    v1_version = str(model_version.version)

    # Set alias champion
    client.set_registered_model_alias(registered_model_name, "champion", v1_version)

    # 6. Save champion model to models/model.joblib
    joblib.dump(best_pipeline, model_save_path)

    # Verify saved model can be loaded
    loaded_model = joblib.load(model_save_path)
    sample_pred = loaded_model.predict(X_val.iloc[:2])

    return {
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
        "data_sha256": original_sha256,
        "run_results": run_results,
        "best_run_id": best_run_id,
        "best_model_name": best_model_name,
        "best_val_mae": best_mae,
        "registered_model_name": registered_model_name,
        "champion_version": v1_version,
        "model_save_path": str(model_save_path),
        "sample_prediction": sample_pred.tolist(),
    }
