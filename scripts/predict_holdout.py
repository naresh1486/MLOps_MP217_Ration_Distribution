"""Generate holdout predictions using the registered champion model (Part 15).

Predicts daily grain distribution (grain_kg) for all 250 records in data/raw/holdout_features.csv
using the Version 2 champion model registered in MLflow.
Saves the results to predictions/holdout_predictions.csv with columns:
record_id,prediction
"""

from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd


def generate_holdout_predictions() -> pd.DataFrame:
    """Load the champion model and generate predictions for holdout_features.csv."""
    holdout_path = Path("data/raw/holdout_features.csv")
    output_path = Path("predictions/holdout_predictions.csv")
    tracking_uri = "sqlite:///mlflow.db"
    registered_model_name = "mp217_ration_model"
    alias = "champion"

    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout dataset not found at: {holdout_path}")

    # 1. Connect to MLflow and verify champion model version
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    alias_info = client.get_model_version_by_alias(registered_model_name, alias)
    champion_version = str(alias_info.version)

    print(f"Loading champion model: {registered_model_name}@{alias}")
    print(f"Confirmed Champion Version: Version {champion_version}")
    assert champion_version == "2", f"Expected champion version 2, found {champion_version}"

    model = mlflow.sklearn.load_model(f"models:/{registered_model_name}@{alias}")

    # 2. Read holdout features
    holdout_df = pd.read_csv(holdout_path)
    print(f"Loaded holdout features: {len(holdout_df)} rows")

    features = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
        "shop_area",
        "block",
    ]

    # 3. Generate raw predictions from champion model
    X_holdout = holdout_df[features].copy()
    raw_predictions = model.predict(X_holdout)

    # 4. Format output DataFrame with exactly two columns: record_id, prediction
    predictions_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": [round(float(p), 2) for p in raw_predictions],
    })

    # 5. Validate output requirements
    assert len(predictions_df) == 250, f"Expected 250 rows, got {len(predictions_df)}"
    assert list(predictions_df.columns) == ["record_id", "prediction"], f"Invalid columns: {predictions_df.columns}"
    assert predictions_df["record_id"].nunique() == 250, "Duplicate record_ids detected"
    assert not predictions_df["prediction"].isnull().any(), "Null predictions found"
    assert not np.isinf(predictions_df["prediction"]).any(), "Infinite predictions found"
    assert pd.api.types.is_numeric_dtype(predictions_df["prediction"]), "Prediction column must be numeric"

    # 6. Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved holdout predictions to: {output_path}")
    print(f"Predictions preview:\n{predictions_df.head(10)}")
    print(f"Prediction summary statistics:\n{predictions_df['prediction'].describe()}")

    return predictions_df


if __name__ == "__main__":
    generate_holdout_predictions()
