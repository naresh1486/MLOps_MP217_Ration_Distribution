#!/usr/bin/env python3
"""Execution script to train candidate regression models, track runs in MLflow,
select the best model, register v1, set champion alias, and persist model.joblib.
"""

import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp217 import train
from mlflow.tracking import MlflowClient


def main():
    print("==================================================")
    print("MP217 — Model Training & MLflow Tracking (v1)")
    print("==================================================")

    result = train.train_and_track_experiments()

    print(f"\nTracking URI: {result['tracking_uri']}")
    print(f"Experiment Name: {result['experiment_name']}")
    print(f"Original train.csv SHA256: {result['data_sha256']}")

    print("\nEvaluated Candidate Model Runs:")
    for r in result["run_results"]:
        print(f"  - Model: {r['model_name']:<25} | Run ID: {r['run_id']} | Validation MAE: {r['val_mae']:.4f} kg")

    print("\nBest Performing Model Selection:")
    print(f"  -> Best Model: {result['best_model_name']}")
    print(f"  -> Best Run ID: {result['best_run_id']}")
    print(f"  -> Best Validation MAE: {result['best_val_mae']:.4f} kg")

    print("\nMLflow Model Registry Status:")
    print(f"  -> Registered Model Name: {result['registered_model_name']}")
    print(f"  -> Registered Version: v{result['champion_version']}")
    
    # Verify champion alias through MlflowClient
    client = MlflowClient(tracking_uri=result["tracking_uri"])
    champion_mv = client.get_model_version_by_alias(result["registered_model_name"], "champion")
    print(f"  -> Champion Alias Verified: Points to Version {champion_mv.version} (Run ID: {champion_mv.run_id})")

    print(f"\nPersisted Model File:")
    print(f"  -> Saved Path: {result['model_save_path']}")
    print(f"  -> Sample Predictions on Validation Set: {result['sample_prediction']}")

    print("\n==================================================")
    print("Training and v1 registration completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
