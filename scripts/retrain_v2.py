"""Retrain v2 workflow, drift monitoring, and champion promotion runner (Part 14).

Executes:
1. Calculates PSI across all numeric features between train_clean.csv and later_clean.csv.
2. Evaluates v1 champion performance per block on later_clean.csv.
3. Partitions later_clean.csv chronologically by date (70% train, 30% check).
4. Retrains v2 on train_clean.csv + first 70% of later_clean.csv.
5. Evaluates v1 and v2 on the last 30% check dataset and evaluates the 5% promotion rule.
6. Logs the run to MLflow, registers Version 2, updates champion alias, and updates findings.yaml.
"""

import hashlib
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from mp217.monitoring import (
    compute_feature_psis,
    evaluate_mae_by_group,
    evaluate_promotion_rule,
    split_later_by_date,
    calculate_psi,
)


def run_retrain_v2_pipeline() -> dict:
    """Execute the complete Part 14 monitoring and retraining pipeline."""
    # 1. Load datasets
    train_path = Path("data/processed/train_clean.csv")
    later_path = Path("data/processed/later_clean.csv")
    raw_train_path = Path("data/raw/train.csv")

    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)

    num_features = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
    ]
    cat_features = ["shop_area", "block"]
    all_features = num_features + cat_features
    target_col = "grain_kg"

    # 2. Calculate PSI for all numeric features
    psi_dict, most_drifted_feature = compute_feature_psis(
        train_df, later_df, num_features, num_bins=10
    )
    # Also calculate PSI for target grain_kg for comprehensive reporting
    target_psi = round(calculate_psi(train_df[target_col], later_df[target_col], num_bins=10), 4)

    print("=== Population Stability Index (PSI, 10 bins) ===")
    for feat, psi_val in psi_dict.items():
        print(f"  {feat}: {psi_val}")
    print(f"  Target ({target_col}): {target_psi}")
    print(f"Most drifted feature: {most_drifted_feature} (PSI = {psi_dict[most_drifted_feature]})\n")

    # 3. Evaluate existing v1 champion on later data by block
    tracking_uri = "sqlite:///mlflow.db"
    registered_model_name = "mp217_ration_model"
    mlflow.set_tracking_uri(tracking_uri)

    v1_model = mlflow.sklearn.load_model(f"models:/{registered_model_name}@champion")

    group_maes, worst_served_group = evaluate_mae_by_group(
        v1_model, later_df, all_features, target=target_col, group_col="block"
    )

    print("=== v1 MAE on later.csv by block ===")
    for block_name, mae_val in group_maes.items():
        print(f"  {block_name}: {mae_val:.4f} kg")
    print(f"Worst-served group: {worst_served_group} (MAE = {group_maes[worst_served_group]:.4f} kg)\n")

    # 4. Partition later data chronologically (first 70% for train, last 30% for check)
    later_train, later_check = split_later_by_date(later_df, date_col="date", train_ratio=0.7)
    print(f"Later data split: {len(later_train)} rows for training, {len(later_check)} rows for comparison")

    # 5. Evaluate v1 on check dataset (last 30%)
    from sklearn.metrics import mean_absolute_error
    v1_check_preds = v1_model.predict(later_check[all_features])
    mae_v1_on_check = round(float(mean_absolute_error(later_check[target_col], v1_check_preds)), 4)
    print(f"v1 MAE on check dataset (last 30%): {mae_v1_on_check:.4f} kg")

    # 6. Prepare combined training data for v2
    combined_train = pd.concat([train_df, later_train], ignore_index=True)
    print(f"Combined training dataset: {len(combined_train)} rows ({len(train_df)} train_clean + {len(later_train)} later_train)")

    # 7. Build and train v2 model
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )

    v2_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=5)),
    ])

    v2_pipeline.fit(combined_train[all_features], combined_train[target_col])

    # 8. Evaluate v2 on the exact same check dataset
    v2_check_preds = v2_pipeline.predict(later_check[all_features])
    mae_v2_on_check = round(float(mean_absolute_error(later_check[target_col], v2_check_preds)), 4)
    print(f"v2 MAE on check dataset (last 30%): {mae_v2_on_check:.4f} kg")

    # 9. Evaluate promotion rule
    decision, improvement_pct = evaluate_promotion_rule(mae_v1_on_check, mae_v2_on_check, required_improvement_pct=5.0)
    print(f"Improvement: {improvement_pct:.2f}% (Required: >= 5.0%)")
    print(f"Promotion Decision: {decision.upper()}\n")

    # 10. MLflow Experiment Tracking & Model Registration
    raw_hash = hashlib.sha256(raw_train_path.read_bytes()).hexdigest()
    mlflow.set_experiment("mp217_ration_distribution")

    with mlflow.start_run(run_name="retrain_v2_gradient_boosting") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Required parameters
        mlflow.log_param("data_sha256", raw_hash)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("training_rows", len(combined_train))
        mlflow.log_param("later_train_ratio", 0.7)

        # Metrics containing 'mae'
        mlflow.log_metric("mae", mae_v2_on_check)
        mlflow.log_metric("mae_v2_check", mae_v2_on_check)
        mlflow.log_metric("mae_v1_check", mae_v1_on_check)
        mlflow.log_metric("improvement_pct", improvement_pct)

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    # Resolve new model version
    client = MlflowClient()
    latest_versions = client.get_latest_versions(registered_model_name)
    v2_version = max(int(v.version) for v in latest_versions)
    print(f"Registered model version: {v2_version}")

    # Apply promotion decision to registry
    if decision == "promote":
        client.set_registered_model_alias(registered_model_name, "champion", v2_version)
        champion_version = v2_version
        print(f"Promoted v2! Champion alias now points to Version {champion_version}")

        # Update local model artifact
        joblib.dump(v2_pipeline, "models/model.joblib")
        print("Updated models/model.joblib with promoted v2 pipeline")
    else:
        champion_version = 1
        print(f"Decision was keep; champion alias remains Version {champion_version}")

    # 11. Update findings.yaml
    findings_paths = [Path("findings.yaml"), Path("reports/findings.yaml"), Path("G230_MP217/findings.yaml")]
    for fpath in findings_paths:
        if fpath.exists():
            with open(fpath, "r") as f:
                data = yaml.safe_load(f)

            data["monitoring"]["psi"] = {
                "cards_attached": float(psi_dict["cards_attached"]),
                "working_hours": float(psi_dict["working_hours"]),
                "biometric_failures": float(psi_dict["biometric_failures"]),
                "temp_c": float(psi_dict["temp_c"]),
                "stock_kg_morning": float(psi_dict["stock_kg_morning"]),
                "days_to_month_end": float(psi_dict["days_to_month_end"]),
            }
            data["monitoring"]["most_drifted_feature"] = most_drifted_feature
            data["monitoring"]["mae_by_group"] = {
                "Block North": float(group_maes["Block North"]),
                "Block South": float(group_maes["Block South"]),
                "Block East": float(group_maes["Block East"]),
            }
            data["monitoring"]["worst_served_group"] = worst_served_group
            data["monitoring"]["mae_v1_on_check"] = float(mae_v1_on_check)
            data["monitoring"]["mae_v2_on_check"] = float(mae_v2_on_check)
            data["monitoring"]["decision"] = decision

            with open(fpath, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            print(f"Updated {fpath}")

    return {
        "psi": psi_dict,
        "target_psi": target_psi,
        "most_drifted_feature": most_drifted_feature,
        "mae_by_group": group_maes,
        "worst_served_group": worst_served_group,
        "mae_v1_on_check": mae_v1_on_check,
        "mae_v2_on_check": mae_v2_on_check,
        "improvement_pct": improvement_pct,
        "decision": decision,
        "run_id": run_id,
        "v2_version": v2_version,
        "champion_version": champion_version,
    }


if __name__ == "__main__":
    run_retrain_v2_pipeline()
