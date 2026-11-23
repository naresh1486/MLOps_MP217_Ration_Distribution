#!/usr/bin/env python3
"""Execution script to evaluate the v1 champion model on the validation dataset,
compute MAE, RMSE, and R2, generate diagnostic plots, and log metrics to MLflow.
"""

import shutil
import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp217 import evaluate


def main():
    print("==================================================")
    print("MP217 — Champion Model Evaluation & Analysis (v1)")
    print("==================================================")

    res = evaluate.evaluate_champion_model()

    meta = res["metadata"]
    metrics = res["metrics"]

    print(f"\nModel Registry Verification:")
    print(f"  -> Model Name: {meta['model_name']}")
    print(f"  -> Alias: {meta['alias']}")
    print(f"  -> Resolved Version: v{meta['resolved_version']}")
    print(f"  -> Training Run ID: {meta['source_run_id']}")
    print(f"  -> Model File Path: {meta['model_file']}")

    print(f"\nEvaluation Dataset Split:")
    print(f"  -> Evaluation Set: Validation Split (N={res['sample_count']} samples, test_size=0.2, seed=42)")

    print(f"\nRegression Performance Metrics:")
    print(f"  -> Mean Absolute Error (MAE):    {metrics['mae']:.4f} kg")
    print(f"  -> Root Mean Squared Error (RMSE): {metrics['rmse']:.4f} kg")
    print(f"  -> R-squared Score (R2):         {metrics['r2']:.4f}")

    print(f"\nPrediction & Residual Distribution Analysis:")
    print(f"  -> Mean of Residuals (y - y_hat): {metrics['residuals_mean']:.4f} kg")
    print(f"  -> Std Dev of Residuals:          {metrics['residuals_std']:.4f} kg")
    print(f"  -> Maximum Absolute Error:        {metrics['max_absolute_error']:.4f} kg")

    print(f"\nGenerated Diagnostic Visualizations:")
    for p in res["plot_files"]:
        print(f"  - {p}")
        # Also copy to G230_MP217/reports/figures
        dest = Path("G230_MP217/reports/figures") / Path(p).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest)

    if res.get("eval_run_id"):
        print(f"\nMLflow Tracking:")
        print(f"  -> Logged Evaluation Run ID: {res['eval_run_id']}")

    print("\n==================================================")
    print("Champion model evaluation completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
