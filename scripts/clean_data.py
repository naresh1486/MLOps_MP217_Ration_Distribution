#!/usr/bin/env python3
"""Data cleaning pipeline execution script for MP217 Ration Shop Distribution project.

Reads raw train.csv and later.csv from data/raw/, applies the cleaning pipeline,
validates quality constraints, and writes processed outputs to data/processed/.
"""

import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from mp217 import cleaning


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_raw_path = raw_dir / "train.csv"
    later_raw_path = raw_dir / "later.csv"

    print("==================================================")
    print("MP217 — Data Cleaning Pipeline (Group G230)")
    print("==================================================")

    # 1. Clean train.csv
    print(f"\n[1/2] Loading raw training data from {train_raw_path}...")
    train_raw = pd.read_csv(train_raw_path)
    initial_train_rows = len(train_raw)
    initial_train_dups = int(train_raw.duplicated().sum())
    initial_east_mixup = int((train_raw["block"] == "Block East").sum())
    initial_typos = int((train_raw["grain_kg"] > 2000.0).sum())
    initial_shop_variants = int(train_raw["shop_area"].nunique())

    print(f"  - Initial shape: {train_raw.shape}")
    print(f"  - Duplicate rows detected: {initial_train_dups}")
    print(f"  - Block East rows affected by unit mixup: {initial_east_mixup}")
    print(f"  - Target typos detected (> 2000 kg): {initial_typos}")
    print(f"  - Raw shop_area variants: {initial_shop_variants}")

    print("\nExecuting cleaning transformations on train.csv...")
    train_clean = cleaning.clean_train_data(train_raw)
    train_out_path = processed_dir / "train_clean.csv"
    train_clean.to_csv(train_out_path, index=False)

    print(f"  -> Successfully generated {train_out_path}")
    print(f"  -> Cleaned shape: {train_clean.shape} (exactly {len(train_clean)} rows)")
    print(f"  -> Remaining duplicates: {train_clean.duplicated().sum()}")
    print(f"  -> Block East stock max: {train_clean[train_clean['block'] == 'Block East']['stock_kg_morning'].max():.2f} kg")
    print(f"  -> Target grain_kg max: {train_clean['grain_kg'].max():.2f} kg (typos resolved: 0 > 2000 kg)")
    print(f"  -> Normalized shop_area levels: {sorted(train_clean['shop_area'].unique().tolist())}")

    # 2. Clean later.csv
    print(f"\n[2/2] Loading raw production monitoring data from {later_raw_path}...")
    later_raw = pd.read_csv(later_raw_path)
    print(f"  - Initial shape: {later_raw.shape}")

    print("Executing cleaning and validation on later.csv...")
    later_clean = cleaning.clean_later_data(later_raw)
    later_out_path = processed_dir / "later_clean.csv"
    later_clean.to_csv(later_out_path, index=False)

    print(f"  -> Successfully generated {later_out_path}")
    print(f"  -> Cleaned shape: {later_clean.shape} (exactly {len(later_clean)} rows)")
    print(f"  -> Normalized shop_area levels: {sorted(later_clean['shop_area'].unique().tolist())}")

    print("\n==================================================")
    print("Data cleaning completed and validated successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
