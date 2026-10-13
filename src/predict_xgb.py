"""
XGBoost inference + submission parquet generation
==================================================

Similar to predict.py (the LightGBM version), only it uses the XGBoost model interface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_ranking, load_submission_template
from src.features import BASELINE_FEATURE_COLS, build_features
from src.train_xgb import _CAT_MAPPINGS, _encode_categoricals, _CAT_MAPPINGS_PATH


def predict_and_submit(
    model_path: str = "models/baseline_xgb.json",
    data_dir: str = "data/raw",
    output_path: str = "submissions/baseline_v1.parquet",
    clip_min: int = 30,
    clip_max: int = 10800,
) -> pd.DataFrame:
    """Run inference + generate the submission parquet.

    Important: before inference you must use the same cat mapping (save one at training time, or refit on the same data).
    For simplicity here we refit the mapping from the training file (assuming the training and inference fields match).
    """
    # 0. First fit the category mapping (learned from the training data)
    print("Learning category mappings from the training data...")
    from src.data_loader import load_training
    df_train = load_training(data_dir)
    df_train_feat = build_features(df_train)
    feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df_train_feat.columns]
    # v0.1: try loading from the persisted file first (preferred), only refit if that fails
    global _CAT_MAPPINGS_PATH
    candidate_paths = list(Path("models/feature_lists").glob("*_cat_mappings.json")) if Path("models/feature_lists").exists() else []
    if candidate_paths:
        # Use the latest cat mappings (override)
        latest = max(candidate_paths, key=lambda p: p.stat().st_mtime)
        _CAT_MAPPINGS_PATH = latest
        # _encode_categoricals(fit=False) automatically loads from _CAT_MAPPINGS_PATH
        _encode_categoricals(df_train_feat[feat_cols], fit=False)
        print(f"  Loaded cat mappings from the persisted file: {latest}")
    else:
        _encode_categoricals(df_train_feat[feat_cols], fit=True)  # only to learn the mapping, the return value is discarded
        print(f"  Refitting the cat mappings")

    # 1. Load the model
    print(f"Loading model: {model_path}")
    booster = xgb.Booster()
    booster.load_model(model_path)

    # 2. Load ranking data + run feature engineering
    print("Loading ranking data...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 3. Select features and encode
    X = df_rank_feat[feat_cols]
    X_enc = _encode_categoricals(X, fit=False)

    # 4. Predict
    print("Predicting...")
    dinfer = xgb.DMatrix(X_enc)
    y_pred = booster.predict(dinfer)

    # 5. Clip to a reasonable range
    y_pred = np.clip(y_pred, clip_min, clip_max)

    # 6. Align to the submission template
    print(f"Aligning to the submission template...")
    template = load_submission_template(data_dir)

    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_feat["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_pred,
    })

    submission = template[["MVT_ID_mvt"]].merge(
        pred_df, on="MVT_ID_mvt", how="left"
    )

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"{n_missing} MVT_ID values had no prediction matched, filling with the global median")
        median_pred = float(np.nanmedian(y_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    # 7. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)

    print(f"\n✅ Submission file generated: {output_path}")
    print(f"  Total rows: {len(submission):,}")
    print(f"  Prediction range: [{submission['TAXITIME_SEC_mvt'].min():.1f}, "
          f"{submission['TAXITIME_SEC_mvt'].max():.1f}] seconds")
    print(f"  Prediction median: {submission['TAXITIME_SEC_mvt'].median():.1f} seconds")

    return submission


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 XGBoost inference")
    parser.add_argument("--model", default="models/baseline_xgb.json")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/jubilant-puzzle_v1.parquet")
    args = parser.parse_args()

    predict_and_submit(
        model_path=args.model,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
