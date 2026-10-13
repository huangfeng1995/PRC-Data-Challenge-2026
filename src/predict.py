"""
Inference + submission parquet generation (LightGBM version)
============================================================

Note: this file is the LightGBM inference version, only for models trained by train_lgb.py.
   In practice predict_xgb.py (XGBoost version, no libomp dependency) is recommended.

What it does:
1. Load the trained LightGBM model
2. Do feature engineering on ranking.parquet (2026-01 + 2026-07)
3. Predict TAXITIME_SEC_mvt
4. Align to the MVT_ID_mvt of the submitting.parquet template and produce a submittable parquet

Important: when doing feature engineering on the ranking data, only "real-time available" fields can be used,
   AOBT_3_flt / BLOCK_TIME_UTC_mvt must not be used (they contain leakage information and are masked in ranking).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data_loader import load_ranking, load_submission_template
from src.features import BASELINE_FEATURE_COLS, build_features


def predict_and_submit(
    model_path: str = "models/baseline_lgb.txt",
    data_dir: str = "data/raw",
    output_path: str = "submissions/baseline_v1.parquet",
    clip_min: int = 30,
    clip_max: int = 7200,
) -> pd.DataFrame:
    """Run inference + generate the submission parquet.

    Args:
        model_path: path to the trained model
        data_dir: data directory (containing ranking.parquet and submitting.parquet)
        output_path: output path of the submission file
        clip_min/max: clipping range of the predictions (seconds)
    """
    # 1. Load the model
    print(f"Loading model: {model_path}")
    booster = lgb.Booster(model_file=model_path)

    # 2. Load ranking data + run feature engineering
    print("Loading ranking data...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 3. Select features (keep only the columns used during training)
    feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df_rank_feat.columns]
    X = df_rank_feat[feat_cols]

    # Categorical column alignment: categories seen during training may be new at inference time, set them to NaN
    # LightGBM natively handles unknown categories, no manual alignment needed

    # 4. Predict
    print("Predicting...")
    y_pred = booster.predict(X)

    # 5. Clip to a reasonable range
    y_pred = np.clip(y_pred, clip_min, clip_max)

    # 6. Align to the submission template
    print(f"Aligning to the submission template...")
    template = load_submission_template(data_dir)

    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_feat["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_pred,
    })

    # Use the MVT_ID order of the template and left-join the predictions
    submission = template[["MVT_ID_mvt"]].merge(
        pred_df, on="MVT_ID_mvt", how="left"
    )

    # Check whether there are missing values
    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"{n_missing} MVT_ID values had no prediction matched, filling with the global median")
        median_pred = float(np.nanmedian(y_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)

    # Check the dtype
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
    parser = argparse.ArgumentParser(description="PRC 2026 inference + submission generation")
    parser.add_argument("--model", default="models/baseline_lgb.txt")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/baseline_v1.parquet")
    args = parser.parse_args()

    predict_and_submit(
        model_path=args.model,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
