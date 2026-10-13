"""
XGBoost full-features inference + generate submission parquet
============================================================

Different from predict_xgb.py:
- Supports all features (basic + aggregate + lead)
- Refits the category mappings from the training file (to ensure cat encoding consistency)
- Recomputes lag/lead from the training file (because ranking data needs the same group/time index)

Note: rank data is a ranking parquet (with TAXITIME_SEC_mvt masked), so the aggregation target has NaN,
in which case count can still be computed, but mean/p90 cannot - this is exactly what we want: on ranking, mean/p90 are NaN.
But computing lag/lead on train and rank data together solves it: the ranking lag window contains real values from train,
so mean/p90 can be computed. This is exactly what the v1 baseline lacked.

Implementation approach:
  1. Load train + ranking merged -> compute lag/lead together (avoids missing data in rank)
  2. Select feature columns separately for train and rank
  3. Run inference with the booster
  4. Merge with the template + save

WARNING: at training time the validation set target is real, so mean/p90 in the aggregates contain validation period data,
which introduces slight "future information leakage". This is a trade-off: so that mean/p90 also has values at inference time.
A stricter approach is to use sliding window OOF to compute lag/lead, which is much more work, so go with the simple approach first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_ranking, load_submission_template, load_training
from src.features import BASELINE_FEATURE_COLS, build_features
from src.features_aggregate import add_aggregate_features
from src.features_laglead import add_lead_features
from src.train_xgb import _encode_categoricals, _CAT_MAPPINGS_PATH


def _build_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic + lag + lead complete features."""
    df = build_features(df)
    df = add_aggregate_features(df)
    df = add_lead_features(df)
    return df


def _get_feat_cols(df: pd.DataFrame, drop_target_stats: bool = True) -> list[str]:
    """All feature columns used for training. Basic + all *_15m / *_30m / *_60m columns (lag + lead).

    v3.5: add the drop_target_stats parameter.
    - drop_target_stats=True (default): keep only cnt_* (no_laglead mode)
    - drop_target_stats=False: keep cnt + mean + p90 (legacy mode)
    """
    base = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    if drop_target_stats:
        agg = [c for c in df.columns if c.endswith("m") and "cnt_" in c]
    else:
        agg = [c for c in df.columns
               if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
    return base + agg


def predict_and_submit(
    model_path: str = "models/baseline_xgb.json",
    data_dir: str = "data/raw",
    output_path: str = "submissions/jubilant-puzzle_v2.parquet",
    clip_min: int = 30,
    clip_max: int = 10800,
    apply_quantile_mapping: bool = False,  # v3: whether to do quantile mapping to align distributions (off by default, wait for v3 validation)
    drop_target_stats: bool = True,  # v3.5: no_laglead mode by default (only cnt)
) -> pd.DataFrame:
    """Run inference + generate submission parquet (full feature version)."""
    # 1. Load training data -> train basic features (only to fit cat mappings)
    print("Loading training data (only to learn cat mappings)...")
    df_train = load_training(data_dir)
    df_train_feat = build_features(df_train)
    # v0.1: prefer loading cat mappings from the persisted file to avoid cross-process drift
    global _CAT_MAPPINGS_PATH
    candidate_paths = list(Path("models/feature_lists").glob("*_cat_mappings.json")) if Path("models/feature_lists").exists() else []
    if candidate_paths:
        _CAT_MAPPINGS_PATH = max(candidate_paths, key=lambda p: p.stat().st_mtime)
        print(f"  Loading cat mappings from persisted file: {_CAT_MAPPINGS_PATH.name}")
    if not candidate_paths:
        from src.train_xgb import _encode_categoricals as _ec
        _ec(df_train_feat[[c for c in BASELINE_FEATURE_COLS if c in df_train_feat.columns]], fit=True)

    # 2. Load the model
    print(f"Loading model: {model_path}")
    booster = xgb.Booster()
    booster.load_model(model_path)

    # 3. Load ranking data -> basic features
    print("Loading ranking data...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 4. Merge train + rank and compute lag/lead together (so that rank's mean/p90 has values)
    print("Merging train + ranking to compute lag/lead ...")
    # Use a marker column to distinguish
    df_train_feat["__is_train__"] = True
    df_rank_feat["__is_train__"] = False
    # Keep the TAXITIME_SEC_mvt column (train has it, rank is NaN), add __is_dep__ and other needed columns
    for col in ["ADEP_mvt", "RUNWAY_mvt", "EOBT_1_flt", "TAXITIME_SEC_mvt"]:
        if col not in df_rank_feat.columns:
            df_rank_feat[col] = np.nan

    common_cols = list(set(df_train_feat.columns) & set(df_rank_feat.columns))
    df_all = pd.concat([df_train_feat[common_cols], df_rank_feat[common_cols]], axis=0, ignore_index=True)
    print(f"  after merge: {len(df_all):,} rows (train {len(df_train_feat):,} + rank {len(df_rank_feat):,})")
    # WARNING: concat turns both sides' category into object, need to convert again
    from src.features import CATEGORICAL_COLS, to_categorical
    cat_in_all = [c for c in CATEGORICAL_COLS if c in df_all.columns]
    df_all = to_categorical(df_all, cat_in_all)

    # Compute lag/lead (on the merged data)
    print("Computing lag features...")
    df_all = add_aggregate_features(df_all)
    print("Computing lead features...")
    df_all = add_lead_features(df_all)

    # 5. Split back into train and rank
    train_mask = df_all["__is_train__"].values
    df_all = df_all.drop(columns=["__is_train__"])

    df_train_full = df_all[train_mask].reset_index(drop=True)
    df_rank_full = df_all[~train_mask].reset_index(drop=True)

    # Key: feature columns must be taken from the merged df_all, so that lag/lead are in the columns
    feat_cols = _get_feat_cols(df_all, drop_target_stats=drop_target_stats)
    print(f"  number of full features: {len(feat_cols)}")

    # Fit category mappings (use train_full, because train data is larger so the category set is more complete)
    _encode_categoricals(df_train_full[feat_cols], fit=True)

    # 6. Select features + encode + inference
    print("Running inference...")
    # Reorder by the column order the model expects (avoid XGBoost feature_names mismatch)
    expected = booster.feature_names
    if expected is None:
        raise RuntimeError("model has no feature_names, cannot align")
    missing = [c for c in expected if c not in df_rank_full.columns]
    extra = [c for c in feat_cols if c not in expected]
    if missing:
        raise RuntimeError(f"inference data is missing columns: {missing[:5]}")
    if extra:
        print(f"  [warn] present in training but unused in inference: {extra[:5]}")
    feat_cols_aligned = expected  # directly use the order the model expects
    X_rank = df_rank_full[feat_cols_aligned]
    X_rank_enc = _encode_categoricals(X_rank, fit=False)
    dinfer = xgb.DMatrix(X_rank_enc)
    y_pred = booster.predict(dinfer)

    # v3: optional quantile mapping - align the rank prediction distribution to the train target distribution
    # (rank has no target, assume train/test have the same distribution; 50% strength is a compromise, avoids hard mapping)
    if apply_quantile_mapping:
        train_target = load_training(data_dir)["TAXITIME_SEC_mvt"]
        train_target = train_target[(train_target > 0) & (train_target < 10800)].values
        rank_quantiles = np.quantile(y_pred, np.linspace(0.001, 0.999, 999))
        target_quantiles = np.quantile(train_target, np.linspace(0.001, 0.999, 999))
        mapped = np.interp(y_pred, rank_quantiles, target_quantiles).astype("float64")
        # 50/50 blend - full mapping may be too strong, not using it at all is wrong
        y_pred = 0.5 * y_pred + 0.5 * mapped
        print(f"  [info] applied 50% quantile mapping (rank->train distribution)")

    y_pred = np.clip(y_pred, clip_min, clip_max)

    # 7. Align with the template
    template = load_submission_template(data_dir)
    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_full["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_pred,
    })
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} MVT_ID have no matching prediction, filling with the global median")
        median_pred = float(np.nanmedian(y_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    # 8. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)

    print(f"\nsubmission file generated: {output_path}")
    print(f"  total rows: {len(submission):,}")
    print(f"  prediction range: [{submission['TAXITIME_SEC_mvt'].min():.1f}, "
          f"{submission['TAXITIME_SEC_mvt'].max():.1f}] seconds")
    print(f"  prediction median: {submission['TAXITIME_SEC_mvt'].median():.1f} seconds")

    return submission


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 XGBoost full-feature inference")
    parser.add_argument("--model", default="models/baseline_xgb.json")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/jubilant-puzzle_v2.parquet")
    parser.add_argument("--keep-target-stats", action="store_true",
                        help="keep mean/p90 columns (legacy mode, default False = no_laglead)")
    args = parser.parse_args()

    predict_and_submit(
        model_path=args.model,
        data_dir=args.data_dir,
        output_path=args.output,
        drop_target_stats=not args.keep_target_stats,
    )


if __name__ == "__main__":
    main()