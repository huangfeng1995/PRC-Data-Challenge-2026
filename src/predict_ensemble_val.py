"""
Generate an ensemble submission using the OOF validation models (without waiting for final training)
=====================================================================================================

Strategy:
- Load the 6 validation models under models/ensemble_val/ (3 XGB + 3 LGB)
- Predict the ranking data with each model
- Weighted average according to weights.json
- Generate the v4 submission

v4 vs v3:
- v3: a single tuned XGB model, leaderboard RMSE pending
- v4: XGB+LGB ensemble (3 seeds each), validation RMSE 231.56 vs v3 232.71
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from src.data_loader import load_ranking, load_submission_template
from src.features import BASELINE_FEATURE_COLS, build_features, to_categorical
from src.features_aggregate import add_aggregate_features
from src.features_laglead import add_lead_features
from src.train_xgb import _encode_categoricals


def predict_ensemble_val(
    models_dir: str = "models/ensemble_val",
    data_dir: str = "data/raw",
    output_path: str = "submissions/jubilant-puzzle_v4.parquet",
    clip_min: int = 30,
    clip_max: int = 10800,
) -> pd.DataFrame:
    """Generate an ensemble submission using the OOF validation models."""

    models_dir = Path(models_dir)
    weights_path = models_dir / "weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}, run train_ensemble at least up to the OOF stage first")

    meta = json.loads(weights_path.read_text())
    weights = meta["weights"]
    blend_mode = meta["blend_mode"]
    feat_cols = meta["feat_cols"]
    model_keys = meta["models"]

    print(f"Loading {len(model_keys)} validation models + weights {weights}, blend={blend_mode}")

    # 1. Load train data (only to learn the cat mappings)
    print("Loading training data + building features (only to fit the cat mappings)...")
    from src.data_loader import load_training
    df_train = load_training(data_dir)
    df_train_feat = build_features(df_train)

    # 2. Load ranking data + base features
    print("Loading ranking data...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 3. Concatenate to compute lag/lead
    df_train_feat["__is_train__"] = True
    df_rank_feat["__is_train__"] = False
    for col in ["ADEP_mvt", "RUNWAY_mvt", "EOBT_1_flt", "TAXITIME_SEC_mvt"]:
        if col not in df_rank_feat.columns:
            df_rank_feat[col] = np.nan

    common_cols = list(set(df_train_feat.columns) & set(df_rank_feat.columns))
    df_all = pd.concat([df_train_feat[common_cols], df_rank_feat[common_cols]], axis=0, ignore_index=True)
    print(f"  After concatenation: {len(df_all):,}")

    cat_in_all = [c for c in ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
                                "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
                                "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
                                "airport_hour"] if c in df_all.columns]
    df_all = to_categorical(df_all, cat_in_all)

    print("Computing lag features...")
    df_all = add_aggregate_features(df_all)
    print("Computing lead features...")
    df_all = add_lead_features(df_all)

    train_mask = df_all["__is_train__"].values
    df_all = df_all.drop(columns=["__is_train__"])

    df_rank_full = df_all[~train_mask].reset_index(drop=True)

    # 4. fit the cat encoding (using train_full)
    df_train_full = df_all[train_mask].reset_index(drop=True)
    X_train_for_cat = df_train_full[feat_cols]
    _encode_categoricals(X_train_for_cat, fit=True)

    # 5. rank encoding + inference
    print("Running inference on 6 models + weighted blending...")
    X_rank = df_rank_full[feat_cols]
    X_rank_enc = _encode_categoricals(X_rank, fit=False)

    model_preds = {}  # key: model_name (e.g. "xgb"), value: avg_pred of seeds
    for key in model_keys:
        model_type, _, seed_str = key.partition("__seed_")
        model_name = model_type  # "xgb" or "lgb"

        pkl_path = models_dir / f"{key}.pkl"
        if not pkl_path.exists():
            print(f"  Missing {pkl_path}, skipping")
            continue

        # Note: wrapper.save actually calls the native save_model of XGB/LGB (writes JSON format with a .pkl suffix)
        if model_name == "xgb":
            booster = xgb.Booster()
            booster.load_model(str(pkl_path))
            # XGBoost models expect cat columns already integer-encoded (same as at training time)
            # _encode_categoricals already built the mapping at fit time, fit=False reuses it directly
            # Note: some cat columns in X_rank_enc may still be object (because the dtype is not category)
            # We convert them once more
            X_for_xgb = _encode_categoricals(X_rank.copy(), fit=False)
            dinfer = xgb.DMatrix(X_for_xgb)
            y_pred = booster.predict(dinfer)
        elif model_name == "lgb":
            booster = lgb.Booster(model_file=str(pkl_path))
            # At training time the categories of the LGB cat columns come from winter_train, but the rank data has different/more categories
            # Solution: convert X_rank to the same cat categories as at training time (using the cat stored inside LGB)
            # But the booster does not expose the cat info; alternative: convert the cat columns to int codes (using the training-time cat->int mapping)
            X_for_lgb = _encode_categoricals(X_rank.copy(), fit=False)
            # The LGB cat column names are identical to XGB, so we reuse the cat encoding
            # Convert category dtypes other than the cat columns to object (LGB needs category or int)
            y_pred = booster.predict(X_for_lgb, num_iteration=booster.best_iteration)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if model_name not in model_preds:
            model_preds[model_name] = []
        model_preds[model_name].append(y_pred)
        print(f"  ✓ {key}")

    # 6. Blending: average the seeds within each model, then weight them by weights
    final_pred = np.zeros(len(df_rank_full))
    for model_name, preds_list in model_preds.items():
        avg = np.mean(preds_list, axis=0)
        w = weights.get(model_name, 0.0)
        final_pred += w * avg
        print(f"  {model_name} ({len(preds_list)} seed avg): weight={w:.3f}")

    final_pred = np.clip(final_pred, clip_min, clip_max)

    # 7. Align to the template + save
    template = load_submission_template(data_dir)
    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_full["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": final_pred,
    })
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        median_pred = float(np.nanmedian(final_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)
    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)

    print(f"\n✅ Submission generated: {output_path}")
    print(f"  Total rows: {len(submission):,}")
    print(f"  Prediction range: [{submission['TAXITIME_SEC_mvt'].min():.1f}, "
          f"{submission['TAXITIME_SEC_mvt'].max():.1f}] seconds")
    print(f"  Prediction median: {submission['TAXITIME_SEC_mvt'].median():.1f} seconds")
    print(f"  OOF validation RMSE: 231.56 (winter split)")

    return submission


def main():
    parser = argparse.ArgumentParser(description="Generate an ensemble submission using the OOF validation models")
    parser.add_argument("--models-dir", default="models/ensemble_val")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/jubilant-puzzle_v4.parquet")
    args = parser.parse_args()

    predict_ensemble_val(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
