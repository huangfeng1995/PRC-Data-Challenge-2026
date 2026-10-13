"""
Three-model ensemble inference script
=====================================

Load the trained XGB + LGB (+ optional CatBoost) models and blend the predictions using the weights learned at training time.

Usage:
    python -m src.predict_ensemble --data-dir data/raw --output submissions/ensemble_v1.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_ranking, load_submission_template, load_training

# Lazy import of the training-side modules to avoid the libomp issue when importing xgboost on macOS
# (because train_xgb imports xgboost at the top level, which fails in the macOS sandbox)
def _build_full_features_lazy(df, feature_set="all", te_mappings=None):
    from src.train_ensemble import build_full_features
    return build_full_features(df, feature_set=feature_set, te_mappings=te_mappings)

def _clear_cache_lazy():
    from src.train_xgb import _clear_cat_cache
    _clear_cat_cache()

def _encode_categoricals_lazy(X, fit=False):
    from src.train_xgb import _encode_categoricals
    return _encode_categoricals(X, fit=fit)

def _load_model_lazy(name, path):
    from src.train_ensemble import ModelWrapper
    return ModelWrapper.load(name, path)


def predict_ensemble(
    data_dir: str = "data/raw",
    models_dir: str = "models",
    output_path: str = "submissions/ensemble_v1.parquet",
    clip_min: int = 30,
    clip_max: int = 10800,
) -> pd.DataFrame:
    """Load the ensemble models + run inference + generate the submission."""
    models_dir = Path(models_dir)

    # 1. Read the weights config
    weights_path = models_dir / "ensemble_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Cannot find {weights_path}, please run train_ensemble.py first")
    config = json.loads(weights_path.read_text())
    weights = config["weights"]
    feat_cols = config["feat_cols"]
    feature_set = config.get("feature_set", "all")
    model_names = config["models"]
    te_mappings = config.get("te_mappings")  # v2: reuse the mapping saved at training time
    blend_mode = config.get("blend_mode", "mean")  # v2: mean or median
    print(f"Config: weights={weights}, models={model_names}, feature_set={feature_set}, blend_mode={blend_mode}")
    print(f"       te_mappings: {len(te_mappings) if te_mappings else 0} features")

    # 2. First fit the categorical encoding mappings (learned from the training data)
    print("\nLearning category mappings from the training data...")
    _clear_cache_lazy()
    df_train = load_training(data_dir)
    df_train_feat, _ = _build_full_features_lazy(df_train, feature_set=feature_set)
    # fit the encoding
    _ = _encode_categoricals_lazy(df_train_feat[feat_cols], fit=True)

    # 3. Load ranking data + build features
    print("\nLoading ranking data...")
    df_rank = load_ranking(data_dir)
    df_rank_feat, _ = _build_full_features_lazy(df_rank, feature_set=feature_set, te_mappings=te_mappings)
    X = df_rank_feat[feat_cols]
    X_enc = _encode_categoricals_lazy(X, fit=False)

    # 4. Load each model (each model has N seed files, average within a seed) + predict
    print("\nLoading models + predicting...")
    y_blend = np.zeros(len(df_rank_feat), dtype="float64")
    total_weight = 0.0

    # v2: support median blending
    # First collect the seed-averaged prediction of each model
    model_seed_avg = {}
    for model_name in model_names:
        ext = {"xgb": "json", "lgb": "txt", "cat": "cbm"}[model_name]
        seed_files = sorted(models_dir.glob(f"ensemble_{model_name}__seed_*.{ext}"))

        if not seed_files:
            print(f"  Cannot find ensemble_{model_name}__seed_*.{ext}, skipping")
            continue

        # v2: try/except per seed, skip bad seeds so they do not crash the whole inference
        seed_preds = []
        for sf in seed_files:
            try:
                wrapper = _load_model_lazy(model_name, str(sf))
                seed_preds.append(wrapper.predict(X_enc))
            except Exception as e:
                print(f"  Loading/inference {sf.name} failed: {e}, skipping this seed")

        if not seed_preds:
            print(f"  All seeds of {model_name} failed, skipping")
            continue

        y_pred_avg = np.mean(seed_preds, axis=0)
        model_seed_avg[model_name] = y_pred_avg
        print(f"  ✓ {model_name}: {len(seed_preds)}/{len(seed_files)} seed average,"
              f" pred range=[{y_pred_avg.min():.0f}, {y_pred_avg.max():.0f}]")

    if not model_seed_avg:
        raise RuntimeError("No model was loaded successfully")

    # Blend according to blend_mode
    if blend_mode == "median":
        preds_matrix = np.column_stack(list(model_seed_avg.values()))
        y_blend = np.median(preds_matrix, axis=1)
        print(f"  Using median blending, {len(model_seed_avg)} models")
    else:
        # mean weighting
        y_blend = np.zeros(len(X_enc), dtype="float64")
        total_weight = 0.0
        for model_name, y_pred_avg in model_seed_avg.items():
            weight = weights.get(model_name, 0)
            y_blend += weight * y_pred_avg
            total_weight += weight
        if total_weight > 0:
            y_blend = y_blend / total_weight
        print(f"  Using mean weighting, {len(model_seed_avg)} models")

    if total_weight == 0:
        raise RuntimeError("No model was loaded successfully")
    y_blend = y_blend / total_weight

    # 5. Clip
    y_blend = np.clip(y_blend, clip_min, clip_max)

    # 6. Align to the submission template
    print(f"\nAligning to the submission template...")
    template = load_submission_template(data_dir)
    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_feat["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_blend,
    })
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"  {n_missing} MVT_ID values were not matched, filling with the global median")
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(float(np.nanmedian(y_blend)))

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    # 7. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)

    print(f"\n✅ Submission file generated: {output_path}")
    print(f"  Total rows: {len(submission):,}")
    print(f"  Prediction range: [{submission['TAXITIME_SEC_mvt'].min():.1f}, {submission['TAXITIME_SEC_mvt'].max():.1f}] seconds")
    print(f"  Prediction median: {submission['TAXITIME_SEC_mvt'].median():.1f} seconds")

    return submission


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 ensemble inference")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="submissions/jubilant-puzzle_v1.parquet")
    args = parser.parse_args()

    predict_ensemble(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
