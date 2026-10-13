"""
XGBoost baseline training script
===============================

Functionality:
1. Load training data
2. Run basic feature engineering
3. Evaluate on winter / summer / kfold three validation sets
4. Record all experiments with MLflow (params + metrics + model artifact)
5. Train the final model (full 12 months) and save to models/

Why XGBoost instead of LightGBM:
- On macOS, LightGBM needs libomp (a system library), which is painful to install
- XGBoost works on macOS with just pip install, no system dependencies needed
- Tree model performance is nearly identical, APIs are very similar

Run:
    python -m src.train_xgb --data-dir data/raw

Visualize MLflow:
    mlflow ui  # then open http://localhost:5000 in the browser
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_training
from src.features import BASELINE_FEATURE_COLS, build_features
from src.features_aggregate import build_features_v2
from src.validation import evaluate, get_validation_splits


# ---------------------------------------------------------------------------
# Default hyperparameters (medium-sized dataset, this is a conservative starting point)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective": "reg:squarederror",  # corresponds to RMSE
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,  # roughly corresponds to LightGBM's num_leaves=127
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "tree_method": "hist",  # fastest algorithm on CPU
    "nthread": -1,  # use all CPU cores
    "verbosity": 0,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Train + validate function
# ---------------------------------------------------------------------------

def train_one_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    sample_weight: np.ndarray | None = None,
) -> tuple[xgb.Booster, dict]:
    """Train one XGBoost model, using early stopping on the validation set. Returns booster and validation metrics.

    v3.7: supports sample_weight (long-tail weighting). None = equal weights (default).
    """
    # XGBoost requires all category columns to be converted to numeric encoding (lightgbm natively supports category, xgb does not)
    X_tr = _encode_categoricals(X_train, fit=True)
    X_va = _encode_categoricals(X_val, fit=False)

    dtrain = xgb.DMatrix(X_tr, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_va, label=y_val)

    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=200,
    )

    y_pred = booster.predict(dval)
    metrics = evaluate(y_val.values, y_pred)
    metrics["best_iter"] = int(booster.best_iteration) + 1
    return booster, metrics


# ---------------------------------------------------------------------------
# Categorical column encoding (XGBoost does not support native category)
# ---------------------------------------------------------------------------

# Module-level cache, avoid re-fitting on every split
_CAT_MAPPINGS: dict[str, dict] = {}


def _encode_categoricals(X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Encode category columns as integers. When fit=True learn the mapping, when fit=False reuse the existing mapping.

    WARNING: fit at training time, do NOT fit at inference time, otherwise you get "category encoding at inference is inconsistent with training".
    v0.1 fix: also store the cat mappings to the _CAT_MAPPINGS_PATH file, keeping them consistent across processes.
    """
    global _CAT_MAPPINGS
    # If the mapping is empty (cross-process), try loading from the persisted file
    if not _CAT_MAPPINGS:
        if _CAT_MAPPINGS_PATH is not None and _CAT_MAPPINGS_PATH.exists():
            _CAT_MAPPINGS = json.loads(_CAT_MAPPINGS_PATH.read_text())

    X = X.copy()
    for col in X.select_dtypes(include="category").columns:
        if fit:
            # Learn the mapping, NaN is represented by -1
            cats = X[col].cat.categories.tolist()
            _CAT_MAPPINGS[col] = {c: i for i, c in enumerate(cats)}
            _CAT_MAPPINGS[col]["__nan__"] = -1
            # Persist (if save_path is set)
            if _CAT_MAPPINGS_PATH is not None:
                _CAT_MAPPINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CAT_MAPPINGS_PATH.write_text(json.dumps(_CAT_MAPPINGS))

        mapping = _CAT_MAPPINGS.get(col, {})
        X[col] = X[col].astype(str)  # convert to str first, avoid categorical setitem error
        X[col] = X[col].map(mapping).fillna(-1).astype("int32")

    return X


_CAT_MAPPINGS_PATH: Path | None = None  # v0.1: persist cat mappings across processes


def _clear_cat_cache() -> None:
    """Clear the cache, called before running a new experiment."""
    _CAT_MAPPINGS.clear()


def _select_feat_cols_from_df(df: pd.DataFrame, feature_set: str) -> list[str]:
    """v3.5: unified feature_set -> column name mapping (used by both cache loading and new training)."""
    if feature_set == "basic":
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    elif feature_set == "aggregate":
        agg_cols = [c for c in df.columns
                    if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
        return agg_cols
    elif feature_set == "all":
        agg_cols = [c for c in df.columns
                    if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns] + agg_cols
    elif feature_set == "no_laglead":
        # v3.5: Phase 1 - keep only basic + cnt_* (no mean/p90, avoid target leak)
        keep_agg_cols = [c for c in df.columns if c.endswith("m") and "cnt_" in c]
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns] + keep_agg_cols
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(
    data_dir: str = "data/raw",
    experiment_name: str = "prc-2026-xgb-baseline",
    params: dict | None = None,
    tracking_uri: str = "mlruns",
    feature_set: str = "basic",  # "basic" | "aggregate" | "all" | "no_laglead"
    cache_path: str | None = None,  # cache features parquet, skip lag/lead recomputation
    tail_quantile: float = 0.99,
    tail_weight: float = 2.0,
    model_version: str = "v0",  # v0: version number
    stage: str = "val",  # "val": holdout validation model (drop certain months); "final": full 12-month training
    holdout_months: list[int] | None = None,  # only effective when stage="val": months removed from the training set
    feat_cols_override: list[str] | None = None,  # v0: explicitly specify feature columns (skip feature_set inference)
) -> dict:
    """Run one full experiment: load -> features -> multi-split validation -> MLflow logging.

    Parameters:
        feature_set: which feature group to use
            - "basic": only basic features (no aggregation), baseline uses this
            - "aggregate": only aggregated congestion features
            - "all": basic + aggregate (contains mean/p90 leak risk, do not use in production)
            - "no_laglead": basic + cnt_dep_* (mean/p90 removed)
        tail_quantile: long-tail threshold (default 0.99 = top 1%)
        tail_weight: long-tail sample weight (default 2.0)
        model_version: version number (e.g. v0 / v7)
        stage: "val" = train with holdout_months removed; "final" = full training
        holdout_months: list of months removed from training when stage="val" (default [1, 7])
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    _clear_cat_cache()

    # MLflow setup
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("tail_quantile", tail_quantile)
        mlflow.log_param("tail_weight", tail_weight)
        mlflow.log_param("stage", stage)
        mlflow.log_param("model_version", model_version)
        if holdout_months is not None:
            mlflow.log_param("holdout_months", ",".join(map(str, holdout_months)))

        # 1. Load + features
        print("=" * 70)
        print("1. Loading training data...")

        # v0.1: set the cat mapping persistence path (saved together with the model) - reused for cross-process inference
        global _CAT_MAPPINGS_PATH
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        seed = params.get("seed", 42)
        _CAT_MAPPINGS_PATH = models_dir / "feature_lists" / f"xgb_{model_version}_{feature_set}_seed{seed}_{stage}_cat_mappings.json"
        # Remove old cat mappings (let fit learn them again)
        if _CAT_MAPPINGS_PATH.exists():
            _CAT_MAPPINGS_PATH.unlink()

        if cache_path and Path(cache_path).exists():
            print(f"  Using cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            # v3.5: also filter by feature_set when loading from cache (avoid mean/p90 leaking into no_laglead experiments)
            if feat_cols_override is not None:
                feat_cols = [c for c in feat_cols_override if c in df.columns]
            else:
                feat_cols = _select_feat_cols_from_df(df, feature_set)
        else:
            df = load_training(data_dir)

            if feature_set == "basic":
                df = build_features(df)
                feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
            elif feature_set == "aggregate":
                from src.features_aggregate import add_aggregate_features
                df = build_features(df)
                df = add_aggregate_features(df)
                # only use aggregate features
                feat_cols = [c for c in df.columns
                             if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
            elif feature_set == "all":
                df = build_features_v2(df, include_aggregate=True, include_lead=True)
                agg_cols = [c for c in df.columns
                            if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
                feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns] + agg_cols
            elif feature_set == "no_laglead":
                # v3.5: Phase 1 - keep only basic + cnt_* (no target-derived lag/lead)
                df = build_features_v2(df, include_aggregate=True, include_lead=True)
                keep_agg_cols = [c for c in df.columns
                                 if c.endswith("m") and "cnt_" in c]
                feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns] + keep_agg_cols
                print(f"  [no_laglead] keeping {len(keep_agg_cols)} cnt columns, dropping all mean/p90")
            elif feat_cols_override is not None:
                # v0: user-defined feature set (e.g. basic30 / cnt42), feature_set is only a naming label
                # By default take the no_laglead path to ensure cnt columns exist, then filter with the override
                df = build_features_v2(df, include_aggregate=True, include_lead=True)
                feat_cols = [c for c in feat_cols_override if c in df.columns]
                print(f"  [feat_cols_override] using {len(feat_cols)} columns")
            else:
                raise ValueError(f"unknown feature_set: {feature_set}")

            # Save to cache
            if cache_path:
                print(f"  Saving cache: {cache_path}")
                df.to_parquet(cache_path, index=False)

        # v0: when stage="val", remove holdout_months from training (default [1, 7])
        if stage == "val":
            if holdout_months is None:
                holdout_months = [1, 7]
            n_before = len(df)
            sched = df["SCHED_TIME_UTC_mvt"]
            month_mask = sched.dt.month.isin(holdout_months)
            df_train_only = df[~month_mask].reset_index(drop=True)
            print(f"\n  [stage=val] removing holdout months from training: {holdout_months}")
            print(f"    train: {len(df_train_only):,} (before removal {n_before:,})")
            print(f"    samples in holdout months: {month_mask.sum():,}")
            df = df_train_only

        # Drop rows with missing labels
        df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
        # v2 outlier filtering (found from EDA: negatives like -12s, extreme value 131167s = 36 hours)
        # > 0 filters out negatives, < 10800 (3 hours) filters extremes, covering 99.99% of real data
        df = df[(df["TAXITIME_SEC_mvt"] > 0) & (df["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
        print(f"training samples after filtering: {len(df):,}")
        mlflow.log_param("n_train_samples", len(df))

        # 2. Prepare X, y
        print(f"\nfeatures used ({len(feat_cols)}): {feat_cols}")

        X = df[feat_cols]
        y = df["TAXITIME_SEC_mvt"]

        # v3.7: long-tail weighting - give top 1% long-tail samples a tail_weight weight
        # Rationale: RMSE is dominated by large long-tail errors, weighting pushes the model to care more about the tail
        if tail_weight != 1.0:
            threshold = y.quantile(tail_quantile)
            sample_weight = np.where(y > threshold, tail_weight, 1.0).astype("float32")
            n_tail = (y > threshold).sum()
            print(f"\n[v3.7 long-tail weighting] threshold p{tail_quantile*100:.0f}={threshold:.0f}s, "
                  f"{n_tail:,} rows ({n_tail/len(y)*100:.1f}%) weight={tail_weight}, "
                  f"remaining weight=1.0")
        else:
            sample_weight = None

        # 3. Multi-split validation
        print("\n2. Running multi-split validation...")
        splits = get_validation_splits(df)

        all_metrics = {}
        for split_name, split in [
            ("winter_main", splits["winter"]),
            ("summer_aux", splits["summer"]),
        ]:
            print(f"\n--- split: {split_name} ({split.val_period}) ---")
            X_tr, y_tr = X[split.train_mask], y[split.train_mask]
            X_va, y_va = X[split.val_mask], y[split.val_mask]
            # v3.7: sample_weight is also split by train_mask (weight only the train part)
            sw_tr = sample_weight[split.train_mask] if sample_weight is not None else None
            print(f"  train: {len(X_tr):,} rows, val: {len(X_va):,} rows")

            booster, metrics = train_one_split(X_tr, y_tr, X_va, y_va, params, sample_weight=sw_tr)
            print(f"  validation metrics: {metrics}")

            for k, v in metrics.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
            all_metrics[split_name] = metrics

        # 4. K-fold time series CV (only look at average RMSE)
        print(f"\n--- K-fold time series CV ({len(splits['kfold'])} folds) ---")
        cv_rmses = []
        for i, split in enumerate(splits["kfold"], 1):
            X_tr, y_tr = X[split.train_mask], y[split.train_mask]
            X_va, y_va = X[split.val_mask], y[split.val_mask]
            sw_tr = sample_weight[split.train_mask] if sample_weight is not None else None

            booster, metrics = train_one_split(X_tr, y_tr, X_va, y_va, params, sample_weight=sw_tr)
            cv_rmses.append(metrics["rmse"])
            print(f"  Fold {i} ({split.val_period}): RMSE={metrics['rmse']:.2f}")
            mlflow.log_metric(f"kfold_fold{i}_rmse", metrics["rmse"])

        cv_mean = float(np.mean(cv_rmses))
        cv_std = float(np.std(cv_rmses))
        mlflow.log_metric("kfold_mean_rmse", cv_mean)
        mlflow.log_metric("kfold_std_rmse", cv_std)
        all_metrics["kfold_mean_rmse"] = cv_mean
        all_metrics["kfold_std_rmse"] = cv_std
        print(f"\nK-fold mean RMSE: {cv_mean:.2f} +/- {cv_std:.2f}")

        # 5. Train the final model (full 12 months, no validation early stopping)
        # Use the K-fold mean best_iter as a reference
        mean_best_iter = int(np.mean([
            m.get("best_iter", 500)
            for m in all_metrics.values()
            if isinstance(m, dict) and "best_iter" in m
        ]))
        print(f"\n3. Training final model (full data, {mean_best_iter} iterations)...")
        X_enc = _encode_categoricals(X, fit=True)
        dfull = xgb.DMatrix(X_enc, label=y, weight=sample_weight)
        final_booster = xgb.train(
            params,
            dfull,
            num_boost_round=mean_best_iter,
            verbose_eval=200,
        )

        # 6. Save model + record feature importance
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        # v0: versioned file name (stage0: validation model; final: 12-month training) - forcibly forbid overwriting baseline_xgb.json
        # Model name format: models/xgb_{version}_{feature_set}_seed{seed}_{stage}.json
        seed = params.get("seed", 42)
        model_name = f"xgb_{model_version}_{feature_set}_seed{seed}_{stage}.json"
        model_path = models_dir / model_name
        final_booster.save_model(str(model_path))
        # v0: bind model <-> feature columns
        feat_list_dir = models_dir / "feature_lists"
        feat_list_dir.mkdir(exist_ok=True, parents=True)
        feat_list_path = feat_list_dir / f"{model_name.replace('.json', '.json')}"
        # The above approach has a bug, fix: directly write the features.json file name
        feat_list_file = feat_list_dir / f"{model_name.replace('.json', '')}_features.json"
        feat_list_file.write_text(json.dumps({
            "model_name": model_name,
            "feature_set": feature_set,
            "stage": stage,
            "model_version": model_version,
            "seed": seed,
            "n_features": len(feat_cols),
            "features": list(feat_cols),
        }, indent=2))
        print(f"  model saved: {model_path}")
        print(f"  feature column list: {feat_list_file}")
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(feat_list_file), artifact_path="features_meta")

        # Feature importance
        importance = final_booster.get_score(importance_type="gain")
        importance_df = pd.DataFrame([
            {"feature": k, "importance_gain": v}
            for k, v in importance.items()
        ]).sort_values("importance_gain", ascending=False)
        importance_path = feat_list_dir / f"{model_name.replace('.json', '')}_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), artifact_path="feature_importance")

        print(f"\nTop 10 important features:\n{importance_df.head(10).to_string(index=False)}")

        # 7. Record into MLflow "tag", to make experiment filtering easier
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("features", "basic_no_aggregate")
        mlflow.set_tag("model", "xgboost")

        # Output run id
        print(f"\n✅ experiment complete, MLflow run_id = {run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "metrics": all_metrics,
            "model_path": str(model_path),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRC 2026 XGBoost baseline")
    parser.add_argument("--data-dir", default="data/raw", help="training data directory")
    parser.add_argument("--experiment-name", default="prc-2026-xgb-baseline", help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default="mlruns", help="MLflow backend URI")
    parser.add_argument("--learning-rate", type=float, default=None, help="override the default learning rate")
    parser.add_argument("--max-depth", type=int, default=None, help="override the default max_depth")
    parser.add_argument(
        "--feature-set",
        default="basic",
        help="feature set label (only used for naming). Values: basic / basic30 / aggregate / all / no_laglead / cnt42 etc. "
             "The actual filtering rule is controlled by --feat-cols (default basic=BASELINE_FEATURE_COLS)",
    )
    parser.add_argument(
        "--feat-cols",
        default=None,
        help="v0: custom feature column list (JSON list file). If not provided, inferred from feature_set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default 42, pass 0 when the user specifies seed0)",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="features cache parquet path (skip lag/lead recomputation)",
    )
    parser.add_argument(
        "--tail-quantile", type=float, default=0.99,
        help="v3.7: long-tail weighting threshold (default 0.99 = top 1%)",
    )
    parser.add_argument(
        "--tail-weight", type=float, default=2.0,
        help="v3.7: long-tail sample weight (default 2.0, 1.0 = no weighting)",
    )
    parser.add_argument(
        "--model-version", type=str, default="v0",
        help="v0: model version number (affects file name, e.g. v0 / v7)",
    )
    parser.add_argument(
        "--stage", type=str, default="val", choices=["val", "final"],
        help="val: train with holdout months removed; final: full 12-month training",
    )
    parser.add_argument(
        "--holdout-months", type=int, nargs="+", default=None,
        help="months removed when stage=val (default [1, 7])",
    )
    args = parser.parse_args()

    # Override hyperparameters (optional)
    overrides = {}
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.max_depth is not None:
        overrides["max_depth"] = args.max_depth
    overrides["seed"] = args.seed

    # Load feat-cols (if provided)
    feat_cols_override = None
    if args.feat_cols is not None and Path(args.feat_cols).exists():
        feat_cols_override = json.loads(Path(args.feat_cols).read_text())

    result = run_experiment(
        data_dir=args.data_dir,
        experiment_name=args.experiment_name,
        params=overrides,
        tracking_uri=args.tracking_uri,
        feature_set=args.feature_set,
        cache_path=args.cache,
        tail_quantile=args.tail_quantile,
        tail_weight=args.tail_weight,
        model_version=args.model_version,
        stage=args.stage,
        holdout_months=args.holdout_months,
        feat_cols_override=feat_cols_override,
    )
    print(f"\n📊 final result:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()