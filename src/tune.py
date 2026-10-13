"""
Optuna auto-tuning script
=========================

Learned from the 2nd place of 2025 jubilant-vase:
"Hyperparameter optimization via Optuna with K-fold CV"

Strategy:
- Use Optuna to search the XGBoost hyperparameters (learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda)
- Objective: RMSE on the winter main validation set
- 30 trials (balancing time and gain)
- After it finishes, output the best parameters, which can be passed directly to train_xgb.py / train_ensemble.py

Usage:
    python -m src.tune --data-dir data/raw --feature-set all --n-trials 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd

from src.data_loader import load_training
from src.features import BASELINE_FEATURE_COLS
from src.train_xgb import _clear_cat_cache, _encode_categoricals
from src.train_ensemble import build_full_features
from src.validation import evaluate, time_split_winter


# Optuna verbose is off by default, we control it ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_xgb(trial: optuna.Trial, X_tr, y_tr, X_va, y_va) -> float:
    """A single trial: train XGBoost and return the validation RMSE."""
    import xgboost as xgb

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "nthread": -1,
        "verbosity": 0,
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
    }

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    y_pred = booster.predict(dval)
    metrics = evaluate(y_va.values, y_pred)
    return metrics["rmse"]


def objective_lgb(trial: optuna.Trial, X_tr, y_tr, X_va, y_va) -> float:
    """A single trial: train LightGBM and return the validation RMSE."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_threads": -1,
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0, log=True),
    }

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature="auto")
    val_set = lgb.Dataset(X_va, label=y_va, categorical_feature="auto", reference=train_set)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    y_pred = booster.predict(X_va, num_iteration=booster.best_iteration)
    metrics = evaluate(y_va.values, y_pred)
    return metrics["rmse"]


def run_tuning(
    data_dir: str = "data/raw",
    feature_set: str = "all",
    model_type: str = "xgb",
    n_trials: int = 50,  # v2: 50 is the sweet spot (referencing the 2025 3rd place v18 case where 160 trials overfit)
    subsample: float = 0.25,  # v2: tune on 25% of the data by default to avoid a timeout
    tracking_uri: str = "mlruns",
    experiment_name: str = "prc-2026-tuning",
    cache_path: str | None = None,
) -> dict:
    """Run Optuna tuning."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_trials", n_trials)

        print("=" * 70)
        print("1. Loading data + building features...")
        if cache_path and Path(cache_path).exists():
            print(f"  Using cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            feat_cols = [c for c in df.columns
                         if c in BASELINE_FEATURE_COLS or
                         (c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)) or
                         c in ["stand_zone", "runway_family", "runway_side"] or
                         c.endswith("_te")]
        else:
            df = load_training(data_dir)
            df, feat_cols = build_full_features(df, feature_set=feature_set)
            if cache_path:
                print(f"  Saving cache: {cache_path}")
                df.to_parquet(cache_path, index=False)
        df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
        df = df[(df["TAXITIME_SEC_mvt"] > 30) & (df["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
        print(f"Samples after filtering: {len(df):,}")
        print(f"Number of features: {len(feat_cols)}")

        _clear_cat_cache()
        X = df[feat_cols].copy()
        X = _encode_categoricals(X, fit=True)
        y = df["TAXITIME_SEC_mvt"]

        # Use the winter main validation
        split = time_split_winter(df)
        X_tr_full, y_tr_full = X[split.train_mask], y[split.train_mask]
        X_va, y_va = X[split.val_mask], y[split.val_mask]

        # v2: subsample for tuning (25% by default, changeable via CLI)
        if subsample < 1.0:
            n_sub = int(len(X_tr_full) * subsample)
            X_tr = X_tr_full.sample(n=n_sub, random_state=42).sort_index()
            y_tr = y_tr_full.loc[X_tr.index]
            print(f"Train (tuning): {len(X_tr):,} (subsample={subsample}), val: {len(X_va):,}")
        else:
            X_tr, y_tr = X_tr_full, y_tr_full
            print(f"Train: {len(X_tr):,}, val: {len(X_va):,}")

        print(f"\n2. Starting Optuna tuning ({model_type}, {n_trials} trials)...")

        # Select the objective
        if model_type == "xgb":
            objective_fn = lambda trial: objective_xgb(trial, X_tr, y_tr, X_va, y_va)
        elif model_type == "lgb":
            objective_fn = lambda trial: objective_lgb(trial, X_tr, y_tr, X_va, y_va)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

        print(f"\n3. Tuning finished!")
        print(f"   Best RMSE: {study.best_value:.4f}")
        print(f"   Best parameters:")
        for k, v in study.best_params.items():
            print(f"     {k}: {v}")

        # Save the results
        results_dir = Path("models")
        results_dir.mkdir(exist_ok=True)
        result_path = results_dir / f"tuned_{model_type}_params.json"
        result_path.write_text(json.dumps({
            "model_type": model_type,
            "feature_set": feature_set,
            "best_rmse": study.best_value,
            "best_params": study.best_params,
        }, indent=2))
        mlflow.log_artifact(str(result_path))
        mlflow.log_metric("best_rmse", study.best_value)

        # Record all trials
        trials_df = study.trials_dataframe()
        trials_csv = results_dir / f"tuned_{model_type}_trials.csv"
        trials_df.to_csv(trials_csv, index=False)
        mlflow.log_artifact(str(trials_csv))

        print(f"\n✅ Tuning results saved:")
        print(f"   - {result_path}")
        print(f"   - {trials_csv}")

        return {
            "run_id": run.info.run_id,
            "best_rmse": study.best_value,
            "best_params": study.best_params,
        }


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 Optuna tuning")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--feature-set", default="all", choices=["basic", "domain", "aggregate", "all"])
    parser.add_argument("--model-type", default="xgb", choices=["xgb", "lgb"])
    parser.add_argument("--n-trials", type=int, default=50, help="default 50 trials (recommended in v2)")
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--subsample", type=float, default=0.25,
                        help="v2: only use N of the training data for tuning (default 0.25 = 25%), avoids 17 hours on 2 million rows")
    parser.add_argument("--cache", default=None, help="path to the features cache parquet (skips recomputing lag/lead)")
    args = parser.parse_args()

    result = run_tuning(
        data_dir=args.data_dir,
        feature_set=args.feature_set,
        model_type=args.model_type,
        n_trials=args.n_trials,
        subsample=args.subsample,
        tracking_uri=args.tracking_uri,
        cache_path=args.cache,
    )
    print(f"\nTuning results:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
