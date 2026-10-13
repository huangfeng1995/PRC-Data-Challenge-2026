"""
LightGBM baseline training script
=================================

Functionality:
1. Load training data
2. Run basic feature engineering
3. Evaluate on winter / summer / kfold three validation sets
4. Record all experiments with MLflow (params + metrics + model artifact)
5. Train the final model (full 12 months) and save to models/

Run:
    python -m src.train_lgb --data-dir data/raw

Visualize MLflow:
    mlflow ui  # then open http://localhost:5000 in the browser
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.data_loader import load_training
from src.features import BASELINE_FEATURE_COLS, build_features
from src.validation import evaluate, get_validation_splits


# ---------------------------------------------------------------------------
# Default hyperparameters (medium-sized dataset, this is a conservative starting point)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": -1,  # use all CPU cores
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
) -> tuple[lgb.Booster, dict]:
    """Train one LightGBM model, using early stopping on the validation set. Returns booster and validation metrics."""
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature="auto")
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature="auto", reference=train_set)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    y_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    metrics = evaluate(y_val.values, y_pred)
    metrics["best_iter"] = booster.best_iteration
    return booster, metrics


def run_experiment(
    data_dir: str = "data/raw",
    experiment_name: str = "prc-2026-baseline",
    params: dict | None = None,
    tracking_uri: str = "mlruns",
) -> dict:
    """Run one full experiment: load -> features -> multi-split validation -> MLflow logging.

    Returns the metrics dict of the last experiment.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    # MLflow setup
    mlflow.set_tracking_uri(tracking_uri)  # local sqlite or file directory
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        # 1. Load + features
        print("=" * 70)
        print("1. Loading training data...")
        df = load_training(data_dir)
        df = build_features(df)

        # Drop rows with missing labels
        df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
        # Outlier filtering: taxi-out values as extreme as negative or > 2 hours are basically data noise
        df = df[(df["TAXITIME_SEC_mvt"] > 30) & (df["TAXITIME_SEC_mvt"] < 7200)].reset_index(drop=True)
        print(f"training samples after filtering: {len(df):,}")

        # 2. Prepare X, y
        feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
        print(f"\nfeatures used ({len(feat_cols)}): {feat_cols}")

        X = df[feat_cols]
        y = df["TAXITIME_SEC_mvt"]

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
            print(f"  train: {len(X_tr):,} rows, val: {len(X_va):,} rows")

            booster, metrics = train_one_split(X_tr, y_tr, X_va, y_va, params)
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

            booster, metrics = train_one_split(X_tr, y_tr, X_va, y_va, params)
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
        mean_best_iter = int(np.mean([m.get("best_iter", 500) for m in all_metrics.values()
                                       if isinstance(m, dict) and "best_iter" in m]))
        print(f"\n3. Training final model (full data, {mean_best_iter} iterations)...")
        full_train = lgb.Dataset(X, label=y, categorical_feature="auto")
        final_booster = lgb.train(
            params,
            full_train,
            num_boost_round=mean_best_iter,
            callbacks=[lgb.log_evaluation(period=200)],
        )

        # 6. Save model + record feature importance
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "baseline_lgb.txt"
        final_booster.save_model(str(model_path))
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # Feature importance (top 30)
        importance_df = pd.DataFrame({
            "feature": feat_cols,
            "importance_gain": final_booster.feature_importance(importance_type="gain"),
            "importance_split": final_booster.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)
        importance_path = models_dir / "baseline_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), artifact_path="feature_importance")

        print(f"\nTop 10 important features:\n{importance_df.head(10).to_string(index=False)}")

        # 7. Record into MLflow "tag", to make experiment filtering easier
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("features", "basic_no_aggregate")

        # Output run id
        print(f"\nexperiment complete, MLflow run_id = {run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "metrics": all_metrics,
            "model_path": str(model_path),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRC 2026 LightGBM baseline")
    parser.add_argument("--data-dir", default="data/raw", help="training data directory")
    parser.add_argument("--experiment-name", default="prc-2026-baseline", help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default="mlruns", help="MLflow backend URI")
    parser.add_argument("--learning-rate", type=float, default=None, help="override the default learning rate")
    parser.add_argument("--num-leaves", type=int, default=None, help="override the default num_leaves")
    args = parser.parse_args()

    # Override hyperparameters (optional)
    overrides = {}
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.num_leaves is not None:
        overrides["num_leaves"] = args.num_leaves

    result = run_experiment(
        data_dir=args.data_dir,
        experiment_name=args.experiment_name,
        params=overrides,
        tracking_uri=args.tracking_uri,
    )
    print(f"\nfinal result:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()