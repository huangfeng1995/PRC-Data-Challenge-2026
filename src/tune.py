"""
Optuna 自动调参脚本
===================

从 2025 第 2 名 jubilant-vase 学到:
"Hyperparameter optimization via Optuna with K-fold CV"

策略:
- 用 Optuna 搜索 XGBoost 超参(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda)
- 目标:在 winter 主验证集上的 RMSE
- 30 次 trials(平衡时间和收益)
- 完成后输出最优参数,可以直接传给 train_xgb.py / train_ensemble.py

使用:
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


# Optuna 默认不开 verbose,自己控制
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_xgb(trial: optuna.Trial, X_tr, y_tr, X_va, y_va) -> float:
    """单次 trial:训练 XGBoost 并返回验证 RMSE。"""
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
    """单次 trial:训练 LightGBM 并返回验证 RMSE。"""
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
    n_trials: int = 50,  # v2:50 是甜蜜点(参考 2025 第 3 名 v18 160 trials 过拟合案例)
    subsample: float = 0.25,  # v2:默认用 25% 数据调参,避免超时
    tracking_uri: str = "mlruns",
    experiment_name: str = "prc-2026-tuning",
    cache_path: str | None = None,
) -> dict:
    """跑 Optuna 调参。"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_trials", n_trials)

        print("=" * 70)
        print("1. 加载数据 + 构造特征...")
        if cache_path and Path(cache_path).exists():
            print(f"  使用缓存: {cache_path}")
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
                print(f"  保存缓存: {cache_path}")
                df.to_parquet(cache_path, index=False)
        df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
        df = df[(df["TAXITIME_SEC_mvt"] > 30) & (df["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
        print(f"过滤后样本: {len(df):,}")
        print(f"特征数: {len(feat_cols)}")

        _clear_cat_cache()
        X = df[feat_cols].copy()
        X = _encode_categoricals(X, fit=True)
        y = df["TAXITIME_SEC_mvt"]

        # 用 winter 主验证
        split = time_split_winter(df)
        X_tr_full, y_tr_full = X[split.train_mask], y[split.train_mask]
        X_va, y_va = X[split.val_mask], y[split.val_mask]

        # v2:subsample 调参(默认 25%,可在 CLI 改)
        if subsample < 1.0:
            n_sub = int(len(X_tr_full) * subsample)
            X_tr = X_tr_full.sample(n=n_sub, random_state=42).sort_index()
            y_tr = y_tr_full.loc[X_tr.index]
            print(f"训练(调参): {len(X_tr):,} (subsample={subsample}), 验证: {len(X_va):,}")
        else:
            X_tr, y_tr = X_tr_full, y_tr_full
            print(f"训练: {len(X_tr):,}, 验证: {len(X_va):,}")

        print(f"\n2. 启动 Optuna 调参({model_type}, {n_trials} trials)...")

        # 选 objective
        if model_type == "xgb":
            objective_fn = lambda trial: objective_xgb(trial, X_tr, y_tr, X_va, y_va)
        elif model_type == "lgb":
            objective_fn = lambda trial: objective_lgb(trial, X_tr, y_tr, X_va, y_va)
        else:
            raise ValueError(f"未知 model_type: {model_type}")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

        print(f"\n3. 调参完成!")
        print(f"   最优 RMSE: {study.best_value:.4f}")
        print(f"   最优参数:")
        for k, v in study.best_params.items():
            print(f"     {k}: {v}")

        # 保存结果
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

        # 记录所有 trials
        trials_df = study.trials_dataframe()
        trials_csv = results_dir / f"tuned_{model_type}_trials.csv"
        trials_df.to_csv(trials_csv, index=False)
        mlflow.log_artifact(str(trials_csv))

        print(f"\n✅ 调参结果已保存:")
        print(f"   - {result_path}")
        print(f"   - {trials_csv}")

        return {
            "run_id": run.info.run_id,
            "best_rmse": study.best_value,
            "best_params": study.best_params,
        }


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 Optuna 调参")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--feature-set", default="all", choices=["basic", "domain", "aggregate", "all"])
    parser.add_argument("--model-type", default="xgb", choices=["xgb", "lgb"])
    parser.add_argument("--n-trials", type=int, default=50, help="默认 50 trials(v2 推荐)")
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--subsample", type=float, default=0.25,
                        help="v2:只用训练数据的 N 做调参(默认 0.25 = 25%),避免 200 万行跑 17 小时")
    parser.add_argument("--cache", default=None, help="features 缓存 parquet 路径(跳过 lag/lead 重算)")
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
    print(f"\n📊 调参结果:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
