"""
LightGBM baseline 训练脚本
==========================

功能:
1. 加载训练数据
2. 跑基础特征工程
3. 在 winter / summer / kfold 三个验证集上评估
4. 用 MLflow 记录所有实验(参数 + 指标 + 模型 artifact)
5. 训练最终模型(全量 12 个月)并保存到 models/

运行:
    python -m src.train_lgb --data-dir data/raw

可视化 MLflow:
    mlflow ui  # 然后浏览器打开 http://localhost:5000
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
# 默认超参(中等规模数据集,这是个保守起点)
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
    "num_threads": -1,  # 用所有 CPU 核心
}


# ---------------------------------------------------------------------------
# 训练 + 验证函数
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
    """训练一个 LightGBM 模型,用验证集早停。返回 booster 和验证指标。"""
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
    """跑一次完整实验:加载 → 特征 → 多切分验证 → MLflow 记录。

    返回最后一次实验的指标字典。
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    # MLflow 设置
    mlflow.set_tracking_uri(tracking_uri)  # 本地 sqlite 或文件目录
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        # 1. 加载 + 特征
        print("=" * 70)
        print("1. 加载训练数据...")
        df = load_training(data_dir)
        df = build_features(df)

        # 丢掉有缺失标签的行
        df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
        # 异常值过滤:taxi-out 极端到负数或 > 2 小时基本是数据噪声
        df = df[(df["TAXITIME_SEC_mvt"] > 30) & (df["TAXITIME_SEC_mvt"] < 7200)].reset_index(drop=True)
        print(f"过滤后训练样本: {len(df):,}")

        # 2. 准备 X, y
        feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
        print(f"\n使用的特征({len(feat_cols)} 个): {feat_cols}")

        X = df[feat_cols]
        y = df["TAXITIME_SEC_mvt"]

        # 3. 多切分验证
        print("\n2. 跑多切分验证...")
        splits = get_validation_splits(df)

        all_metrics = {}
        for split_name, split in [
            ("winter_main", splits["winter"]),
            ("summer_aux", splits["summer"]),
        ]:
            print(f"\n--- 切分: {split_name} ({split.val_period}) ---")
            X_tr, y_tr = X[split.train_mask], y[split.train_mask]
            X_va, y_va = X[split.val_mask], y[split.val_mask]
            print(f"  训练: {len(X_tr):,} 行, 验证: {len(X_va):,} 行")

            booster, metrics = train_one_split(X_tr, y_tr, X_va, y_va, params)
            print(f"  验证指标: {metrics}")

            for k, v in metrics.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
            all_metrics[split_name] = metrics

        # 4. K 折时序 CV(只看平均 RMSE)
        print(f"\n--- K 折时序 CV ({len(splits['kfold'])} 折) ---")
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
        print(f"\nK 折平均 RMSE: {cv_mean:.2f} ± {cv_std:.2f}")

        # 5. 训练最终模型(全量 12 个月,不用验证集早停)
        # 用 K 折平均 best_iter 作为参考
        mean_best_iter = int(np.mean([m.get("best_iter", 500) for m in all_metrics.values()
                                       if isinstance(m, dict) and "best_iter" in m]))
        print(f"\n3. 训练最终模型(全量数据,迭代 {mean_best_iter} 轮)...")
        full_train = lgb.Dataset(X, label=y, categorical_feature="auto")
        final_booster = lgb.train(
            params,
            full_train,
            num_boost_round=mean_best_iter,
            callbacks=[lgb.log_evaluation(period=200)],
        )

        # 6. 保存模型 + 记录 feature importance
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "baseline_lgb.txt"
        final_booster.save_model(str(model_path))
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # 特征重要性(top 30)
        importance_df = pd.DataFrame({
            "feature": feat_cols,
            "importance_gain": final_booster.feature_importance(importance_type="gain"),
            "importance_split": final_booster.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)
        importance_path = models_dir / "baseline_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), artifact_path="feature_importance")

        print(f"\nTop 10 重要特征:\n{importance_df.head(10).to_string(index=False)}")

        # 7. 记录到 MLflow 的"tag",方便筛选实验
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("features", "basic_no_aggregate")

        # 输出 run id
        print(f"\n✅ 实验完成,MLflow run_id = {run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "metrics": all_metrics,
            "model_path": str(model_path),
        }


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRC 2026 LightGBM baseline")
    parser.add_argument("--data-dir", default="data/raw", help="训练数据目录")
    parser.add_argument("--experiment-name", default="prc-2026-baseline", help="MLflow 实验名")
    parser.add_argument("--tracking-uri", default="mlruns", help="MLflow 后端 URI")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖默认学习率")
    parser.add_argument("--num-leaves", type=int, default=None, help="覆盖默认 num_leaves")
    args = parser.parse_args()

    # 覆盖超参(可选)
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
    print(f"\n📊 最终结果:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
