"""
XGBoost baseline 训练脚本
==========================

功能:
1. 加载训练数据
2. 跑基础特征工程
3. 在 winter / summer / kfold 三个验证集上评估
4. 用 MLflow 记录所有实验(参数 + 指标 + 模型 artifact)
5. 训练最终模型(全量 12 个月)并保存到 models/

为什么是 XGBoost 而不是 LightGBM:
- macOS 上 LightGBM 需要 libomp(系统库),安装麻烦
- XGBoost 在 macOS 上 pip install 即可,不需要任何系统依赖
- 树模型性能几乎一致,API 非常相近

运行:
    python -m src.train_xgb --data-dir data/raw

可视化 MLflow:
    mlflow ui  # 然后浏览器打开 http://localhost:5000
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
# 默认超参(中等规模数据集,这是个保守起点)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective": "reg:squarederror",  # 对应 RMSE
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,  # 跟 LightGBM 的 num_leaves=127 粗略对应
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "tree_method": "hist",  # CPU 上最快的算法
    "nthread": -1,  # 用所有 CPU 核心
    "verbosity": 0,
    "seed": 42,
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
    sample_weight: np.ndarray | None = None,
) -> tuple[xgb.Booster, dict]:
    """训练一个 XGBoost 模型,用验证集早停。返回 booster 和验证指标。

    v3.7:支持 sample_weight(长尾加权)。None = 等权(默认)。
    """
    # XGBoost 需要把所有 category 列转成数值编码(lightgbm 原生支持 category,xgb 不行)
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
# 类别列编码(XGBoost 不支持原生 category)
# ---------------------------------------------------------------------------

# 模块级缓存,避免在每个 split 都重新 fit
_CAT_MAPPINGS: dict[str, dict] = {}


def _encode_categoricals(X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """把 category 列用整数编码。fit=True 时学习映射,fit=False 时用已有映射。

    ⚠️ 注意:训练时 fit,推理时不能 fit,否则会出现"推理时类别编码跟训练不一致"的问题。
    v0.1 修复:同时把 cat 映射存到 _CAT_MAPPINGS_PATH 文件,跨进程保持一致。
    """
    global _CAT_MAPPINGS
    # 如果映射空(跨进程),尝试从持久化文件加载
    if not _CAT_MAPPINGS:
        if _CAT_MAPPINGS_PATH is not None and _CAT_MAPPINGS_PATH.exists():
            _CAT_MAPPINGS = json.loads(_CAT_MAPPINGS_PATH.read_text())

    X = X.copy()
    for col in X.select_dtypes(include="category").columns:
        if fit:
            # 学习映射,NaN 用 -1 表示
            cats = X[col].cat.categories.tolist()
            _CAT_MAPPINGS[col] = {c: i for i, c in enumerate(cats)}
            _CAT_MAPPINGS[col]["__nan__"] = -1
            # 持久化(如果有设 save_path)
            if _CAT_MAPPINGS_PATH is not None:
                _CAT_MAPPINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CAT_MAPPINGS_PATH.write_text(json.dumps(_CAT_MAPPINGS))

        mapping = _CAT_MAPPINGS.get(col, {})
        X[col] = X[col].astype(str)  # 先转 str,避免 categorical setitem 报错
        X[col] = X[col].map(mapping).fillna(-1).astype("int32")

    return X


_CAT_MAPPINGS_PATH: Path | None = None  # v0.1:跨进程持久化 cat 映射


def _clear_cat_cache() -> None:
    """清除缓存,在跑新实验前调用。"""
    _CAT_MAPPINGS.clear()


def _select_feat_cols_from_df(df: pd.DataFrame, feature_set: str) -> list[str]:
    """v3.5:统一 feature_set → 列名映射(同时给 cache 加载和新训练用)。"""
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
        # v3.5:Phase 1 — 只保留 basic + cnt_*(没 mean/p90,避免 target leak)
        keep_agg_cols = [c for c in df.columns if c.endswith("m") and "cnt_" in c]
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns] + keep_agg_cols
    else:
        raise ValueError(f"未知 feature_set: {feature_set}")


# ---------------------------------------------------------------------------
# 跑实验
# ---------------------------------------------------------------------------

def run_experiment(
    data_dir: str = "data/raw",
    experiment_name: str = "prc-2026-xgb-baseline",
    params: dict | None = None,
    tracking_uri: str = "mlruns",
    feature_set: str = "basic",  # "basic" | "aggregate" | "all" | "no_laglead"
    cache_path: str | None = None,  # 缓存 features parquet,跳过 lag/lead 重算
    tail_quantile: float = 0.99,
    tail_weight: float = 2.0,
    model_version: str = "v0",  # v0:版本号
    stage: str = "val",  # "val": holdout 验证模型(剔除某些月);"final": 12 月全量训练
    holdout_months: list[int] | None = None,  # 仅 stage="val" 时生效:从训练集中剔除
    feat_cols_override: list[str] | None = None,  # v0:显式指定特征列(跳过按 feature_set 推断)
) -> dict:
    """跑一次完整实验:加载 → 特征 → 多切分验证 → MLflow 记录。

    参数:
        feature_set: 用哪组特征
            - "basic": 只用基础特征(无聚合),baseline 用这个
            - "aggregate": 只用聚合拥堵特征
            - "all": 基础 + 聚合(含 mean/p90 leak 风险,不要在生产用)
            - "no_laglead": 基础 + cnt_dep_*(删 mean/p90)
        tail_quantile: 长尾阈值(默认 0.99 = top 1%)
        tail_weight: 长尾样本权重(默认 2.0)
        model_version: 版本号(如 v0 / v7)
        stage: "val"=剔除 holdout_months 训练;"final"=全量训练
        holdout_months: stage="val" 时,从训练剔除的月份列表(默认 [1, 7])
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    _clear_cat_cache()

    # MLflow 设置
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

        # 1. 加载 + 特征
        print("=" * 70)
        print("1. 加载训练数据...")

        # v0.1:设置 cat 映射持久化路径(和模型一起保存)— 跨进程推理时复用
        global _CAT_MAPPINGS_PATH
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        seed = params.get("seed", 42)
        _CAT_MAPPINGS_PATH = models_dir / "feature_lists" / f"xgb_{model_version}_{feature_set}_seed{seed}_{stage}_cat_mappings.json"
        # 清空旧 cat 映射(让 fit 重新学)
        if _CAT_MAPPINGS_PATH.exists():
            _CAT_MAPPINGS_PATH.unlink()

        if cache_path and Path(cache_path).exists():
            print(f"  使用缓存: {cache_path}")
            df = pd.read_parquet(cache_path)
            # v3.5:cache 加载时也按 feature_set 过滤(避免 mean/p90 漏到 no_laglead 实验)
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
                # 只用聚合特征
                feat_cols = [c for c in df.columns
                             if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
            elif feature_set == "all":
                df = build_features_v2(df, include_aggregate=True, include_lead=True)
                agg_cols = [c for c in df.columns
                            if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
                feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns] + agg_cols
            elif feature_set == "no_laglead":
                # v3.5:Phase 1 — 只保留 basic + cnt_*(没有 target-derived lag/lead)
                df = build_features_v2(df, include_aggregate=True, include_lead=True)
                keep_agg_cols = [c for c in df.columns
                                 if c.endswith("m") and "cnt_" in c]
                feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns] + keep_agg_cols
                print(f"  [no_laglead] 保留 {len(keep_agg_cols)} 个 cnt 列,删除所有 mean/p90")
            elif feat_cols_override is not None:
                # v0:用户自定义特征集(如 basic30 / cnt42),feature_set 仅做命名标签
                # 默认走 no_laglead 路径确保 cnt 列存在,然后用 override 筛选
                df = build_features_v2(df, include_aggregate=True, include_lead=True)
                feat_cols = [c for c in feat_cols_override if c in df.columns]
                print(f"  [feat_cols_override] 使用 {len(feat_cols)} 列")
            else:
                raise ValueError(f"未知 feature_set: {feature_set}")

            # 保存到缓存
            if cache_path:
                print(f"  保存缓存: {cache_path}")
                df.to_parquet(cache_path, index=False)

        # v0:stage="val" 时,从训练剔除 holdout_months(默认 [1, 7])
        if stage == "val":
            if holdout_months is None:
                holdout_months = [1, 7]
            n_before = len(df)
            sched = df["SCHED_TIME_UTC_mvt"]
            month_mask = sched.dt.month.isin(holdout_months)
            df_train_only = df[~month_mask].reset_index(drop=True)
            print(f"\n  [stage=val] 从训练剔除 holdout 月份 {holdout_months}")
            print(f"    训练: {len(df_train_only):,} (剔除前 {n_before:,})")
            print(f"    holdout 月份的样本数: {month_mask.sum():,}")
            df = df_train_only

        # 丢掉有缺失标签的行
        df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
        # v2 异常值过滤(从 EDA 发现:有负数 -12s,有极端值 131167s=36 小时)
        # > 0 过滤掉负数,< 10800(3 小时)过滤极端值,覆盖 99.99% 真实数据
        df = df[(df["TAXITIME_SEC_mvt"] > 0) & (df["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
        print(f"过滤后训练样本: {len(df):,}")
        mlflow.log_param("n_train_samples", len(df))

        # 2. 准备 X, y
        print(f"\n使用的特征({len(feat_cols)} 个): {feat_cols}")

        X = df[feat_cols]
        y = df["TAXITIME_SEC_mvt"]

        # v3.7:长尾加权 —— 给 top 1% 长尾样本 tail_weight 权重
        # 原理:RMSE 被长尾大误差主导,加权重让模型更重视长尾预测
        if tail_weight != 1.0:
            threshold = y.quantile(tail_quantile)
            sample_weight = np.where(y > threshold, tail_weight, 1.0).astype("float32")
            n_tail = (y > threshold).sum()
            print(f"\n[v3.7 长尾加权] 阈值 p{tail_quantile*100:.0f}={threshold:.0f}s, "
                  f"{n_tail:,} 行 ({n_tail/len(y)*100:.1f}%) 权重={tail_weight}, "
                  f"其余权重=1.0")
        else:
            sample_weight = None

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
            # v3.7:sample_weight 也按 train_mask 切(只对 train 部分加权)
            sw_tr = sample_weight[split.train_mask] if sample_weight is not None else None
            print(f"  训练: {len(X_tr):,} 行, 验证: {len(X_va):,} 行")

            booster, metrics = train_one_split(X_tr, y_tr, X_va, y_va, params, sample_weight=sw_tr)
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
        print(f"\nK 折平均 RMSE: {cv_mean:.2f} ± {cv_std:.2f}")

        # 5. 训练最终模型(全量 12 个月,不用验证集早停)
        # 用 K 折平均 best_iter 作为参考
        mean_best_iter = int(np.mean([
            m.get("best_iter", 500)
            for m in all_metrics.values()
            if isinstance(m, dict) and "best_iter" in m
        ]))
        print(f"\n3. 训练最终模型(全量数据,迭代 {mean_best_iter} 轮)...")
        X_enc = _encode_categoricals(X, fit=True)
        dfull = xgb.DMatrix(X_enc, label=y, weight=sample_weight)
        final_booster = xgb.train(
            params,
            dfull,
            num_boost_round=mean_best_iter,
            verbose_eval=200,
        )

        # 6. 保存模型 + 记录 feature importance
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        # v0:版本化文件名(stage0: 验证模型;final: 12 月训练)— 强制禁止覆盖 baseline_xgb.json
        # 模型名格式:models/xgb_{version}_{feature_set}_seed{seed}_{stage}.json
        seed = params.get("seed", 42)
        model_name = f"xgb_{model_version}_{feature_set}_seed{seed}_{stage}.json"
        model_path = models_dir / model_name
        final_booster.save_model(str(model_path))
        # v0:模型↔特征列绑定
        feat_list_dir = models_dir / "feature_lists"
        feat_list_dir.mkdir(exist_ok=True, parents=True)
        feat_list_path = feat_list_dir / f"{model_name.replace('.json', '.json')}"
        # 上面的写法有 bug,修正:直接写 features.json 文件名
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
        print(f"  模型保存: {model_path}")
        print(f"  特征列清单: {feat_list_file}")
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(feat_list_file), artifact_path="features_meta")

        # 特征重要性
        importance = final_booster.get_score(importance_type="gain")
        importance_df = pd.DataFrame([
            {"feature": k, "importance_gain": v}
            for k, v in importance.items()
        ]).sort_values("importance_gain", ascending=False)
        importance_path = feat_list_dir / f"{model_name.replace('.json', '')}_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), artifact_path="feature_importance")

        print(f"\nTop 10 重要特征:\n{importance_df.head(10).to_string(index=False)}")

        # 7. 记录到 MLflow 的"tag",方便筛选实验
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("features", "basic_no_aggregate")
        mlflow.set_tag("model", "xgboost")

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
    parser = argparse.ArgumentParser(description="PRC 2026 XGBoost baseline")
    parser.add_argument("--data-dir", default="data/raw", help="训练数据目录")
    parser.add_argument("--experiment-name", default="prc-2026-xgb-baseline", help="MLflow 实验名")
    parser.add_argument("--tracking-uri", default="mlruns", help="MLflow 后端 URI")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖默认学习率")
    parser.add_argument("--max-depth", type=int, default=None, help="覆盖默认 max_depth")
    parser.add_argument(
        "--feature-set",
        default="basic",
        help="特征集标签(仅用于命名)。取值:basic / basic30 / aggregate / all / no_laglead / cnt42 等。"
             "实际过滤规则由 --feat-cols 控制(默认 basic=BASELINE_FEATURE_COLS)",
    )
    parser.add_argument(
        "--feat-cols",
        default=None,
        help="v0:自定义特征列清单(JSON 列表文件)。如未提供,按 feature_set 推断",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子(默认 42,用户指定 seed0 时传 0)",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="features 缓存 parquet 路径(跳过 lag/lead 重算)",
    )
    parser.add_argument(
        "--tail-quantile", type=float, default=0.99,
        help="v3.7:长尾加权阈值(默认 0.99=top 1%)",
    )
    parser.add_argument(
        "--tail-weight", type=float, default=2.0,
        help="v3.7:长尾样本权重(默认 2.0,1.0=不加权)",
    )
    parser.add_argument(
        "--model-version", type=str, default="v0",
        help="v0:模型版本号(影响文件名,如 v0 / v7)",
    )
    parser.add_argument(
        "--stage", type=str, default="val", choices=["val", "final"],
        help="val:剔除 holdout 月份训练;final:12 月全量训练",
    )
    parser.add_argument(
        "--holdout-months", type=int, nargs="+", default=None,
        help="stage=val 时剔除的月份(默认 [1, 7])",
    )
    args = parser.parse_args()

    # 覆盖超参(可选)
    overrides = {}
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.max_depth is not None:
        overrides["max_depth"] = args.max_depth
    overrides["seed"] = args.seed

    # 加载 feat-cols(如提供)
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
    print(f"\n📊 最终结果:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
