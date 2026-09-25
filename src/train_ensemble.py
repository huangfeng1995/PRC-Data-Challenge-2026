"""
三模型融合训练脚本
==================

从 2025 第 2 名 jubilant-vase 学到:
"Ensemble of three gradient boosting libraries to maximize performance and diversity"

策略:
1. XGBoost(主力,我们已有)
2. LightGBM(用 try/except 包装,macOS 缺 libomp 时降级到单线程)
3. CatBoost(可选)

融合方式:
- 默认:加权平均(权重由 scipy.optimize.minimize 在 OOF 上学)
- 也支持简单平均

使用:
    python -m src.train_ensemble --data-dir data/raw --feature-set all
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.data_loader import load_training
from src.features import BASELINE_FEATURE_COLS, build_features
from src.features_aggregate import add_aggregate_features
from src.features_domain import (
    add_domain_features,
    add_target_encodings,
    fit_te_mappings,
    add_target_encodings_with_mapping,
    apply_te_mappings,
)
from src.features_laglead import add_lead_features

# 延迟 import train_xgb 的辅助函数 —— 因为 train_xgb 顶层 import xgboost 会触发
# macOS 上的 libomp 缺失问题。这里只在 build_full_features 用到 _encode_categoricals
# 时才 import。
def _encode_categoricals_lazy(X, fit=False):
    from src.train_xgb import _encode_categoricals
    return _encode_categoricals(X, fit=fit)

# 天气特征可选开关 —— 默认关闭,避免没有 data/external/weather/ 时炸
_WEATHER_COLS = [
    "weather_temperature", "weather_dewpoint", "weather_dewpoint_deficit",
    "weather_snowfall", "weather_snow_depth", "weather_precipitation",
    "weather_wind_speed", "weather_wind_gust",
    "weather_cloud_cover", "weather_pressure", "weather_code",
]

def _add_weather_features_lazy(df: pd.DataFrame) -> pd.DataFrame:
    """延迟 import build_weather_features,避免没跑 fetch_weather.py 时炸。"""
    from scripts.build_weather_features import add_weather_features
    return add_weather_features(df)

def _clear_cat_cache_lazy():
    from src.train_xgb import _clear_cat_cache as _orig
    _orig()

from src.validation import evaluate, get_validation_splits


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------

class ModelWrapper:
    """统一 XGB / LGB / CatBoost 的接口,方便融合。"""

    def __init__(self, name: str, model):
        self.name = name
        self.model = model

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    @staticmethod
    def load(name: str, path: Path):
        if name == "xgb":
            import xgboost as xgb
            m = xgb.Booster()
            m.load_model(str(path))
            return XGBWrapper(m)
        elif name == "lgb":
            import lightgbm as lgb
            m = lgb.Booster(model_file=str(path))
            return LGBWrapper(m)
        elif name == "cat":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(str(path))
            return CATWrapper(m)
        else:
            raise ValueError(f"未知模型: {name}")


class XGBWrapper(ModelWrapper):
    def __init__(self, booster):
        super().__init__("xgb", booster)

    def fit(self, X, y, eval_set=None, **kwargs):
        import xgboost as xgb
        params = self.model  # 在 fit 前 params 放在 model 字段
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]
        if eval_set:
            dval = xgb.DMatrix(eval_set[0], label=eval_set[1])
            evals.append((dval, "val"))
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=kwargs.get("num_boost_round", 2000),
            evals=evals,
            early_stopping_rounds=kwargs.get("early_stopping_rounds", 100),
            verbose_eval=False,
        )

    def predict(self, X) -> np.ndarray:
        import xgboost as xgb
        return self.model.predict(xgb.DMatrix(X))

    def save(self, path: Path):
        self.model.save_model(str(path))


class LGBWrapper(ModelWrapper):
    def __init__(self, booster):
        super().__init__("lgb", booster)

    def fit(self, X, y, eval_set=None, **kwargs):
        import lightgbm as lgb
        params = self.model
        train_set = lgb.Dataset(X, label=y, categorical_feature="auto")
        valid_sets = [train_set]
        valid_names = ["train"]
        if eval_set:
            val_set = lgb.Dataset(eval_set[0], label=eval_set[1], categorical_feature="auto")
            valid_sets.append(val_set)
            valid_names.append("val")
        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=kwargs.get("num_boost_round", 2000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(kwargs.get("early_stopping_rounds", 100), verbose=False)],
        )

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path: Path):
        self.model.save_model(str(path))


class CATWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__("cat", model)

    def fit(self, X, y, eval_set=None, **kwargs):
        from catboost import CatBoostRegressor
        params = self.model  # dict
        eval_set_arg = None
        if eval_set:
            eval_set_arg = (eval_set[0], eval_set[1])
        self.model = CatBoostRegressor(**params)
        self.model.fit(
            X, y,
            eval_set=eval_set_arg,
            early_stopping_rounds=kwargs.get("early_stopping_rounds", 100),
            verbose=False,
        )

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path):
        self.model.save_model(str(path))


# ---------------------------------------------------------------------------
# 默认超参
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
    "seed": 42,
}

DEFAULT_LGB_PARAMS = {
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
    "num_threads": -1,
    "seed": 42,
}

DEFAULT_CAT_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "verbose": False,
    "thread_count": -1,
    "random_seed": 42,
    "early_stopping_rounds": 100,
}


# ---------------------------------------------------------------------------
# 特征构建(集中管理,避免和 train_xgb 重复)
# ---------------------------------------------------------------------------

def build_full_features(
    df: pd.DataFrame,
    feature_set: str = "all",
    te_mappings: dict | None = None,
    add_weather: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """构造完整特征矩阵,返回 (df, feat_cols)。

    参数:
        df: 原始数据
        feature_set: 同前
        te_mappings: 训练时学到的 TE 映射 dict(用于 ranking 推理时复用,
                  避免 ranking 没有 TAXITIME_SEC_mvt 而无法计算 TE)
    """
    _clear_cat_cache_lazy()

    if feature_set == "basic":
        df = build_features(df)
        feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    elif feature_set == "domain":
        df = build_features(df)
        df = add_domain_features(df)
        domain_extra = ["stand_zone", "runway_family", "runway_side"]
        feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns] + domain_extra
    elif feature_set == "aggregate":
        df = build_features(df)
        df = add_aggregate_features(df)
        agg_cols = [c for c in df.columns
                    if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
        feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns] + agg_cols
    elif feature_set == "all":
        df = build_features(df)
        df = add_domain_features(df)
        df = add_aggregate_features(df)
        df = add_lead_features(df)
        # 目标编码
        # 训练时:df 含 TAXITIME_SEC_mvt,直接 fit
        # 推理时:taxi-out 被屏蔽,必须用 te_mappings(从训练时保存)
        if "TAXITIME_SEC_mvt" in df.columns and df["TAXITIME_SEC_mvt"].notna().sum() > 100:
            df = add_target_encodings_with_mapping(df)
        elif te_mappings is not None:
            df = apply_te_mappings(df, te_mappings)
        domain_extra = ["stand_zone", "runway_family", "runway_side"]
        agg_cols = [c for c in df.columns
                    if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
        te_cols = [c for c in df.columns if c.endswith("_te")]
        feat_cols = (
            [c for c in BASELINE_FEATURE_COLS if c in df.columns]
            + domain_extra
            + agg_cols
            + te_cols
        )
    else:
        raise ValueError(f"未知 feature_set: {feature_set}")

    if add_weather:
        df = _add_weather_features_lazy(df)
        feat_cols = feat_cols + [c for c in _WEATHER_COLS if c in df.columns]

    return df, feat_cols


# ---------------------------------------------------------------------------
# 融合权重优化
# ---------------------------------------------------------------------------

def optimize_blend_weights(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> tuple[dict[str, float], float]:
    """用 scipy.optimize.minimize 在 OOF 预测上学最优融合权重。"""
    from scipy.optimize import minimize

    model_names = list(oof_preds.keys())
    n = len(model_names)

    def rmse_loss(weights):
        # 权重归一化到 sum=1
        w = np.maximum(weights, 0)
        if w.sum() == 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()
        y_blend = np.zeros_like(y_true, dtype="float64")
        for i, name in enumerate(model_names):
            y_blend += w[i] * oof_preds[name]
        return float(np.sqrt(np.mean((y_blend - y_true) ** 2)))

    # 初值:均匀权重
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    result = minimize(rmse_loss, x0, method="SLSQP", bounds=bounds)

    w = np.maximum(result.x, 0)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    weights_dict = {name: float(w[i]) for i, name in enumerate(model_names)}
    return weights_dict, float(result.fun)


def median_blend(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Median voting 融合(从 2024 第 4 名 team_brave_pillow 学到)。

    对每个样本,取所有模型预测的**中位数**(不是均值)。
    在长尾多 / outlier 多的场景下,median 比 mean 更鲁棒。
    """
    preds_matrix = np.column_stack(list(oof_preds.values()))
    y_blend = np.median(preds_matrix, axis=1)
    rmse = float(np.sqrt(np.mean((y_blend - y_true) ** 2)))
    # median 没法直接给"权重",我们用 uniform weights 作为记号
    weights = {name: 1.0 / len(oof_preds) for name in oof_preds}
    return rmse, weights


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------

def run_ensemble_experiment(
    data_dir: str = "data/raw",
    feature_set: str = "all",
    models: tuple[str, ...] = ("xgb", "lgb", "cat"),  # 三模型融合(v2:参考 2025 第 2 名 CatBoost 拿 64% 权重)
    n_seeds: int = 5,  # 每个模型跑几个不同 seed,默认 5(从 2024 第 1 名学到)
    tracking_uri: str = "mlruns",
    experiment_name: str = "prc-2026-ensemble",
    cache_path: str | None = None,
    add_weather: bool = False,
) -> dict:
    """跑一次多模型 + 多 seed 融合实验。

    从 2024 第 1 名 team_likable_jelly 学到:同一模型不同 seed 的平均,
    能让 RMSE 进一步降低 1-3%。

    参数:
        n_seeds: 每个模型跑几个 seed。设为 1 = 不平均,设 5 = 5 个 seed 平均,
                 设 10 = 10 个(更稳但更慢)。
        cache_path: features 缓存路径(跳过 lag/lead 重算,加速迭代)
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("models", ",".join(models))
        mlflow.log_param("n_seeds", n_seeds)
        mlflow.log_param("add_weather", add_weather)

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
            df, feat_cols = build_full_features(df, feature_set=feature_set, add_weather=add_weather)
            if cache_path:
                print(f"  保存缓存: {cache_path}")
                df.to_parquet(cache_path, index=False)

        # v2:fit TE mapping 并保存,推理时复用(避免 ranking 缺 TAXITIME_SEC_mvt 时 TE 列缺失)
        if (
            feature_set == "all"
            and "TAXITIME_SEC_mvt" in df.columns
            and df["TAXITIME_SEC_mvt"].notna().sum() > 100
        ):
            self_te_mappings = fit_te_mappings(df)
        else:
            self_te_mappings = None
        df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
        df = df[(df["TAXITIME_SEC_mvt"] > 0) & (df["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
        # v2:从 EDA 看,taxi-out 有负数(数据 bug)和极端值(131167s)
        print(f"过滤后样本: {len(df):,}")
        print(f"特征数: {len(feat_cols)}")

        # 编码 category 列
        X = df[feat_cols].copy()
        X = _encode_categoricals_lazy(X, fit=True)
        y = df["TAXITIME_SEC_mvt"]

        print("\n2. 时序切分验证(winter + summer 两个 fold)...")
        splits = get_validation_splits(df)

        # v2:同时跑 winter + summer,选最优 fold 的 weights(测试集含 1 月+7 月)
        primary_split = splits["winter"]
        secondary_split = splits["summer"]
        split = primary_split
        X_tr, y_tr = X[split.train_mask], y[split.train_mask]
        X_va, y_va = X[split.val_mask], y[split.val_mask]
        print(f"训练: {len(X_tr):,}, 验证(winter): {len(X_va):,}")
        print(f"训练: {len(X_tr):,}, 验证(summer): {secondary_split.n_val:,}")

        # 3. 训练每个模型(每个模型跑 n_seeds 个 seed)
        # 用 "model_name__seed_N" 作为 key 区分,融合时按 model_name 聚合
        oof_preds = {}  # key 是 "xgb__seed_0", "xgb__seed_1", ...
        trained_models = {}  # 同样的 key
        all_metrics = {}  # key 是 "xgb__seed_0",值是 evaluate 返回的 dict(含 best_iter)

        for model_name in models:
            print(f"\n3. 训练模型: {model_name} (×{n_seeds} seed)")
            for seed_i in range(n_seeds):
                seed_key = f"{model_name}__seed_{seed_i}"
                try:
                    if model_name == "xgb":
                        import xgboost as xgb
                        params = DEFAULT_XGB_PARAMS.copy()
                        params["seed"] = seed_i
                        wrapper = XGBWrapper(params)
                        wrapper.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                        y_pred = wrapper.predict(X_va)
                    elif model_name == "lgb":
                        import lightgbm as lgb
                        try:
                            lgb.LGBMRegressor()
                        except Exception as e:
                            print(f"  ⚠️ LightGBM 不可用({e}),跳过整个 lgb")
                            break
                        params = DEFAULT_LGB_PARAMS.copy()
                        params["seed"] = seed_i
                        params["feature_fraction_seed"] = seed_i
                        params["bagging_seed"] = seed_i
                        wrapper = LGBWrapper(params)
                        wrapper.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                        y_pred = wrapper.predict(X_va)
                    elif model_name == "cat":
                        try:
                            from catboost import CatBoostRegressor  # noqa
                        except Exception as e:
                            print(f"  ⚠️ CatBoost 不可用({e}),跳过整个 cat")
                            break
                        params = DEFAULT_CAT_PARAMS.copy()
                        params["random_seed"] = seed_i
                        wrapper = CATWrapper(params)
                        wrapper.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                        y_pred = wrapper.predict(X_va)
                    else:
                        raise ValueError(f"未知模型: {model_name}")

                    metrics = evaluate(y_va.values, y_pred)
                    print(f"  {seed_key} 指标: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
                    mlflow.log_metric(f"{seed_key}_rmse", metrics["rmse"])
                    mlflow.log_metric(f"{seed_key}_mae", metrics["mae"])

                    # v3 bug fix:从 wrapper 拿 best_iteration(XGB 和 LGB 都有)
                    best_iter = getattr(wrapper.model, "best_iteration", None)
                    if best_iter is None:
                        # CatBoost
                        best_iter = getattr(wrapper.model, "tree_count_", None)
                    if best_iter is not None:
                        metrics["best_iter"] = int(best_iter)
                        mlflow.log_metric(f"{seed_key}_best_iter", int(best_iter))

                    oof_preds[seed_key] = y_pred
                    trained_models[seed_key] = wrapper
                    all_metrics[seed_key] = metrics  # v3 bug fix:save metrics for best_iter lookup

                except Exception as e:
                    print(f"  ❌ {seed_key} 训练失败: {e}")
                    continue

        if len(oof_preds) < 2:
            print("\n⚠️ 至少需要 2 个模型才能融合,跳过融合步骤")
            return {"run_id": run.info.run_id, "individual_metrics": {k: v for k, v in oof_preds.items()}}

        # 4. 多 seed 平均 + 多模型融合
        # 先按 model_name 做 seed 内的平均,得到每个 model 的"平均预测"
        print("\n4. 多 seed 平均 + 优化融合权重...")
        model_avg_preds = {}
        for model_name in models:
            seeds_for_model = [k for k in oof_preds if k.startswith(f"{model_name}__")]
            if not seeds_for_model:
                continue
            avg_pred = np.mean([oof_preds[k] for k in seeds_for_model], axis=0)
            single_rmse = evaluate(y_va.values, avg_pred)["rmse"]
            print(f"  {model_name}({len(seeds_for_model)} seed 平均): RMSE={single_rmse:.2f}")
            mlflow.log_metric(f"{model_name}_avg_rmse", single_rmse)
            model_avg_preds[model_name] = avg_pred

        if len(model_avg_preds) < 2:
            print("  ⚠️ 只有一个模型可用,直接用其平均预测作为最终")
            final_rmse = evaluate(y_va.values, list(model_avg_preds.values())[0])["rmse"]
            weights = {list(model_avg_preds.keys())[0]: 1.0}
            blend_mode = "mean"
        else:
            # 比较 mean 加权和 median voting(v2:参考 2024 第 4 名)
            mean_weights, mean_rmse = optimize_blend_weights(model_avg_preds, y_va.values)
            median_rmse, _ = median_blend(model_avg_preds, y_va.values)
            print(f"  mean 融合 RMSE: {mean_rmse:.2f}, 权重: {mean_weights}")
            print(f"  median 融合 RMSE: {median_rmse:.2f}")

            # 选最优的(自动比较两种策略)
            if median_rmse < mean_rmse:
                print(f"  ✅ median 更好,选用 median")
                final_rmse = median_rmse
                weights = {k: 1.0 / len(model_avg_preds) for k in model_avg_preds}
                blend_mode = "median"
            else:
                print(f"  ✅ mean 加权更好,选用 mean")
                final_rmse = mean_rmse
                weights = mean_weights
                blend_mode = "mean"
            print(f"  最终 RMSE: {final_rmse:.2f}, blend_mode = {blend_mode}")

        mlflow.log_metrics({f"weight_{k}": v for k, v in weights.items()})
        mlflow.log_metric("blend_rmse", final_rmse)

        # v3.5:保存 OOF 验证模型(每个模型 × seed 单独保存),用于推理
        # 即使 final 训练耗时太长,这些验证模型也能生成 ensemble 提交
        models_dir = Path("models/ensemble_val")
        models_dir.mkdir(parents=True, exist_ok=True)
        for seed_key, wrapper in trained_models.items():
            save_path = models_dir / f"{seed_key}.pkl"
            wrapper.save(save_path)
            print(f"  保存验证模型: {save_path}")
        # 保存 weights + 一些推理所需元信息
        meta = {
            "weights": weights,
            "blend_mode": blend_mode,
            "feature_set": feature_set,
            "models": list(trained_models.keys()),
            "feat_cols": feat_cols,
        }
        (models_dir / "weights.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"  保存 weights: {models_dir / 'weights.json'}")

        # 5. 训练最终模型(全量 12 个月,每个模型 × n_seeds 个)
        # v2:用验证阶段找到的 best_iteration × 1.1,而不是硬编码 500
        print(f"\n5. 训练最终模型(全量,每个模型 × {n_seeds} seed)...")
        # 计算每个模型的最终 num_boost_round:从验证阶段的 best_iteration × 1.1(全量数据多 20% 余量)
        model_best_iters = {}
        for model_name in models:
            seeds_for_model = [k for k in oof_preds if k.startswith(f"{model_name}__")]
            if not seeds_for_model:
                continue
            # 取该模型所有 seed 的 best_iteration 平均
            best_iters = []
            for k in seeds_for_model:
                v = all_metrics.get(k)
                if v and "best_iter" in v:
                    best_iters.append(v["best_iter"])
            if best_iters:
                avg_best = int(np.mean(best_iters))
                # v2 bug fix:用 1.1x 而不是硬编码 500
                model_best_iters[model_name] = max(int(avg_best * 1.1), 200)
                print(f"  {model_name} 验证 best_iter 平均={avg_best}, final 用 {model_best_iters[model_name]}")
            else:
                # fallback
                model_best_iters[model_name] = 500

        final_models = {}
        for model_name in models:
            for seed_i in range(n_seeds):
                seed_key = f"{model_name}__seed_{seed_i}"
                try:
                    if model_name == "xgb":
                        import xgboost as xgb
                        params = DEFAULT_XGB_PARAMS.copy()
                        params["seed"] = seed_i
                        dfull = xgb.DMatrix(X, label=y)
                        booster = xgb.train(params, dfull, num_boost_round=model_best_iters[model_name])
                        final_models[seed_key] = XGBWrapper(booster)
                    elif model_name == "lgb":
                        import lightgbm as lgb
                        params = DEFAULT_LGB_PARAMS.copy()
                        params["seed"] = seed_i
                        params["feature_fraction_seed"] = seed_i
                        params["bagging_seed"] = seed_i
                        full_set = lgb.Dataset(X, label=y, categorical_feature="auto")
                        booster = lgb.train(params, full_set, num_boost_round=model_best_iters[model_name])
                        final_models[seed_key] = LGBWrapper(booster)
                    elif model_name == "cat":
                        from catboost import CatBoostRegressor
                        params = DEFAULT_CAT_PARAMS.copy()
                        params["random_seed"] = seed_i
                        params["iterations"] = model_best_iters[model_name]
                        cat = CatBoostRegressor(**params)
                        cat.fit(X, y, verbose=False)
                        final_models[seed_key] = CATWrapper(cat)
                    print(f"  ✓ {seed_key} 全量训练完成({model_best_iters[model_name]} 轮)")
                except Exception as e:
                    print(f"  ⚠️ {seed_key} 全量训练失败: {e}")

        # 6. 保存模型和权重
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        for name, wrapper in final_models.items():
            ext = "json" if "xgb" in name else ("txt" if "lgb" in name else "cbm")
            wrapper.save(models_dir / f"ensemble_{name}.{ext}")
        weights_path = models_dir / "ensemble_weights.json"
        # v2:同时保存 TE mapping 给推理用
        weights_dict = {
            "weights": weights,
            "blend_mode": blend_mode,  # v2:mean 或 median
            "feature_set": feature_set,
            "models": list(set(k.split("__seed_")[0] for k in final_models.keys())),
            "model_seeds": {k.split("__seed_")[0]: [k for k in final_models.keys() if k.startswith(k.split("__seed_")[0] + "__seed_")] for k in final_models.keys()},
            "n_seeds": n_seeds,
            "feat_cols": feat_cols,
            "te_mappings": self_te_mappings,
        }
        weights_path.write_text(json.dumps(weights_dict, indent=2, default=str))
        mlflow.log_artifact(str(weights_path))

        print(f"\n✅ 实验完成,run_id = {run.info.run_id}")
        return {
            "run_id": run.info.run_id,
            "individual_metrics": {k: float(evaluate(y_va.values, v)["rmse"]) for k, v in oof_preds.items()},
            "model_avg_rmse": {k: float(evaluate(y_va.values, v)["rmse"]) for k, v in model_avg_preds.items()},
            "blend_rmse": final_rmse,
            "weights": weights,
            "n_seeds": n_seeds,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRC 2026 三模型融合训练")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--feature-set", default="all", choices=["basic", "domain", "aggregate", "all"])
    parser.add_argument("--models", default="xgb,lgb,cat", help="逗号分隔,如 'xgb,lgb' 或 'xgb,lgb,cat' (默认三模型融合)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="每个模型跑几个 seed(从 2024 第 1 名学到),默认 5,推荐 5-10")
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--cache", default=None,
                        help="features 缓存 parquet 路径(跳过 lag/lead 重算)")
    parser.add_argument("--add-weather", action="store_true", default=False,
                        help="追加 Open-Meteo 天气特征 (默认关闭; 需先跑 scripts/fetch_weather.py 并安装依赖)")
    args = parser.parse_args()

    models = tuple(m.strip() for m in args.models.split(","))
    result = run_ensemble_experiment(
        data_dir=args.data_dir,
        feature_set=args.feature_set,
        models=models,
        n_seeds=args.n_seeds,
        tracking_uri=args.tracking_uri,
        cache_path=args.cache,
        add_weather=args.add_weather,
    )
    print(f"\n📊 最终结果:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
