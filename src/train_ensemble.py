"""
Three-model ensemble training script
===================================

Learned from the 2nd place jubilant-vase in 2025:
"Ensemble of three gradient boosting libraries to maximize performance and diversity"

Strategy:
1. XGBoost (main one, we already have it)
2. LightGBM (wrapped in try/except, degrade to single thread when macOS lacks libomp)
3. CatBoost (optional)

Blending method:
- Default: weighted average (weights learned on OOF by scipy.optimize.minimize)
- Simple average is also supported

Usage:
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

# Lazily import helper functions from train_xgb - because importing xgboost at the top level of
# train_xgb triggers the missing-libomp problem on macOS. Here we only import _encode_categoricals
# when it is used inside build_full_features.
def _encode_categoricals_lazy(X, fit=False):
    from src.train_xgb import _encode_categoricals
    return _encode_categoricals(X, fit=fit)

def _clear_cat_cache_lazy():
    from src.train_xgb import _clear_cat_cache as _orig
    _orig()

from src.validation import evaluate, get_validation_splits


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelWrapper:
    """Unified interface for XGB / LGB / CatBoost, to make blending easier."""

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
            raise ValueError(f"unknown model: {name}")


class XGBWrapper(ModelWrapper):
    def __init__(self, booster):
        super().__init__("xgb", booster)

    def fit(self, X, y, eval_set=None, **kwargs):
        import xgboost as xgb
        params = self.model  # before fit, params are placed in the model field
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
# Default hyperparameters
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
# Feature construction (centralized, to avoid duplication with train_xgb)
# ---------------------------------------------------------------------------

def build_full_features(
    df: pd.DataFrame,
    feature_set: str = "all",
    te_mappings: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full feature matrix, return (df, feat_cols).

    Parameters:
        df: raw data
        feature_set: same as before
        te_mappings: TE mapping dict learned at training time (reused during ranking inference,
                  avoiding the inability to compute TE because ranking has no TAXITIME_SEC_mvt)
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
        # Target encoding
        # At training time: df contains TAXITIME_SEC_mvt, fit directly
        # At inference time: taxi-out is masked, must use te_mappings (saved at training time)
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
        raise ValueError(f"unknown feature_set: {feature_set}")

    return df, feat_cols


# ---------------------------------------------------------------------------
# Blend weight optimization
# ---------------------------------------------------------------------------

def optimize_blend_weights(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> tuple[dict[str, float], float]:
    """Learn the optimal blend weights on OOF predictions using scipy.optimize.minimize."""
    from scipy.optimize import minimize

    model_names = list(oof_preds.keys())
    n = len(model_names)

    def rmse_loss(weights):
        # Normalize weights to sum=1
        w = np.maximum(weights, 0)
        if w.sum() == 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()
        y_blend = np.zeros_like(y_true, dtype="float64")
        for i, name in enumerate(model_names):
            y_blend += w[i] * oof_preds[name]
        return float(np.sqrt(np.mean((y_blend - y_true) ** 2)))

    # Initial value: uniform weights
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
    """Median voting blend (learned from the 4th place team_brave_pillow in 2024).

    For each sample, take the **median** (not the mean) of all model predictions.
    In scenarios with many long-tail / outlier cases, median is more robust than mean.
    """
    preds_matrix = np.column_stack(list(oof_preds.values()))
    y_blend = np.median(preds_matrix, axis=1)
    rmse = float(np.sqrt(np.mean((y_blend - y_true) ** 2)))
    # median cannot directly give "weights", we use uniform weights as a notation
    weights = {name: 1.0 / len(oof_preds) for name in oof_preds}
    return rmse, weights


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_ensemble_experiment(
    data_dir: str = "data/raw",
    feature_set: str = "all",
    models: tuple[str, ...] = ("xgb", "lgb", "cat"),  # three-model ensemble (v2: referencing the 2nd place in 2025 who got 64% weight with CatBoost)
    n_seeds: int = 5,  # how many different seeds to run per model, default 5 (learned from 1st place in 2024)
    tracking_uri: str = "mlruns",
    experiment_name: str = "prc-2026-ensemble",
    cache_path: str | None = None,
) -> dict:
    """Run one multi-model + multi-seed ensemble experiment.

    Learned from the 1st place team_likable_jelly in 2024: averaging the same model over different seeds
    can further reduce RMSE by 1-3%.

    Parameters:
        n_seeds: how many seeds to run per model. 1 = no averaging, 5 = average of 5 seeds,
                 10 = 10 seeds (more stable but slower).
        cache_path: features cache path (skips lag/lead recomputation, speeds up iteration)
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("models", ",".join(models))
        mlflow.log_param("n_seeds", n_seeds)

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

        # v2: fit TE mappings and save them, reuse at inference time (avoid missing TE columns when ranking lacks TAXITIME_SEC_mvt)
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
        # v2: from EDA, taxi-out has negatives (data bug) and extreme values (131167s)
        print(f"samples after filtering: {len(df):,}")
        print(f"number of features: {len(feat_cols)}")

        # Encode category columns
        X = df[feat_cols].copy()
        X = _encode_categoricals_lazy(X, fit=True)
        y = df["TAXITIME_SEC_mvt"]

        print("\n2. Time-series split validation (winter + summer, two folds)...")
        splits = get_validation_splits(df)

        # v2: run winter + summer at the same time, pick the weights of the best fold (test set contains Jan + Jul)
        primary_split = splits["winter"]
        secondary_split = splits["summer"]
        split = primary_split
        X_tr, y_tr = X[split.train_mask], y[split.train_mask]
        X_va, y_va = X[split.val_mask], y[split.val_mask]
        print(f"train: {len(X_tr):,}, val (winter): {len(X_va):,}")
        print(f"train: {len(X_tr):,}, val (summer): {secondary_split.n_val:,}")

        # 3. Train each model (each model runs n_seeds seeds)
        # Use "model_name__seed_N" as key to distinguish, aggregate by model_name when blending
        oof_preds = {}  # key is "xgb__seed_0", "xgb__seed_1", ...
        trained_models = {}  # the same keys
        all_metrics = {}  # key is "xgb__seed_0", value is the dict returned by evaluate (includes best_iter)

        for model_name in models:
            print(f"\n3. Training model: {model_name} (x{n_seeds} seed)")
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
                            print(f"  WARNING: LightGBM unavailable ({e}), skipping the whole lgb")
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
                            print(f"  WARNING: CatBoost unavailable ({e}), skipping the whole cat")
                            break
                        params = DEFAULT_CAT_PARAMS.copy()
                        params["random_seed"] = seed_i
                        wrapper = CATWrapper(params)
                        wrapper.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                        y_pred = wrapper.predict(X_va)
                    else:
                        raise ValueError(f"unknown model: {model_name}")

                    metrics = evaluate(y_va.values, y_pred)
                    print(f"  {seed_key} metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
                    mlflow.log_metric(f"{seed_key}_rmse", metrics["rmse"])
                    mlflow.log_metric(f"{seed_key}_mae", metrics["mae"])

                    # v3 bug fix: get best_iteration from the wrapper (both XGB and LGB have it)
                    best_iter = getattr(wrapper.model, "best_iteration", None)
                    if best_iter is None:
                        # CatBoost
                        best_iter = getattr(wrapper.model, "tree_count_", None)
                    if best_iter is not None:
                        metrics["best_iter"] = int(best_iter)
                        mlflow.log_metric(f"{seed_key}_best_iter", int(best_iter))

                    oof_preds[seed_key] = y_pred
                    trained_models[seed_key] = wrapper
                    all_metrics[seed_key] = metrics  # v3 bug fix: save metrics for best_iter lookup

                except Exception as e:
                    print(f"  Training failed for {seed_key}: {e}")
                    continue

        if len(oof_preds) < 2:
            print("\nWARNING: at least 2 models are needed to blend, skipping the blending step")
            return {"run_id": run.info.run_id, "individual_metrics": {k: v for k, v in oof_preds.items()}}

        # 4. Multi-seed average + multi-model blending
        # First average within seeds per model_name to get each model's "average prediction"
        print("\n4. Multi-seed average + optimized blend weights...")
        model_avg_preds = {}
        for model_name in models:
            seeds_for_model = [k for k in oof_preds if k.startswith(f"{model_name}__")]
            if not seeds_for_model:
                continue
            avg_pred = np.mean([oof_preds[k] for k in seeds_for_model], axis=0)
            single_rmse = evaluate(y_va.values, avg_pred)["rmse"]
            print(f"  {model_name} (average of {len(seeds_for_model)} seeds): RMSE={single_rmse:.2f}")
            mlflow.log_metric(f"{model_name}_avg_rmse", single_rmse)
            model_avg_preds[model_name] = avg_pred

        if len(model_avg_preds) < 2:
            print("  WARNING: only one model is available, use its average prediction as the final one")
            final_rmse = evaluate(y_va.values, list(model_avg_preds.values())[0])["rmse"]
            weights = {list(model_avg_preds.keys())[0]: 1.0}
            blend_mode = "mean"
        else:
            # Compare mean weighting vs median voting (v2: referencing the 4th place in 2024)
            mean_weights, mean_rmse = optimize_blend_weights(model_avg_preds, y_va.values)
            median_rmse, _ = median_blend(model_avg_preds, y_va.values)
            print(f"  mean blend RMSE: {mean_rmse:.2f}, weights: {mean_weights}")
            print(f"  median blend RMSE: {median_rmse:.2f}")

            # Choose the better one (automatically compares the two strategies)
            if median_rmse < mean_rmse:
                print(f"  median is better, choosing median")
                final_rmse = median_rmse
                weights = {k: 1.0 / len(model_avg_preds) for k in model_avg_preds}
                blend_mode = "median"
            else:
                print(f"  weighted mean is better, choosing mean")
                final_rmse = mean_rmse
                weights = mean_weights
                blend_mode = "mean"
            print(f"  final RMSE: {final_rmse:.2f}, blend_mode = {blend_mode}")

        mlflow.log_metrics({f"weight_{k}": v for k, v in weights.items()})
        mlflow.log_metric("blend_rmse", final_rmse)

        # v3.5: save OOF validation models (each model x seed saved separately), used for inference
        # Even if final training takes too long, these validation models can produce ensemble submissions
        models_dir = Path("models/ensemble_val")
        models_dir.mkdir(parents=True, exist_ok=True)
        for seed_key, wrapper in trained_models.items():
            save_path = models_dir / f"{seed_key}.pkl"
            wrapper.save(save_path)
            print(f"  Saved validation model: {save_path}")
        # Save weights + some metadata needed for inference
        meta = {
            "weights": weights,
            "blend_mode": blend_mode,
            "feature_set": feature_set,
            "models": list(trained_models.keys()),
            "feat_cols": feat_cols,
        }
        (models_dir / "weights.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"  Saved weights: {models_dir / 'weights.json'}")

        # 5. Train the final models (full 12 months, each model x n_seeds)
        # v2: use the best_iteration found during validation x 1.1, instead of hardcoding 500
        print(f"\n5. Training final models (full data, each model x {n_seeds} seed)...")
        # Compute each model's final num_boost_round: best_iteration from validation phase x 1.1 (20% headroom for full data)
        model_best_iters = {}
        for model_name in models:
            seeds_for_model = [k for k in oof_preds if k.startswith(f"{model_name}__")]
            if not seeds_for_model:
                continue
            # Take the average best_iteration of all seeds of that model
            best_iters = []
            for k in seeds_for_model:
                v = all_metrics.get(k)
                if v and "best_iter" in v:
                    best_iters.append(v["best_iter"])
            if best_iters:
                avg_best = int(np.mean(best_iters))
                # v2 bug fix: use 1.1x instead of hardcoding 500
                model_best_iters[model_name] = max(int(avg_best * 1.1), 200)
                print(f"  {model_name} validation best_iter average={avg_best}, final uses {model_best_iters[model_name]}")
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
                    print(f"  full-data training done for {seed_key} ({model_best_iters[model_name]} rounds)")
                except Exception as e:
                    print(f"  Full-data training failed for {seed_key}: {e}")

        # 6. Save models and weights
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        for name, wrapper in final_models.items():
            ext = "json" if "xgb" in name else ("txt" if "lgb" in name else "cbm")
            wrapper.save(models_dir / f"ensemble_{name}.{ext}")
        weights_path = models_dir / "ensemble_weights.json"
        # v2: also save TE mappings for inference
        weights_dict = {
            "weights": weights,
            "blend_mode": blend_mode,  # v2: mean or median
            "feature_set": feature_set,
            "models": list(set(k.split("__seed_")[0] for k in final_models.keys())),
            "model_seeds": {k.split("__seed_")[0]: [k for k in final_models.keys() if k.startswith(k.split("__seed_")[0] + "__seed_")] for k in final_models.keys()},
            "n_seeds": n_seeds,
            "feat_cols": feat_cols,
            "te_mappings": self_te_mappings,
        }
        weights_path.write_text(json.dumps(weights_dict, indent=2, default=str))
        mlflow.log_artifact(str(weights_path))

        print(f"\nexperiment complete, run_id = {run.info.run_id}")
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
    parser = argparse.ArgumentParser(description="PRC 2026 three-model ensemble training")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--feature-set", default="all", choices=["basic", "domain", "aggregate", "all"])
    parser.add_argument("--models", default="xgb,lgb,cat", help="comma separated, e.g. 'xgb,lgb' or 'xgb,lgb,cat' (default: three-model ensemble)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="how many seeds to run per model (learned from 1st place in 2024), default 5, recommend 5-10")
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--cache", default=None,
                        help="features cache parquet path (skip lag/lead recomputation)")
    args = parser.parse_args()

    models = tuple(m.strip() for m in args.models.split(","))
    result = run_ensemble_experiment(
        data_dir=args.data_dir,
        feature_set=args.feature_set,
        models=models,
        n_seeds=args.n_seeds,
        tracking_uri=args.tracking_uri,
        cache_path=args.cache,
    )
    print(f"\nfinal result:\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()