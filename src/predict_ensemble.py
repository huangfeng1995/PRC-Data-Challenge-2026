"""
三模型融合推理脚本
==================

加载训练好的 XGB + LGB(+ 可选 CatBoost) 模型,按训练时学到的权重融合预测。

使用:
    python -m src.predict_ensemble --data-dir data/raw --output submissions/ensemble_v1.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_ranking, load_submission_template, load_training

# 延迟 import 训练侧模块,避免 macOS 上 import xgboost 时的 libomp 问题
# (因为 train_xgb 顶层 import xgboost,在 macOS 沙箱里会失败)
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
    """加载 ensemble 模型 + 推理 + 生成提交。"""
    models_dir = Path(models_dir)

    # 1. 读权重配置
    weights_path = models_dir / "ensemble_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到 {weights_path},请先跑 train_ensemble.py")
    config = json.loads(weights_path.read_text())
    weights = config["weights"]
    feat_cols = config["feat_cols"]
    feature_set = config.get("feature_set", "all")
    model_names = config["models"]
    te_mappings = config.get("te_mappings")  # v2:从训练保存的 mapping 复用
    blend_mode = config.get("blend_mode", "mean")  # v2:mean 或 median
    print(f"配置: weights={weights}, models={model_names}, feature_set={feature_set}, blend_mode={blend_mode}")
    print(f"       te_mappings: {len(te_mappings) if te_mappings else 0} 个特征")

    # 2. 先 fit 类别编码映射(从训练数据学)
    print("\n从训练数据学习类别映射...")
    _clear_cache_lazy()
    df_train = load_training(data_dir)
    df_train_feat, _ = _build_full_features_lazy(df_train, feature_set=feature_set)
    # fit 编码
    _ = _encode_categoricals_lazy(df_train_feat[feat_cols], fit=True)

    # 3. 加载 ranking 数据 + 构造特征
    print("\n加载 ranking 数据...")
    df_rank = load_ranking(data_dir)
    df_rank_feat, _ = _build_full_features_lazy(df_rank, feature_set=feature_set, te_mappings=te_mappings)
    X = df_rank_feat[feat_cols]
    X_enc = _encode_categoricals_lazy(X, fit=False)

    # 4. 加载每个模型(每个模型有 N 个 seed 文件,做 seed 内平均) + 预测
    print("\n加载模型 + 预测...")
    y_blend = np.zeros(len(df_rank_feat), dtype="float64")
    total_weight = 0.0

    # v2:支持 median blending
    # 先收集每个模型的 seed 平均预测
    model_seed_avg = {}
    for model_name in model_names:
        ext = {"xgb": "json", "lgb": "txt", "cat": "cbm"}[model_name]
        seed_files = sorted(models_dir.glob(f"ensemble_{model_name}__seed_*.{ext}"))

        if not seed_files:
            print(f"  ⚠️ 找不到 ensemble_{model_name}__seed_*.{ext},跳过")
            continue

        # v2:try/except 单 seed,坏的 seed 跳过,不让它崩溃整推理
        seed_preds = []
        for sf in seed_files:
            try:
                wrapper = _load_model_lazy(model_name, str(sf))
                seed_preds.append(wrapper.predict(X_enc))
            except Exception as e:
                print(f"  ⚠️ 加载/推理 {sf.name} 失败:{e},跳过该 seed")

        if not seed_preds:
            print(f"  ❌ {model_name} 所有 seed 都失败,跳过")
            continue

        y_pred_avg = np.mean(seed_preds, axis=0)
        model_seed_avg[model_name] = y_pred_avg
        print(f"  ✓ {model_name}: {len(seed_preds)}/{len(seed_files)} seed 平均,"
              f" pred range=[{y_pred_avg.min():.0f}, {y_pred_avg.max():.0f}]")

    if not model_seed_avg:
        raise RuntimeError("没有任何模型成功加载")

    # 按 blend_mode 融合
    if blend_mode == "median":
        preds_matrix = np.column_stack(list(model_seed_avg.values()))
        y_blend = np.median(preds_matrix, axis=1)
        print(f"  使用 median blending, {len(model_seed_avg)} 个模型")
    else:
        # mean加权
        y_blend = np.zeros(len(X_enc), dtype="float64")
        total_weight = 0.0
        for model_name, y_pred_avg in model_seed_avg.items():
            weight = weights.get(model_name, 0)
            y_blend += weight * y_pred_avg
            total_weight += weight
        if total_weight > 0:
            y_blend = y_blend / total_weight
        print(f"  使用 mean加权, {len(model_seed_avg)} 个模型")

    if total_weight == 0:
        raise RuntimeError("没有任何模型成功加载")
    y_blend = y_blend / total_weight

    # 5. 裁剪
    y_blend = np.clip(y_blend, clip_min, clip_max)

    # 6. 对齐到提交模板
    print(f"\n对齐到提交模板...")
    template = load_submission_template(data_dir)
    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_feat["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_blend,
    })
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"  ⚠️ {n_missing} 个 MVT_ID 未匹配,用全局中位数填充")
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(float(np.nanmedian(y_blend)))

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    # 7. 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)

    print(f"\n✅ 提交文件已生成: {output_path}")
    print(f"  总行数: {len(submission):,}")
    print(f"  预测值范围: [{submission['TAXITIME_SEC_mvt'].min():.1f}, {submission['TAXITIME_SEC_mvt'].max():.1f}] 秒")
    print(f"  预测值中位数: {submission['TAXITIME_SEC_mvt'].median():.1f} 秒")

    return submission


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 Ensemble 推理")
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
