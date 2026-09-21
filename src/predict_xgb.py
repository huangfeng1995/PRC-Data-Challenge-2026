"""
XGBoost 推理 + 生成提交 parquet
==============================

跟 predict.py (LightGBM 版) 类似,只是用了 XGBoost 的 model 接口。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_ranking, load_submission_template
from src.features import BASELINE_FEATURE_COLS, build_features
from src.train_xgb import _CAT_MAPPINGS, _encode_categoricals, _CAT_MAPPINGS_PATH


def predict_and_submit(
    model_path: str = "models/baseline_xgb.json",
    data_dir: str = "data/raw",
    output_path: str = "submissions/baseline_v1.parquet",
    clip_min: int = 30,
    clip_max: int = 10800,
) -> pd.DataFrame:
    """跑推理 + 生成提交 parquet。

    ⚠️ 重要:推理前必须先用同一份 cat 映射(从训练保存一份,或者重新 fit 同一份数据)。
    这里为了简洁,直接从训练文件重新 fit 一遍映射(假设训练和推理字段一致)。
    """
    # 0. 先 fit 类别映射(从训练数据学)
    print("从训练数据学习类别映射...")
    from src.data_loader import load_training
    df_train = load_training(data_dir)
    df_train_feat = build_features(df_train)
    feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df_train_feat.columns]
    # v0.1:尝试从持久化文件加载(优先),失败才重新 fit
    global _CAT_MAPPINGS_PATH
    candidate_paths = list(Path("models/feature_lists").glob("*_cat_mappings.json")) if Path("models/feature_lists").exists() else []
    if candidate_paths:
        # 用最新的 cat mappings(覆盖)
        latest = max(candidate_paths, key=lambda p: p.stat().st_mtime)
        _CAT_MAPPINGS_PATH = latest
        # _encode_categoricals(fit=False) 会自动从 _CAT_MAPPINGS_PATH 加载
        _encode_categoricals(df_train_feat[feat_cols], fit=False)
        print(f"  从持久化文件加载 cat 映射: {latest}")
    else:
        _encode_categoricals(df_train_feat[feat_cols], fit=True)  # 只为学映射,丢弃返回值
        print(f"  重新 fit cat 映射")

    # 1. 加载模型
    print(f"加载模型: {model_path}")
    booster = xgb.Booster()
    booster.load_model(model_path)

    # 2. 加载 ranking 数据 + 跑特征
    print("加载 ranking 数据...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 3. 选择特征并编码
    X = df_rank_feat[feat_cols]
    X_enc = _encode_categoricals(X, fit=False)

    # 4. 预测
    print("预测...")
    dinfer = xgb.DMatrix(X_enc)
    y_pred = booster.predict(dinfer)

    # 5. 裁剪到合理范围
    y_pred = np.clip(y_pred, clip_min, clip_max)

    # 6. 对齐到提交模板
    print(f"对齐到提交模板...")
    template = load_submission_template(data_dir)

    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_feat["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_pred,
    })

    submission = template[["MVT_ID_mvt"]].merge(
        pred_df, on="MVT_ID_mvt", how="left"
    )

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"⚠️ 有 {n_missing} 个 MVT_ID 没匹配到预测值,用全局中位数填充")
        median_pred = float(np.nanmedian(y_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    # 7. 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)

    print(f"\n✅ 提交文件已生成: {output_path}")
    print(f"  总行数: {len(submission):,}")
    print(f"  预测值范围: [{submission['TAXITIME_SEC_mvt'].min():.1f}, "
          f"{submission['TAXITIME_SEC_mvt'].max():.1f}] 秒")
    print(f"  预测值中位数: {submission['TAXITIME_SEC_mvt'].median():.1f} 秒")

    return submission


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 XGBoost 推理")
    parser.add_argument("--model", default="models/baseline_xgb.json")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/jubilant-puzzle_v1.parquet")
    args = parser.parse_args()

    predict_and_submit(
        model_path=args.model,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
