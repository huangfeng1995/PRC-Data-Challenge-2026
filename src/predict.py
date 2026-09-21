"""
推理 + 生成提交 parquet (LightGBM 版本)
======================================

⚠️ 注意:这个文件是 LightGBM 推理版本,只用于 train_lgb.py 训练出来的模型。
   实际推荐使用 predict_xgb.py(XGBoost 版本,无 libomp 依赖)。

功能:
1. 加载训练好的 LightGBM 模型
2. 对 ranking.parquet(2026-01 + 2026-07)做特征工程
3. 预测 TAXITIME_SEC_mvt
4. 对齐到 submitting.parquet 模板的 MVT_ID_mvt,生成可提交的 parquet

⚠️ 重要:对 ranking 数据做特征工程时,只能用"可实时获取"的字段,
   不能用 AOBT_3_flt / BLOCK_TIME_UTC_mvt(它们包含泄漏信息,且 ranking 里被屏蔽了)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data_loader import load_ranking, load_submission_template
from src.features import BASELINE_FEATURE_COLS, build_features


def predict_and_submit(
    model_path: str = "models/baseline_lgb.txt",
    data_dir: str = "data/raw",
    output_path: str = "submissions/baseline_v1.parquet",
    clip_min: int = 30,
    clip_max: int = 7200,
) -> pd.DataFrame:
    """跑推理 + 生成提交 parquet。

    参数:
        model_path: 训练好的模型路径
        data_dir: 数据目录(包含 ranking.parquet 和 submitting.parquet)
        output_path: 提交文件输出路径
        clip_min/max: 预测值裁剪范围(秒)
    """
    # 1. 加载模型
    print(f"加载模型: {model_path}")
    booster = lgb.Booster(model_file=model_path)

    # 2. 加载 ranking 数据 + 跑特征
    print("加载 ranking 数据...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 3. 选择特征(只保留训练时用过的列)
    feat_cols = [c for c in BASELINE_FEATURE_COLS if c in df_rank_feat.columns]
    X = df_rank_feat[feat_cols]

    # 类别列对齐:训练时见过的 category,推理时也可能见到新的,设为 NaN
    # LightGBM 原生会处理 unknown category,不用手动对齐

    # 4. 预测
    print("预测...")
    y_pred = booster.predict(X)

    # 5. 裁剪到合理范围
    y_pred = np.clip(y_pred, clip_min, clip_max)

    # 6. 对齐到提交模板
    print(f"对齐到提交模板...")
    template = load_submission_template(data_dir)

    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_feat["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_pred,
    })

    # 用模板的 MVT_ID 顺序,左连接预测值
    submission = template[["MVT_ID_mvt"]].merge(
        pred_df, on="MVT_ID_mvt", how="left"
    )

    # 检查是否有缺失
    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"⚠️ 有 {n_missing} 个 MVT_ID 没匹配到预测值,用全局中位数填充")
        median_pred = float(np.nanmedian(y_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)

    # 检查类型
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
    parser = argparse.ArgumentParser(description="PRC 2026 推理 + 生成提交")
    parser.add_argument("--model", default="models/baseline_lgb.txt")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/baseline_v1.parquet")
    args = parser.parse_args()

    predict_and_submit(
        model_path=args.model,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
