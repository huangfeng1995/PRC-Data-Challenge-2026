"""
XGBoost full-features 推理 + 生成提交 parquet
============================================

跟 predict_xgb.py 不同:
- 支持所有特征(basic + aggregate + lead)
- 从训练文件重新 fit 类别映射(确保 cat 编码一致)
- 从训练文件重新算 lag/lead(因为 ranking 数据需要同样的 group/time 索引)

注意:rank 数据是 ranking parquet(屏蔽 TAXITIME_SEC_mvt),所以聚合 target 会有 NaN,
这时 count 还能算,mean/p90 算不了 —— 这正是我们想要的:在 ranking 上 mean/p90 就是 NaN。
但 train 和 rank 数据一起算 lag/lead 就能解决:ranking 的 lag 窗口里包含 train 的真实值,
mean/p90 就能算出来。这正是 v1 baseline 缺的。

实现思路:
  1. 加载 train + ranking 合并 → 一起算 lag/lead(避免 rank 缺数据)
  2. 分别 train 和 rank 选特征列
  3. 用 booster 推理
  4. 按模板 merge + 保存

⚠️ 注意:训练时验证集 target 是真实的,聚合里的 mean/p90 包含验证期数据,
这会引入轻微的"未来信息泄漏"。这是 trade-off:为了推理时 mean/p90 也有值。
更严格的做法是用 sliding window OOF 算 lag/lead,工作量会大很多,先按简单方案上。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_ranking, load_submission_template, load_training
from src.features import BASELINE_FEATURE_COLS, build_features
from src.features_aggregate import add_aggregate_features
from src.features_laglead import add_lead_features
from src.train_xgb import _encode_categoricals, _CAT_MAPPINGS_PATH


def _build_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """基础 + lag + lead 完整特征。"""
    df = build_features(df)
    df = add_aggregate_features(df)
    df = add_lead_features(df)
    return df


def _get_feat_cols(df: pd.DataFrame, drop_target_stats: bool = True) -> list[str]:
    """所有用于训练的特征列。基础 + 所有 *_15m / *_30m / *_60m 列(lag + lead)。

    v3.5:增加 drop_target_stats 参数。
    - drop_target_stats=True (默认):只保留 cnt_*(no_laglead 模式)
    - drop_target_stats=False:保留 cnt + mean + p90(legacy 模式)
    """
    base = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    if drop_target_stats:
        agg = [c for c in df.columns if c.endswith("m") and "cnt_" in c]
    else:
        agg = [c for c in df.columns
               if c.endswith("m") and ("cnt_" in c or "mean_" in c or "p90_" in c)]
    return base + agg


def predict_and_submit(
    model_path: str = "models/baseline_xgb.json",
    data_dir: str = "data/raw",
    output_path: str = "submissions/jubilant-puzzle_v2.parquet",
    clip_min: int = 30,
    clip_max: int = 10800,
    apply_quantile_mapping: bool = False,  # v3:是否做 quantile mapping 对齐分布(默认关闭,等 v3 验证)
    drop_target_stats: bool = True,  # v3.5:默认 no_laglead 模式(只用 cnt)
) -> pd.DataFrame:
    """跑推理 + 生成提交 parquet(完整特征版)。"""
    # 1. 加载训练数据 → 训练基础特征(只为 fit cat 映射)
    print("加载训练数据(只为学 cat 映射)...")
    df_train = load_training(data_dir)
    df_train_feat = build_features(df_train)
    # v0.1:优先从持久化文件加载 cat 映射,避免跨进程漂移
    global _CAT_MAPPINGS_PATH
    candidate_paths = list(Path("models/feature_lists").glob("*_cat_mappings.json")) if Path("models/feature_lists").exists() else []
    if candidate_paths:
        _CAT_MAPPINGS_PATH = max(candidate_paths, key=lambda p: p.stat().st_mtime)
        print(f"  从持久化加载 cat 映射: {_CAT_MAPPINGS_PATH.name}")
    if not candidate_paths:
        from src.train_xgb import _encode_categoricals as _ec
        _ec(df_train_feat[[c for c in BASELINE_FEATURE_COLS if c in df_train_feat.columns]], fit=True)

    # 2. 加载模型
    print(f"加载模型: {model_path}")
    booster = xgb.Booster()
    booster.load_model(model_path)

    # 3. 加载 ranking 数据 → 基础特征
    print("加载 ranking 数据...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 4. 合并 train + rank 一起算 lag/lead(这样 rank 的 mean/p90 才有值)
    print("合并 train + ranking 算 lag/lead ...")
    # 用一个标记列区分
    df_train_feat["__is_train__"] = True
    df_rank_feat["__is_train__"] = False
    # 保持 TAXITIME_SEC_mvt 列(train 有,rank 是 NaN),加 __is_dep__ 等需要的列
    for col in ["ADEP_mvt", "RUNWAY_mvt", "EOBT_1_flt", "TAXITIME_SEC_mvt"]:
        if col not in df_rank_feat.columns:
            df_rank_feat[col] = np.nan

    common_cols = list(set(df_train_feat.columns) & set(df_rank_feat.columns))
    df_all = pd.concat([df_train_feat[common_cols], df_rank_feat[common_cols]], axis=0, ignore_index=True)
    print(f"  合并后: {len(df_all):,} 行 (train {len(df_train_feat):,} + rank {len(df_rank_feat):,})")
    # ⚠️ concat 会把两边的 category 统一成 object,要重新转
    from src.features import CATEGORICAL_COLS, to_categorical
    cat_in_all = [c for c in CATEGORICAL_COLS if c in df_all.columns]
    df_all = to_categorical(df_all, cat_in_all)

    # 算 lag/lead(在合并数据上)
    print("算 lag features...")
    df_all = add_aggregate_features(df_all)
    print("算 lead features...")
    df_all = add_lead_features(df_all)

    # 5. 拆分回 train 和 rank
    train_mask = df_all["__is_train__"].values
    df_all = df_all.drop(columns=["__is_train__"])

    df_train_full = df_all[train_mask].reset_index(drop=True)
    df_rank_full = df_all[~train_mask].reset_index(drop=True)

    # 关键:特征列要从合并后的 df_all 取,这样 lag/lead 才在列里
    feat_cols = _get_feat_cols(df_all, drop_target_stats=drop_target_stats)
    print(f"  完整特征数: {len(feat_cols)}")

    # fit 类别映射(用 train_full,因为 train 数据量大,category 集合更全)
    _encode_categoricals(df_train_full[feat_cols], fit=True)

    # 6. 选特征 + 编码 + 推理
    print("推理...")
    # 按模型期望的列顺序重排(避免 XGBoost feature_names mismatch)
    expected = booster.feature_names
    if expected is None:
        raise RuntimeError("模型没有 feature_names,无法对齐")
    missing = [c for c in expected if c not in df_rank_full.columns]
    extra = [c for c in feat_cols if c not in expected]
    if missing:
        raise RuntimeError(f"推理数据缺列: {missing[:5]}")
    if extra:
        print(f"  [warn] 训练有但推理不用: {extra[:5]}")
    feat_cols_aligned = expected  # 直接用模型期望的顺序
    X_rank = df_rank_full[feat_cols_aligned]
    X_rank_enc = _encode_categoricals(X_rank, fit=False)
    dinfer = xgb.DMatrix(X_rank_enc)
    y_pred = booster.predict(dinfer)

    # v3:可选 quantile mapping — 把 rank 的预测分布对齐到 train target 分布
    # (rank 没 target,假设 train/test 同期分布;50% 强度是折中,避免硬映射)
    if apply_quantile_mapping:
        train_target = load_training(data_dir)["TAXITIME_SEC_mvt"]
        train_target = train_target[(train_target > 0) & (train_target < 10800)].values
        rank_quantiles = np.quantile(y_pred, np.linspace(0.001, 0.999, 999))
        target_quantiles = np.quantile(train_target, np.linspace(0.001, 0.999, 999))
        mapped = np.interp(y_pred, rank_quantiles, target_quantiles).astype("float64")
        # 50/50 混合 — 完全映射可能过强,完全不用又不对
        y_pred = 0.5 * y_pred + 0.5 * mapped
        print(f"  [info] 应用 50% quantile mapping (rank→train 分布)")

    y_pred = np.clip(y_pred, clip_min, clip_max)

    # 7. 对齐模板
    template = load_submission_template(data_dir)
    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_full["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": y_pred,
    })
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        print(f"⚠️ 有 {n_missing} 个 MVT_ID 没匹配到预测值,用全局中位数填充")
        median_pred = float(np.nanmedian(y_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    # 8. 保存
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
    parser = argparse.ArgumentParser(description="PRC 2026 XGBoost 完整特征推理")
    parser.add_argument("--model", default="models/baseline_xgb.json")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/jubilant-puzzle_v2.parquet")
    parser.add_argument("--keep-target-stats", action="store_true",
                        help="保留 mean/p90 列(legacy 模式,默认 False = no_laglead)")
    args = parser.parse_args()

    predict_and_submit(
        model_path=args.model,
        data_dir=args.data_dir,
        output_path=args.output,
        drop_target_stats=not args.keep_target_stats,
    )


if __name__ == "__main__":
    main()
