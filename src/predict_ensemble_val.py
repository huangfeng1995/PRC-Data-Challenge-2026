"""
用 OOF 验证模型(没等 final 训练完成)生成 ensemble 提交
============================================================

策略:
- 加载 models/ensemble_val/ 下的 6 个验证模型(3 XGB + 3 LGB)
- 用每个模型预测 rank 数据
- 按 weights.json 加权平均
- 生成 v4 提交

v4 vs v3:
- v3:单个 XGB tuned 模型,leaderboard RMSE 待揭晓
- v4:XGB+LGB ensemble(3 seeds each),验证集 RMSE 231.56 vs v3 232.71
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from src.data_loader import load_ranking, load_submission_template
from src.features import BASELINE_FEATURE_COLS, build_features, to_categorical
from src.features_aggregate import add_aggregate_features
from src.features_laglead import add_lead_features
from src.train_xgb import _encode_categoricals


def predict_ensemble_val(
    models_dir: str = "models/ensemble_val",
    data_dir: str = "data/raw",
    output_path: str = "submissions/jubilant-puzzle_v4.parquet",
    clip_min: int = 30,
    clip_max: int = 10800,
) -> pd.DataFrame:
    """用 OOF 验证模型生成 ensemble 提交。"""

    models_dir = Path(models_dir)
    weights_path = models_dir / "weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path},先跑 train_ensemble 至少到 OOF 阶段")

    meta = json.loads(weights_path.read_text())
    weights = meta["weights"]
    blend_mode = meta["blend_mode"]
    feat_cols = meta["feat_cols"]
    model_keys = meta["models"]

    print(f"加载 {len(model_keys)} 个验证模型 + weights {weights}, blend={blend_mode}")

    # 1. 加载 train 数据(只为学 cat 映射)
    print("加载训练数据 + 构造特征(只为 fit cat 映射)...")
    from src.data_loader import load_training
    df_train = load_training(data_dir)
    df_train_feat = build_features(df_train)

    # 2. 加载 ranking 数据 + 基础特征
    print("加载 ranking 数据...")
    df_rank = load_ranking(data_dir)
    df_rank_feat = build_features(df_rank)

    # 3. 合并算 lag/lead
    df_train_feat["__is_train__"] = True
    df_rank_feat["__is_train__"] = False
    for col in ["ADEP_mvt", "RUNWAY_mvt", "EOBT_1_flt", "TAXITIME_SEC_mvt"]:
        if col not in df_rank_feat.columns:
            df_rank_feat[col] = np.nan

    common_cols = list(set(df_train_feat.columns) & set(df_rank_feat.columns))
    df_all = pd.concat([df_train_feat[common_cols], df_rank_feat[common_cols]], axis=0, ignore_index=True)
    print(f"  合并后: {len(df_all):,}")

    cat_in_all = [c for c in ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
                                "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
                                "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
                                "airport_hour"] if c in df_all.columns]
    df_all = to_categorical(df_all, cat_in_all)

    print("算 lag features...")
    df_all = add_aggregate_features(df_all)
    print("算 lead features...")
    df_all = add_lead_features(df_all)

    train_mask = df_all["__is_train__"].values
    df_all = df_all.drop(columns=["__is_train__"])

    df_rank_full = df_all[~train_mask].reset_index(drop=True)

    # 4. fit cat 编码(用 train_full)
    df_train_full = df_all[train_mask].reset_index(drop=True)
    X_train_for_cat = df_train_full[feat_cols]
    _encode_categoricals(X_train_for_cat, fit=True)

    # 5. rank 编码 + 推理
    print("推理 6 个模型 + 加权融合...")
    X_rank = df_rank_full[feat_cols]
    X_rank_enc = _encode_categoricals(X_rank, fit=False)

    model_preds = {}  # key: model_name (e.g. "xgb"), value: avg_pred of seeds
    for key in model_keys:
        model_type, _, seed_str = key.partition("__seed_")
        model_name = model_type  # "xgb" 或 "lgb"

        pkl_path = models_dir / f"{key}.pkl"
        if not pkl_path.exists():
            print(f"  ⚠️ 缺失 {pkl_path},跳过")
            continue

        # 注:wrapper.save 实际调 XGB/LGB 的 native save_model(写入 .pkl 后缀的 JSON 格式)
        if model_name == "xgb":
            booster = xgb.Booster()
            booster.load_model(str(pkl_path))
            # XGBoost 模型期望已经整数编码过的 cat(和训练时一致)
            # _encode_categoricals 在 fit 时已建立了映射,fit=False 直接用
            # 注意:X_rank_enc 里有些 cat 列可能还是 object(因为 dtype 不是 category)
            # 我们再转换一次
            X_for_xgb = _encode_categoricals(X_rank.copy(), fit=False)
            dinfer = xgb.DMatrix(X_for_xgb)
            y_pred = booster.predict(dinfer)
        elif model_name == "lgb":
            booster = lgb.Booster(model_file=str(pkl_path))
            # LGB 在训练时 cat 列的 categories 来自 winter_train,但 rank 数据有不同/更多 categories
            # 解决:把 X_rank 转成和训练时同样 cat categories(用 LGB 内部 store 的 cat)
            # 但 booster 不暴露 cat info;替代办法:把 cat 列转成 int code (用训练时的 cat→int 映射)
            X_for_lgb = _encode_categoricals(X_rank.copy(), fit=False)
            # LGB 训练时 cat 列名是 XGB 同名,所以我们复用 cat encoding
            # 把 cat 列以外的 category dtype 转 object(LGB 需要的是 category or int)
            y_pred = booster.predict(X_for_lgb, num_iteration=booster.best_iteration)
        else:
            raise ValueError(f"未知模型: {model_name}")

        if model_name not in model_preds:
            model_preds[model_name] = []
        model_preds[model_name].append(y_pred)
        print(f"  ✓ {key}")

    # 6. 融合:每个 model 内 seed 平均,然后按 weights 加权
    final_pred = np.zeros(len(df_rank_full))
    for model_name, preds_list in model_preds.items():
        avg = np.mean(preds_list, axis=0)
        w = weights.get(model_name, 0.0)
        final_pred += w * avg
        print(f"  {model_name} ({len(preds_list)} seed avg): weight={w:.3f}")

    final_pred = np.clip(final_pred, clip_min, clip_max)

    # 7. 对齐模板 + 保存
    template = load_submission_template(data_dir)
    pred_df = pd.DataFrame({
        "MVT_ID_mvt": df_rank_full["MVT_ID_mvt"].values,
        "TAXITIME_SEC_mvt": final_pred,
    })
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    n_missing = submission["TAXITIME_SEC_mvt"].isna().sum()
    if n_missing > 0:
        median_pred = float(np.nanmedian(final_pred))
        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].fillna(median_pred)
    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, index=False)

    print(f"\n✅ 提交生成: {output_path}")
    print(f"  总行数: {len(submission):,}")
    print(f"  预测值范围: [{submission['TAXITIME_SEC_mvt'].min():.1f}, "
          f"{submission['TAXITIME_SEC_mvt'].max():.1f}] 秒")
    print(f"  预测值中位数: {submission['TAXITIME_SEC_mvt'].median():.1f} 秒")
    print(f"  OOF 验证 RMSE: 231.56 (winter split)")

    return submission


def main():
    parser = argparse.ArgumentParser(description="用 OOF 验证模型生成 ensemble 提交")
    parser.add_argument("--models-dir", default="models/ensemble_val")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="submissions/jubilant-puzzle_v4.parquet")
    args = parser.parse_args()

    predict_ensemble_val(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
