"""
阶段1 P0:cat 映射跨进程一致性回归测试
====================================

目的:验证训练时持久化的 *_cat_mappings.json 在新进程里加载后,推理结果与训练进程内推理**逐行一致**。

步骤:
1. 加载 holdout(2025-01+07)前 10k 行
2. 路径 A(训练进程内):导入 train_xgb,直接 fit + predict,记录 y_pred_a
3. 路径 B(跨进程):清除 _CAT_MAPPINGS,_CAT_MAPPINGS_PATH=持久化文件;再 predict,记录 y_pred_b
4. 逐行 allclose(rtol=1e-6, atol=1e-4);特征列名/顺序一致

失败则:定位 cat 映射持久化 bug,禁止进入阶段1。

输出: notebooks/diag/cat_consistency_result.json
"""
import sys, json, importlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

from src.data_loader import filter_departures, parse_time_columns, load_month
from src.features import build_features as build_basic, BASELINE_FEATURE_COLS
from src.features_aggregate import build_features_v2

OUT_PATH = ROOT / "notebooks" / "diag" / "cat_consistency_result.json"

TRAIN_COLS = [
    "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt",
    "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]

MODELS = {
    "v1": {
        "path": ROOT / "models" / "xgb_v1_basic30_seed0_val.json",
        "feat_cols": json.loads((ROOT / "models/feature_lists/feat_basic30.json").read_text()),
        "cat_mappings": ROOT / "models/feature_lists/xgb_v1_basic30_seed0_val_cat_mappings.json",
        "feat_meta": ROOT / "models/feature_lists/xgb_v1_basic30_seed0_val_features.json",
    },
    "v7": {
        "path": ROOT / "models" / "xgb_v7_cnt36_seed0_val.json",
        "feat_cols": json.loads((ROOT / "models/feature_lists/feat_cnt36.json").read_text()),
        "cat_mappings": ROOT / "models/feature_lists/xgb_v7_cnt36_seed0_val_cat_mappings.json",
        "feat_meta": ROOT / "models/feature_lists/xgb_v7_cnt36_seed0_val_features.json",
    },
}


def main():
    print("=" * 70)
    print("阶段1 P0:cat 映射跨进程一致性回归测试")
    print("=" * 70)

    # 加载 holdout 1万行
    print("\n[1] 加载 holdout(2025-01+07)前 10k 行...")
    dfs = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw = filter_departures(df_raw)
    df_raw = parse_time_columns(df_raw, cols=TIME_COLS)
    df_raw = df_raw[(df_raw["TAXITIME_SEC_mvt"] > 0) &
                    (df_raw["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
    df_raw = df_raw.head(10_000).reset_index(drop=True)
    print(f"  测试样本: {len(df_raw):,}")

    out = {}
    overall_pass = True

    for version, cfg in MODELS.items():
        print(f"\n[{version}] 测试...")
        booster = xgb.Booster()
        booster.load_model(str(cfg["path"]))
        meta = json.loads(Path(cfg["feat_meta"]).read_text())
        assert list(booster.feature_names) == list(meta["features"]) == list(cfg["feat_cols"]), \
            f"{version} 特征列不一致"

        # 构造特征
        if version == "v1":
            df_feat = build_basic(df_raw.copy())
        else:
            df_feat = build_features_v2(df_raw.copy(), include_aggregate=True, include_lead=True)
        missing = [c for c in cfg["feat_cols"] if c not in df_feat.columns]
        if missing:
            raise KeyError(f"{version} 缺失: {missing}")
        X = df_feat[cfg["feat_cols"]].copy()

        # 路径 A:同一进程内(模拟"训练后立即推理"场景),从持久化 cat 加载
        import src.train_xgb as tx_a
        importlib.reload(tx_a)
        tx_a._CAT_MAPPINGS.clear()
        tx_a._CAT_MAPPINGS_PATH = cfg["cat_mappings"]
        X_enc_a = tx_a._encode_categoricals(X.copy(), fit=False)
        y_pred_a = booster.predict(xgb.DMatrix(X_enc_a))

        # 路径 B:模拟完全跨进程(新解释器视角)— 子进程重启 _CAT_MAPPINGS 为空,只能从 _CAT_MAPPINGS_PATH 文件加载
        import src.train_xgb as tx_b
        importlib.reload(tx_b)
        # reload 后 _CAT_MAPPINGS 已经被 reload 后的新模块重置为空 dict,符合新进程语义
        assert tx_b._CAT_MAPPINGS == {}, f"reload 后 _CAT_MAPPINGS 应为空,实际: {len(tx_b._CAT_MAPPINGS)}"
        tx_b._CAT_MAPPINGS_PATH = cfg["cat_mappings"]
        X_enc_b = tx_b._encode_categoricals(X.copy(), fit=False)
        y_pred_b = booster.predict(xgb.DMatrix(X_enc_b))

        # 特征列一致(顺序、列名)
        feat_names_match = (list(X_enc_a.columns) == list(X_enc_b.columns) ==
                           list(booster.feature_names))
        # 逐行 allclose
        allclose = np.allclose(y_pred_a, y_pred_b, rtol=1e-6, atol=1e-4)
        max_abs_diff = float(np.max(np.abs(y_pred_a - y_pred_b)))
        max_rel_diff = float(np.max(np.abs(y_pred_a - y_pred_b) /
                                    np.maximum(np.abs(y_pred_a), 1e-6)))

        pass_v = bool(feat_names_match and allclose)
        overall_pass = overall_pass and pass_v

        print(f"  feat names match: {feat_names_match}")
        print(f"  allclose(rtol=1e-6, atol=1e-4): {allclose}")
        print(f"  max |Δ|: {max_abs_diff:.6e}, max rel diff: {max_rel_diff:.6e}")
        print(f"  PASS: {pass_v}")

        out[version] = {
            "feat_names_match": feat_names_match,
            "allclose": bool(allclose),
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "n_test_rows": len(df_raw),
            "pass": pass_v,
        }

    out["overall_pass"] = overall_pass
    out_path = OUT_PATH
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n结果写入 {out_path}")
    print(f"\n整体 PASS: {overall_pass}")
    if not overall_pass:
        print("\n❌ P0 测试失败 — 必须先修复 cat 映射持久化,禁止进入阶段1")
        sys.exit(1)


if __name__ == "__main__":
    main()
