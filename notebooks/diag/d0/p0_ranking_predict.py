"""
P0 九模型 booster 预测 ranking.parquet 真榜 (2026-01+07)
=========================================================
- booster: P0 LGB×3 + CAT×3 + v15 XGB×3 (json)
- 特征: FEATS71 整数编码, 与训练一致
- TE/LUT/offset 只用训练折 2025-02..12, 不碰 ranking
- 时间戳先 .astype("datetime64[ns]") (原始 us)
- 9 模型简单平均, clip [30, 10800]
- 输出 submissions/jubilant-puzzle_v20.parquet
- 不上传, 不重训, 不检查训练
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

TRAIN_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]

FEATS71 = [
    "sched_hour", "sched_dow", "sched_month", "sched_day", "sched_minute",
    "is_weekend", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "airport_hour",
    "is_domestic_estimate", "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt",
    "is_diverted",
    "iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
    "dep_mvt_cnt_airport_5m", "dep_mvt_cnt_airport_10m", "dep_mvt_cnt_airport_15m",
    "dep_queue_n_airport",
    "dep_mvt_cnt_runway_5m", "dep_mvt_cnt_runway_10m", "dep_mvt_cnt_runway_15m",
    "dep_queue_n_runway",
    "ref_was_estimated",
    "mvt_n_ahead", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
    "mvt_w_actual_15", "mvt_w_actual_30",
    "mvt_q_peak", "mvt_q_mean",
    "mvt_na_J", "mvt_na_H", "mvt_na_M", "mvt_na_L",
    "mvt_na_heavy_frac", "mvt_pos_rw", "mvt_na_heavy_x_pos",
    "mvt_mu_rwy_5", "mvt_w_actual_5",
    "dep_cnt_m_5", "dep_cnt_m_15", "dep_cnt_m_30",
    "aobt_cnt_m_5", "aobt_cnt_m_15", "aobt_cnt_m_30",
    "eobt_ahead_cnt_5", "eobt_ahead_cnt_15", "eobt_ahead_cnt_30",
    "dly_aobt_eobt", "dly_aobt_iobt", "dly_aobt_sched", "dly_lobt_aobt",
    "t_unimp_r7",
    "rwy_month_share", "ap_month_hourly_density",
]
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]
SEEDS = [42, 123, 2024]

CAT_MAP = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
P0_DIR = D0 / "p0_models"
V15_XGB_DIR = ROOT / "models" / "v15_clean_retrain"
SUB_PATH = ROOT / "submissions" / "jubilant-puzzle_v20.parquet"

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L, _r8_drift_lib as D


def to_ns(s):
    """原始数据是 datetime64[us, UTC] (微秒), 显式转 ns naive, 避免浮点误差"""
    s2 = pd.to_datetime(s)
    if hasattr(s2.dtype, "tz") and s2.dtype.tz is not None:
        s2 = s2.dt.tz_localize(None)
    return s2.astype("datetime64[ns]")


def build_features(df_raw, ref_dep, lut, label):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    dfb = build_basic(df)
    df_full = add_queue_features(dfb.copy(), include_a=True, include_b=True,
                                 include_c=True, include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    for k, v in D.compute_drift_features(df_full, ref_dep=ref_dep, label=label).items():
        df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    for k, v in L.compute_extra_features(df_full[dep_mask], label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    for k, v in L.compute_mvt_anchor_features(df_full[dep_mask],
                                              t_unimp_vals=t_vals[dep_mask],
                                              label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    # R5: 时间戳统一 ns
    df_full["MVT_TIME_UTC_mvt"] = to_ns(df_full["MVT_TIME_UTC_mvt"])
    return df_full


def encode_X_lgb(df):
    X = df[FEATS71].copy()
    for col in HIGH_CARD_CATS:
        if col in X.columns:
            m = CAT_MAP.get(col, {})
            X[col] = X[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def encode_X_xgb(df):
    X = df[FEATS71].copy()
    for col in FEATS71:
        if col in X.columns:
            if X[col].dtype.name == "category":
                X[col] = X[col].cat.codes.astype("int32").astype("float32")
            elif X[col].dtype.name == "object":
                X[col] = X[col].astype(str).map(CAT_MAP.get(col, {})).fillna(-1).astype("int32").astype("float32")
    for col in HIGH_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype("float32")
    return X.astype("float32")


def encode_X_cat(df):
    X = df[FEATS71].copy()
    for col in LOW_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype("object").where(X[col].notna(), "__NA__").astype(str)
    for col in HIGH_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype(str).map(CAT_MAP.get(col, {})).fillna(-1).astype("int32")
    return X


def main():
    t0 = time.time()
    print("=" * 70)
    print("[P0 真榜推理] ranking.parquet → jubilant-puzzle_v20.parquet")
    print("=" * 70)

    # ============= [1] 加载训练折用于 LUF/offset, 严格 2025-02..12 =============
    print("\n[1] 加载训练折 2025-02..12 用于 LUT/offset")
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr_dfs = []
    for m in train_months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        tr_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
    df_tr_dep_clean = df_tr_raw[
        (df_tr_raw["PHASE_mvt"] == "DEP") &
        (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
        (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)
    ].copy()
    lut = L.build_t_unimp_lut(df_tr_dep_clean)
    print(f"  训练 DEP (2-12, TX∈(0,10800]): {len(df_tr_dep_clean):,}")
    print(f"  LUT global_q30 = {lut['global_q30']:.0f}s")

    # ============= [2] 加载 ranking, 仅取对应列 =============
    print("\n[2] 加载 ranking (2026-01+07, 不含 TAXITIME/BLOCK)")
    rank_cols = [c for c in TRAIN_COLS if c != "TAXITIME_SEC_mvt"]
    # 显式不读 BLOCK_TIME_UTC_mvt (用户协议: 不碰 DEP 的 TAXITIME 或 BLOCK)
    # 但 build_features 需要 MVT_TIME_UTC_mvt 等时间戳; ranking 一定有
    rank_df = pd.read_parquet("data/raw/ranking.parquet", columns=rank_cols)
    print(f"  ranking 总: {len(rank_df):,}")
    print(f"  ranking DEP: {(rank_df['PHASE_mvt']=='DEP').sum():,}")

    # ============= [3] 重建 ranking 特征 =============
    print("\n[3] 重建 ranking 特征 (LUT/offset 用训练折)")
    t1 = time.time()
    # parse_time_columns 必需要所有 TIME_COLS; BLOCK 在 ranking 不存在, 会被 parse_time_columns
    # 自动转为 NaT, 后续流程已过滤
    rank_full = build_features(rank_df, df_tr_raw, lut, label="rank")
    rank_dep = rank_full[rank_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    print(f"  rank_dep: {len(rank_dep):,}  ({time.time()-t1:.0f}s)")

    # 自检
    expected_ids = rank_df[rank_df["PHASE_mvt"] == "DEP"]["MVT_ID_mvt"].astype(float).values
    actual_ids = rank_dep["MVT_ID_mvt"].astype(float).values
    assert len(expected_ids) == len(actual_ids), \
        f"ranking DEP 行数不匹配: expected {len(expected_ids)}, got {len(actual_ids)}"
    print(f"  ✓ ranking DEP 行数 {len(rank_dep):,} 匹配")

    # ============= [4] 编码 =============
    X_rk_lgb = encode_X_lgb(rank_dep)
    X_rk_xgb = encode_X_xgb(rank_dep)
    X_rk_cat = encode_X_cat(rank_dep)
    print(f"  X_rk_lgb shape: {X_rk_lgb.shape}")
    print(f"  X_rk_xgb shape: {X_rk_xgb.shape}")
    print(f"  X_rk_cat shape: {X_rk_cat.shape}")

    # ============= [5] 9 模型推理 =============
    print("\n[5] 9 模型推理")
    preds = {}

    # LGB (3 个 P0 booster)
    print("  LGB:")
    for sd in SEEDS:
        b = lgb.Booster(model_file=str(P0_DIR / f"p0_lgb_seed{sd}.txt"))
        p = b.predict(X_rk_lgb).astype(np.float64)
        preds[f"lgb{sd}"] = p
        print(f"    lgb{sd} done", flush=True)

    # XGB (用 v15 json booster, 同训练折同 71 特征)
    print("  XGB (v15 json booster):")
    for sd in SEEDS:
        b = xgb.Booster()
        b.load_model(str(V15_XGB_DIR / f"xgb_seed{sd}.json"))
        d = xgb.DMatrix(X_rk_xgb.values, feature_names=FEATS71)
        p = b.predict(d).astype(np.float64)
        preds[f"xgb{sd}"] = p
        print(f"    xgb{sd} done", flush=True)

    # CAT (3 个 P0 booster)
    print("  CAT:")
    for sd in SEEDS:
        m = CatBoostRegressor()
        m.load_model(str(P0_DIR / f"p0_cat_seed{sd}.cbm"))
        p = m.predict(X_rk_cat).astype(np.float64)
        preds[f"cat{sd}"] = p
        print(f"    cat{sd} done", flush=True)

    # ============= [6] 集成 + clip =============
    print("\n[6] 9 模型集成 + clip [30, 10800]")
    ens_raw = np.mean(np.stack(list(preds.values())), axis=0)
    ens = np.clip(ens_raw, 30.0, 10800.0)
    print(f"  ens_raw: min={ens_raw.min():.1f}, max={ens_raw.max():.1f}, "
          f"med={np.median(ens_raw):.0f}")
    print(f"  ens_clip: min={ens.min():.1f}, max={ens.max():.1f}, "
          f"med={np.median(ens):.0f}")
    n_clipped = int(((ens_raw < 30) | (ens_raw > 10800)).sum())
    print(f"  被 clip 调整的行: {n_clipped:,}")

    # ============= [7] 输出 submitting.parquet =============
    print("\n[7] 输出 jubilant-puzzle_v20.parquet")
    # 关键: 顺序必须和 submitting.parquet 完全一致 (按 MVT_ID_mvt 匹配)
    submitting = pd.read_parquet("data/raw/submitting.parquet")
    print(f"  submitting.parquet 行数: {len(submitting):,}")
    assert set(submitting.columns) == {"MVT_ID_mvt", "TAXITIME_SEC_mvt"}, \
        f"submitting 列异常: {submitting.columns.tolist()}"

    # 建一个 MVT_ID → pred 的映射
    pred_map = pd.DataFrame({
        "MVT_ID_mvt": rank_dep["MVT_ID_mvt"].astype(float).values,
        "TAXITIME_SEC_mvt": np.round(ens, 0).astype(int),
    })
    # 按 submitting 顺序合并
    out = submitting[["MVT_ID_mvt"]].merge(pred_map, on="MVT_ID_mvt", how="left")
    assert len(out) == len(submitting), f"合并后行数不匹配: {len(out)} vs {len(submitting)}"
    assert out["TAXITIME_SEC_mvt"].notna().all(), "NaN 行数: " + \
        f"{int(out['TAXITIME_SEC_mvt'].isna().sum())}"
    assert (out["TAXITIME_SEC_mvt"] >= 30).all(), "负/过小预测: " + \
        f"{int((out['TAXITIME_SEC_mvt']<30).sum())}"
    assert (out["TAXITIME_SEC_mvt"] <= 10800).all(), "过大预测: " + \
        f"{int((out['TAXITIME_SEC_mvt']>10800).sum())}"

    # MVT_ID 集合+顺序完全匹配
    exp_ids = submitting["MVT_ID_mvt"].astype(float).values
    out_ids = out["MVT_ID_mvt"].astype(float).values
    assert set(exp_ids) == set(out_ids), "MVT_ID 集合不匹配"
    assert np.all(exp_ids == out_ids), "MVT_ID 顺序不匹配"
    print(f"  ✓ MVT_ID 集合+顺序 完全对齐 ({len(out):,} 行)")

    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SUB_PATH, index=False)
    print(f"  ✓ 落盘: {SUB_PATH}")
    print(f"  文件大小: {SUB_PATH.stat().st_size:,} bytes")

    # ============= [8] 分布自检 =============
    print("\n[8] 真榜预测分布自检")
    def q(s, p): return float(np.percentile(s, p))
    print(f"  n      = {len(ens):,}")
    print(f"  min    = {ens.min():.0f}")
    print(f"  p25    = {q(ens, 25):.0f}")
    print(f"  median = {q(ens, 50):.0f}")
    print(f"  p75    = {q(ens, 75):.0f}")
    print(f"  p95    = {q(ens, 95):.0f}")
    print(f"  max    = {ens.max():.0f}")
    print(f"  >10800: {(ens > 10800).sum():,}")
    print(f"  >3600 : {(ens > 3600).sum():,}")
    print(f"  >1800 : {(ens > 1800).sum():,}")

    # 与 holdout ENS 对比
    holdout_pred = pd.read_parquet(D0 / "p0_holdout_predictions.parquet")
    h_ens = holdout_pred["TAXI_PRED_s_ENS"].values.astype(np.float64)
    print(f"\n  --- 与 holdout ENS (n={len(h_ens):,}) 对比 ---")
    print(f"  {'指标':10s} {'真榜 (rank)':>15s} {'holdout':>15s} {'Δ':>10s}")
    for name, v in [("min", ens.min()), ("p25", q(ens, 25)),
                    ("median", q(ens, 50)), ("p75", q(ens, 75)),
                    ("p95", q(ens, 95)), ("max", ens.max())]:
        hv = float(np.percentile(h_ens, {"min": 0, "p25": 25, "median": 50,
                                          "p75": 75, "p95": 95, "max": 100}[name]))
        diff = v - hv
        print(f"  {name:10s} {v:>15.0f} {hv:>15.0f} {diff:>+10.1f}")

    # 落 JSON
    out_meta = {
        "task": "p0_ranking_predict_v20",
        "n_ranking_dep": int(len(rank_dep)),
        "ensemble_method": "9-model simple mean + clip [30, 10800]",
        "models": ["lgb42", "lgb123", "lgb2024", "xgb42_v15", "xgb123_v15", "xgb2024_v15",
                   "cat42", "cat123", "cat2024"],
        "n_clipped_by_clip": n_clipped,
        "submission_path": str(SUB_PATH),
        "submission_rows": int(len(out)),
        "mvt_id_match": "✓ set+order 完全对齐 submitting.parquet",
        "distribution_rank": {
            "n": int(len(ens)), "min": float(ens.min()),
            "p25": q(ens, 25), "median": q(ens, 50),
            "p75": q(ens, 75), "p95": q(ens, 95),
            "max": float(ens.max()),
            "n_pred_gt_1800": int((ens > 1800).sum()),
            "n_pred_gt_3600": int((ens > 3600).sum()),
            "n_pred_gt_10800": int((ens > 10800).sum()),
        },
        "vs_holdout_ENS": {
            "median_diff": q(ens, 50) - float(np.median(h_ens)),
            "p95_diff": q(ens, 95) - float(np.percentile(h_ens, 95)),
            "max_diff": float(ens.max() - h_ens.max()),
        },
        "model_files": {
            "lgb": [f"notebooks/diag/d0/p0_models/p0_lgb_seed{sd}.txt" for sd in SEEDS],
            "xgb": [f"models/v15_clean_retrain/xgb_seed{sd}.json" for sd in SEEDS],
            "cat": [f"notebooks/diag/d0/p0_models/p0_cat_seed{sd}.cbm" for sd in SEEDS],
        },
        "submission_status": "NOT uploaded (等用户确认)",
        "total_time_sec": round(time.time() - t0, 1),
    }
    meta_fn = D0 / "p0_v20_predict_meta.json"
    meta_fn.write_text(json.dumps(out_meta, indent=2, ensure_ascii=False, default=float))
    print(f"\n  元数据: {meta_fn}")
    print(f"  总耗时: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()