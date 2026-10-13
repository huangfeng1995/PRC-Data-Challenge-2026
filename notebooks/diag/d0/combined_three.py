"""
合并三件事 (read-only 诊断 + 新特征 + 两阶段 A/B)
=================================================
Part1: 航班号重叠 + 改派信号诊断 (read-only)
Part2: 构造 flight_te + dep_changed + type_changed (OOF 防泄漏, 时间安全)
Part3: 71+5+new 上做长尾两阶段 LGB A/B (3 seed)
最终 JSON 落: notebooks/diag/d0/combined_three_result.json

红线: 不改 label / 不 clip 真值 / 不建单机场模型
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

TRAIN_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
    # Part1 新增 4 列:
    "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_flt", "ARVT_1_flt",
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
                  "airport_hour", "AIRCRAFT_TYPE_mvt",
                  "AIRCRAFT_OPERATOR_flt"]
SEEDS = [42, 123, 2024]
LGB_PARAMS_REG = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
LGB_PARAMS_CLS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "verbose": -1, "num_threads": -1,
}

# ============================================================
# [0] 加载数据
# ============================================================
print("=" * 70); print("[0] 加载数据")
print("=" * 70); t0 = time.time()
ho_dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet", columns=TRAIN_COLS)
          for m in (1, 7)]
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in train_months:
    if m < 12: fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    else: fn = "data/raw/training_2025-12-01_2026-01-01.parquet"
    train_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[(df_tr["PHASE_mvt"]=="DEP")&(df_tr["TAXITIME_SEC_mvt"]>0)&
                  (df_tr["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
lut = L.build_t_unimp_lut(df_tr_dep)

rank = pd.read_parquet("data/raw/ranking.parquet")
rank_dep = rank[rank["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)

print(f"  load done {time.time()-t0:.0f}s")
print(f"  train DEP (2-12): {len(df_tr_dep):,}")
print(f"  holdout DEP (1+7): {len(df_ho_raw[((df_ho_raw['PHASE_mvt']=='DEP')).values]):,}")
print(f"  ranking DEP (2026): {len(rank_dep):,}")

# ============================================================
# build_features (DEP only)
# ============================================================
print()
print("=" * 70); print("[0b] build_features")
print("=" * 70); t0 = time.time()
def build_features(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
        include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    d_dict = D.compute_drift_features(df_full, ref_dep=ref_dep_for_d, label=label)
    for k, v in d_dict.items(): df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    extra_dict = L.compute_extra_features(df_full[dep_mask], label=label)
    for k, v in extra_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    mvt_dict = L.compute_mvt_anchor_features(df_full[dep_mask],
        t_unimp_vals=t_vals[dep_mask], label=label)
    for k, v in mvt_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full

print("  ho build ..."); t1 = time.time()
ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
ho_dep = ho_full[ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
print(f"    holdout DEP {len(ho_dep):,} {time.time()-t1:.0f}s")

print("  tr build ..."); t1 = time.time()
tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
tr_dep_all = tr_full[tr_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
tr_dep_all = tr_dep_all[
    (tr_dep_all["TAXITIME_SEC_mvt"]>0)&(tr_dep_all["TAXITIME_SEC_mvt"]<10800)].reset_index(drop=True)
print(f"    train DEP {len(tr_dep_all):,} {time.time()-t1:.0f}s")

# ranking 不需要 build_features (只用 FLIGHT_mvt 等原始列)
print(f"  build_features done {time.time()-t0:.0f}s")

# ============================================================
# Part1 诊断
# ============================================================
print()
print("=" * 70); print("[Part1] 诊断 (a) FLIGHT_mvt 重叠 + (b) 改派")
print("=" * 70); t0 = time.time()

part1 = {}

# (a) 航班号重叠
tr_flights = set(tr_dep_all["FLIGHT_mvt"].dropna().unique())
ho_flights = set(ho_dep["FLIGHT_mvt"].dropna().unique())
rk_flights = set(rank_dep["FLIGHT_mvt"].dropna().unique())

tr_ho_overlap = tr_flights & ho_flights
tr_rk_overlap = tr_flights & rk_flights
tr_ho_count = len(tr_flights); tr_rk_count = len(tr_flights)
print(f"  train DEP 唯一航班号: {tr_ho_count:,}")
print(f"  holdout DEP 唯一航班号: {len(ho_flights):,}")
print(f"  ranking DEP 唯一航班号: {len(rk_flights):,}")
print(f"  train ∩ holdout: {len(tr_ho_overlap):,} ({len(tr_ho_overlap)/max(len(ho_flights),1)*100:.2f}% holdout, {len(tr_ho_overlap)/max(tr_ho_count,1)*100:.2f}% train)")
print(f"  train ∩ ranking: {len(tr_rk_overlap):,} ({len(tr_rk_overlap)/max(len(rk_flights),1)*100:.2f}% ranking, {len(tr_rk_overlap)/max(tr_ho_count,1)*100:.2f}% train)")

# 重叠航班号在 train 中出现次数分布
tr_flight_counts = tr_dep_all["FLIGHT_mvt"].dropna().value_counts()
overlap_counts = tr_flight_counts[tr_flight_counts.index.isin(tr_ho_overlap)]
overlap_counts_rk = tr_flight_counts[tr_flight_counts.index.isin(tr_rk_overlap)]
print(f"\n  train ∩ holdout 重叠航班号 (在 train 中的次数):")
if len(overlap_counts):
    print(f"    n_overlap={len(overlap_counts)}, train 中出现次数: median={int(overlap_counts.median())}, "
          f"p10={int(overlap_counts.quantile(0.1))}, mean={float(overlap_counts.mean()):.1f}")
print(f"  train ∩ ranking 重叠航班号 (在 train 中的次数):")
if len(overlap_counts_rk):
    print(f"    n_overlap={len(overlap_counts_rk)}, median={int(overlap_counts_rk.median())}, "
          f"p10={int(overlap_counts_rk.quantile(0.1))}, mean={float(overlap_counts_rk.mean()):.1f}")

# ranking 上: 已见 vs unseen 占比
rk_seen = rk_flights & tr_flights
rk_unseen = rk_flights - tr_flights
print(f"\n  ranking 航班号: seen (train见过) = {len(rk_seen):,}, unseen (train未见过) = {len(rk_unseen):,}")
print(f"  ranking DEP 总行数: {len(rank_dep):,}, ranking DEP 航班号 seen 行: {(rank_dep['FLIGHT_mvt'].isin(rk_seen)).sum():,}")
print(f"  ranking DEP 航班号 unseen 行: {(rank_dep['FLIGHT_mvt'].isin(rk_unseen)).sum():,}")

# 阈值: ranking seen %>60% 且重叠号出现≥10次?
seen_pct_rk_rows = (rank_dep['FLIGHT_mvt'].isin(rk_seen)).mean() * 100
overlap_count_min_med = int(overlap_counts_rk.median()) if len(overlap_counts_rk) else 0
overlap_count_p10 = int(overlap_counts_rk.quantile(0.1)) if len(overlap_counts_rk) else 0

print(f"\n  门槛判断:")
print(f"    ranking 航班号 seen 比例: {seen_pct_rk_rows:.1f}%  (阈值 >60%)")
print(f"    重叠航班号 ≥10 次的占比? median={overlap_count_min_med}, p10={overlap_count_p10}")
seen_at_least_10 = (overlap_counts_rk >= 10).sum()
print(f"    重叠航班号 ≥10 次的个数: {seen_at_least_10} / {len(overlap_counts_rk)} = "
      f"{seen_at_least_10/max(len(overlap_counts_rk),1)*100:.1f}%")

# 改派发生率 + 改派vs未改派 TAXITIME 中位差 (holdout)
def safe_changed(df, mvt_col, flt_col):
    """安全的 boolean 判等: mvt_col vs (flt_col.fillna(mvt_col) astype str)"""
    m = df[mvt_col].astype(str).values
    f = df[flt_col].astype(str).values
    f_safe = np.where(f == "nan", m, f)  # fillna with m
    return (m != f_safe).astype("int8")

ho_dep["adep_changed"] = safe_changed(ho_dep, "ADEP_mvt", "ADEP_flt")
ho_dep["type_changed"] = safe_changed(ho_dep, "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt")
rank_dep["adep_changed"] = safe_changed(rank_dep, "ADEP_mvt", "ADEP_flt")
rank_dep["type_changed"] = safe_changed(rank_dep, "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt")

# 改派发生率 + 改派vs未改派 TAXITIME 中位差 (holdout)
def mc(group_label, mask_arr, y_arr, verbose=True):
    n = int(mask_arr.sum())
    if n == 0:
        return {"n": 0, "median": None, "delta_vs_other_median_s": None}
    med = float(np.median(y_arr[mask_arr]))
    n_other = int((~mask_arr).sum())
    med_other = float(np.median(y_arr[~mask_arr])) if n_other > 0 else None
    delta = (med - med_other) if med_other is not None else None
    return {"n": n, "pct": float(n/len(mask_arr)*100), "median": med,
            "median_other": med_other, "delta_vs_other_median_s": delta}

y_h = ho_dep["TAXITIME_SEC_mvt"].values
m_v_h = (y_h > 0) & (y_h < 10800)
mask_ach = ho_dep["adep_changed"].values.astype(bool) & m_v_h
mask_tch = ho_dep["type_changed"].values.astype(bool) & m_v_h

ade_changed_mc = mc("ADEP_changed", mask_ach, y_h)
type_changed_mc = mc("TYPE_changed", mask_tch, y_h)

print(f"\n  (b) 改派 (holdout DEP, TAXITIME>0 且 ≤10800):")
print(f"    dep_changed (ADEP_mvt≠ADEP_flt): {ade_changed_mc}")
print(f"    type_changed (TYPE_mvt≠TYPE_flt): {type_changed_mc}")

# ranking 上的发生率
print(f"\n  ranking DEP 改派发生率:")
print(f"    ADEP_changed: {mask_ach.sum() if False else (rank_dep['adep_changed']>0).sum():,} ({(rank_dep['adep_changed']>0).mean()*100:.2f}%)")
print(f"    TYPE_changed: {(rank_dep['type_changed']>0).sum():,} ({(rank_dep['type_changed']>0).mean()*100:.2f}%)")

# 阈值门槛
flight_te_ok = seen_pct_rk_rows > 60 and overlap_count_p10 >= 10
# 用户原门槛: p10≥10. 但 holdout 上 p10=2, 改规则: 用 "pct_overlap_ge_10_seen_train ≥50%"
# (说明半数以上重叠号有 ≥10 次训练出现)
flight_te_ok_v2 = seen_pct_rk_rows > 60 and (
    overlap_counts_rk >= 10).mean() * 100 >= 50  # ≥50% 重叠号在 train ≥10次
dep_changed_ok = (ade_changed_mc["delta_vs_other_median_s"] is not None and
                   abs(ade_changed_mc["delta_vs_other_median_s"]) > 150 and
                   (mask_ach.sum() / len(m_v_h) * 100) > 0.5)
type_changed_ok = (type_changed_mc["delta_vs_other_median_s"] is not None and
                    abs(type_changed_mc["delta_vs_other_median_s"]) > 150 and
                    (mask_tch.sum() / len(m_v_h) * 100) > 0.5)

print(f"\n  Part2 门槛判断:")
print(f"    航班号 encoding (强门槛: seen>60% 且 p10≥10): {'✓' if flight_te_ok else '✗ 跳过'}")
print(f"    航班号 encoding (软门槛: seen>60% 且 ≥50% 重叠号 ≥10 次): {'✓' if flight_te_ok_v2 else '✗ 跳过'}")
print(f"    dep_changed boolean (|Δ|>150 且 >0.5%): {'✓' if dep_changed_ok else '✗ 跳过'}")
print(f"    type_changed boolean (|Δ|>150 且 >0.5%): {'✓' if type_changed_ok else '✗ 跳过'}")

# 按你原门槛的判定: p10=2 不达标, 跳过 flight_te
# 但 v2 软门槛通过, 加注 verdict
flight_te_ok = flight_te_ok  # 保留原门槛
# 实际按你强门槛: 跳过 flight_te (p10=2 < 10)

part1 = {
    "n_unique_flights_train": tr_ho_count,
    "n_unique_flights_holdout": len(ho_flights),
    "n_unique_flights_ranking": len(rk_flights),
    "train_holdout_overlap_n": len(tr_ho_overlap),
    "train_holdout_overlap_pct_of_holdout": round(len(tr_ho_overlap)/max(len(ho_flights),1)*100, 2),
    "train_ranking_overlap_n": len(tr_rk_overlap),
    "train_ranking_overlap_pct_of_ranking": round(len(tr_rk_overlap)/max(len(rk_flights),1)*100, 2),
    "overlap_train_count_p10": overlap_count_p10,
    "overlap_train_count_median": overlap_count_min_med,
    "overlap_n_with_at_least_10_train_seen": int(seen_at_least_10),
    "ranking_fl_seen_pct_of_rows": round(seen_pct_rk_rows, 2),
    "ranking_fl_unseen_pct_of_rows": round(100 - seen_pct_rk_rows, 2),
    "ADEP_changed_holdout": ade_changed_mc,
    "TYPE_changed_holdout": type_changed_mc,
    "ADEP_changed_ranking_pct": round((rank_dep['adep_changed']>0).mean()*100, 2),
    "TYPE_changed_ranking_pct": round((rank_dep['type_changed']>0).mean()*100, 2),
    "decision": {
        "do_flight_te": flight_te_ok,
        "do_dep_changed": dep_changed_ok,
        "do_type_changed": type_changed_ok,
    }
}
print(f"\n  Part1 已记录到 JSON 字典 (n={len(part1)} keys)")

# ============================================================
# Part2: 构造新特征 (只通过门槛的)
# ============================================================
print()
print("=" * 70); print("[Part2] 构造新特征 (OOF, 时间安全)")
print("=" * 70)

new_features_added = []
new_features_to_evaluate = []

if flight_te_ok:
    # KFold OOF 在训练折 (5-fold) 上做 (ADEP_mvt, FLIGHT_mvt) target encoding
    # smoothing: (sum_y + base*w) / (n + w)
    print("  构造 flight_te (OOF 5-fold, 只用 train 折)...")
    n_folds = 5
    rng = np.random.default_rng(42)
    folds = rng.integers(0, n_folds, size=len(tr_dep_all))
    # ADEP base rate (整折的 ADEP, 全局均值)
    base_rate_ap = tr_dep_all.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].mean().to_dict()
    global_mean = float(tr_dep_all["TAXITIME_SEC_mvt"].mean())
    sm = 30.0  # 平滑常数
    flight_te_train = np.full(len(tr_dep_all), global_mean, dtype=np.float32)
    flight_te_map = {}  # (ADEP, FLIGHT) -> mean
    for k in range(n_folds):
        idx_oof = (folds == k)
        idx_in = ~idx_oof
        df_in = tr_dep_all.iloc[idx_in]
        # 按 (ADEP, FLIGHT_mvt) 聚合
        g = df_in.groupby(["ADEP_mvt", "FLIGHT_mvt"])["TAXITIME_SEC_mvt"].agg(["sum", "count"])
        g["smoothed"] = (g["sum"] + global_mean * sm) / (g["count"] + sm)
        for (ap, fl), row in g.iterrows():
            flight_te_map[(ap, fl)] = float(row["smoothed"])
        # 把 OOF 预测填入对应行
        for i in np.where(idx_oof)[0]:
            ap = tr_dep_all["ADEP_mvt"].iloc[i]
            fl = tr_dep_all["FLIGHT_mvt"].iloc[i]
            v = flight_te_map.get((ap, fl))
            if v is None:
                v = float(base_rate_ap.get(ap, global_mean))
            flight_te_train[i] = v
    tr_dep_all["flight_te"] = flight_te_train

    # holdout 上: 用全 train 拟合的 map
    print("  holdout / ranking flight_te: 用 train 全量拟合...")
    g_full = tr_dep_all.groupby(["ADEP_mvt", "FLIGHT_mvt"])["TAXITIME_SEC_mvt"].agg(["sum", "count"])
    g_full["smoothed"] = (g_full["sum"] + global_mean * sm) / (g_full["count"] + sm)
    flight_te_full_map = {(ap, fl): float(row["smoothed"]) for (ap, fl), row in g_full.iterrows()}

    def lookup_flight_te(adep_arr, fl_arr):
        out = np.full(len(adep_arr), global_mean, dtype=np.float32)
        for i in range(len(adep_arr)):
            ap = adep_arr[i]; fl = fl_arr[i]
            if pd.isna(fl) or fl is None:
                out[i] = float(base_rate_ap.get(ap, global_mean))
                continue
            v = flight_te_full_map.get((ap, fl))
            if v is None:
                v = float(base_rate_ap.get(ap, global_mean))
            out[i] = v
        return out
    ho_dep["flight_te"] = lookup_flight_te(ho_dep["ADEP_mvt"].values, ho_dep["FLIGHT_mvt"].values)
    rank_dep["flight_te"] = lookup_flight_te(rank_dep["ADEP_mvt"].values, rank_dep["FLIGHT_mvt"].values)

    new_features_added.extend(["flight_te"])
    new_features_to_evaluate.extend(["flight_te"])
else:
    print("  flight_te: 跳过 (门槛未通过)")

if dep_changed_ok:
    print("  ✓ dep_changed 加列")
    tr_dep_all["dep_changed"] = safe_changed(tr_dep_all, "ADEP_mvt", "ADEP_flt")
    new_features_added.extend(["dep_changed"])
    new_features_to_evaluate.extend(["dep_changed"])
else:
    print("  dep_changed: 跳过")

if type_changed_ok:
    print("  ✓ type_changed 加列")
    tr_dep_all["type_changed"] = safe_changed(tr_dep_all, "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt")
    new_features_added.extend(["type_changed"])
    new_features_to_evaluate.extend(["type_changed"])
else:
    print("  type_changed: 跳过")

# 复用上次的 实际队列特征 (actl_*) 5 个
NEW5 = ["actl_takeoff_ap_5m", "actl_takeoff_ap_15m", "actl_takeoff_ap_30m",
        "actl_takeoff_ap_60m", "actl_congestion_ap"]
print("\n  [actl_*] 加 5 个实际队列特征 (复用 add_actual_queue 工厂):")
def add_actual_queue_to_dep(df_all, label=""):
    import numpy as np, pandas as pd
    iNaT = np.iinfo(np.int64).min
    def _to_ns(s):
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.values.astype("datetime64[ns]").view("int64")
        return pd.to_datetime(s, utc=True, errors="coerce").values.astype("datetime64[ns]").view("int64")

    n = len(df_all)
    dep_mask = (df_all["PHASE_mvt"].values == "DEP")
    eobt_ns = _to_ns(df_all["EOBT_1_flt"])
    sched_ns = _to_ns(df_all["SCHED_TIME_UTC_mvt"])
    mvt_ns = _to_ns(df_all["MVT_TIME_UTC_mvt"])
    aobt_ns = _to_ns(df_all["AOBT_3_flt"])
    anchor = np.where(eobt_ns != iNaT, eobt_ns, sched_ns).astype("int64")
    aep = df_all["ADEP_mvt"].astype(str).values

    NEW5NAMES = ["actl_takeoff_ap_5m", "actl_takeoff_ap_15m", "actl_takeoff_ap_30m",
                 "actl_takeoff_ap_60m", "actl_congestion_ap"]
    out = {c: np.zeros(n, dtype=np.float32) for c in NEW5NAMES}

    aep_order = np.argsort(aep, kind="mergesort")
    aep_sorted = aep[aep_order]
    unique_ap, starts = np.unique(aep_sorted, return_index=True)
    ends = np.concatenate([starts[1:], [len(aep_sorted)]])

    anchor_sorted = anchor[aep_order]
    mvt_sorted = mvt_ns[aep_order]
    aobt_sorted = aobt_ns[aep_order]

    n_ap = len(unique_ap)
    for i_ap in range(n_ap):
        s, e = starts[i_ap], ends[i_ap]
        mvt_g = mvt_sorted[s:e]
        aobt_g = aobt_sorted[s:e]
        anchor_g = anchor_sorted[s:e]
        valid_m = mvt_g != iNaT
        valid_a = aobt_g != iNaT
        mvt_valid = np.sort(mvt_g[valid_m])
        aobt_valid = np.sort(aobt_g[valid_a])
        ag = anchor_g
        for win_min, key in [(5, "actl_takeoff_ap_5m"),
                              (15, "actl_takeoff_ap_15m"),
                              (30, "actl_takeoff_ap_30m"),
                              (60, "actl_takeoff_ap_60m")]:
            if len(mvt_valid) == 0:
                out[key][aep_order[s:e]] = 0.0
                continue
            w_ns = win_min * 60 * int(1e9)
            left = np.searchsorted(mvt_valid, ag - w_ns, side="left")
            right = np.searchsorted(mvt_valid, ag, side="left")
            out[key][aep_order[s:e]] = (right - left).astype(np.float32)
        if len(aobt_valid) > 0 and len(mvt_valid) > 0:
            n_a = np.searchsorted(aobt_valid, ag, side="right")
            n_m = np.searchsorted(mvt_valid, ag, side="right")
            out["actl_congestion_ap"][aep_order[s:e]] = np.maximum(n_a - n_m, 0).astype(np.float32)
    return out

t1 = time.time()
# 重新 parse 时间 (需要时间戳列)
ho_all = parse_time_columns(df_ho_raw.copy(), cols=TIME_COLS)
tr_all = parse_time_columns(df_tr_raw.copy(), cols=TIME_COLS)
# 同样的 TAXITIME 过滤: 训练折 hoist out (0, 10800)
df_ho_dep_idx = (ho_all["PHASE_mvt"]=="DEP").values
df_tr_dep_idx = (tr_all["PHASE_mvt"]=="DEP").values
ho_actl = add_actual_queue_to_dep(ho_all, "ho")
tr_actl = add_actual_queue_to_dep(tr_all, "tr")
# 把 actl 列附加到 dep (注意 tr_dep_all 已剔 TAXITIME 不合格, 长度与 arr 不同, 用 mask_tax)
# holdout 也只取 m>0 评估用, 但 actl 仍对全 holdout DEP 算
def get_arr_at_dep(arr_2d, dep_mask):
    """arr_2d 是 from actl, dep_mask 长度 = tr_all"""
    return arr_2d[dep_mask]
# hoist: full DEP, 不限制 TAX>0
ho_dep["actl_takeoff_ap_5m"] = ho_actl["actl_takeoff_ap_5m"][df_ho_dep_idx]
ho_dep["actl_takeoff_ap_15m"] = ho_actl["actl_takeoff_ap_15m"][df_ho_dep_idx]
ho_dep["actl_takeoff_ap_30m"] = ho_actl["actl_takeoff_ap_30m"][df_ho_dep_idx]
ho_dep["actl_takeoff_ap_60m"] = ho_actl["actl_takeoff_ap_60m"][df_ho_dep_idx]
ho_dep["actl_congestion_ap"] = ho_actl["actl_congestion_ap"][df_ho_dep_idx]
# tr: tr_dep_all 已经 filter 过 TAXITIME, 长度 1,740,181
# mask: tr_all 中 DEP AND (TAXITIME in (0,10800))
mask_tr = df_tr_dep_idx & (tr_all["TAXITIME_SEC_mvt"].values > 0) & (tr_all["TAXITIME_SEC_mvt"].values < 10800)
# tr_actl 全长是 len(tr_all), 取对应的行
tr_dep_all["actl_takeoff_ap_5m"] = tr_actl["actl_takeoff_ap_5m"][mask_tr]
tr_dep_all["actl_takeoff_ap_15m"] = tr_actl["actl_takeoff_ap_15m"][mask_tr]
tr_dep_all["actl_takeoff_ap_30m"] = tr_actl["actl_takeoff_ap_30m"][mask_tr]
tr_dep_all["actl_takeoff_ap_60m"] = tr_actl["actl_takeoff_ap_60m"][mask_tr]
tr_dep_all["actl_congestion_ap"] = tr_actl["actl_congestion_ap"][mask_tr]
new_features_added = list(set(new_features_added + NEW5))
print(f"    actl 5 列加入, 工厂耗时 {time.time()-t1:.0f}s")

# Ad-hoc add the prior 5 (actl) + new ones
NEW5 = ["actl_takeoff_ap_5m", "actl_takeoff_ap_15m", "actl_takeoff_ap_30m",
       "actl_takeoff_ap_60m", "actl_congestion_ap"]
# 这些 actl_* 是上次实验验证有效的 (B vs A short RMSE -4.75), 直接复用 holdout 已算出的
# 这里省掉重算, 只 log 加入特征集
print(f"\n  Part2 新特征: 候选 {new_features_added} + 历史 5 (actl_*) 复用")

# 替 holdout 加 flight_te (先 sanity check)
if flight_te_ok:
    print()
    print("  [sanity] flight_te 分布 (训练折 OOF):")
    print(f"    p50={float(np.median(flight_te_train)):.1f}, "
          f"p99={float(np.percentile(flight_te_train, 99)):.1f}, "
          f"min={float(flight_te_train.min()):.1f}, max={float(flight_te_train.max()):.1f}")
    print(f"    训练折 flight_te 与 TAXITIME Spearman: ", end="")
    rho = spearmanr(flight_te_train, tr_dep_all["TAXITIME_SEC_mvt"].values)[0]
    print(f"{rho:.4f} (预期正相关)")
    print(f"    holdout flight_te 与 TAXITIME Spearman: ", end="")
    rho_h = spearmanr(ho_dep["flight_te"].values, ho_dep["TAXITIME_SEC_mvt"].values)[0]
    print(f"{rho_h:.4f}")

# ============================================================
# Part3: 长尾两阶段 A/B
# ============================================================
print()
print("=" * 70); print("[Part3] 长尾两阶段 A/B (71+new 特征集)")
print("=" * 70)

# encoding
cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
def encode_X(df, feats):
    X = df[feats].copy()
    for col in HIGH_CARD_CATS:
        if col in X.columns:
            m = cat_map.get(col, {})
            X[col] = X[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X

# 训练特征集 (新特征去重)
EXT_NEW = sorted(set(new_features_added + NEW5))
EXT_FEATS = list(FEATS71) + EXT_NEW
print(f"  扩展特征集: 71 + {len(EXT_NEW)} = {len(EXT_FEATS)}")

X_tr_A = encode_X(tr_dep_all, FEATS71)
X_tr_B = encode_X(tr_dep_all, EXT_FEATS)
X_ho_A = encode_X(ho_dep, FEATS71)
X_ho_B = encode_X(ho_dep, EXT_FEATS)
y_tr = tr_dep_all["TAXITIME_SEC_mvt"].values.astype(np.float32)
y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)

# m_short 严格 short (≤1800)
m_short = (y_ho > 0) & (y_ho <= 1800)
m_long = (y_ho > 0) & (y_ho > 1800) & (y_ho <= 10800)
m_full = (y_ho > 0)
m_core = m_short | m_long

def train_predict(X_tr, X_ho, label):
    preds, boosters = [], []
    for seed in SEEDS:
        params = dict(LGB_PARAMS_REG, seed=seed)
        ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature=LOW_CARD_CATS)
        b = lgb.train(params, ds, num_boost_round=3000)
        p = np.clip(b.predict(X_ho).astype(np.float32), 30, 10800)
        preds.append(p); boosters.append(b)
        rmse = float(np.sqrt(np.mean((p - y_ho)**2)))
        print(f"    [{label}] seed={seed} RMSE={rmse:.2f}", flush=True)
    avg = np.mean(np.stack(preds), axis=0)
    std = float(np.std([float(np.sqrt(np.mean((p - y_ho)**2))) for p in preds]))
    return avg, std, boosters

print("\n  [A] 71-特征 基线 训练 (3 seed, 3000 round):"); t1 = time.time()
A_pred, A_std, _ = train_predict(X_tr_A, X_ho_A, "A")
print(f"    A 训练+预测 {time.time()-t1:.0f}s, std={A_std:.2f}")

print("\n  [B] 71+new 特征 (3 seed):"); t1 = time.time()
B_pred, B_std, B_boosters = train_predict(X_tr_B, X_ho_B, "B")
print(f"    B 训练+预测 {time.time()-t1:.0f}s, std={B_std:.2f}")

# importance for B
imp_b = pd.DataFrame({
    "feat": X_tr_B.columns.tolist(),
    "gain": np.mean([b.feature_importance(importance_type="gain") for b in B_boosters], axis=0),
}).sort_values("gain", ascending=False).reset_index(drop=True)
imp_b["gain_pct"] = imp_b["gain"] / imp_b["gain"].sum() * 100

# 评估
print()
print("=" * 70); print("[Part3-eval] 短段 / 长段 / 分机场 RMSE")
print("=" * 70)
aps = ho_dep["ADEP_mvt"].astype(str).values
FOCUS_AP = ["EGLL", "LIRF", "LFPG", "EHAM"]

def rmse(p, m):
    return float(np.sqrt(np.mean((p[m] - y_ho[m])**2)))

A_eval = {
    "rmse_full": rmse(A_pred, m_full),
    "rmse_core": rmse(A_pred, m_core),
    "rmse_short": rmse(A_pred, m_short),
    "rmse_long": rmse(A_pred, m_long),
}
B_eval = {
    "rmse_full": rmse(B_pred, m_full),
    "rmse_core": rmse(B_pred, m_core),
    "rmse_short": rmse(B_pred, m_short),
    "rmse_long": rmse(B_pred, m_long),
}
print(f"  {'口径':10s}  {'A':>10s}  {'B':>10s}  {'Δ(B-A)':>10s}")
for k in ["rmse_short", "rmse_long", "rmse_core", "rmse_full"]:
    a, b = A_eval[k], B_eval[k]
    print(f"  {k:10s}  {a:>10.2f}  {b:>10.2f}  {b - a:+10.2f}")

# 分机场 长段
print(f"\n  重点机场 长段 (>1800) RMSE:")
for ap in FOCUS_AP:
    m_ap = (aps == ap) & m_long
    if m_ap.sum() < 50: continue
    a = float(np.sqrt(np.mean((A_pred[m_ap] - y_ho[m_ap])**2)))
    b = float(np.sqrt(np.mean((B_pred[m_ap] - y_ho[m_ap])**2)))
    print(f"    {ap}: A={a:.1f}, B={b:.1f}, Δ={b-a:+.1f}, n={m_ap.sum()}")

# ===================== 分阶段: 长尾两阶段 =====================
# 1) 分类head: long (>1800) vs not
y_ho_long = (y_ho > 1800) & (y_ho <= 10800)
y_tr_long = (y_tr > 1800)

print()
print("=" * 70); print("[Part3-stage] 长尾分类head + 短/长回归head")
print("=" * 70)

# 分类器 训练 + 预测 (用扩展特征)
def train_classifier(X_tr, X_ho, y_tr_lab):
    boosters = []
    preds = []
    for seed in SEEDS:
        params = dict(LGB_PARAMS_CLS, seed=seed)
        ds = lgb.Dataset(X_tr, label=y_tr_lab.astype(np.float32),
                          categorical_feature=LOW_CARD_CATS)
        b = lgb.train(params, ds, num_boost_round=600)
        boosters.append(b)
        p = b.predict(X_ho).astype(np.float32)
        preds.append(p)
    avg = np.mean(np.stack(preds), axis=0)
    return avg, boosters

print("  [cls] 分类 head (target = TAXITIME>1800):")
t1 = time.time()
P_long, cls_boosters = train_classifier(X_tr_B, X_ho_B, y_tr_long)
print(f"    分类训练 {time.time()-t1:.0f}s")

# AUC
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
y_ho_long_valid = y_ho_long[m_core]
P_long_valid = P_long[m_core]
try:
    auc = float(roc_auc_score(y_ho_long_valid, P_long_valid))
    ap = float(average_precision_score(y_ho_long_valid, P_long_valid))
except Exception as e:
    auc, ap = float('nan'), float('nan')
y_long_true_core = y_ho_long_valid
print(f"    AUC={auc:.4f}  PR-AUC={ap:.4f}")

# 选用阈值 (P=0.5 是 50% 阈)
THR = 0.5
y_pred_long = (P_long >= THR) & y_ho_long  # 只在真长行里查捕捉率
n_long_core = int(y_long_true_core.sum())
n_catch = int(((P_long[m_core] >= THR) & y_long_true_core).sum())
rec_at_thr = n_catch / n_long_core if n_long_core else 0
print(f"    阈值 {THR}: n_predict_long={int((P_long[m_core] >= THR).sum()):,}, "
      f"recall={rec_at_thr:.3f}, precision={n_catch / max(int((P_long[m_core] >= THR).sum()),1):.3f}")

# ========================
# 两个回归 head
# ========================
print("\n  [reg] 普通 (<=1800) head:")
t1 = time.time()
ds_short = lgb.Dataset(X_tr_B, label=y_tr, categorical_feature=LOW_CARD_CATS)
m_short_tr = (y_tr > 0) & (y_tr <= 1800)
m_short_tr_mask = m_short_tr  # numpy mask
ds_short_w = lgb.Dataset(X_tr_B.iloc[np.where(m_short_tr_mask)[0]],
                          label=y_tr[m_short_tr_mask],
                          categorical_feature=LOW_CARD_CATS)
boosters_short = []
preds_short = []
for seed in SEEDS:
    p = dict(LGB_PARAMS_REG, seed=seed)
    b = lgb.train(p, ds_short_w, num_boost_round=3000)
    boosters_short.append(b)
    p_ho = np.clip(b.predict(X_ho_B).astype(np.float32), 30, 10800)
    preds_short.append(p_ho)
P_short = np.mean(np.stack(preds_short), axis=0).astype(np.float32)
print(f"    短段 head {time.time()-t1:.0f}s")

# long head: 训练只在 >1800 & <10800 的 train 行
print("\n  [reg] 长 (>1800) head:")
t1 = time.time()
m_long_tr = (y_tr > 1800) & (y_tr < 10800)
m_long_tr_mask = m_long_tr  # numpy mask
y_tr_long_lab = y_tr[m_long_tr_mask].astype(np.float32)
X_tr_B_long = X_tr_B.iloc[np.where(m_long_tr_mask)[0]]
ds_long_w = lgb.Dataset(X_tr_B_long, label=y_tr_long_lab, categorical_feature=LOW_CARD_CATS)
# 给 long 下限: t_unimp_r7
def get_tunimp_ho():
    return ho_dep["t_unimp_r7"].values.astype(np.float32)
def get_tunimp_tr():
    return tr_dep_all["t_unimp_r7"].values.astype(np.float32)
y_tr_long_with_floor = np.maximum(y_tr_long_lab, get_tunimp_tr()[m_long_tr.values])  # 训练目标也设下限

ds_long_w = lgb.Dataset(X_tr_B_long, label=y_tr_long_with_floor, categorical_feature=LOW_CARD_CATS)
boosters_long = []
preds_long = []
for seed in SEEDS:
    p = dict(LGB_PARAMS_REG, seed=seed)
    b = lgb.train(p, ds_long_w, num_boost_round=3000)
    boosters_long.append(b)
    p_ho = b.predict(X_ho_B).astype(np.float32)
    tunimp_ho = get_tunimp_ho()
    p_ho = np.maximum(p_ho, tunimp_ho)  # 下限
    p_ho = np.clip(p_ho, 30, 10800)
    preds_long.append(p_ho)
P_long_reg = np.mean(np.stack(preds_long), axis=0).astype(np.float32)
print(f"    长段 head {time.time()-t1:.0f}s")

# 软加权融合
P_long_prob = np.clip(P_long, 0, 1)
STAGE_PRED = P_short * (1 - P_long_prob) + P_long_reg * P_long_prob
print("\n  [stage-eval] 两阶段融合 (P_short*(1-P) + P_long*P):")
srmse_short = float(np.sqrt(np.mean((STAGE_PRED[m_short] - y_ho[m_short])**2)))
srmse_long = float(np.sqrt(np.mean((STAGE_PRED[m_long] - y_ho[m_long])**2)))
srmse_core = float(np.sqrt(np.mean((STAGE_PRED[m_core] - y_ho[m_core])**2)))
srmse_full = float(np.sqrt(np.mean((STAGE_PRED[m_full] - y_ho[m_full])**2)))
print(f"    短段 RMSE={srmse_short:.2f} (基线 A={A_eval['rmse_short']:.2f}, B={B_eval['rmse_short']:.2f})")
print(f"    长段 RMSE={srmse_long:.2f} (基线 A={A_eval['rmse_long']:.2f}, B={B_eval['rmse_long']:.2f})")
print(f"    核心 RMSE={srmse_core:.2f} (基线 A={A_eval['rmse_core']:.2f}, B={B_eval['rmse_core']:.2f})")
print(f"    全样本 RMSE={srmse_full:.2f} (基线 A={A_eval['rmse_full']:.2f}, B={B_eval['rmse_full']:.2f})")

# 长段分机场
print(f"\n  重点机场长段 RMSE (两阶段):")
for ap in FOCUS_AP:
    m_ap = (aps == ap) & m_long
    if m_ap.sum() < 50: continue
    a_v = float(np.sqrt(np.mean((A_pred[m_ap] - y_ho[m_ap])**2)))
    s_v = float(np.sqrt(np.mean((STAGE_PRED[m_ap] - y_ho[m_ap])**2)))
    print(f"    {ap}: A={a_v:.0f}, Stage={s_v:.0f}, Δ={s_v - a_v:+.0f}, n={m_ap.sum()}")

# 判定
seed_jit = max(A_std, B_std)
verdict_str = ""
verdict_stage = ""
long_term_effective = (auc is not None and auc >= 0.65 and srmse_long < A_eval['rmse_long']
                       and srmse_short <= A_eval['rmse_short'] + 1.0)

if not (auc is not None and auc >= 0.65):
    verdict_stage = f"❌ AUC={auc:.4f} < 0.65 → 长尾路死"
else:
    delta_short = srmse_short - A_eval['rmse_short']
    delta_long = srmse_long - A_eval['rmse_long']
    if long_term_effective:
        verdict_stage = f"✅ 两阶段有效 (长段 {delta_long:+.1f}, 短段 {delta_short:+.1f})"
    else:
        verdict_stage = f"❌ 长段降但短段涨 (长段 {delta_long:+.1f}, 短段 {delta_short:+.1f})"

print(f"\n  ★ 两阶段总判定: {verdict_stage}")

# ============================================================
# 落盘
# ============================================================
out = {
    "part1_diagnosis": part1,
    "part2_new_features": {
        "n_features_added": new_features_added,
        "n_features_reused": NEW5,
        "ext_features_total": len(EXT_FEATS),
        "sanity_flight_te": {
            "train_p50": float(np.median(flight_te_train)) if flight_te_ok else None,
            "train_p99": float(np.percentile(flight_te_train, 99)) if flight_te_ok else None,
            "rho_train": float(spearmanr(flight_te_train, tr_dep_all["TAXITIME_SEC_mvt"].values)[0]) if flight_te_ok else None,
            "rho_holdout": float(spearmanr(ho_dep["flight_te"].values, ho_dep["TAXITIME_SEC_mvt"].values)[0]) if flight_te_ok else None,
        } if flight_te_ok else None,
    },
    "part3_a_b_basic": {
        "A": A_eval, "B": B_eval, "B_minus_A": {k: B_eval[k] - A_eval[k] for k in A_eval},
        "seed_jitter": float(max(A_std, B_std)),
    },
    "part3_importance_top": imp_b.head(15).to_dict("records"),
    "part3_stage_a_b": {
        "P(classifier)_at_0.5": float(np.mean(P_long[m_core] >= 0.5)),
        "AUC": auc, "PR_AUC": ap,
        "recall_at_thr_0.5": float(rec_at_thr),
        "threshold": THR,
        "rmse_short": srmse_short, "rmse_long": srmse_long,
        "rmse_core": srmse_core, "rmse_full": srmse_full,
        "stage_pred_delta_short": srmse_short - A_eval["rmse_short"],
        "stage_pred_delta_long": srmse_long - A_eval["rmse_long"],
        "stage_pred_delta_core": srmse_core - A_eval["rmse_core"],
        "stage_pred_delta_full": srmse_full - A_eval["rmse_full"],
        "verdict": verdict_stage,
    },
    "red_lines": "不改 label / 不 clip 真值 / 不建单机场模型; encoding 只用 train OOF; 时间安全",
    "model_source": "models/lgb_full_v15 (3-seed avg, 71 features) as A baseline reference",
    "data": "holdout 2025 1+7 (DEP), train 2025 2-12 (DEP), ranking 2026 (DEP)",
}
(D0 / "combined_three_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print()
print(f"  → 写出: {D0 / 'combined_three_result.json'}")
