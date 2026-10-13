"""
AOBT_NaT 兜底诊断 (read-only, 不重训、不上传、不改真值)
======================================================

Step0: ranking/holdout AOBT 字段非空率口径
Step1: AOBT_NaT (5321) vs AOBT_valid 刻画
Step2: 兜底模拟 — 条件均值替换, 报 RMSE 改善 / valid 段代价
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

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

# ============================================================
# [Step0] 口径
# ============================================================
print("=" * 70); print("[Step0] ranking AOBT 口径")
print("=" * 70)

rank = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
print(f"  ranking 总行数: {len(rank):,}")
print(f"  PHASE_mvt 分布: {rank['PHASE_mvt'].value_counts().to_dict()}")
nn = rank["AOBT_3_flt"].notna().sum()
print(f"  AOBT_3_flt 非空率: {nn}/{len(rank)} ({nn/len(rank)*100:.2f}%)")
print(f"  AOBT 缺失行数: {len(rank) - int(nn):,}")

rank_dep = rank[rank["PHASE_mvt"] == "DEP"].copy()
nn_dep = rank_dep["AOBT_3_flt"].notna().sum()
print(f"  ranking DEP AOBT 非空: {nn_dep}/{len(rank_dep)} ({nn_dep/len(rank_dep)*100:.2f}%)")
print(f"  ranking DEP AOBT 缺失: {len(rank_dep) - int(nn_dep):,} ({(len(rank_dep)-int(nn_dep))/len(rank_dep)*100:.2f}%)")
print(f"  → 真榜推理: 缺失率 ~{(len(rank_dep) - int(nn_dep))/len(rank_dep)*100:.2f}%, 用 isna() 直接识别")

# ============================================================
# [1] holdout 预测 (复用 v15 booster)
# ============================================================
print()
print("=" * 70); print("[1] load holdout + 预测")
print("=" * 70); t0 = time.time()
ho_dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet", columns=TRAIN_COLS)
          for m in (1, 7)]
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in train_months:
    if m < 12:
        fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    else:
        fn = "data/raw/training_2025-12-01_2026-01-01.parquet"
    train_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[(df_tr["PHASE_mvt"]=="DEP")&(df_tr["TAXITIME_SEC_mvt"]>0)&
                  (df_tr["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
lut = L.build_t_unimp_lut(df_tr_dep)

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

ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
ho_dep = ho_full[ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)

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

X_ho = encode_X(ho_dep, FEATS71).reset_index(drop=True)
y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)
boosters = [lgb.Booster(model_file=str(ROOT / "models/lgb_full_v15" / f"lgb_seed{s}.txt"))
            for s in (42, 123, 2024)]
pred = np.mean(np.stack([np.clip(b.predict(X_ho).astype(np.float32), 30, 10800) for b in boosters]), axis=0)
print(f"  预测完成 {time.time()-t0:.0f}s, RMSE={np.sqrt(np.mean((pred-y_ho)**2)):.2f}")

m_full = y_ho > 0
# 全部 DEP 都视为样本 (包含 y_ho = 0/NaT 的行也算入 N_FULL, 但仅当 y_ho > 0 才会进入 RMSE 计算).
# 这里 m_full 是 axis-aligned 索引, 等长于 ho_dep 全长.
m_aobt_na_h = ho_dep["AOBT_3_flt"].isna().values & (y_ho > 0)
m_aobt_v_h = ho_dep["AOBT_3_flt"].notna().values & (y_ho > 0)
m_aobt_na = m_aobt_na_h
m_aobt_v = m_aobt_v_h
n_full = int(m_full.sum())
n_na = int(m_aobt_na.sum())
n_v = int(m_aobt_v.sum())
print(f"  holdout DEP 全样本: {n_full:,}")
print(f"  AOBT_NaT (NaT):       {n_na:,} ({n_na/n_full*100:.2f}%)")
print(f"  AOBT_valid:           {n_v:,} ({n_v/n_full*100:.2f}%)")

# ============================================================
# [Step1] AOBT_NaT 刻画
# ============================================================
print()
print("=" * 70); print("[Step1] AOBT_NaT 刻画")
print("=" * 70)

resid = y_ho - pred
err2 = resid**2
print(f"\n  全样本 SSE = {err2[m_full].sum():.3e}, mean RMSE = {np.sqrt(err2[m_full].mean()):.2f}")
print(f"  AOBT_NaT SSE = {err2[m_aobt_na].sum():.3e} ({err2[m_aobt_na].sum()/err2[m_full].sum()*100:.1f}%)")
print(f"  AOBT_valid SSE = {err2[m_aobt_v].sum():.3e}")

# TAXITIME 分布
print(f"\n  AOBT_NaT (n={n_na}) TAXITIME 分布:")
y_na = y_ho[m_aobt_na]
print(f"    median={np.median(y_na):.0f} p99={np.percentile(y_na,99):.0f} max={y_na.max():.0f}")
print(f"    >10800 (跟真榜同口径): {(y_na > 10800).sum()} ({(y_na > 10800).sum()/n_na*100:.2f}%)")
print(f"    >3600:  {(y_na > 3600).sum()} ({(y_na > 3600).sum()/n_na*100:.2f}%)")
print(f"    >1800:  {(y_na > 1800).sum()} ({(y_na > 1800).sum()/n_na*100:.2f}%)")

print(f"\n  AOBT_valid (n={n_v}) TAXITIME 分布:")
y_v = y_ho[m_aobt_v]
print(f"    median={np.median(y_v):.0f} p99={np.percentile(y_v,99):.0f} max={y_v.max():.0f}")
print(f"    >10800: {(y_v > 10800).sum()} ({(y_v > 10800).sum()/n_v*100:.2f}%)")

# top 圈出 AOBT_NaT 的事前预测特征组合
print("\n  按 ADEP 看 AOBT 缺失率:")
airport_na_pct = (ho_dep[m_full].groupby("ADEP_mvt")
                   .agg(total=("AOBT_3_flt", lambda s: len(s)),
                        aobt_na=("AOBT_3_flt", lambda s: int(s.isna().sum())))
                   .assign(pct=lambda d: d["aobt_na"]/d["total"]*100)
                   .sort_values("pct", ascending=False).head(15))
print(airport_na_pct.to_string())

print("\n  按 OPERATOR 看 AOBT 缺失率 (top 15):")
op_na_pct = (ho_dep[m_full].groupby("AIRCRAFT_OPERATOR_flt")
              .agg(total=("AOBT_3_flt", lambda s: len(s)),
                   aobt_na=("AOBT_3_flt", lambda s: int(s.isna().sum())))
              .assign(pct=lambda d: d["aobt_na"]/d["total"]*100)
              .sort_values("pct", ascending=False).head(15))
print(op_na_pct.to_string())

print("\n  按 AIRCRAFT_TYPE 看 AOBT 缺失率 (top 15):")
at_na_pct = (ho_dep[m_full].groupby("AIRCRAFT_TYPE_mvt")
             .agg(total=("AOBT_3_flt", lambda s: len(s)),
                  aobt_na=("AOBT_3_flt", lambda s: int(s.isna().sum())))
             .assign(pct=lambda d: d["aobt_na"]/d["total"]*100)
             .sort_values("pct", ascending=False).head(15))
print(at_na_pct.to_string())

print("\n  按 (ADEP, hour) 看 top 缺率 (pre-forecast time 可知):")
ho_dep["_hour"] = ho_dep["sched_hour"].astype(int)
ap_hr = ho_dep[m_full].groupby(["ADEP_mvt", "_hour"]).agg(
    total=("AOBT_3_flt", lambda s: len(s)),
    aobt_na=("AOBT_3_flt", lambda s: int(s.isna().sum()))
).assign(pct=lambda d: d["aobt_na"]/d["total"]*100).sort_values("pct", ascending=False).head(15)
print(ap_hr.to_string())

# ============================================================
# 召回 / 精确：在 holdout 上用规则圈出 AOBT_NaT
# ============================================================
print()
print("=" * 70); print("[Step1b] 圈出召回/精确 (holdout 上事后评估)")
print("=" * 70)

# 用预测时点可得特征 → 标 AOBT_NaT 候选
# 圈出规则: 单纯 AOBT.isna() 完美 recall=100, precision=100 (这是 ground truth)
# 但真榜推理时 ranking 同样有 AOBT 字段 → 直接 isna() 也完美
# 这就是 Step0 答案: ranking 真榜上 AOBT 100% 可得, 直接 isna() 即可
na_mask_holdout = ho_dep["AOBT_3_flt"].isna().values & m_full
n_recall_target = int(na_mask_holdout.sum())
n_killed = n_recall_target  # 直接 isna() = 完美 recall
n_wrong_kill = 0  # 没误杀
print(f"  用 isna() 直接圈: recall={n_killed/n_na*100:.1f}%, precision=100%, 误杀=0")

# 用"事前特征"组合近似: 看 AOBT_NaT 占比最高的机场/运营商/机型/h时段
# 取阈值: ≥20% 缺率的机场 → 召回多少
high_na_ap = set(airport_na_pct[airport_na_pct["pct"] >= 20].index.tolist())
high_na_op = set(op_na_pct[op_na_pct["pct"] >= 20].index.tolist())
high_na_at = set(at_na_pct[at_na_pct["pct"] >= 20].index.tolist())

# holdout 上这些行的 aobt_na 分布
m_ap_high = m_full & ho_dep["ADEP_mvt"].isin(high_na_ap).values
m_op_high = m_full & ho_dep["AIRCRAFT_OPERATOR_flt"].isin(high_na_op).values
m_at_high = m_full & ho_dep["AIRCRAFT_TYPE_mvt"].isin(high_na_at).values
m_union = m_full & (ho_dep["ADEP_mvt"].isin(high_na_ap).values
                     | ho_dep["AIRCRAFT_OPERATOR_flt"].isin(high_na_op).values
                     | ho_dep["AIRCRAFT_TYPE_mvt"].isin(high_na_at).values)

print(f"\n  阈值 ≥20% 缺率, 候选规则:")
print(f"    ADEP 高缺率 ({len(high_na_ap)} 个): {sorted(high_na_ap)[:20]}")
print(f"    OPERATOR 高缺率 ({len(high_na_op)} 个): {sorted(map(str, high_na_op))[:20]}")
print(f"    AIRCRAFT_TYPE 高缺率 ({len(high_na_at)} 个): {sorted(map(str, high_na_at))[:20]}")

def eval_rule(name, mask):
    n_rule = int(mask.sum())
    if n_rule == 0:
        print(f"  [{name}] 不圈出任何航段")
        return None
    tp = int((mask & na_mask_holdout).sum())  # 圈内确实是 aobt_na (召回)
    fp = int((mask & ~na_mask_holdout).sum())  # 圈内是 aobt_valid (误杀)
    n_predicted = n_rule
    n_actual_na = n_na
    recall = tp / max(n_actual_na, 1) * 100
    precision = tp / max(n_rule, 1) * 100
    print(f"  [{name}] 圈={n_rule:,} TP={tp} FP={fp}  precision={precision:.1f}%  recall={recall:.1f}%")
    return {"n_rule": n_rule, "TP": tp, "FP": fp, "precision": precision, "recall": recall}

eval_rule("ADEP≥20%", m_ap_high)
eval_rule("OPERATOR≥20%", m_op_high)
eval_rule("TYPE≥20%", m_at_high)
eval_rule("三者并集", m_union)

# 也尝试更严格阈值
print("\n  阈值 ≥50% 缺率:")
v_ap50 = set(airport_na_pct[airport_na_pct["pct"] >= 50].index.tolist())
v_op50 = set(op_na_pct[op_na_pct["pct"] >= 50].index.tolist())
v_at50 = set(at_na_pct[at_na_pct["pct"] >= 50].index.tolist())
mask_ap50 = m_full & ho_dep["ADEP_mvt"].isin(v_ap50).values
mask_op50 = m_full & ho_dep["AIRCRAFT_OPERATOR_flt"].isin(v_op50).values
mask_at50 = m_full & ho_dep["AIRCRAFT_TYPE_mvt"].isin(v_at50).values
mask_u50 = m_full & (ho_dep["ADEP_mvt"].isin(v_ap50).values
                      | ho_dep["AIRCRAFT_OPERATOR_flt"].isin(v_op50).values
                      | ho_dep["AIRCRAFT_TYPE_mvt"].isin(v_at50).values)
print(f"    ADEP≥50% ({len(v_ap50)}): {sorted(v_ap50)}")
eval_rule("ADEP≥50%", mask_ap50)
eval_rule("OPERATOR≥50%", mask_op50)
eval_rule("TYPE≥50%", mask_at50)
eval_rule("三者并集≥50%", mask_u50)

# ============================================================
# [Step2] 兜底模拟
# ============================================================
print()
print("=" * 70); print("[Step2] 兜底模拟")
print("=" * 70)

# 训练折 train 2-12 月 (与模型训折一致, 不混 holdout 信息)
df_tr_full_for_fallback = df_tr_raw.copy()
df_tr_full_for_fallback = parse_time_columns(df_tr_full_for_fallback, cols=TIME_COLS)
df_tr_dep_full_for_fb = df_tr_full_for_fallback[
    (df_tr_full_for_fallback["PHASE_mvt"]=="DEP")].copy()
df_tr_dep_full_for_fb = df_tr_dep_full_for_fb[
    (df_tr_dep_full_for_fb["TAXITIME_SEC_mvt"]>0)&
    (df_tr_dep_full_for_fb["TAXITIME_SEC_mvt"]<10800)].reset_index(drop=True)
print(f"  train 折 DEP 总行: {len(df_tr_dep_full_for_fb):,}")
print(f"  train 折 AOBT_NaT: {df_tr_dep_full_for_fb['AOBT_3_flt'].isna().sum():,}")

# 训练折按 (ADEP, hour) 分组的 TAXITIME 条件均值
df_tr_dep_full_for_fb["_hour"] = pd.to_datetime(df_tr_dep_full_for_fb["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce").dt.hour
df_tr_dep_full_for_fb["_hour"] = df_tr_dep_full_for_fb["_hour"].fillna(-1).astype(int)
ap_hr_mean = df_tr_dep_full_for_fb.groupby(["ADEP_mvt", "_hour"])["TAXITIME_SEC_mvt"].mean()
ap_hr_med = df_tr_dep_full_for_fb.groupby(["ADEP_mvt", "_hour"])["TAXITIME_SEC_mvt"].median()
ap_mean = df_tr_dep_full_for_fb.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].mean()
global_mean = float(df_tr_dep_full_for_fb["TAXITIME_SEC_mvt"].mean())

def get_fallback(ap, hr):
    """优先 (ap, hour), 回退 ap, 再回退 global"""
    try:
        v = ap_hr_mean.loc[(ap, hr)]
        if not pd.isna(v):
            return float(v)
    except KeyError:
        pass
    try:
        v = ap_mean.loc[ap]
        if not pd.isna(v):
            return float(v)
    except KeyError:
        pass
    return global_mean

# A: 基线 (不兜底)
err2_a = resid**2  # 全长
sse_a = float(err2_a[m_full].sum())
rmse_a = float(np.sqrt(err2_a[m_full].mean()))

# B: AOBT_NaT 直接用 (ADEP, hour) 兜底均值 (holdout 验证)
pred_b = pred.copy()
fallback_used = []
for i in np.where(m_aobt_na)[0]:
    ap = ho_dep.iloc[i]["ADEP_mvt"]
    hr = int(ho_dep.iloc[i]["sched_hour"])
    fb = get_fallback(ap, hr)
    pred_b[i] = fb
    fallback_used.append(fb)
err2_b = (y_ho - pred_b)**2
sse_b = float(err2_b[m_full].sum())
rmse_b = float(np.sqrt(err2_b[m_full].mean()))
print(f"\n  [A] 基线 (LGB 预测): RMSE={rmse_a:.2f}, SSE={sse_a:.3e}")
print(f"  [B] AOBT_NaT 用 (ADEP,hour) 兜底均值: RMSE={rmse_b:.2f}, SSE={sse_b:.3e}")
print(f"    全样本改善: Δ RMSE={rmse_b - rmse_a:+.3f}, Δ SSE={sse_b - sse_a:+.3e} ({(sse_b-sse_a)/sse_a*100:+.2f}%)")
rmse_a_na = float(np.sqrt(err2_a[m_aobt_na].mean()))
rmse_b_na = float(np.sqrt(err2_b[m_aobt_na].mean()))
print(f"    AOBT_NaT 段: {rmse_a_na:.2f} → {rmse_b_na:.2f}  ({rmse_b_na - rmse_a_na:+.2f})")
rmse_a_v = float(np.sqrt(err2_a[m_aobt_v].mean()))
rmse_b_v = float(np.sqrt(err2_b[m_aobt_v].mean()))
print(f"    AOBT_valid 段 (误杀代价): {rmse_a_v:.2f} → {rmse_b_v:.2f}  ({rmse_b_v - rmse_a_v:+.2f})")

# C: AOBT_NaT 用全 global 均值
pred_c = pred.copy()
for i in np.where(m_aobt_na)[0]:
    pred_c[i] = global_mean
err2_c = (y_ho - pred_c)**2
rmse_c = float(np.sqrt(err2_c[m_full].mean()))
print(f"\n  [C] AOBT_NaT 用全 global均值: RMSE={rmse_c:.2f}")

# D: 对比 — 规则 >10800 (跟体检里 long taxi 兜底思路对比)
m_gt10k = m_full & (y_ho > 10800)
pred_d = pred.copy()
for i in np.where(m_gt10k)[0]:
    ap = ho_dep.iloc[i]["ADEP_mvt"]
    hr = int(ho_dep.iloc[i]["sched_hour"])
    pred_d[i] = get_fallback(ap, hr)
err2_d = (y_ho - pred_d)**2
rmse_d = float(np.sqrt(err2_d[m_full].mean()))
n_gt10k = int(m_gt10k.sum())
sse_d = float(err2_d[m_full].sum())
print(f"\n  [D] 仅 >10800 ({n_gt10k} 行) 兜底 (对比): RMSE={rmse_d:.2f}, SSE={sse_d:.3e}")
print(f"      >10800 段 RMSE: {float(np.sqrt(err2_d[m_gt10k].mean())):.2f}")

# E: AOBT_NaT OR >10800 同时兜底
m_either = m_full & (m_aobt_na | m_gt10k)
pred_e = pred.copy()
for i in np.where(m_either)[0]:
    ap = ho_dep.iloc[i]["ADEP_mvt"]
    hr = int(ho_dep.iloc[i]["sched_hour"])
    pred_e[i] = get_fallback(ap, hr)
err2_e = (y_ho - pred_e)**2
rmse_e = float(np.sqrt(err2_e[m_full].mean()))
n_e = int(m_either.sum())
print(f"\n  [E] AOBT_NaT OR >10800 ({n_e:,} 行) 都兜底: RMSE={rmse_e:.2f}")
rmse_e_na = float(np.sqrt(err2_e[m_aobt_na].mean()))
rmse_e_gt = float(np.sqrt(err2_e[m_gt10k].mean()))
print(f"    AOBT_NaT 段: {rmse_e_na:.2f}")
print(f"    >10800 段: {rmse_e_gt:.2f}")

# F: AOBT_NaT 圈出一些保守 subset, 兜底. 比如用 (ADEP, OPERATOR, TYPE) 三阈值并集
m_f = m_union  # 三者并集 (≥20% 缺率)
pred_f = pred.copy()
n_correct = 0
for i in np.where(m_f)[0]:
    ap = ho_dep.iloc[i]["ADEP_mvt"]
    hr = int(ho_dep.iloc[i]["sched_hour"])
    pred_f[i] = get_fallback(ap, hr)
err2_f = (y_ho - pred_f)**2
rmse_f = float(np.sqrt(err2_f[m_full].mean()))
rmse_f_na = float(np.sqrt(err2_f[m_aobt_na].mean()))
rmse_f_v = float(np.sqrt(err2_f[m_aobt_v].mean()))
print(f"\n  [F] 三特征 ≥20% 缺率 并集圈 (~{int(m_f.sum()):,} 行) 兜底: RMSE={rmse_f:.2f}")
print(f"    AOBT_NaT 段: {rmse_f_na:.2f}")
print(f"    AOBT_valid 段 (误杀): {rmse_f_v:.2f}")

print()
print(f"  ★ 全样本 RMSE 改善一览:")
print(f"    A (基线):              {rmse_a:.2f}")
print(f"    B (AOBT_NaT 兜底=全均值): {rmse_b:.2f}  Δ={rmse_b-rmse_a:+.2f}")
print(f"    C (AOBT_NaT 兜底=global):  {rmse_c:.2f}  Δ={rmse_c-rmse_a:+.2f}")
print(f"    D (>10800 兜底):         {rmse_d:.2f}  Δ={rmse_d-rmse_a:+.2f}")
print(f"    E (两者都兜底):          {rmse_e:.2f}  Δ={rmse_e-rmse_a:+.2f}")
print(f"    F (≥20%缺率组 兜底):     {rmse_f:.2f}  Δ={rmse_f-rmse_a:+.2f}")

# ============================================================
# 落盘
# ============================================================
out = {
    "step0_ranking_aobt": {
        "ranking_total": int(len(rank)),
        "ranking_dep": int(len(rank_dep)),
        "aobt_non_null_count": int(nn_dep),
        "aobt_na_count": int(len(rank_dep) - int(nn_dep)),
        "aobt_na_pct": float((len(rank_dep) - int(nn_dep))/len(rank_dep)*100),
        "verdict": ("ranking AOBT 100% 字段可得, 直接 isna() 圈出即可, 不需要事先代理"),
    },
    "step1_holdout_chars": {
        "n_full_dep": n_full,
        "n_aobt_na": n_na,
        "n_aobt_valid": n_v,
        "aobt_na_sse_pct": float(err2[m_aobt_na].sum()/err2[m_full].sum()*100),
        "aobt_na_tax_dist": {
            "median": float(np.median(y_na)),
            "p99": float(np.percentile(y_na,99)),
            "max": float(y_na.max()),
            "n_gt_1800": int((y_na > 1800).sum()),
            "n_gt_3600": int((y_na > 3600).sum()),
            "n_gt_10800": int((y_na > 10800).sum()),
        },
        "aobt_valid_tax_dist": {
            "median": float(np.median(y_v)),
            "p99": float(np.percentile(y_v,99)),
        },
        "top_airport_na_pct": airport_na_pct.head(15).to_dict("records"),
        "top_operator_na_pct": op_na_pct.head(15).to_dict("records"),
        "top_type_na_pct": at_na_pct.head(15).to_dict("records"),
        "top_ap_hour_na_pct": ap_hr.head(15).to_dict("records"),
    },
    "rules_eval": {
        "isna_direct": {"TP": n_na, "FP": 0, "FN": 0, "precision_pct": 100, "recall_pct": 100,
                         "note": "ranking 真榜直接用 isna() 圈出"},
        "ADEP_geq_20pct": eval_rule("ADEP≥20%", m_ap_high),
        "OPERATOR_geq_20pct": eval_rule("OPERATOR≥20%", m_op_high),
        "TYPE_geq_20pct": eval_rule("TYPE≥20%", m_at_high),
        "union_geq_20pct": eval_rule("union≥20%", m_union),
    },
    "step2_fallback_sim": {
        "A_baseline_rmse": rmse_a,
        "B_aobtna_ap_hr_mean": {"rmse_full": rmse_b, "rmse_aobtna": rmse_b_na, "rmse_valid": rmse_b_v,
                                 "delta_rmse": rmse_b - rmse_a, "delta_pct": (rmse_b-rmse_a)/rmse_a*100,
                                 "SSE_full_a": sse_a, "SSE_full_b": sse_b,
                                 "SSE_aobtna_drop": float((err2[m_aobt_na] - err2_b[m_aobt_na]).sum()),
                                 "fallback_used_n": len(fallback_used),
                                 "fallback_mean": float(np.mean(fallback_used)),
                                 "fallback_median": float(np.median(fallback_used))},
        "C_aobtna_global": rmse_c,
        "D_gt10k_only": rmse_d,
        "E_aobtna_or_gt10k": rmse_e,
        "F_union_geq_20pct": rmse_f,
        "best_is": ("E" if (rmse_e < rmse_b and rmse_e < rmse_d) else "B"),
        "verdict": (
            f"全样本 RMSE 改善: B {rmse_b-rmse_a:+.2f}, D {rmse_d-rmse_a:+.2f}, E {rmse_e-rmse_a:+.2f}, F {rmse_f-rmse_a:+.2f}"
        ),
    },
    "validity_on_ranking": {
        "ranking_aobt_field_observable": True,
        "ranking_aobt_missing_rate": float((len(rank_dep) - int(nn_dep))/len(rank_dep)*100),
        "rule_can_be_applied": True,
        "recommendation": "真榜: 直接对 ranking DEP where AOBT_3_flt.isna() 圈出, 替换为 (ADEP, sched_hour) 训练折条件均值 (T-Unimp 级 fallback). 见 skills/osn-leaderboard-upload.md 类似流程, 在推理后处理层做.",
    },
    "red_lines": "不改 label / 不 clip 真值 / 不建单机场模型; 条件均值只用训练折",
}
(D0 / "aobt_nat_diagnose.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print(f"  → 写出: {D0 / 'aobt_nat_diagnose.json'}")
