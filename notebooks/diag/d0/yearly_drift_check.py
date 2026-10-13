"""
2025→2026 同月跨年分布漂移体检
=====================================
- ranking (2026-01+07) vs training (2025-01+07)
- 逐原始字段 + 工程特征, 比分布
- 输出: 稳定白名单 + 漂移清单 + 总体判定

只读, 不训练, 不发版
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns

# ====== 配置 ======
TRAIN_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
    "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_flt", "ARVT_1_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt", "ARVT_1_flt"]

# 工程特征 (71 基础 + offset) - 这些不会出现在 raw 表里, 我们基于 raw + 时间做时差
NUMERIC_VARS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "ARVT_3_flt",
                "ARVT_1_flt"]
CATEGORICAL_VARS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                    "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt",
                    "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
                    "AIRCRAFT_OPERATOR_flt", "FLIGHT_mvt", "FLIGHT_RULE_mvt",
                    "ADEP_flt", "ADES_flt", "ADES_FILED_flt"]


def numeric_stats(df, col, df_ref, col_ref):
    """数值列漂移: 缺失率, p1/p50/p99, KS 检验.
    时间列只 KS 检验 + 月分布; 不强行打 percentile."""
    a = df[col].dropna()
    b = df_ref[col_ref].dropna()
    if len(a) < 10 or len(b) < 10:
        return {"missing_pct_2026": float(df[col].isna().mean() * 100),
                "missing_pct_2025": float(df_ref[col_ref].isna().mean() * 100),
                "n_2026": int(len(a)), "n_2025": int(len(b)),
                "skip": "insufficient"}
    if pd.api.types.is_datetime64_any_dtype(a):
        # a 是 ranking 2026 数据, b 是 training 2025 数据 (按 main 调用)
        # 数据是 datetime64[us, UTC] — 内部整数是 μs 偏移
        b_ns = b.values.astype("int64")  # 2025
        a_ns = a.values.astype("int64")  # 2026
        ks, p = ks_2samp(a_ns, b_ns)
        iNaT = np.iinfo(np.int64).min
        a_mo = pd.to_datetime(a).dt.month.to_numpy()  # 2026
        b_mo = pd.to_datetime(b).dt.month.to_numpy()  # 2025
        a_valid = a_ns != iNaT
        b_valid = b_ns != iNaT
        a_m = pd.DataFrame({"m": a_mo[a_valid], "v": a_ns[a_valid]}).groupby("m")["v"].median()
        b_m = pd.DataFrame({"m": b_mo[b_valid], "v": b_ns[b_valid]}).groupby("m")["v"].median()
        return {
            "type": "datetime",
            "missing_pct_2026": float(df[col].isna().mean() * 100),
            "missing_pct_2025": float(df_ref[col_ref].isna().mean() * 100),
            "n_2026": int(len(a)), "n_2025": int(len(b)),
            "ks_stat": float(ks), "ks_p": float(p),
            "month_p50_2026": {int(k): int(v) for k, v in a_m.items()},  # 2026 ranking
            "month_p50_2025": {int(k): int(v) for k, v in b_m.items()},  # 2025 training
            "datetime_unit": "us",  # 重要: 数据用微秒
            "drift_flag": "DRIFT" if p < 0.01 and ks > 0.05 else ("WATCH" if p < 0.05 else "STABLE"),
            "drift_note": "时间戳 KS 受年份漂移影响, 看 month_p50 对比",
        }
    ks, p = ks_2samp(a.values, b.values)
    return {
        "type": "numeric",
        "missing_pct_2026": float(df[col].isna().mean() * 100),
        "missing_pct_2025": float(df_ref[col_ref].isna().mean() * 100),
        "n_2026": int(len(a)), "n_2025": int(len(b)),
        "p1_2026": float(np.percentile(a, 1)), "p1_2025": float(np.percentile(b, 1)),
        "p50_2026": float(np.median(a)), "p50_2025": float(np.median(b)),
        "p99_2026": float(np.percentile(a, 99)), "p99_2025": float(np.percentile(b, 99)),
        "mean_2026": float(np.mean(a)), "mean_2025": float(np.mean(b)),
        "ks_stat": float(ks), "ks_p": float(p),
        "drift_flag": "DRIFT" if p < 0.01 and ks > 0.05 else ("WATCH" if p < 0.05 else "STABLE"),
    }


def categorical_stats(df, col, df_ref, col_ref):
    """类别列漂移: 基数, top10 重叠率, 新增/消失类"""
    a = df[col].astype(str).fillna("__nan__")
    b = df_ref[col_ref].astype(str).fillna("__nan__")
    cnt_a = a.value_counts()
    cnt_b = b.value_counts()
    set_a, set_b = set(cnt_a.index), set(cnt_b.index)
    new_cats = set_a - set_b
    miss_cats = set_b - set_a
    top10_a = set(cnt_a.head(10).index)
    top10_b = set(cnt_b.head(10).index)
    overlap = len(top10_a & top10_b)
    return {
        "cardinality_2026": int(len(set_a)),
        "cardinality_2025": int(len(set_b)),
        "n_new_2026_only": int(len(new_cats)),
        "n_missing_2025_only": int(len(miss_cats)),
        "top10_overlap": int(overlap),
        "top10_new_2026": sorted(list(top10_a - top10_b))[:5],
        "top10_lost": sorted(list(top10_b - top10_a))[:5],
        "drift_flag": "DRIFT" if len(new_cats) > len(set_b) * 0.10 else
                      ("WATCH" if len(new_cats) > len(set_b) * 0.05 else "STABLE"),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("[年度漂移体检] 2025-01+07 (training) vs 2026-01+07 (ranking)")
    print("=" * 70)

    # ====== 加载数据 ======
    print("\n[1] 加载 2025 holdout + 2026 ranking")
    df_2025 = pd.concat([
        load_month("data/raw/training_2025-01-01_2025-02-01.parquet", columns=TRAIN_COLS),
        load_month("data/raw/training_2025-07-01_2025-08-01.parquet", columns=TRAIN_COLS),
    ], ignore_index=True)
    df_2025 = parse_time_columns(df_2025, cols=TIME_COLS)
    df_2025 = df_2025[df_2025["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)

    df_2026 = pd.read_parquet("data/raw/ranking.parquet", columns=TRAIN_COLS)
    df_2026 = parse_time_columns(df_2026, cols=[c for c in TIME_COLS if c != "TAXITIME_SEC_mvt"])
    df_2026 = df_2026[df_2026["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    print(f"  2025 DEP (1+7): {len(df_2025):,}")
    print(f"  2026 DEP (1+7): {len(df_2026):,}")

    # ====== 数值列体检 ======
    print("\n" + "=" * 70)
    print("[2] 数值特征漂移 (KS 检验)")
    print("=" * 70)
    num_results = {}
    for col in NUMERIC_VARS:
        if col not in df_2025.columns or col not in df_2026.columns:
            continue
        s = numeric_stats(df_2026, col, df_2025, col)
        num_results[col] = s
        if "skip" in s:
            print(f"  {col:>28}: n_2026={s['n_2026']} n_2025={s['n_2025']} SKIP")
        elif s.get("type") == "datetime":
            print(f"  {col:>28} [DT]: miss26={s['missing_pct_2026']:>5.2f}% "
                  f"miss25={s['missing_pct_2025']:>5.2f}% "
                  f"KS={s['ks_stat']:.3f} (跨年, 受年份影响)")
            for m in sorted(s["month_p50_2025"].keys()):
                v25 = s["month_p50_2025"].get(m)  # 2025 training
                v26 = s["month_p50_2026"].get(m)  # 2026 ranking
                unit = s.get("datetime_unit", "ns")
                if v25 and v26:
                    d25 = pd.to_datetime(np.int64(v25), unit=unit).strftime("%Y-%m-%d %H:%M")
                    d26 = pd.to_datetime(np.int64(v26), unit=unit).strftime("%Y-%m-%d %H:%M")
                    actual_diff = (pd.to_datetime(np.int64(v26), unit=unit) -
                                   pd.to_datetime(np.int64(v25), unit=unit)).total_seconds() / 86400
                    drift_from_one_year = abs(actual_diff - 365)
                    print(f"    月={m}: 2025={d25}  2026={d26}  "
                          f"actual_diff={actual_diff:+.1f}天  "
                          f"|Δ-1年|={drift_from_one_year:.2f}天 → "
                          f"{'STABLE' if drift_from_one_year < 30 else 'DRIFT'}")
                elif v25:
                    d25 = pd.to_datetime(np.int64(v25), unit=unit).strftime("%Y-%m-%d %H:%M")
                    print(f"    月={m}: 2025={d25}  2026=? (无数据)")
                else:
                    print(f"    月={m}: 2025=?  2026=? (无数据)")
            print(f"    → 跨年时间戳均值+1年=期望, 若 diff < 30天 → 同月跨年稳定")
        else:
            print(f"  {col:>28}: miss26={s['missing_pct_2026']:>5.2f}% "
                  f"miss25={s['missing_pct_2025']:>5.2f}% "
                  f"p50_26={s['p50_2026']:>10.1f} p50_25={s['p50_2025']:>10.1f} "
                  f"KS={s['ks_stat']:.3f} p={s['ks_p']:.2e} → {s['drift_flag']}")

    # ====== 类别列体检 ======
    print("\n" + "=" * 70)
    print("[3] 类别特征漂移 (基数, top10 重叠)")
    print("=" * 70)
    cat_results = {}
    for col in CATEGORICAL_VARS:
        if col not in df_2025.columns or col not in df_2026.columns:
            continue
        s = categorical_stats(df_2026, col, df_2025, col)
        cat_results[col] = s
        print(f"  {col:>28}: card26={s['cardinality_2026']:>6,} "
              f"card25={s['cardinality_2025']:>6,} "
              f"new26={s['n_new_2026_only']:>4} "
              f"miss25={s['n_missing_2025_only']:>4} "
              f"top10_overlap={s['top10_overlap']:>2} → {s['drift_flag']}")

    # ====== 关键信号 ======
    print("\n" + "=" * 70)
    print("[4] 关键信号 (按月跨年对比)")
    print("=" * 70)

    # 4.1 AOBT 缺失
    aobt_nat_2026 = df_2026["AOBT_3_flt"].isna().mean() * 100
    aobt_nat_2025 = df_2025["AOBT_3_flt"].isna().mean() * 100
    print(f"\n  4.1 AOBT 缺失率:")
    print(f"    2025-01+07: {aobt_nat_2025:.2f}%  ({df_2025['AOBT_3_flt'].isna().sum():,} / {len(df_2025):,})")
    print(f"    2026-01+07: {aobt_nat_2026:.2f}  ({df_2026['AOBT_3_flt'].isna().sum():,} / {len(df_2026):,})")
    print(f"    跨年绝对差: {aobt_nat_2026 - aobt_nat_2025:+.2f} pct")

    # 按月
    for m in [1, 7]:
        s_25 = df_2025[pd.to_datetime(df_2025["MVT_TIME_UTC_mvt"]).dt.month == m]
        s_26 = df_2026[pd.to_datetime(df_2026["MVT_TIME_UTC_mvt"]).dt.month == m]
        print(f"    月={m}: 2025 AOBT_NaT={(s_25['AOBT_3_flt'].isna().mean()*100):.2f}%  "
              f"2026 AOBT_NaT={(s_26['AOBT_3_flt'].isna().mean()*100):.2f}%")

    # 4.2 IOBT 缺失 / EOBT 缺失
    for col in ["IOBT_flt", "EOBT_1_flt", "LOBT_flt", "BLOCK_TIME_UTC_mvt"]:
        if col not in df_2025.columns:
            continue
        n25 = df_2025[col].isna().mean() * 100
        n26 = df_2026[col].isna().mean() * 100
        print(f"  {col} 缺失率: 2025={n25:.2f}%  2026={n26:.2f}%  Δ={n26-n25:+.2f}")

    # 4.3 计划时间差 (EOBT/IOBT/SCHED/LOBT 两两差)
    print(f"\n  4.3 计划时间差 (秒):")
    pairs = [("EOBT_1_flt", "SCHED_TIME_UTC_mvt", "EOBT-SCHED"),
             ("IOBT_flt", "SCHED_TIME_UTC_mvt", "IOBT-SCHED"),
             ("EOBT_1_flt", "IOBT_flt", "EOBT-IOBT"),
             ("LOBT_flt", "EOBT_1_flt", "LOBT-EOBT"),
             ("AOBT_3_flt", "EOBT_1_flt", "AOBT-EOBT")]
    pair_results = {}
    for c1, c2, lbl in pairs:
        for year_label, df in [("2025", df_2025), ("2026", df_2026)]:
            m = df[c1].notna() & df[c2].notna()
            diff = (df[c1][m] - df[c2][m]).dt.total_seconds()
            r = {"label": lbl, "year": year_label,
                 "n": int(m.sum()),
                 "p1": float(np.percentile(diff, 1)),
                 "p50": float(np.median(diff)),
                 "p99": float(np.percentile(diff, 99)),
                 "mean": float(np.mean(diff)),
                 "frac_neg": float((diff < 0).mean() * 100)}
            print(f"    {year_label} {lbl:18s}: n={r['n']:>7,}  "
                  f"p50={r['p50']:>+8.1f}s  p99={r['p99']:>+8.1f}s  "
                  f"frac_neg={r['frac_neg']:.2f}%")
            pair_results.setdefault(lbl, {})[year_label] = r

    # 4.4 BLOCK/AOBT 对照 (BLOCK 在 ranking DEP 100% 缺失, 已验证)
    blk_2026 = df_2026["BLOCK_TIME_UTC_mvt"].isna().mean() * 100
    blk_2025 = df_2025["BLOCK_TIME_UTC_mvt"].isna().mean() * 100
    print(f"\n  4.4 BLOCK_TIME_UTC_mvt 缺失率: 2025={blk_2025:.2f}%  2026={blk_2026:.2f}%  Δ={blk_2026-blk_2025:+.2f}")

    # 4.5 RUNWAY, STAND 覆盖
    print(f"\n  4.5 跑道/机位 (ADEP, RUNWAY) 与 (ADEP, STAND) 组合:")
    for y, df in [("2025", df_2025), ("2026", df_2026)]:
        rw = df.groupby(["ADEP_mvt", "RUNWAY_mvt"]).size()
        st = df.groupby(["ADEP_mvt", "STAND_mvt"]).size()
        print(f"    {y}: (ADEP,RUNWAY) 组数={rw.shape[0]:,}, (ADEP,STAND) 组数={st.shape[0]:,}")
        # top 5 ADEP 各组均架次
        top_ap = df["ADEP_mvt"].astype(str).value_counts().head(5)
        print(f"    {y} top5 ADEP:")
        for ap in top_ap.index[:3]:
            n_ap = (df["ADEP_mvt"] == ap).sum()
            n_rw = ((df["ADEP_mvt"] == ap) & df["RUNWAY_mvt"].notna()).sum()
            n_st = ((df["ADEP_mvt"] == ap) & df["STAND_mvt"].notna()).sum()
            print(f"      {ap}: n={n_ap:,}  has_rwy={n_rw/n_ap*100:.1f}%  has_std={n_st/n_ap*100:.1f}%")

    # 4.6 时间分布 (小时/星期)
    print(f"\n  4.6 MVT_TIME 小时/星期分布:")
    for y, df in [("2025", df_2025), ("2026", df_2026)]:
        hr = pd.to_datetime(df["MVT_TIME_UTC_mvt"]).dt.hour
        dw = pd.to_datetime(df["MVT_TIME_UTC_mvt"]).dt.dayofweek
        print(f"    {y} hour dist: " +
                  ", ".join(f"{h:02d}={int((hr==h).sum()):>6,}" for h in range(0, 24, 6)))
        print(f"    {y} dow dist:  " +
                  ", ".join(f"{d}={int((dw==d).sum()):>6,}" for d in range(7)))

    # 4.7 跨天 (SCHED_TIME 与 MVT_TIME 跨天比例)
    cross_2025 = float((pd.to_datetime(df_2025["MVT_TIME_UTC_mvt"]).dt.normalize() !=
                       pd.to_datetime(df_2025["SCHED_TIME_UTC_mvt"]).dt.normalize()).mean() * 100)
    cross_2026 = float((pd.to_datetime(df_2026["MVT_TIME_UTC_mvt"]).dt.normalize() !=
                       pd.to_datetime(df_2026["SCHED_TIME_UTC_mvt"]).dt.normalize()).mean() * 100)
    print(f"\n  4.7 跨天 (MVT day != SCHED day):")
    print(f"    2025: {cross_2025:.2f}%")
    print(f"    2026: {cross_2026:.2f}%  Δ={cross_2026-cross_2025:+.2f}")

    # ====== 5. 极端候选行 ======
    print(f"\n" + "=" * 70)
    print("[5] 极端候选行 (AOBT 缺失 + 计划时间异常)")
    print("=" * 70)
    e25 = df_2025[
        df_2025["AOBT_3_flt"].isna() |
        ((df_2025["AOBT_3_flt"] - df_2025["IOBT_flt"]).dt.total_seconds().abs() > 1800)
    ]
    e26 = df_2026[
        df_2026["AOBT_3_flt"].isna() |
        ((df_2026["AOBT_3_flt"] - df_2026["IOBT_flt"]).dt.total_seconds().abs() > 1800)
    ]
    print(f"  2025: 极端候选 (AOBT_NaT 或 |AOBT-IOBT|>1800s) = {len(e25):,}")
    print(f"  2026: 极端候选 (AOBT_NaT 或 |AOBT-IOBT|>1800s) = {len(e26):,}")
    print(f"  占比:  2025={len(e25)/len(df_2025)*100:.2f}%  2026={len(e26)/len(df_2026)*100:.2f}%")
    if len(e25) > 0:
        m25 = e25["EOBT_1_flt"].isna().mean() * 100
        m26 = e26["EOBT_1_flt"].isna().mean() * 100
        print(f"  这类行 EOBT_NaT 占比: 2025={m25:.2f}%  2026={m26:.2f}%")

    # 按月
    print(f"\n  按月极端行分布:")
    for m in [1, 7]:
        n25 = ((df_2025["AOBT_3_flt"].isna() &
                (df_2025["EOBT_1_flt"].notna() | df_2025["IOBT_flt"].notna()) &
                (pd.to_datetime(df_2025["MVT_TIME_UTC_mvt"]).dt.month == m)).sum())
        n26 = ((df_2026["AOBT_3_flt"].isna() &
                (df_2026["EOBT_1_flt"].notna() | df_2026["IOBT_flt"].notna()) &
                (pd.to_datetime(df_2026["MVT_TIME_UTC_mvt"]).dt.month == m)).sum())
        print(f"    月={m}: 2025={n25:,}  2026={n26:,}  Δ={n26-n25:+}")

    print(f"\n" + "=" * 70)
    print("[总结]")
    print("=" * 70)
    # 时间戳按"跨年 +1y 是否成立"判定; 不算 KS
    n_dt_stable = sum(1 for col, v in num_results.items()
                       if v.get("type") == "datetime"
                       and all(abs((pd.to_datetime(np.int64(v.get("month_p50_2026", {}).get(m, 0)),
                                                     unit="us") -
                                    pd.to_datetime(np.int64(v.get("month_p50_2025", {}).get(m, 0)),
                                                     unit="us")).total_seconds() / 86400 - 365) < 30
                                  for m in v.get("month_p50_2025", {})
                                  if v.get("month_p50_2026", {}).get(m)))
    n_dt_drift = sum(1 for col, v in num_results.items()
                      if v.get("type") == "datetime") - n_dt_stable
    n_dt_skipped = sum(1 for col, v in num_results.items() if "skip" in v)
    print(f"  时间戳 (按月跨年 +1y 判定): STABLE={n_dt_stable}  DRIFT={n_dt_drift}  SKIP={n_dt_skipped}")

    # 类别特征 (基数, top10 重叠, new26 比例)
    n_cat_stable = sum(1 for v in cat_results.values() if v.get("drift_flag") == "STABLE")
    n_cat_drift = sum(1 for v in cat_results.values() if v.get("drift_flag") == "DRIFT")
    n_cat_watch = sum(1 for v in cat_results.values() if v.get("drift_flag") == "WATCH")
    print(f"  类别 (基数/重叠): STABLE={n_cat_stable}  WATCH={n_cat_watch}  DRIFT={n_cat_drift}")

    # 总判定
    if n_dt_stable >= 6 and n_cat_stable >= 5 and n_dt_drift <= 2:
        verdict = "分布基本一致(病因Y)"
    elif n_dt_stable + n_cat_stable >= 8 and n_dt_drift + n_cat_drift <= 5:
        verdict = "轻微漂移(主稳, 少数高基数类别新增)"
    else:
        verdict = "系统性漂移(病因X)"
    print(f"  总判定: {verdict}")

    # 落 JSON
    out = {
        "task": "yearly_drift_check_2025vs2026",
        "n_2025_dep": int(len(df_2025)),
        "n_2026_dep": int(len(df_2026)),
        "numeric_stats": num_results,
        "categorical_stats": cat_results,
        "key_signals": {
            "aobt_nat_2025_pct": aobt_nat_2025,
            "aobt_nat_2026_pct": aobt_nat_2026,
            "block_nat_2025_pct": blk_2025,
            "block_nat_2026_pct": blk_2026,
            "time_diff_pairs": pair_results,
            "cross_day_2025_pct": cross_2025,
            "cross_day_2026_pct": cross_2026,
        },
        "extreme_candidates": {
            "n_2025_aobt_nat_with_eobt_iobt": int(len(e25)),
            "n_2026_aobt_nat_with_eobt_iobt": int(len(e26)),
        },
        "verdict": verdict,
        "stable_count_datetime": n_dt_stable,
        "drift_count_datetime": n_dt_drift,
        "skip_count_datetime": n_dt_skipped,
        "stable_count_categorical": n_cat_stable,
        "drift_count_categorical": n_cat_drift,
        "watch_count_categorical": n_cat_watch,
        "total_time_sec": round(time.time() - t0, 1),
    }
    fn = D0 / "yearly_drift_check.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()