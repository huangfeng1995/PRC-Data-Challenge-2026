"""
全量数据质量审计 (只读)
========================
- A: 恒等式/逻辑一致性 (TAXITIME vs MVT-BLOCK, TAXI<=0, BLOCK>=MVT, BLOCK==MVT)
- B: BLOCK 来源检查 (BLOCK 顶替 SCHED/EOBT/TOBT/AOBT/IOBT/LOBT; 哨兵 -9223372037 / NaT)
- C: MVT 来源检查 (哨兵/NaT, MVT 与其他时间列相等, 时间戳异常, 重复 MVT)
- D: 时间链逻辑 (SCHED≈EOBT≈TOBT < BLOCK < MVT 顺序违反)
- E: 中档极端行重点 (1800-3600, 3600-10800 逐机场错误命中率, 异常簇)
- F: 机制裁决 (错误 vs 真实极端) + ranking 可识别性

不训练、不改 label、不 clip/winsorize
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns

# ===== 配置 =====
ALL_TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
# rank=None (训练+holdout 合计)
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt", "ARVT_1_flt"]
MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                   if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
              for m in range(1, 13)}
ALL_TIME_FOR_BLOCK = ["BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                       "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "LOBT_flt"]

iNaT_NS = np.iinfo(np.int64).min
SENTINEL_THRESHOLD = np.iinfo(np.int64).min + 10_000_000_000  # iNaT 附近容差


def ns_int(series):
    """datetime64[ns, UTC] -> int64; NaT -> iNaT_NS; 已 int64 保留"""
    if pd.api.types.is_datetime64_any_dtype(series):
        s2 = series
        if s2.dt.tz is not None:
            s2 = s2.dt.tz_localize(None)
        return s2.values.view("int64")
    if pd.api.types.is_integer_dtype(series):
        return series.values
    return pd.to_datetime(series, utc=True, errors="coerce").values.view("int64")


def abs_diff_seconds(a, b):
    """ns 差 / 1e9, NaT 产生 NaN"""
    d = (a - b).astype("float64") / 1e9
    return d


def load_full_dep():
    """加载 1-12 月全部 DEP, 加 holdout (1+7 月重叠)"""
    print("  加载 2025 1-12 月 DEP...")
    dfs = []
    for m in ALL_TRAIN_MONTHS:
        fn = MONTH_FILE[m]
        df = pd.read_parquet(fn, columns=[
            "MVT_ID_mvt", "PHASE_mvt",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "ARVT_3_flt", "ARVT_1_flt",
            "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "TAXITIME_SEC_mvt", "ADES_flt", "ADES_FILED_flt", "ADEP_flt",
            "FLIGHT_mvt", "AIRCRAFT_TYPE_mvt",
        ])
        df = df[df["PHASE_mvt"] == "DEP"].copy()
        df = parse_time_columns(df, cols=[c for c in TIME_COLS if c in df.columns])
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"  DEP 全量: {len(df_all):,}")
    return df_all


def load_rank_dep():
    print("  加载 ranking DEP (2026 1+7)...")
    df = pd.read_parquet("data/raw/ranking.parquet", columns=[
        "MVT_ID_mvt", "PHASE_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "ARVT_3_flt", "ARVT_1_flt",
        "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
        "ADES_flt", "ADES_FILED_flt", "ADEP_flt",
        "FLIGHT_mvt", "AIRCRAFT_TYPE_mvt",
    ])
    df = df[df["PHASE_mvt"] == "DEP"].copy()
    df = parse_time_columns(df, cols=[c for c in TIME_COLS if c in df.columns])
    print(f"  ranking DEP: {len(df):,}")
    return df


def per_airport_table(series_mask, df, top_n=10):
    """mask 命中行按 ADEP_mvt 计数, 给出机场表"""
    sub = df[series_mask]
    cnt = sub["ADEP_mvt"].astype(str).value_counts()
    total = len(sub)
    return [(ap, int(n), round(n/total*100, 1) if total else 0.0)
            for ap, n in cnt.head(top_n).items()]


def main():
    t_start = time.time()
    out = {}

    # ====== 加载 ======
    print("=" * 70)
    print("[全量数据质量审计] 只读, 不训练, 不改 label")
    print("=" * 70)
    df = load_full_dep()
    rank = load_rank_dep()

    # 统一为 ns int
    print("\n  转换时间戳到 ns int64...")
    time_ns = {}
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
              "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "ARVT_3_flt", "ARVT_1_flt"]:
        if c in df.columns:
            time_ns[c] = ns_int(df[c])
    y = df["TAXITIME_SEC_mvt"].values.astype("float64")
    ap = df["ADEP_mvt"].astype(str).values

    # ====== A. 恒等式与逻辑一致性 ======
    print("\n" + "=" * 70)
    print("[A] 恒等式与逻辑一致性")
    print("=" * 70)

    mvt_ns = time_ns["MVT_TIME_UTC_mvt"]
    blk_ns = time_ns["BLOCK_TIME_UTC_mvt"]
    valid_mvt = mvt_ns != iNaT_NS
    valid_blk = blk_ns != iNaT_NS

    # A1: TAXITIME vs MVT - BLOCK (容差 2s)
    derived_taxi = (mvt_ns - blk_ns).astype("float64") / 1e9
    valid_pair = valid_mvt & valid_blk
    diff_taxi = y - derived_taxi
    m_mismatch = valid_pair & (np.abs(diff_taxi) > 2.0)
    n_mismatch = int(m_mismatch.sum())
    # 若 TAXITIME<=0 或缺失的也算 invalid
    invalid_tax = (~np.isfinite(y)) | (y <= 0)
    m_mismatch_clean = m_mismatch & (~invalid_tax)
    print(f"\n  A1 TAXITIME vs MVT-BLOCK 不符 (容差2s, TX>0):")
    print(f"    总命中: {n_mismatch:,}  (TX>0 干净: {int(m_mismatch_clean.sum()):,})")
    if n_mismatch:
        print(f"    差值分布: p50={float(np.median(diff_taxi[m_mismatch_clean])):.1f}s, "
              f"p90={float(np.percentile(diff_taxi[m_mismatch_clean],90)):.1f}s")
        out["A1"] = {"n_mismatch_total": n_mismatch, "n_mismatch_clean_TX_pos": int(m_mismatch_clean.sum()),
                     "diff_p50_s": round(float(np.median(diff_taxi[m_mismatch_clean])), 2),
                     "per_airport_top10": per_airport_table(m_mismatch_clean, df)}

    # A2a: TAXITIME <= 0
    m_tx_bad = (np.isfinite(y)) & (y <= 0)
    print(f"\n  A2a TAXITIME <= 0: {int(m_tx_bad.sum()):,}")
    out["A2a_TX_le_zero"] = (int(m_tx_bad.sum()),
                              per_airport_table(m_tx_bad, df) if m_tx_bad.sum() else [])

    # A2b: BLOCK >= MVT (无效: BLOCK 应该 < MVT, 但 BLOCK 在 ranking 100% 缺失)
    m_inv_blk_mvt = valid_pair & (blk_ns >= mvt_ns)
    print(f"  A2b BLOCK >= MVT: {int(m_inv_blk_mvt.sum()):,}")
    out["A2b_BLOCK_ge_MVT"] = (int(m_inv_blk_mvt.sum()),
                                per_airport_table(m_inv_blk_mvt, df) if m_inv_blk_mvt.sum() else [])

    # A2c: BLOCK == MVT (BLOCK 与 MVT 同秒)
    m_eq = valid_pair & (blk_ns == mvt_ns)
    print(f"  A2c BLOCK == MVT: {int(m_eq.sum()):,}")
    out["A2c_BLOCK_eq_MVT"] = (int(m_eq.sum()),
                                per_airport_table(m_eq, df) if m_eq.sum() else [])

    # ====== B. BLOCK 来源检查 ======
    print("\n" + "=" * 70)
    print("[B] BLOCK 来源检查")
    print("=" * 70)

    # B1: BLOCK 哨兵 (负值) - 但已是 datetime64, 不能直接是负数, 所以检查 NaT
    m_blk_nat = ~valid_blk
    print(f"\n  B1 BLOCK NaT: {int(m_blk_nat.sum()):,}")
    out["B1_BLOCK_NaT"] = (int(m_blk_nat.sum()),
                              per_airport_table(m_blk_nat, df) if m_blk_nat.sum() else [])

    # B2: BLOCK 与其他时间列相等 (容差 60s)
    block_imposter = {}
    for c in ALL_TIME_FOR_BLOCK:
        if c == "BLOCK_TIME_UTC_mvt":
            continue
        if c not in time_ns:
            continue
        c_ns = time_ns[c]
        valid_c = c_ns != iNaT_NS
        # 同 DEP 内, BLOCK ≈ C 且 BLOCK NaT 已被 valid_blk 过滤
        # 这里看的是 BLOCK == C (同 ns 时间戳)
        m_both_valid = valid_blk & valid_c
        m_eq_block_c = m_both_valid & (blk_ns == c_ns)
        n_eq = int(m_eq_block_c.sum())
        m_close = m_both_valid & (np.abs(blk_ns - c_ns) <= 60 * int(1e9))
        n_close = int(m_close.sum())
        if n_close > 0:
            tax = y[m_close]
            sse = float(np.sum((tax - tax.mean()) ** 2)) if len(tax) > 1 else 0.0
            block_imposter[c] = {
                "n_exact_eq": n_eq, "n_close_60s": n_close,
                "taxi_median": round(float(np.nanmedian(tax)), 1) if len(tax) else None,
                "sse_total_in_block_rows": sse,
                "per_airport_top10": per_airport_table(m_close, df),
            }
            print(f"    BLOCK ≈ {c} (≤60s, 同 ns): n={n_close:,}  exact={n_eq:,}  "
                  f"taxi_med={float(np.nanmedian(tax)):.1f}s")
    out["B2_BLOCK_imposter"] = block_imposter

    # ====== C. MVT 来源检查 ======
    print("\n" + "=" * 70)
    print("[C] MVT 来源检查")
    print("=" * 70)
    m_mvt_nat = ~valid_mvt
    print(f"\n  C1 MVT NaT: {int(m_mvt_nat.sum()):,}")
    out["C1_MVT_NaT"] = (int(m_mvt_nat.sum()),
                          per_airport_table(m_mvt_nat, df) if m_mvt_nat.sum() else [])

    # MVT 重复: 同 (ADEP, MVT_TIME) 出现 >1 次
    if valid_mvt.sum() > 0:
        sub = df[valid_mvt][["MVT_TIME_UTC_mvt", "ADEP_mvt", "RUNWAY_mvt"]].copy()
        sub["MVT_ns"] = mvt_ns[valid_mvt]
        dup = sub.groupby(["ADEP_mvt", "MVT_ns"]).size()
        m_dup = dup[dup > 1]
        n_dup_rows = int(m_dup.sum())
        print(f"\n  C2 MVT 重复 (同 (ADEP,MVT_TIME) 出现 >1): "
              f"组数={len(m_dup):,}, 总行={n_dup_rows:,}")
        # 机场分布
        if len(m_dup) > 0:
            sub_dup = sub.merge(m_dup.rename("n"), on=["ADEP_mvt", "MVT_ns"])
            out["C2_MVT_dup"] = {
                "n_groups": int(len(m_dup)), "n_rows_total": n_dup_rows,
                "per_airport_top10": [(ap, int(n), 0.0)
                                        for ap, n in sub_dup["ADEP_mvt"].astype(str).value_counts().head(10).items()]}
        else:
            out["C2_MVT_dup"] = {"n_groups": 0, "n_rows_total": 0}

    # MVT 与其他时间列相等 (60s 容差)
    mvt_imposter = {}
    for c in ALL_TIME_FOR_BLOCK + ["AOBT_3_flt", "ARVT_3_flt"]:
        if c not in time_ns:
            continue
        c_ns = time_ns[c]
        valid_c = c_ns != iNaT_NS
        m_both = valid_mvt & valid_c
        m_close = m_both & (np.abs(mvt_ns - c_ns) <= 60 * int(1e9))
        n_close = int(m_close.sum())
        if n_close > 0:
            tax = y[m_close]
            mvt_imposter[c] = {
                "n_close_60s": n_close,
                "taxi_median": round(float(np.nanmedian(tax)), 1) if len(tax) else None,
                "per_airport_top10": per_airport_table(m_close, df),
            }
            print(f"    MVT ≈ {c} (≤60s): n={n_close:,}  "
                  f"taxi_med={float(np.nanmedian(tax)):.1f}s")
    out["C3_MVT_imposter"] = mvt_imposter

    # ====== D. 时间链逻辑 ======
    print("\n" + "=" * 70)
    print("[D] 时间链逻辑 (SCHED≈EOBT < BLOCK < MVT)")
    print("=" * 70)
    aobt_ns = time_ns["AOBT_3_flt"]
    iobt_ns = time_ns["IOBT_flt"]
    eobt_ns = time_ns["EOBT_1_flt"]
    sched_ns = time_ns["SCHED_TIME_UTC_mvt"]

    # D1: SCHED 晚于 MVT 数小时
    m_sched_late_mvt = valid_mvt & (sched_ns != iNaT_NS) & (mvt_ns - sched_ns > 3600 * int(1e9))
    print(f"\n  D1 SCHED 晚于 MVT >1h: {int(m_sched_late_mvt.sum()):,}")
    out["D1_SCHED_late_MVT_gt1h"] = (int(m_sched_late_mvt.sum()),
                                       per_airport_table(m_sched_late_mvt, df) if m_sched_late_mvt.sum() else [])

    # D2: BLOCK > MVT
    print(f"  D2 BLOCK > MVT (严格): {int(m_inv_blk_mvt.sum()):,}")
    out["D2_BLOCK_gt_MVT"] = (int(m_inv_blk_mvt.sum()),
                                per_airport_table(m_inv_blk_mvt, df) if m_inv_blk_mvt.sum() else [])

    # D3: SCHED 早于 AOBT 数小时 (SCHE > AOBT - 3600)
    valid_aobt = aobt_ns != iNaT_NS
    m_sched_early_aobt = valid_aobt & (sched_ns != iNaT_NS) & (sched_ns + 3600 * int(1e9) < aobt_ns)
    print(f"  D3 SCHED 早于 AOBT >1h: {int(m_sched_early_aobt.sum()):,}")
    out["D3_SCHED_early_AOBT_gt1h"] = (int(m_sched_early_aobt.sum()),
                                          per_airport_table(m_sched_early_aobt, df) if m_sched_early_aobt.sum() else [])

    # D4: MVT 早于 AOBT (AOBT > MVT)
    valid_aobt_v = (aobt_ns != iNaT_NS) & valid_mvt
    m_aobt_after_mvt = valid_aobt_v & (aobt_ns > mvt_ns)
    print(f"  D4 AOBT > MVT (推出晚于起飞): {int(m_aobt_after_mvt.sum()):,}")
    out["D4_AOBT_after_MVT"] = (int(m_aobt_after_mvt.sum()),
                                  per_airport_table(m_aobt_after_mvt, df) if m_aobt_after_mvt.sum() else [])

    # ====== E. 中档极端行重点排查 ======
    print("\n" + "=" * 70)
    print("[E] 中档极端行 (1800-3600, 3600-10800) 重点排查")
    print("=" * 70)
    m_long_mid = (y > 1800) & (y <= 3600)
    m_long_hi = (y > 3600) & (y <= 10800)
    print(f"  1800-3600: {int(m_long_mid.sum()):,}")
    print(f"  3600-10800: {int(m_long_hi.sum()):,}")

    def audit_long_band(mask, name):
        sub = df[mask]
        if len(sub) == 0:
            return
        print(f"\n  === {name} (n={len(sub):,}) ===")
        # 错误类型: BLOCK 顶替 (≈ SCHED 或 EOBT)
        if "BLOCK_TIME_UTC_mvt" in time_ns:
            blk_v = time_ns["BLOCK_TIME_UTC_mvt"][mask]
            valid_b = blk_v != iNaT_NS
            sched_v = time_ns["SCHED_TIME_UTC_mvt"][mask]
            eobt_v = time_ns["EOBT_1_flt"][mask]
            sched_valid = sched_v != iNaT_NS
            eobt_valid = eobt_v != iNaT_NS
            m_imp_sched = (sched_valid & valid_b & (np.abs(sched_v - blk_v) <= 60 * int(1e9)))
            m_imp_eobt = (eobt_valid & valid_b & (np.abs(eobt_v - blk_v) <= 60 * int(1e9)))
            m_imp_sched_e = m_imp_sched & m_imp_eobt
            print(f"    BLOCK 顶替 SCHED (≤60s): {int(m_imp_sched.sum()):,}  ({int(m_imp_sched.sum())*100/len(sub):.1f}%)")
            print(f"    BLOCK 顶替 EOBT (≤60s): {int(m_imp_eobt.sum()):,}  ({int(m_imp_eobt.sum())*100/len(sub):.1f}%)")
            print(f"    BLOCK 顶替 SCHED+EOBT (≤60s): {int(m_imp_sched_e.sum()):,}")
        # 关键时间列 NaT
        for col in ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]:
            c_v = time_ns[col][mask]
            n_nat = int((c_v == iNaT_NS).sum())
            if n_nat:
                print(f"    {col} NaT: {n_nat:,}")
        # ADEP_flt 缺失
        if "ADEP_flt" in df.columns:
            adep_flt = df["ADEP_flt"].values[mask]
            adep_flt_str = adep_flt.astype(str)
            missing_mask_local = (adep_flt_str == "nan") | (adep_flt_str == "<NA>") | pd.isna(adep_flt)
            n_adep_flt_missing = int(missing_mask_local.sum())
            print(f"    ADEP_flt 缺失/字面nan: {n_adep_flt_missing:,}")
        # ADES_flt
        if "ADES_flt" in df.columns:
            ades_flt = df["ADES_flt"].values[mask]
            nan_count = int(pd.isna(ades_flt).sum())
            print(f"    ADES_flt 缺失: {nan_count:,}")
        # 命中数 vs 推算数
        # 这与 v15 的 anomaly_rule_match 接近
        # 打印 TAXITIME 中位
        y_sub = y[mask]
        print(f"    TAXITIME: p50={float(np.median(y_sub)):.0f}, p99={float(np.percentile(y_sub, 99)):.0f}")

    audit_long_band(m_long_mid, "1800-3600")
    audit_long_band(m_long_hi, "3600-10800")

    # ====== F. 异常簇高 SSE 排查 ======
    print("\n" + "=" * 70)
    print("[F] 高 SSE 异常簇 (按机场 × 错误类型)")
    print("=" * 70)
    # 对 1800+ 段, 找 BLOCK 顶替簇 (机场 × BLOCK 与 SCHED/EOBT 相同的组)
    cluster_results = {}
    if "BLOCK_TIME_UTC_mvt" in time_ns:
        blk_v = time_ns["BLOCK_TIME_UTC_mvt"]
        sched_v = time_ns["SCHED_TIME_UTC_mvt"]
        eobt_v = time_ns["EOBT_1_flt"]
        valid_b = blk_v != iNaT_NS
        valid_s = sched_v != iNaT_NS
        valid_e = eobt_v != iNaT_NS
        # BLOCK 顶替 SCHED
        m_imp_sched = valid_b & valid_s & (np.abs(blk_v - sched_v) <= 60 * int(1e9))
        # BLOCK 顶替 EOBT
        m_imp_eobt = valid_b & valid_e & (np.abs(blk_v - eobt_v) <= 60 * int(1e9))
        # ADEP_flt 缺失
        adep_flt_str = df["ADEP_flt"].astype(str).values
        m_flt_missing = (adep_flt_str == "nan") | (adep_flt_str == "<NA>")
        # 4 类错误指纹
        sig_combos = {
            "BLOCK=BLOCK_S_EED+D_EB": m_imp_sched & m_imp_eobt,
            "BLOCK=BLOCK_S_EED_only": m_imp_sched & (~m_imp_eobt),
            "BLOCK=BLOCK_E_EB_only": (~m_imp_sched) & m_imp_eobt,
            "BLOCK=BLOCK_noimposter": valid_b & (~m_imp_sched) & (~m_imp_eobt),
            "ADEP_flt_missing": m_flt_missing,
        }
        print(f"  4 类错误指纹 + 全部 (BENIGN):")
        for label, mask_ in sig_combos.items():
            sub = df[mask_]
            if len(sub) < 5:
                continue
            sse = float(np.sum((y[mask_] - y[mask_].mean()) ** 2))
            sse_pct = sse / float(np.sum((y - y.mean()) ** 2)) * 100
            ap_cnt = sub["ADEP_mvt"].astype(str).value_counts().head(5)
            ap_str = ", ".join(f"{ap}={n}" for ap, n in ap_cnt.items())
            # 限长尾带 (>=1800) 的命中率
            n_long = int((y[mask_] > 1800).sum())
            print(f"    {label:30s} n={len(sub):>7,}  SSE={sse*1e-9:>7.2f}e9 ({sse_pct:>5.1f}%) "
                  f"long(>1800)={n_long:>6,}  top5: {ap_str}")
            cluster_results[label] = {
                "n_total": int(len(sub)),
                "n_long_>1800": n_long,
                "sse_share_pct": round(sse_pct, 2),
                "top_airports": [(ap, int(n)) for ap, n in ap_cnt.items()],
            }
        out["F_clusters"] = cluster_results

    # ====== ranking 可识别性 ======
    print("\n" + "=" * 70)
    print("[F] ranking 可识别性 (用 BLOCK==SCHED+EOBT 信号)")
    print("=" * 70)
    rank_ns = {}
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
              "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]:
        if c in rank.columns:
            rank_ns[c] = ns_int(rank[c])
    rank_y_known_na = bool("MVT_TIME_UTC_mvt" in rank.columns)  # ranking 没有真值
    # 看 ranking 字段可用
    print(f"  ranking 字段可用: {list(rank.columns)[:15]}")
    # BLOCK_TIME_UTC_mvt 在 ranking 中已知 100% NaT
    rank_blk_nat = pd.isna(rank["BLOCK_TIME_UTC_mvt"]).values if "BLOCK_TIME_UTC_mvt" in rank.columns else None
    if rank_blk_nat is not None:
        print(f"  ranking BLOCK NaT: {int(rank_blk_nat.sum()):,}/{len(rank):,}")
    # 检查 ranking 中可识别的字段
    avail_rank_cols = [c for c in rank.columns if c in [
        "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
        "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt",
        "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "FLIGHT_mvt",
        "AIRCRAFT_TYPE_mvt",
    ]]
    print(f"  ranking 中可用关键列: {avail_rank_cols}")

    # ranking 无法识别 BLOCK 顶替, 因为 BLOCK 是 NaT; 但 ADEP_flt 缺失是 ranking 可用信号
    if "ADEP_flt" in rank.columns:
        rank_adep_flt_str = rank["ADEP_flt"].astype(str).values
        rank_adep_flt_missing = (rank_adep_flt_str == "nan") | (rank_adep_flt_str == "<NA>")
        print(f"  ranking ADEP_flt 缺失: {int(rank_adep_flt_missing.sum()):,}  "
              f"({rank_adep_flt_missing.mean()*100:.2f}%)")
        # 与训练段 (m_long_to_compare) 比较
        if "ADEP_flt_missing" in cluster_results:
            print(f"  ↔ 训练段 ADEP_flt_missing n={cluster_results['ADEP_flt_missing']['n_total']:,}, "
                  f"ranking n={int(rank_adep_flt_missing.sum()):,}")

    # ====== 输出 ======
    out["summary"] = {
        "total_DEP": len(df_all := df),
        "n_train_holdout_combined": True,
        "n_ranking_DEP": len(rank),
    }
    fn = D0 / "data_quality_audit.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n  → 结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start):.0f}s")


if __name__ == "__main__":
    main()