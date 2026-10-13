"""
全量数据质量审计 v2 — 修正 BLOCK 损坏判定
=========================================
修正1: BLOCK 精确等于 SCHED (≤2s) 且 MVT-SCHED > T (T=10800/14400/21600) — LIRF 取证口径
修正2: 基线断言 — LIRF BLOCK==SCHED ~20.7%, ranking LIRF NM 缺失 ~383, holdout LIRF >10800=54
修正3: 矛盾簇 (BLOCK≥MVT, AOBT>MVT) 抽 20 行报告全部时间列 + ranking 统计
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

ALL_TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "ARVT_3_flt", "ARVT_1_flt"]
MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                   if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
              for m in range(1, 13)}

iNaT_NS = np.iinfo(np.int64).min


def ns_int(series):
    """转 ns 整数. 原始数据是 datetime64[us, UTC] (微秒精度),
    view('int64') 返回的是 μs 数, 必须 × 1e3 转纳秒, 否则是 1970 时间戳"""
    if pd.api.types.is_datetime64_any_dtype(series):
        s2 = series
        if s2.dt.tz is not None:
            s2 = s2.dt.tz_localize(None)
        # 如果是 datetime64[us], 内部 view 是 μs, 必须 ×1e3
        v = s2.values.view("int64")
        if s2.dtype == "datetime64[us]":
            v = v * 1000
        return v
    if pd.api.types.is_integer_dtype(series):
        return series.values
    return pd.to_datetime(series, utc=True, errors="coerce").values.view("int64")


def load_full_dep():
    print("  加载 2025 1-12 月 DEP...")
    dfs = []
    for m in ALL_TRAIN_MONTHS:
        df = pd.read_parquet(MONTH_FILE[m], columns=[
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
    return df


def load_holdout_dep():
    """只 1+7 月, 用来报告 LIRF >10800 = 54 等锚点"""
    dfs = []
    for m in [1, 7]:
        df = pd.read_parquet(MONTH_FILE[m], columns=[
            "MVT_ID_mvt", "PHASE_mvt",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt",
            "ADEP_mvt", "TAXITIME_SEC_mvt",
        ])
        df = df[df["PHASE_mvt"] == "DEP"].copy()
        df = parse_time_columns(df, cols=[c for c in TIME_COLS if c in df.columns])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    t0 = time.time()
    out = {}

    df = load_full_dep()
    rank = load_rank_dep()
    ho = load_holdout_dep()

    time_ns = {}
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
              "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "ARVT_3_flt", "ARVT_1_flt"]:
        if c in df.columns:
            time_ns[c] = ns_int(df[c])
    y = df["TAXITIME_SEC_mvt"].values.astype("float64")
    ap = df["ADEP_mvt"].astype(str).values

    mvt_ns = time_ns["MVT_TIME_UTC_mvt"]
    blk_ns = time_ns["BLOCK_TIME_UTC_mvt"]
    sched_ns = time_ns["SCHED_TIME_UTC_mvt"]
    aobt_ns = time_ns["AOBT_3_flt"]
    valid_mvt = mvt_ns != iNaT_NS
    valid_blk = blk_ns != iNaT_NS
    valid_sched = sched_ns != iNaT_NS
    valid_aobt = aobt_ns != iNaT_NS

    # =========================================================
    # 修正 2: 基线断言 — 先验证脚本读取正确
    # =========================================================
    print("\n" + "=" * 70)
    print("[基线断言 — 必须全部命中]")
    print("=" * 70)

    # LIRF 全12月 exact BLOCK==SCHED (≤2s) 比例
    valid_pair_b_s = valid_blk & valid_sched
    diff_blk_sched = np.abs(blk_ns - sched_ns)
    m_exact_eq = valid_pair_b_s & (diff_blk_sched <= 2 * int(1e9))
    is_lirf = (ap == "LIRF")
    n_lirf = int(is_lirf.sum())
    n_lirf_eq = int((is_lirf & m_exact_eq).sum())
    pct_lirf_eq = round(n_lirf_eq / n_lirf * 100, 2) if n_lirf else 0.0
    print(f"  LIRF 全12月: n={n_lirf:,}")
    print(f"  LIRF exact BLOCK==SCHED (≤2s): {n_lirf_eq:,}  ({pct_lirf_eq}%)  ← 期望 ≈20.7%")
    if abs(pct_lirf_eq - 20.7) > 1.5:
        print(f"  ⚠ 基线 LIRF 偏差 {pct_lirf_eq - 20.7:+.2f}%, 排查读取/join bug")

    # ranking LIRF NM 缺失 (ADEP_flt 空 或 AOBT 哨兵)
    rank_ap = rank["ADEP_mvt"].astype(str).values
    rank_adep_flt = rank["ADEP_flt"].astype(str).values
    rank_aobt_ns = ns_int(rank["AOBT_3_flt"])
    rank_mvt_ns = ns_int(rank["MVT_TIME_UTC_mvt"])
    rank_sched_ns = ns_int(rank["SCHED_TIME_UTC_mvt"])
    rank_y_check = y[is_lirf][:5]  # placeholder
    is_lirf_rank = (rank_ap == "LIRF")
    # ADEP_flt 空 (字面 nan/空)
    adep_flt_missing = (rank_adep_flt == "nan") | (rank_adep_flt == "<NA>")
    # AOBT 哨兵 (NaT)
    aobt_nat = (rank_aobt_ns == iNaT_NS)
    # NM = 至少一个不可用 (空或哨兵)
    nm_lirf = is_lirf_rank & (adep_flt_missing | aobt_nat)
    n_lirf_rank = int(is_lirf_rank.sum())
    n_lirf_nm = int(nm_lirf.sum())
    print(f"  ranking LIRF: n={n_lirf_rank:,}")
    print(f"  ranking LIRF NM (ADEP_flt空 OR AOBT哨兵): {n_lirf_nm:,}  ← 期望 ≈383")

    # holdout LIRF >10800
    ho_y = ho["TAXITIME_SEC_mvt"].values.astype("float64")
    ho_ap = ho["ADEP_mvt"].astype(str).values
    n_lirf_gt10k = int(((ho_ap == "LIRF") & (ho_y > 10800)).sum())
    print(f"  holdout (1+7) LIRF >10800: {n_lirf_gt10k}  ← 期望 54")

    # ranking BLOCK 100% NaT
    n_blk_nat = int(pd.isna(rank["BLOCK_TIME_UTC_mvt"]).sum())
    print(f"  ranking BLOCK NaT: {n_blk_nat}/{len(rank)}  ← 期望 100%")

    # =========================================================
    # 修正 1: BLOCK 损坏判定 — exact BLOCK==SCHED (≤2s), 按 MVT-SCHED > T 划分
    # =========================================================
    print("\n" + "=" * 70)
    print("[修正1] BLOCK 损坏判定: exact BLOCK==SCHED (≤2s)")
    print("=" * 70)

    # 全部机场的 exact BLOCK==SCHED 比例
    m_exact = m_exact_eq
    print(f"\n  全量 DEP exact BLOCK==SCHED (≤2s): {int(m_exact.sum()):,} / {len(df):,}  "
          f"({m_exact.mean()*100:.2f}%)")
    # 逐机场
    ap_exact = {}
    ap_total = {}
    for ap_name in np.unique(ap):
        arr_mask = ap == ap_name
        if arr_mask.sum() < 100:
            continue
        ap_exact[str(ap_name)] = int((arr_mask & m_exact).sum())
        ap_total[str(ap_name)] = int(arr_mask.sum())
    rows = []
    for ap_name in sorted(ap_total.keys(), key=lambda a: -ap_exact[a]/max(ap_total[a],1)):
        n_e, n_t = ap_exact[ap_name], ap_total[ap_name]
        rows.append((ap_name, n_e, n_t, round(n_e/n_t*100, 2)))
    print(f"\n  {'机场':6s} {'n_exact_eq':>10s} {'n_total':>10s} {'%':>8s}")
    for ap_name, n_e, n_t, pct in rows:
        print(f"  {ap_name:6s} {n_e:>10,d} {n_t:>10,d} {pct:>7.2f}%")

    # MVT-SCHED > T 阈值下的 LIRF 占大头
    diff_mvt_sched = (mvt_ns - sched_ns).astype("float64") / 1e9  # seconds
    for T in [10800, 14400, 21600]:
        m_exact_long = m_exact & (diff_mvt_sched > T)
        n = int(m_exact_long.sum())
        # 其中 LIRF
        n_lirf_T = int((is_lirf & m_exact_long).sum())
        # 全部机场 top5
        cnt = pd.Series(ap[m_exact_long]).value_counts().head(5)
        cnt_str = "; ".join(f"{a}={c}" for a, c in cnt.items())
        print(f"\n  T={T}s: n_exact_eq & MVT-SCHED>{T}s = {n:,}  (LIRF={n_lirf_T}, {n_lirf_T*100/n if n else 0:.1f}%)  top5: {cnt_str}")

    # =========================================================
    # 修正 3: 矛盾簇取证 — 抽 20 行全部时间列 + ranking 统计
    # =========================================================
    print("\n" + "=" * 70)
    print("[修正3] 矛盾簇取证: BLOCK≥MVT, AOBT>MVT")
    print("=" * 70)

    # 簇 1: BLOCK ≥ MVT (353 行)
    m_inv = valid_blk & valid_mvt & (blk_ns >= mvt_ns)
    sse_inv = float(np.sum((y[m_inv] - y[m_inv].mean())**2))
    sse_total = float(np.sum((y - y.mean())**2))
    print(f"\n  簇 1: BLOCK ≥ MVT: {int(m_inv.sum()):,}")
    print(f"    SSE = {sse_inv:.2f}  ({sse_inv/sse_total*100:.3f}% of total)")
    print(f"    TAXITIME p50={float(np.median(y[m_inv])):.0f}, "
          f"p90={float(np.percentile(y[m_inv], 90)):.0f}")

    # 抽 20 行取证
    sample_idx = np.where(m_inv)[0][:20]
    print(f"\n  簇 1 前20行取证 (MVT ID, ADEP, TAXI, 全部时间戳):")
    print(f"  {'MVT_ID':>10s}  {'ADEP':6s}  {'TAXI':>6s}  "
          f"{'MVT':>19s}  {'BLOCK':>19s}  {'SCHED':>19s}  {'AOBT':>19s}  {'EOBT':>19s}")
    for i in sample_idx:
        def fmt(ns):
            if ns == iNaT_NS: return "NaT"
            ts = pd.to_datetime(ns, unit="ns")
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {df['MVT_ID_mvt'].iloc[i]:>10}  {ap[i]:6s}  {int(y[i]):>6d}  "
              f"{fmt(mvt_ns[i]):>19s}  {fmt(blk_ns[i]):>19s}  {fmt(sched_ns[i]):>19s}  "
              f"{fmt(aobt_ns[i]):>19s}  {fmt(time_ns['EOBT_1_flt'][i]):>19s}")

    # 簇 2: AOBT > MVT (834 行)
    m_aobt_inv = valid_aobt & valid_mvt & (aobt_ns > mvt_ns)
    sse_aobt_inv = float(np.sum((y[m_aobt_inv] - y[m_aobt_inv].mean())**2))
    print(f"\n  簇 2: AOBT > MVT: {int(m_aobt_inv.sum()):,}")
    print(f"    SSE = {sse_aobt_inv:.2f}  ({sse_aobt_inv/sse_total*100:.3f}% of total)")
    print(f"    TAXITIME p50={float(np.median(y[m_aobt_inv])):.0f}, "
          f"p90={float(np.percentile(y[m_aobt_inv], 90)):.0f}")

    sample_idx2 = np.where(m_aobt_inv)[0][:20]
    print(f"\n  簇 2 前20行取证:")
    print(f"  {'MVT_ID':>10s}  {'ADEP':6s}  {'TAXI':>6s}  "
          f"{'MVT':>19s}  {'BLOCK':>19s}  {'SCHED':>19s}  {'AOBT':>19s}  {'EOBT':>19s}")
    for i in sample_idx2:
        def fmt(ns):
            if ns == iNaT_NS: return "NaT"
            ts = pd.to_datetime(ns, unit="ns")
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {df['MVT_ID_mvt'].iloc[i]:>10}  {ap[i]:6s}  {int(y[i]):>6d}  "
              f"{fmt(mvt_ns[i]):>19s}  {fmt(blk_ns[i]):>19s}  {fmt(sched_ns[i]):>19s}  "
              f"{fmt(aobt_ns[i]):>19s}  {fmt(time_ns['EOBT_1_flt'][i]):>19s}")

    # ranking 上的同类矛盾 (用 AOBT/MVT 可见)
    print("\n  --- ranking 同类矛盾 (AOBT/MVT 可见) ---")
    rank_aobt = rank_aobt_ns
    rank_mvt = rank_mvt_ns
    rank_sched = rank_sched_ns
    rank_ap_v = rank_ap
    valid_rk_aobt = rank_aobt != iNaT_NS
    valid_rk_mvt = rank_mvt != iNaT_NS
    valid_rk_sched = rank_sched != iNaT_NS
    m_rk_aobt_inv = valid_rk_aobt & valid_rk_mvt & (rank_aobt > rank_mvt)
    m_rk_aobt_inv_lirf = m_rk_aobt_inv & (rank_ap_v == "LIRF")
    print(f"  ranking AOBT > MVT: {int(m_rk_aobt_inv.sum()):,}  (LIRF: {int(m_rk_aobt_inv_lirf.sum())})")

    # ranking 上 ADEP_flt 缺失 或 AOBT 哨兵
    m_rk_nm = adep_flt_missing | aobt_nat
    n_m_rk_nm = int(m_rk_nm.sum())
    print(f"  ranking ADEP_flt空 或 AOBT哨兵 (NM): {n_m_rk_nm:,}")
    print(f"  ranking LIRF NM: {int((m_rk_nm & is_lirf_rank).sum())}")

    # ranking NM 上可比性
    rank_summable = rank[~m_rk_nm]
    print(f"  ranking 可用行 (无NM): {len(rank_summable):,}")

    # =========================================================
    # 汇总
    # =========================================================
    out["baselines"] = {
        "LIRF_BLOCK_eq_SCHED_pct": pct_lirf_eq,
        "LIRF_BLOCK_eq_SCHED_n": n_lirf_eq,
        "LIRF_total": n_lirf,
        "ranking_LIRF_NM_n": n_lirf_nm,
        "holdout_LIRF_gt10800": n_lirf_gt10k,
        "ranking_BLOCK_NaT_total": n_blk_nat,
        "ranking_total_RANK": len(rank),
    }
    out["exact_block_eq_sched_per_airport"] = [
        {"ap": ap, "n_exact": n_e, "n_total": n_t, "pct": pct}
        for ap, n_e, n_t, pct in rows
    ]
    out["block_eq_sched_in_long_window"] = {}
    for T in [10800, 14400, 21600]:
        m_exact_long = m_exact & (diff_mvt_sched > T)
        n = int(m_exact_long.sum())
        cnt = pd.Series(ap[m_exact_long]).value_counts().head(5).to_dict()
        out["block_eq_sched_in_long_window"][f"T_{T}s"] = {
            "n": n, "LIRF_n": int((is_lirf & m_exact_long).sum()),
            "top5": {k: int(v) for k, v in cnt.items()},
        }
    out["contradiction_clusters"] = {
        "BLOCK_ge_MVT": {
            "n": int(m_inv.sum()),
            "sse_share_pct": round(sse_inv / sse_total * 100, 4),
            "taxi_p50": round(float(np.median(y[m_inv])), 2),
            "taxi_p90": round(float(np.percentile(y[m_inv], 90)), 2),
            "sample_20": [
                    {"MVT_ID": int(df["MVT_ID_mvt"].iloc[i]),
                     "ADEP": ap[i],
                     "TAXI": int(y[i]),
                     "MVT": str(pd.to_datetime(mvt_ns[i], unit="ns")),
                     "BLOCK": str(pd.to_datetime(blk_ns[i], unit="ns")),
                     "SCHED": str(pd.to_datetime(sched_ns[i], unit="ns")),
                     "AOBT": str(pd.to_datetime(aobt_ns[i], unit="ns")) if aobt_ns[i] != iNaT_NS else "NaT",
                     "EOBT": str(pd.to_datetime(time_ns["EOBT_1_flt"][i], unit="ns")) if time_ns["EOBT_1_flt"][i] != iNaT_NS else "NaT"}
                    for i in sample_idx
                ],
        },
        "AOBT_gt_MVT": {
            "n": int(m_aobt_inv.sum()),
            "sse_share_pct": round(sse_aobt_inv / sse_total * 100, 4),
            "taxi_p50": round(float(np.median(y[m_aobt_inv])), 2),
            "taxi_p90": round(float(np.percentile(y[m_aobt_inv], 90)), 2),
            "sample_20": [
                    {"MVT_ID": int(df["MVT_ID_mvt"].iloc[i]),
                     "ADEP": ap[i],
                     "TAXI": int(y[i]),
                     "MVT": str(pd.to_datetime(mvt_ns[i], unit="ns")),
                     "BLOCK": str(pd.to_datetime(blk_ns[i], unit="ns")),
                     "SCHED": str(pd.to_datetime(sched_ns[i], unit="ns")),
                     "AOBT": str(pd.to_datetime(aobt_ns[i], unit="ns")) if aobt_ns[i] != iNaT_NS else "NaT",
                     "EOBT": str(pd.to_datetime(time_ns["EOBT_1_flt"][i], unit="ns")) if time_ns["EOBT_1_flt"][i] != iNaT_NS else "NaT"}
                    for i in sample_idx2
                ],
        },
    }
    out["ranking_recognizability"] = {
        "AOBT_gt_MVT_in_ranking": int(m_rk_aobt_inv.sum()),
        "AOBT_gt_MVT_in_ranking_LIRF": int(m_rk_aobt_inv_lirf.sum()),
        "NM_in_ranking_total": n_m_rk_nm,
        "NM_in_ranking_LIRF": int((m_rk_nm & is_lirf_rank).sum()),
        "ranking_usable_rows": int(len(rank_summable)),
    }

    fn = D0 / "data_quality_audit_v2.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n  → 结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()