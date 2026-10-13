"""
Step0 排队机制信号检验 — 跑道级精确排队位置 + 服务间隔
==================================================
- holdout: 2025-01+07 DEP
- 训练折 (2025-02..12) 仅用于统计 (ADEP, RUNWAY, hour) throughput -> 服务间隔 s
- k: 精确排在本航班 AOBT 之前、且同 (ADEP, RUNWAY) 的航班数, 排除已起飞 (MVT > 本航班 AOBT)
- Spearman(k*s, true - t_unimp) 与按 k 分档的均值

时间戳: 必须 .astype("datetime64[ns]") 后再差, 否则 pandas 会把 ns 扩成 us (×1000)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
import _r7_lib as L

TRAIN_COLS = [
    "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "AOBT_3_flt", "IOBT_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "AOBT_3_flt", "IOBT_flt"]


def main():
    t_start = time.time()
    print("=" * 70)
    print("[Step0 排队机制信号检验] holdout 2025-01+07, throughput 训练折 2025-02..12")
    print("=" * 70)

    # ============= 加载 =============
    print("\n[1] 加载 holdout 2025-01+07")
    ho_dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet",
                         columns=TRAIN_COLS) for m in [1, 7]]
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
    df_ho_dep = df_ho_raw[(df_ho_raw["PHASE_mvt"] == "DEP") &
                          (df_ho_raw["TAXITIME_SEC_mvt"] > 0)].copy().reset_index(drop=True)
    print(f"  holdout DEP (TX>0): {len(df_ho_dep):,}")

    # 时间戳必须 astype datetime64[ns] 后再差
    df_ho_dep = parse_time_columns(df_ho_dep, cols=TIME_COLS)
    print(f"  AOBT dtype: {df_ho_dep['AOBT_3_flt'].dtype}")
    print(f"  MVT  dtype: {df_ho_dep['MVT_TIME_UTC_mvt'].dtype}")

    # ============= Step0.1 训练折 throughput 统计 =============
    print("\n[2] 训练折 (2025-02..12) 统计 (ADEP, RUNWAY, hour) throughput")
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr_dfs = []
    for m in train_months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        tr_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
    df_tr_dep = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                          (df_tr_raw["TAXITIME_SEC_mvt"] > 0)].copy().reset_index(drop=True)
    df_tr_dep = parse_time_columns(df_tr_dep, cols=TIME_COLS)
    print(f"  train DEP: {len(df_tr_dep):,}")

    # 按 (ADEP, RUNWAY, hour) 起飞架次 (用 SCHED_TIME 作为粗小时)
    tr_dep_for_thr = df_tr_dep.assign(
        hour=df_tr_dep["SCHED_TIME_UTC_mvt"].dt.hour
    )
    thr = (tr_dep_for_thr.groupby(["ADEP_mvt", "RUNWAY_mvt", "hour"])
                          .size().rename("dep_cnt_per_hour")
                          .reset_index())
    # s = 3600 / throughput (秒/架次)
    thr["service_interval_s"] = 3600.0 / thr["dep_cnt_per_hour"]
    # 全局兜底
    global_throughput = len(tr_dep_for_thr) / (10 * 30 * 24)  # ~10 月 × 30d × 24h
    global_s = 3600.0 / global_throughput
    print(f"  训练折 (ADEP,RUNWAY,hour) 组数: {len(thr):,}")
    print(f"  throughput 分布: p10={thr['dep_cnt_per_hour'].quantile(0.1):.1f}, "
          f"p50={thr['dep_cnt_per_hour'].median():.1f}, "
          f"p90={thr['dep_cnt_per_hour'].quantile(0.9):.1f}")
    print(f"  service_interval s: p10={thr['service_interval_s'].quantile(0.1):.0f}, "
          f"p50={thr['service_interval_s'].median():.0f}, "
          f"p90={thr['service_interval_s'].quantile(0.9):.0f}  "
          f"(global_s={global_s:.0f}s)")

    # ============= Step0.2 训练折 t_unimp LUT =============
    print("\n[3] t_unimp LUT (训练折)")
    lut = L.build_t_unimp_lut(df_tr_dep)
    t_vals, _ = L.lookup_t_unimp(df_tr_dep, lut)
    df_tr_dep["t_unimp_r7"] = t_vals
    # 取 holdout 的 LUT lookup
    t_ho_vals, _ = L.lookup_t_unimp(df_ho_dep, lut)
    df_ho_dep["t_unimp_r7"] = t_ho_vals
    n_tunimp_ho = (df_ho_dep["t_unimp_r7"].notna().sum())
    print(f"  holdout t_unimp 命中率: {n_tunimp_ho:,} / {len(df_ho_dep):,} "
          f"({n_tunimp_ho/len(df_ho_dep)*100:.2f}%)")

    # ============= Step0.3 精确排队位置 k =============
    print("\n[4] 精确排队位置 k (按 ADEP+RUNWAY 内 AOBT 排序, MVT 之后排除)")
    # 关键: AOBT/MVT 都强制 datetime64[ns] 后再 int64 view
    iNaT = np.iinfo(np.int64).min
    aobt_ns = df_ho_dep["AOBT_3_flt"].dt.tz_localize(None).astype("datetime64[ns]").view("int64").values
    mvt_ns = df_ho_dep["MVT_TIME_UTC_mvt"].dt.tz_localize(None).astype("datetime64[ns]").view("int64").values
    # 单位校验
    print(f"  aobt_ns 范围: [{aobt_ns[aobt_ns!=iNaT].min() if (aobt_ns!=iNaT).any() else 0:,}, "
          f"{aobt_ns[aobt_ns!=iNaT].max() if (aobt_ns!=iNaT).any() else 0:,}]  "
          f"差值单位应是 1e9 ~ 1s")
    print(f"  mvt_ns 范围: [{mvt_ns[mvt_ns!=iNaT].min() if (mvt_ns!=iNaT).any() else 0:,}, "
          f"{mvt_ns[mvt_ns!=iNaT].max() if (mvt_ns!=iNaT).any() else 0:,}]")

    # 按 (ADEP, RUNWAY) 分组排序
    adep = df_ho_dep["ADEP_mvt"].astype(str).values
    runway = df_ho_dep["RUNWAY_mvt"].astype(str).values
    keys = pd.DataFrame({"k": list(zip(adep, runway))})
    # 仅 DEP, 已在 dep_dep 上
    n = len(df_ho_dep)
    k_arr = np.zeros(n, dtype=np.int32)

    # 按 (ADEP, RUNWAY) 分组
    grp_keys = pd.DataFrame({"ADEP_mvt": adep, "RUNWAY_mvt": runway,
                             "aobt_ns": aobt_ns, "mvt_ns": mvt_ns})
    grp_keys["orig_idx"] = np.arange(n)

    # 安全的双重循环: group by
    grouped = grp_keys.groupby(["ADEP_mvt", "RUNWAY_mvt"], sort=False, observed=True)
    n_groups = 0
    for (ap, rw), sub in grouped:
        n_groups += 1
        idx = sub["orig_idx"].values
        a = sub["aobt_ns"].values.copy()
        m = sub["mvt_ns"].values.copy()
        valid_a = a != iNaT
        if valid_a.sum() == 0:
            k_arr[idx] = 0
            continue
        # 双重循环: 对每个有效 aobt 找 (aobt_i 在 sub 中 ≤ aobt_self, 且 mvt > aobt_self)
        # O(n²) 在 holdout 上 ~344k 行, sub 大小通常 <10k, 总约 10^8 操作, 接受
        order = np.argsort(a, kind="mergesort")  # 按 aobt 升序
        a_sorted = a[order]
        m_sorted = m[order]
        # 对每个原始位置, k = arr_i - arr_self
        # 但要排除 mvt_self <= aobt_self 的: 已在起飞的, 它不应该占前面位置
        # 算法: 对每个 sorted index j, 找 a_sorted[j'] <= a_sorted[j] 且 m_sorted[j'] > a_sorted[j]
        # 简单做法: O(n²) but group size 通常 < 10k
        for jj in range(len(a_sorted)):
            ai = a_sorted[jj]
            mi = m_sorted[jj]
            if ai == iNaT:
                continue
            # 前面 (aobt <= ai) 且 (mvt > ai) 的数量
            cnt_le = np.searchsorted(a_sorted, ai, side="right")
            # 在 a_sorted[0:cnt_le] 中, mvt > ai 的个数
            m_among_le = m_sorted[:cnt_le]
            cnt = int((m_among_le > ai).sum())  # 排除自己 (自己 mvt 也 > ai)
            k_arr[idx[order[jj]]] = cnt
    print(f"  计算完: groups={n_groups}, 耗时 {time.time()-f_t:.0f}s" if False else "")
    df_ho_dep["rwy_k"] = k_arr

    # ============= Step0.4 服务间隔 s 查 holdout =============
    print("\n[5] 服务间隔 s (查训练折统计的 (ADEP,RUNWAY,hour))")
    df_ho_dep["hour"] = df_ho_dep["SCHED_TIME_UTC_mvt"].dt.hour
    ho_keys = df_ho_dep[["ADEP_mvt", "RUNWAY_mvt", "hour"]].astype(str)
    thr_idx = thr.set_index(["ADEP_mvt", "RUNWAY_mvt", "hour"])
    s_arr = np.full(n, global_s, dtype=np.float64)
    for i in range(n):
        ap = str(df_ho_dep["ADEP_mvt"].iloc[i])
        rw = str(df_ho_dep["RUNWAY_mvt"].iloc[i])
        h = int(df_ho_dep["hour"].iloc[i])
        try:
            v = thr_idx.loc[(ap, rw, h), "service_interval_s"]
            s_arr[i] = float(v)
        except KeyError:
            pass
    df_ho_dep["rwy_s"] = s_arr
    df_ho_dep["rwy_ks"] = df_ho_dep["rwy_k"] * df_ho_dep["rwy_s"]

    # ============= Step0.5 信号检验 =============
    print("\n[6] 信号检验")
    # 关注 t_unimp 命中的样本
    valid = df_ho_dep["t_unimp_r7"].notna().values
    print(f"  有效样本 (t_unimp 命中): {valid.sum():,} / {len(df_ho_dep):,}")

    y_ho = df_ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    t_unimp = df_ho_dep["t_unimp_r7"].values.astype(np.float64)
    true_minus_t = y_ho - t_unimp
    ks_vals = df_ho_dep["rwy_ks"].values.astype(np.float64)

    rho_ks, _ = spearmanr(ks_vals[valid], true_minus_t[valid])
    rho_k, _ = spearmanr(df_ho_dep["rwy_k"].values[valid],
                          true_minus_t[valid])
    print(f"  Spearman(rwy_ks, true - t_unimp) = {rho_ks:.4f}")
    print(f"  Spearman(rwy_k,  true - t_unimp) = {rho_k:.4f}")

    print(f"\n  按 k 分档 (true - t_unimp) 均值/中位 (基于有效样本):")
    k_vals = df_ho_dep["rwy_k"].values
    bins = [(-0.5, 0.5, "k=0"),
            (0.5, 1.5, "k=1"),
            (1.5, 2.5, "k=2"),
            (2.5, 3.5, "k=3"),
            (3.5, 4.5, "k=4"),
            (4.5, 100.5, "k≥5")]
    print(f"  {'档':>8s} {'n':>8s} {'mean':>10s} {'median':>10s} {'p90':>10s} {'p99':>10s}")
    for lo, hi, lbl in bins:
        mk = valid & (k_vals > lo) & (k_vals <= hi)
        if mk.sum() < 5:
            print(f"  {lbl:>8s} {mk.sum():>8d} (样本少)")
            continue
        m_t = true_minus_t[mk]
        print(f"  {lbl:>8s} {mk.sum():>8d} "
              f"{np.mean(m_t):>10.1f} {np.median(m_t):>10.1f} "
              f"{np.percentile(m_t, 90):>10.1f} {np.percentile(m_t, 99):>10.1f}")

    # rwy_ks 分布
    print(f"\n  rwy_ks 分布: min={ks_vals[valid].min():.1f}, "
          f"p50={np.median(ks_vals[valid]):.1f}, "
          f"p99={np.percentile(ks_vals[valid], 99):.1f}, "
          f"max={ks_vals[valid].max():.1f}")

    # 排名 (按 rwy_ks 分档)
    print(f"\n  按 rwy_ks 分档 (true - t_unimp) 均值 (低分档应有干净k):")
    for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        th = np.quantile(ks_vals[valid], q)
        mk = valid & (ks_vals <= th)
        print(f"  ks≤p{int(q*100):>2d}({th:>6.0f}s):  n={mk.sum():>6,}  "
              f"mean(t-t)={np.mean(true_minus_t[mk]):>+8.1f}")

    # 判断
    print(f"\n[判断]")
    print(f"  Spearman(rwy_ks, true-t_unimp) = {rho_ks:.4f}  (门槛 0.1)")
    monotonic = True
    prev_med = -np.inf
    for lo, hi, lbl in bins:
        mk = valid & (k_vals > lo) & (k_vals <= hi)
        if mk.sum() < 50:
            continue
        m_t = np.median(true_minus_t[mk])
        if m_t < prev_med - 10:
            monotonic = False
            print(f"  ! k分档单调破坏于 {lbl} (median={m_t:.0f} < prev={prev_med:.0f})")
        prev_med = m_t
    pass_step0 = (rho_ks >= 0.10) and monotonic
    print(f"  通过 Step0? {pass_step0}  (Spearman≥0.1 且 k分档单调)")
    if not pass_step0:
        print(f"  → 停止, 不进 Step1")
    else:
        print(f"  → 进入 Step1 (重训验证)")

    # JSON 落盘
    import json
    sig = {
        "task": "rwy_queue_signal_step0",
        "n_holdout_dep": int(len(df_ho_dep)),
        "n_valid_t_unimp": int(valid.sum()),
        "n_groups_ap_runway": int(n_groups),
        "global_service_interval_s": global_s,
        "spearman_ks": float(rho_ks),
        "spearman_k": float(rho_k),
        "rwy_k_distribution": {
            "k=0_n": int((k_vals == 0).sum()),
            "k=1_n": int((k_vals == 1).sum()),
            "k=2_n": int((k_vals == 2).sum()),
            "k=3_n": int((k_vals == 3).sum()),
            "k=4_n": int((k_vals == 4).sum()),
            "k≥5_n": int((k_vals >= 5).sum()),
        },
        "pass_step0": bool(pass_step0),
        "monotonic": bool(monotonic),
        "k_bin_means": {lbl: float(np.mean(true_minus_t[valid & (k_vals > lo) & (k_vals <= hi)]))
                        for lo, hi, lbl in bins if (valid & (k_vals > lo) & (k_vals <= hi)).sum() >= 5},
        "k_bin_n": {lbl: int((valid & (k_vals > lo) & (k_vals <= hi)).sum())
                    for lo, hi, lbl in bins},
        "total_time_sec": round(time.time() - t_start, 1),
    }
    fn = D0 / "rwy_queue_signal_step0.json"
    fn.write_text(json.dumps(sig, indent=2, ensure_ascii=False))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()