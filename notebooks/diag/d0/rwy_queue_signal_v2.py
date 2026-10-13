"""
Step0 排队信号检验 (v2 修正) — 跑道级精确排队
==========================================
修正 k 定义:
  t_rwy_i = AOBT_i + t_unimp_offset_corrected_i
  k_new_i = 同(ADEP,RUNWAY) 中, AOBT_j < AOBT_i 且 MVT_j > t_rwy_i 的架数

对照:
  k_old_i = 同(ADEP,RUNWAY) 中, AOBT_j < AOBT_i 且 MVT_j > AOBT_i 的架数 (上一版)
"""
from __future__ import annotations
import sys, json, time
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
    print("[Step0 v2 修正排队信号检验] holdout 2025-01+07 (剔 AOBT_NaT)")
    print("=" * 70)

    # ============= 加载 =============
    print("\n[1] 加载")
    ho_dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet",
                         columns=TRAIN_COLS) for m in [1, 7]]
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
    df_ho_dep = df_ho_raw[(df_ho_raw["PHASE_mvt"] == "DEP") &
                          (df_ho_raw["TAXITIME_SEC_mvt"] > 0)].copy().reset_index(drop=True)
    df_ho_dep = parse_time_columns(df_ho_dep, cols=TIME_COLS)
    print(f"  holdout DEP (TX>0): {len(df_ho_dep):,}")

    aobt_nat_mask = df_ho_dep["AOBT_3_flt"].isna().values
    print(f"  AOBT_NaT 行: {aobt_nat_mask.sum():,} ({aobt_nat_mask.mean()*100:.2f}%)")
    # 标注不混入: 单独存, 主分析只取 AOBT 非空
    df_ho_dep_main = df_ho_dep[~aobt_nat_mask].copy().reset_index(drop=True)
    print(f"  holdout 主样本 (AOBT 非空): {len(df_ho_dep_main):,}")

    # ============= 训练折 throughput =============
    print("\n[2] 训练折 (ADEP,RUNWAY,hour) throughput → 服务间隔 s")
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
    df_tr_dep["hour"] = df_tr_dep["SCHED_TIME_UTC_mvt"].dt.hour
    thr = (df_tr_dep.groupby(["ADEP_mvt", "RUNWAY_mvt", "hour"])
                     .size().rename("dep_cnt").reset_index())
    thr["s"] = 3600.0 / thr["dep_cnt"]
    global_throughput = len(df_tr_dep) / (10 * 30 * 24)
    global_s = 3600.0 / global_throughput
    print(f"  (ADEP,RUNWAY,hour) 组数: {len(thr):,}, global_s={global_s:.1f}s")
    thr_idx = thr.set_index(["ADEP_mvt", "RUNWAY_mvt", "hour"])["s"].to_dict()

    # ============= LUT t_unimp =============
    print("\n[3] t_unimp LUT (训练折)")
    lut = L.build_t_unimp_lut(df_tr_dep)
    t_main, _ = L.lookup_t_unimp(df_ho_dep_main, lut)
    df_ho_dep_main["t_unimp_r7"] = t_main
    n_t = (~np.isnan(t_main)).sum()
    print(f"  holdout 主样本 t_unimp 命中: {n_t:,} / {len(df_ho_dep_main):,} "
          f"({n_t/len(df_ho_dep_main)*100:.2f}%)")

    # t_unimp 缺失的行用 整体中位数 兜底, 后续不删
    t_filled = np.where(np.isnan(t_main), float(np.nanmedian(t_main)), t_main)
    df_ho_dep_main["t_unimp_filled"] = t_filled
    print(f"  t_unimp 中位: {np.nanmedian(t_main):.0f}s, "
          f"用此填充缺失 (n={int(np.isnan(t_main).sum())} 行)")

    # ============= 时间戳转 ns (timezone 处理) =============
    print("\n[4] 时间戳转 ns int64 view")
    iNaT = np.iinfo(np.int64).min
    aobt_ns = df_ho_dep_main["AOBT_3_flt"].dt.tz_localize(None).astype("datetime64[ns]").view("int64").values
    mvt_ns = df_ho_dep_main["MVT_TIME_UTC_mvt"].dt.tz_localize(None).astype("datetime64[ns]").view("int64").values
    # t_rwy = AOBT + t_unimp (秒)
    t_rwy_ns = (aobt_ns.astype(np.float64) + t_filled.astype(np.float64) * 1e9).astype(np.int64)
    print(f"  aobt range valid:  [{aobt_ns[aobt_ns!=iNaT].min():,} , "
          f"{aobt_ns[aobt_ns!=iNaT].max():,}]")
    print(f"  mvt  range valid:  [{mvt_ns[mvt_ns!=iNaT].min():,} , "
          f"{mvt_ns[mvt_ns!=iNaT].max():,}]")
    print(f"  t_rwy range valid: [{t_rwy_ns[t_rwy_ns!=iNaT].min():,} , "
          f"{t_rwy_ns[t_rwy_ns!=iNaT].max():,}]")

    # ============= 服务间隔查表 =============
    df_ho_dep_main["hour"] = df_ho_dep_main["SCHED_TIME_UTC_mvt"].dt.hour
    adep = df_ho_dep_main["ADEP_mvt"].astype(str).values
    runway = df_ho_dep_main["RUNWAY_mvt"].astype(str).values
    hour = df_ho_dep_main["hour"].values.astype(np.int64)
    s_arr = np.full(len(df_ho_dep_main), global_s, dtype=np.float64)
    for i in range(len(df_ho_dep_main)):
        v = thr_idx.get((adep[i], runway[i], int(hour[i])))
        if v is not None:
            s_arr[i] = float(v)
    df_ho_dep_main["rwy_s"] = s_arr

    # ============= 排队位置计算 (按 (ADEP, RUNWAY) 内 groupby) =============
    print("\n[5] 计算 k_old (MVT_j > AOBT_i) 与 k_new (MVT_j > t_rwy_i) - 按 (ADEP,RUNWAY) groupby")
    n = len(df_ho_dep_main)
    k_old = np.zeros(n, dtype=np.int32)
    k_new = np.zeros(n, dtype=np.int32)

    grp = pd.DataFrame({
        "ADEP_mvt": adep, "RUNWAY_mvt": runway,
        "aobt_ns": aobt_ns, "mvt_ns": mvt_ns, "t_rwy_ns": t_rwy_ns,
        "orig_idx": np.arange(n),
    }).groupby(["ADEP_mvt", "RUNWAY_mvt"], sort=False, observed=True)

    n_groups = 0
    for (ap, rw), sub in grp:
        n_groups += 1
        idx = sub["orig_idx"].values
        a = sub["aobt_ns"].values.copy()
        m = sub["mvt_ns"].values.copy()
        t_rwy = sub["t_rwy_ns"].values.copy()
        # 按 AOBT 升序排序
        order = np.argsort(a, kind="mergesort")
        a_s = a[order]
        m_s = m[order]
        t_rwy_s = t_rwy[order]
        # 对每个 sorted j: 找前面 (AOBT <= a_s[j]) 中 mvt > threshold 的个数
        # threshold_old = a_s[j]; threshold_new = t_rwy_s[j]
        n_grp = len(order)
        # 先 vectorize 计算"前面段"边界
        # cnt_le_pos[j] = 满足 AOBT <= a_s[j] 的最大下标
        cnt_le = np.searchsorted(a_s, a_s, side="right")
        for jj in range(n_grp):
            cnt_l = cnt_le[jj]
            ai = a_s[jj]
            ti = t_rwy_s[jj]
            mi = m_s[jj]
            # m_s[0:cnt_l] 是 AOBT ≤ ai 的所有前序 (按含本组)
            seg = m_s[:cnt_l]
            k_old[jj] = int((seg > ai).sum())
            k_new[jj] = int((seg > ti).sum())
        # 散回原位
        k_old[idx[order]] = k_old[:n_grp] if False else 0  # 不可用
        # 直接写
        for jj in range(n_grp):
            k_old[idx[order[jj]]] = int((m_s[:cnt_le[jj]] > a_s[jj]).sum())
            k_new[idx[order[jj]]] = int((m_s[:cnt_le[jj]] > t_rwy_s[jj]).sum())
    df_ho_dep_main["k_old"] = k_old
    df_ho_dep_main["k_new"] = k_new

    # ============= 信号检验 =============
    print("\n[6] 信号检验 (主样本: 剔 AOBT_NaT)")
    y = df_ho_dep_main["TAXITIME_SEC_mvt"].values.astype(np.float64)
    t_unimp = df_ho_dep_main["t_unimp_filled"].values.astype(np.float64)
    true_minus_t = y - t_unimp
    s = df_ho_dep_main["rwy_s"].values.astype(np.float64)
    valid = np.isfinite(true_minus_t) & np.isfinite(s) & (s > 0)

    # ============= k_old 检验 (对照) =============
    k_old_vals = df_ho_dep_main["k_old"].values
    rho_old, _ = spearmanr(k_old_vals[valid], true_minus_t[valid])
    print(f"\n  === k_old (MVT_j > AOBT_i) — 旧版对照 ===")
    print(f"  Spearman(k_old×s, true-t) = {rho_old:.4f}")
    print(f"  k_old 分布:")
    for thr_label, lo, hi in [("k=0", -0.5, 0.5), ("k=1", 0.5, 1.5), ("k=2", 1.5, 2.5),
                               ("k=3", 2.5, 3.5), ("k=4", 3.5, 4.5), ("k≥5", 4.5, 1e6)]:
        mk = valid & (k_old_vals > lo) & (k_old_vals <= hi)
        if mk.sum() < 5:
            continue
        print(f"    {thr_label}: n={mk.sum():>7,}  "
              f"mean(true-t)={np.mean(true_minus_t[mk]):>+7.1f}  "
              f"median={np.median(true_minus_t[mk]):>+7.1f}")

    # ============= k_new 检验 (新) =============
    k_new_vals = df_ho_dep_main["k_new"].values
    ks_new = k_new_vals * s
    rho_new, _ = spearmanr(ks_new[valid], true_minus_t[valid])
    rho_new_k, _ = spearmanr(k_new_vals[valid], true_minus_t[valid])
    print(f"\n  === k_new (MVT_j > t_rwy_i) — 修正版 ===")
    print(f"  Spearman(k_new×s, true-t) = {rho_new:.4f}  (门槛 ≥ +0.10)")
    print(f"  Spearman(k_new,    true-t) = {rho_new_k:.4f}")
    print(f"  k_new 分布:")
    ks_bins = [("k=0", -0.5, 0.5), ("k=1", 0.5, 1.5), ("k=2", 1.5, 2.5),
               ("k=3", 2.5, 3.5), ("k=4", 3.5, 4.5), ("k≥5", 4.5, 1e6)]
    new_means = []
    new_ns = []
    for thr_label, lo, hi in ks_bins:
        mk = valid & (k_new_vals > lo) & (k_new_vals <= hi)
        if mk.sum() < 5:
            continue
        m_t = true_minus_t[mk]
        new_means.append(float(np.mean(m_t)))
        new_ns.append(int(mk.sum()))
        print(f"    {thr_label}: n={mk.sum():>7,}  "
              f"mean(true-t)={np.mean(m_t):>+7.1f}  "
              f"median={np.median(m_t):>+7.1f}")

    # 单调检验 (从 k=0 → k≥5 mean 必须单调非降)
    monotonic = True
    for i in range(len(new_means) - 1):
        if new_means[i + 1] < new_means[i] - 5:  # 容差 5s
            print(f"  ! 单调破坏: {ks_bins[i][0]} mean={new_means[i]:+.1f} → "
                  f"{ks_bins[i+1][0]} mean={new_means[i+1]:+.1f}")
            monotonic = False

    # ============= 决策 =============
    pass_step0 = (rho_new >= 0.10) and monotonic
    print(f"\n[判断]")
    print(f"  Spearman(k_new×s) = {rho_new:.4f} ≥ 0.10 ? {rho_new >= 0.10}")
    print(f"  k_new 分档单调上升? {monotonic}")
    print(f"  通过 Step0? {pass_step0}")
    if pass_step0:
        print(f"  → 进入 Step1 重训")
    else:
        print(f"  → 排队方向彻底关闭, 不再重试")

    # 落 JSON
    sig = {
        "task": "rwy_queue_signal_step0_v2",
        "scope": "holdout 2025-01+07 DEP, AOBT 非空 (剔 AOBT_NaT)",
        "n_total_holdout_dep": int(len(df_ho_dep)),
        "n_aobt_nat_excluded": int(aobt_nat_mask.sum()),
        "n_main": int(len(df_ho_dep_main)),
        "n_groups_ap_runway": int(n_groups),
        "global_service_interval_s": global_s,
        "old_version": {
            "definition": "k_old = #{j: AOBT_j < AOBT_i ∧ MVT_j > AOBT_i, j in same(ADEP,RUNWAY)}",
            "spearman_kxS": float(rho_old),
            "k_distribution": {str(int(lo*10+1) if hi<10 else "5+"):
                                int((valid & (k_old_vals>lo) & (k_old_vals<=hi)).sum())
                                for _, lo, hi in ks_bins},
        },
        "new_version": {
            "definition": "t_rwy_i = AOBT_i + t_unimp_r7_corrected; k_new = #{j: AOBT_j < AOBT_i ∧ MVT_j > t_rwy_i, j in same(ADEP,RUNWAY)}",
            "spearman_kxS": float(rho_new),
            "spearman_k_only": float(rho_new_k),
            "k_distribution": {str(int(lo*10+1) if hi<10 else "5+"):
                                int((valid & (k_new_vals>lo) & (k_new_vals<=hi)).sum())
                                for _, lo, hi in ks_bins},
            "k_bin_mean_true_minus_t": {ks_bins[i][0]: round(new_means[i], 2)
                                        for i in range(len(new_means))},
        },
        "monotonic": bool(monotonic),
        "pass_step0": bool(pass_step0),
        "total_time_sec": round(time.time() - t_start, 1),
    }
    fn = D0 / "rwy_queue_signal_v2_step0.json"
    fn.write_text(json.dumps(sig, indent=2, ensure_ascii=False))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()