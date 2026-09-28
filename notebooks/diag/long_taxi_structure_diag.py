"""
长滑行结构独立诊断
==================
第二信源独立复算,不采信任何已有结论。只分析、不训练、不上传。

产出:
1. SSE 分档表 + 敏感性(任务1)
2. 长滑行 AOBT 打标:真实/疑似脏值/不确定(任务2)
3. 分机场 × 分档 分布(任务3)
4. 可预测信号强弱排序(任务4)
5. 抽样表(任务5)

数据约束(环境实际):
- holdout_v2_preds.parquet: 344k 行全量 holdout(2025-01+07 DEP),有 true_y + 4 模型预测,
  无原始时间戳 → 任务1 用"均值基准"(环境无 LGB 模型文件,用户规则允许)
- data/sample/training_*.parquet: 12×700 行小样本(每机场每月 40 DEP+30 ARR),
  有全部原始时间戳 → 任务2-5 用标准 holdout 月(1月+7月)DEP 子集 ~800 行
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features_queue import add_queue_features  # noqa: E402

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = ROOT / "notebooks" / "diag"
OUT_JSON = OUT_DIR / "long_taxi_structure_diag.json"
OUT_MD = OUT_DIR / "long_taxi_structure_diag.md"

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _to_dt(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def _rmse(err):
    return float(np.sqrt(np.mean(np.asarray(err, dtype="float64") ** 2)))


# ---------------------------------------------------------------------------
# 任务1: SSE 分档 + 敏感性(全量 holdout 344k,均值基准)
# ---------------------------------------------------------------------------

def task1_sse_bins():
    print("=== 任务1: SSE 分档 + 敏感性(均值基准) ===")
    preds = pd.read_parquet(ROOT / "notebooks" / "diag" / "holdout_v2_preds.parquet")
    assert (preds["PHASE_mvt"] == "DEP").all()
    y = preds["true_y"].to_numpy(dtype="float64")

    # 基准: 用训练期均值。环境无独立训练均值,用 holdout 均值会有 in-sample 偏差。
    # 选方案: 用每个机场在 holdout 外的均值不可得 → 报告两种:
    #  (a) 全局均值(含轻微乐观偏差,标注)
    #  (b) 分机场均值(更强基准)
    preds["ADEP_mvt"] = preds["ADEP_mvt"].astype(str)
    global_mean = float(y.mean())
    ap_means = preds.groupby("ADEP_mvt")["true_y"].transform("mean").to_numpy()

    results = {"global_mean": global_mean, "note": (
        "环境无 LGB 模型文件,按用户规则用均值基准粗算。"
        "baseline_global=全 holdout DEP 均值(轻微乐观);"
        "baseline_ap=分机场均值(更接近常用生产基准)。"
    )}

    bin_edges = [0, 1800, 3600, 7200, 10800, np.inf]
    bin_labels = ["<=1800", "(1800,3600]", "(3600,7200]", "(7200,10800]", ">10800"]
    bin_idx = np.clip(np.digitize(y, bin_edges[1:-1], right=True), 0, 4)

    for name, pred in [("baseline_global", np.full_like(y, global_mean)),
                        ("baseline_ap", ap_means)]:
        err = y - pred
        sse_total = float(np.sum(err ** 2))
        rows = []
        for i, lab in enumerate(bin_labels):
            m = bin_idx == i
            n = int(m.sum())
            sse_bin = float(np.sum(err[m] ** 2)) if n else 0.0
            rows.append({
                "bin": lab,
                "n": n,
                "pct_rows": n / len(y) * 100,
                "y_mean": float(y[m].mean()) if n else None,
                "y_median": float(np.median(y[m])) if n else None,
                "rmse_bin": _rmse(err[m]) if n else None,
                "sse": sse_bin,
                "pct_sse": sse_bin / sse_total * 100 if sse_total else 0.0,
            })
        # 敏感性: 长滑行(>1800)档 RMSE 降到 target 时,整盘 RMSE
        long_mask = bin_idx >= 1  # bins 1..4
        sse_short = float(np.sum(err[~long_mask] ** 2))
        n_short = int((~long_mask).sum())
        n_long = int(long_mask.sum())
        sensitivity = []
        for tgt in [500.0, 300.0, 0.0]:
            sse_new = sse_short + n_long * tgt ** 2
            rmse_new = float(np.sqrt(sse_new / len(y)))
            sensitivity.append({
                "long_bin_target_rmse": tgt,
                "overall_rmse": rmse_new,
                "delta_vs_current": rmse_new - _rmse(err),
            })
        results[name] = {
            "overall_rmse": _rmse(err),
            "overall_mae": float(np.mean(np.abs(err))),
            "sse_total": sse_total,
            "bins": rows,
            "sensitivity": sensitivity,
            "n_long(>1800)": n_long,
            "pct_long(>1800)": n_long / len(y) * 100,
        }
        print(f"  {name}: overall RMSE={results[name]['overall_rmse']:.2f}, n>1800={n_long} ({n_long/len(y)*100:.2f}%)")
        for r in rows:
            print(f"    {r['bin']:>14s} n={r['n']:>6d} ({r['pct_rows']:5.2f}%) rmse={r['rmse_bin'] if r['rmse_bin'] else 0:7.1f} sse%={r['pct_sse']:5.2f}")
        for s in sensitivity:
            print(f"    long→{s['long_bin_target_rmse']:.0f}: overall RMSE={s['overall_rmse']:.2f} (Δ{s['delta_vs_current']:+.2f})")
    return results


# ---------------------------------------------------------------------------
# 任务2-5: 小样本原始时间戳分析(2025-01 + 2025-07 DEP)
# ---------------------------------------------------------------------------

def load_all_sample_dep():
    """加载 sample 全年 DEP 行 + 全部原始时间戳(用于任务4 全年 Spearman)。"""
    dfs = []
    for f in sorted((ROOT / "data" / "sample").glob("training_*.parquet")):
        d = pd.read_parquet(f)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
               "AOBT_3_flt", "ARVT_3_flt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt"]:
        df[c] = _to_dt(df[c])
    return df[df["PHASE_mvt"] == "DEP"].copy()


def load_holdout_sample():
    """加载 sample 中标准 holdout 月(1、7月)的 DEP 行。"""
    dep = load_all_sample_dep()
    sched_month = dep["SCHED_TIME_UTC_mvt"].dt.month
    return dep[sched_month.isin([1, 7])].copy()


def task2_classify(dep: pd.DataFrame):
    """长滑行(>1800)AOBT 打标。"""
    print("\n=== 任务2: 长滑行 AOBT 打标 ===")
    long = dep[dep["TAXITIME_SEC_mvt"] > 1800].copy()
    print(f"  长滑行样本数: {len(long)}")

    mvt = long["MVT_TIME_UTC_mvt"]
    blk = long["BLOCK_TIME_UTC_mvt"]
    sch = long["SCHED_TIME_UTC_mvt"]
    aobt = long["AOBT_3_flt"]

    long["aobt_missing"] = aobt.isna()
    long["abs_aobt_minus_blk_sec"] = (aobt - blk).dt.total_seconds().abs()
    long["mvt_minus_aobt_sec"] = (mvt - aobt).dt.total_seconds()
    long["abs_blk_minus_sched_sec"] = (blk - sch).dt.total_seconds().abs()
    long["mvt_minus_sched_sec"] = (mvt - sch).dt.total_seconds()

    # 规则
    cond_real = (
        (~long["aobt_missing"])
        & (long["abs_aobt_minus_blk_sec"] <= 300)
        & (long["mvt_minus_aobt_sec"] > 1800)
    )
    cond_dirty = (
        long["aobt_missing"]
        | (long["abs_aobt_minus_blk_sec"] > 1800)
        | ((long["abs_blk_minus_sched_sec"] <= 60) & (long["mvt_minus_sched_sec"] > 2700))
    )
    # 优先 dirty 判定(更安全),再 real,其余 uncertain
    long["label"] = "uncertain"
    long.loc[cond_real, "label"] = "real"
    long.loc[cond_dirty, "label"] = "dirty"

    # 跨天 ±86400 与 MVT 日期差 > 30d 命中数
    cross_day_hits = int((long["abs_aobt_minus_blk_sec"].sub(86400).abs() < 60).sum())
    long["mvt_blk_days"] = (mvt.dt.normalize() - blk.dt.normalize()).dt.days
    long["mvt_aobt_days"] = (mvt.dt.normalize() - aobt.dt.normalize()).dt.days
    date_diff_30d = int(((long["mvt_blk_days"].abs() > 30) | (long["mvt_aobt_days"].abs() > 30)).sum())

    # uncertain 细分原因
    u = long[long["label"] == "uncertain"]
    long["uncertain_reason"] = ""
    mask_a = u["aobt_missing"]  # 不会发生(dirty 已拦截),防御
    mask_b = (~u["aobt_missing"]) & (u["abs_aobt_minus_blk_sec"] > 300) & (u["abs_aobt_minus_blk_sec"] <= 1800)
    mask_c = (~u["aobt_missing"]) & (u["abs_aobt_minus_blk_sec"] <= 300) & (u["mvt_minus_aobt_sec"] <= 1800)
    long.loc[u.index[mask_b], "uncertain_reason"] = "aobt_blk_diff_300_1800"
    long.loc[u.index[mask_c], "uncertain_reason"] = "aobt_ok_but_mvt_aobt_le_1800"
    long.loc[u.index[mask_a], "uncertain_reason"] = "aobt_missing_fallback"
    unc_reason = long.loc[long["label"] == "uncertain", "uncertain_reason"].value_counts().to_dict()

    counts = long["label"].value_counts().to_dict()
    print(f"  分布: {counts}, uncertain细分={unc_reason}, 跨天±86400命中={cross_day_hits}, 日期差>30d命中={date_diff_30d}")

    summary = {
        "n_long": int(len(long)),
        "counts": {k: int(v) for k, v in counts.items()},
        "pct": {k: float(v / len(long) * 100) for k, v in counts.items()},
        "uncertain_reasons": {k: int(v) for k, v in unc_reason.items()},
        "cross_day_86400_hits": cross_day_hits,
        "date_diff_gt_30d_hits": date_diff_30d,
        "rules": {
            "real": "AOBT有效 & |AOBT-BLOCK|<=300s & MVT-AOBT>1800",
            "dirty": "AOBT缺失 | |AOBT-BLOCK|>1800s | (|BLOCK-SCHED|<=60s & MVT-SCHED>2700s)",
            "uncertain": "其余(细分: aobt_blk_diff_300_1800 / aobt_ok_but_mvt_aobt_le_1800)",
        },
    }
    return long, summary


def task3_by_airport(long: pd.DataFrame):
    """分机场 × 分三档 真实/脏值/不确定 占比 + TAXITIME 中位/p99/max。"""
    print("\n=== 任务3: 分机场 × 分档 ===")
    def bucket(t):
        if t <= 3600:
            return "1800-3600"
        if t <= 10800:
            return "3600-10800"
        return ">10800"

    long = long.copy()
    long["bucket"] = long["TAXITIME_SEC_mvt"].map(bucket)
    out = {}
    for ap, g in long.groupby(long["ADEP_mvt"].astype(str)):
        ap_d = {}
        for b in ["1800-3600", "3600-10800", ">10800"]:
            gb = g[g["bucket"] == b]
            if len(gb) == 0:
                ap_d[b] = {"n": 0}
                continue
            lab_counts = gb["label"].value_counts().to_dict()
            ap_d[b] = {
                "n": int(len(gb)),
                "real": int(lab_counts.get("real", 0)),
                "dirty": int(lab_counts.get("dirty", 0)),
                "uncertain": int(lab_counts.get("uncertain", 0)),
                "pct_real": float(lab_counts.get("real", 0) / len(gb) * 100),
                "pct_dirty": float(lab_counts.get("dirty", 0) / len(gb) * 100),
                "pct_uncertain": float(lab_counts.get("uncertain", 0) / len(gb) * 100),
                "taxitime_median": float(gb["TAXITIME_SEC_mvt"].median()),
                "taxitime_p99": float(gb["TAXITIME_SEC_mvt"].quantile(0.99)),
                "taxitime_max": float(gb["TAXITIME_SEC_mvt"].max()),
            }
        out[ap] = ap_d
        print(f"  {ap}: 1800-3600 n={ap_d['1800-3600'].get('n',0)}, "
              f"3600-10800 n={ap_d['3600-10800'].get('n',0)}, >10800 n={ap_d['>10800'].get('n',0)}")
    return out


def _compute_signals(dep: pd.DataFrame) -> pd.DataFrame:
    """对给定 DEP 帧计算全部信号(队列 + 计划 + 距离)。"""
    dep = dep.copy()
    dep = add_queue_features(dep, include_a=True, include_b=True, include_c=True,
                              include_d=True, include_e=False,
                              windows_dep_mvt=(5, 10, 15, 30, 60))
    dep["sched_hour"] = dep["SCHED_TIME_UTC_mvt"].dt.hour
    dep["eobt_minus_iobt_sec"] = (dep["EOBT_1_flt"] - dep["IOBT_flt"]).dt.total_seconds()
    dep["sched_minus_iobt_sec"] = (dep["SCHED_TIME_UTC_mvt"] - dep["IOBT_flt"]).dt.total_seconds()
    dep["sched_minus_eobt_sec"] = (dep["SCHED_TIME_UTC_mvt"] - dep["EOBT_1_flt"]).dt.total_seconds()
    dep["lobt_minus_iobt_sec"] = (dep["LOBT_flt"] - dep["IOBT_flt"]).dt.total_seconds()
    dep["ap_hour"] = dep["ADEP_mvt"].astype(str) + "_" + dep["SCHED_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d-%H")
    dep["ap_hour_dep_cnt"] = dep.groupby("ap_hour")["MVT_ID_mvt"].transform("count")
    dep["ap_hour_rwy_nunique"] = dep.groupby("ap_hour")["RUNWAY_mvt"].transform("nunique")
    try:
        dist = pd.read_csv(ROOT / "data" / "external" / "xplane" / "_reports"
                            / "stand_rwy_distance_2025-01-01_2025-02-01.csv")
        dist = (dist[["MVT_ID_mvt", "stand_to_rwy_m"]]
                .drop_duplicates(subset=["MVT_ID_mvt"])
                .rename(columns={"stand_to_rwy_m": "stand_rwy_dist_m"}))
        dep = dep.merge(dist, on="MVT_ID_mvt", how="left", validate="m:1")
        print(f"  stand-rwy dist matched {int(dep['stand_rwy_dist_m'].notna().sum())}/{len(dep)}")
    except Exception as e:
        print(f"  stand-rwy distance unavailable: {e}")
        dep["stand_rwy_dist_m"] = np.nan
    return dep


def task4_signals(dep: pd.DataFrame, long: pd.DataFrame, dep_allyear: pd.DataFrame):
    """对'真实'长滑行,与普通航班(<=1800)比较信号分位差异,判定强弱。

    证据两级:
    (a) holdout 样本上 real vs normal 的分位差 + Cohen's d(主,但 n_real 小,功效低)
    (b) 全年 sample(4800 DEP)上 信号 × TAXITIME 的 Spearman ρ(辅,功效更高)
    """
    print("\n=== 任务4: 可预测信号强弱 ===")
    dep = _compute_signals(dep)
    dep_allyear = _compute_signals(dep_allyear)

    # 真实长滑行 & 普通组(普通=非长滑行;长滑行 38 行全排除,避免脏值污染对照)
    real_long_ids = set(long.loc[long["label"] == "real", "MVT_ID_mvt"])
    dep["is_real_long"] = dep["MVT_ID_mvt"].isin(real_long_ids)
    dep["is_normal"] = dep["TAXITIME_SEC_mvt"] <= 1800
    dep["ADEP_mvt"] = dep["ADEP_mvt"].astype(str)

    sig_defs = {
        # 信号名: (列, 说明)
        "dep_mvt_cnt_runway_15m": ("dep_mvt_cnt_runway_15m", "同跑道过去15min实际起飞架次"),
        "dep_mvt_cnt_runway_60m": ("dep_mvt_cnt_runway_60m", "同跑道过去60min实际起飞架次"),
        "dep_mvt_cnt_airport_15m": ("dep_mvt_cnt_airport_15m", "同机场过去15min实际起飞架次"),
        "dep_mvt_cnt_airport_60m": ("dep_mvt_cnt_airport_60m", "同机场过去60min实际起飞架次"),
        "dep_queue_n_runway": ("dep_queue_n_runway", "同跑道瞬时队列长度"),
        "dep_queue_n_airport": ("dep_queue_n_airport", "同机场瞬时队列长度"),
        "dep_queue_heavy_n_airport": ("dep_queue_heavy_n_airport", "同机场重型机队列"),
        "ap_hour_dep_cnt": ("ap_hour_dep_cnt", "机场×小时计划拥堵(计划DEP数)"),
        "ap_hour_rwy_nunique": ("ap_hour_rwy_nunique", "机场×小时活跃跑道数(计划)"),
        "stand_rwy_dist_m": ("stand_rwy_dist_m", "stand到跑道haversine距离"),
        "eobt_minus_iobt_sec": ("eobt_minus_iobt_sec", "EOBT相对IOBT后移(流量调控代理)"),
        "sched_minus_iobt_sec": ("sched_minus_iobt_sec", "SCHED相对IOBT后移"),
        "sched_minus_eobt_sec": ("sched_minus_eobt_sec", "SCHED相对EOBT后移"),
        "lobt_minus_iobt_sec": ("lobt_minus_iobt_sec", "LOBT相对IOBT后移"),
        "sched_hour": ("sched_hour", "计划小时(0-23)"),
    }

    real = dep[dep["is_real_long"]].copy()
    norm = dep[dep["is_normal"]].copy()
    print(f"  real_long n={len(real)}, normal n={len(norm)}")

    rows = []
    for name, (col, desc) in sig_defs.items():
        if col not in dep.columns:
            rows.append({"signal": name, "desc": desc, "verdict": "unavailable"})
            continue
        rv = real[col].dropna().to_numpy(dtype="float64")
        nv = norm[col].dropna().to_numpy(dtype="float64")
        if len(rv) < 5 or len(nv) < 5:
            rows.append({"signal": name, "desc": desc, "verdict": "insufficient_data",
                          "n_real": int(len(rv)), "n_norm": int(len(nv))})
            continue
        # 分位数对比
        def qs(x):
            return {"p10": float(np.percentile(x, 10)),
                    "p25": float(np.percentile(x, 25)),
                    "p50": float(np.percentile(x, 50)),
                    "p75": float(np.percentile(x, 75)),
                    "p90": float(np.percentile(x, 90)),
                    "mean": float(np.mean(x))}
        rq, nq = qs(rv), qs(nv)
        # 总体 Cohen's d
        pooled_sd = float(np.sqrt((np.var(rv) + np.var(nv)) / 2)) or 1.0
        cohens_d = float((np.mean(rv) - np.mean(nv)) / pooled_sd)
        # 按机场分层 Cohen's d(控制机场混杂),加权平均
        real_ap = dep[dep["is_real_long"]][["ADEP_mvt", col]].dropna()
        norm_ap = dep[dep["is_normal"]][["ADEP_mvt", col]].dropna()
        d_strat_num, d_strat_den = 0.0, 0.0
        for ap, g_r in real_ap.groupby("ADEP_mvt"):
            g_n = norm_ap[norm_ap["ADEP_mvt"] == ap]
            if len(g_r) < 1 or len(g_n) < 3:
                continue
            sd_ap = float(np.sqrt((np.var(g_r[col]) + np.var(g_n[col])) / 2)) or 1.0
            d_ap = float((g_r[col].mean() - g_n[col].mean()) / sd_ap)
            w = len(g_r) * len(g_n) / (len(g_r) + len(g_n))
            d_strat_num += w * d_ap
            d_strat_den += w
        cohens_d_strat = d_strat_num / d_strat_den if d_strat_den > 0 else np.nan
        # Mann-Whitney U (双侧,正态近似 z)
        from scipy.stats import mannwhitneyu, spearmanr
        try:
            u_stat, u_p = mannwhitneyu(rv, nv, alternative="two-sided")
        except Exception:
            u_stat, u_p = np.nan, np.nan
        # 与 TAXITIME 的 spearman: holdout(主) + 全年(辅,功效高)
        both = dep[[col, "TAXITIME_SEC_mvt"]].dropna()
        try:
            rho_ho, rho_ho_p = spearmanr(both[col], both["TAXITIME_SEC_mvt"]) if len(both) > 10 else (np.nan, np.nan)
        except Exception:
            rho_ho, rho_ho_p = np.nan, np.nan
        both_yr = dep_allyear[[col, "TAXITIME_SEC_mvt"]].dropna()
        try:
            rho_yr, rho_yr_p = spearmanr(both_yr[col], both_yr["TAXITIME_SEC_mvt"]) if len(both_yr) > 10 else (np.nan, np.nan)
        except Exception:
            rho_yr, rho_yr_p = np.nan, np.nan

        # 判定: 主用分层 d(控制机场混杂),辅以 rho_ho/rho_yr 方向一致性
        abs_d = abs(cohens_d_strat) if not np.isnan(cohens_d_strat) else 0.0
        abs_rho_ho = abs(rho_ho) if not np.isnan(rho_ho) else 0.0
        abs_rho_yr = abs(rho_yr) if not np.isnan(rho_yr) else 0.0
        same_dir = (np.sign(rho_ho) == np.sign(rho_yr)) if not (np.isnan(rho_ho) or np.isnan(rho_yr)) else False
        if abs_d >= 0.8 or (abs_rho_ho >= 0.3 and same_dir) or (abs_rho_yr >= 0.3 and same_dir):
            verdict = "strong"
        elif abs_d >= 0.3 or (abs_rho_ho >= 0.15 and same_dir) or (abs_rho_yr >= 0.15 and same_dir):
            verdict = "weak"
        else:
            verdict = "none"

        rows.append({
            "signal": name, "desc": desc,
            "n_real": int(len(rv)), "n_norm": int(len(nv)),
            "real_quantiles": rq, "norm_quantiles": nq,
            "cohens_d": cohens_d, "cohens_d_strat": float(cohens_d_strat) if not np.isnan(cohens_d_strat) else None,
            "mwu_p": float(u_p) if not np.isnan(u_p) else None,
            "spearman_rho_holdout": float(rho_ho) if not np.isnan(rho_ho) else None,
            "spearman_p_holdout": float(rho_ho_p) if not np.isnan(rho_ho_p) else None,
            "spearman_rho_allyear": float(rho_yr) if not np.isnan(rho_yr) else None,
            "spearman_p_allyear": float(rho_yr_p) if not np.isnan(rho_yr_p) else None,
            "verdict": verdict,
        })
        print(f"    {name:30s} d_strat={cohens_d_strat if not np.isnan(cohens_d_strat) else float('nan'):+.2f} "
              f"rho_ho={rho_ho if not np.isnan(rho_ho) else float('nan'):+.2f} "
              f"rho_yr={rho_yr if not np.isnan(rho_yr) else float('nan'):+.2f} verdict={verdict}")

    # 排序: strong > weak > none,按 |cohens_d_strat| 降序
    rows.sort(key=lambda r: (
        {"strong": 0, "weak": 1, "none": 2, "insufficient_data": 3, "unavailable": 4}[r["verdict"]],
        -abs(r.get("cohens_d_strat") or 0),
    ))
    match_rate = float(dep["stand_rwy_dist_m"].notna().mean())
    print(f"  stand_rwy_dist_m match rate: {match_rate:.1%}")
    return rows, dep, {"stand_rwy_dist_match_rate": match_rate}


def task5_samples(long: pd.DataFrame):
    """每机场从真实/脏值各抽5行,打印关键时间戳。"""
    print("\n=== 任务5: 抽样表 ===")
    cols = ["MVT_ID_mvt", "ADEP_mvt", "label", "uncertain_reason",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt",
            "TAXITIME_SEC_mvt", "mvt_minus_aobt_sec", "abs_aobt_minus_blk_sec",
            "abs_blk_minus_sched_sec", "mvt_minus_sched_sec"]
    samples = []
    for ap, g in long.groupby(long["ADEP_mvt"].astype(str)):
        for lab in ["real", "dirty"]:
            gg = g[g["label"] == lab]
            n_take = min(5, len(gg))
            if n_take == 0:
                continue
            take = gg.sample(n=n_take, random_state=RANDOM_STATE)
            for _, r in take.iterrows():
                row = {}
                for c in cols:
                    v = r[c]
                    if isinstance(v, pd.Timestamp):
                        row[c] = str(v)
                    elif isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                        row[c] = float(v)
                    elif pd.isna(v):
                        row[c] = None
                    else:
                        row[c] = v
                samples.append(row)
    print(f"  抽样行数: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    out = {"meta": {
        "random_state": RANDOM_STATE,
        "note": "独立复算。任务1 用全量 holdout 预测 parquet + 均值基准(环境无 LGB 模型);"
                "任务2-5 用 sample 中 SCHED 月份∈{1,7} 的 DEP(小样本,~800 行)。"
                "stand_rwy_dist 仅 2025-01 单月的 dist 文件可用,跨月匹配率约 39%。"
                "天气数据未接入。",
        "data_limits": {
            "lgb_model_unavailable": "models/ 下无 .json/.cbm LGB 模型文件",
            "holdout_preds_only": "holdout_v2_preds.parquet 仅含 true_y+pred,无 BLOCK/AOBT",
            "sample_only": "data/sample/ 为 40 DEP/机场/月 小样本",
        },
    }}

    out["task1_sse_sensitivity"] = task1_sse_bins()

    dep_allyear = load_all_sample_dep()
    dep = dep_allyear[dep_allyear["SCHED_TIME_UTC_mvt"].dt.month.isin([1, 7])].copy()
    out["meta"]["n_allyear_sample_dep"] = int(len(dep_allyear))
    out["meta"]["n_holdout_sample_dep"] = int(len(dep))
    print(f"\nallyear sample DEP: {len(dep_allyear)}, holdout DEP: {len(dep)},"
          f" 长滑行(>1800) n={int((dep['TAXITIME_SEC_mvt']>1800).sum())}")

    long, t2 = task2_classify(dep)
    out["task2_labels"] = t2
    out["task3_airport_bucket"] = task3_by_airport(long)
    t4, dep_full, t4_meta = task4_signals(dep, long, dep_allyear)
    out["task4_signals"] = t4
    out["task4_meta"] = t4_meta
    out["task5_samples"] = task5_samples(long)

    # 可建模样本(真实)确切行数: holdout 实测 + 全年同规则推算
    out["modelable_real_count"] = int((long["label"] == "real").sum())
    long_all, t2_all = task2_classify(dep_allyear)
    out["modelable_real_count_allyear"] = int((long_all["label"] == "real").sum())
    out["task2_labels_allyear"] = t2_all

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {OUT_JSON}")

    # 简版 MD
    md = []
    md.append("# 长滑行结构独立诊断\n")
    md.append(f"- 随机种子: {RANDOM_STATE}")
    md.append(f"- 任务1: 全量 holdout 344k 行,均值基准(环境无 LGB 模型文件)")
    md.append(f"- 任务2-5: sample 2025-01+07 DEP n={out['meta']['n_holdout_sample_dep']},"
              f" 长滑行(>1800) n={t2['n_long']}\n")
    md.append("## 任务1: SSE 分档 + 敏感性\n")
    for name in ["baseline_global", "baseline_ap"]:
        r = out["task1_sse_sensitivity"][name]
        md.append(f"### {name} (overall RMSE={r['overall_rmse']:.2f})")
        md.append("| bin | n | %行 | RMSE | %SSE |")
        md.append("|---|---|---|---|---|")
        for b in r["bins"]:
            md.append(f"| {b['bin']} | {b['n']} | {b['pct_rows']:.2f}% | "
                      f"{b['rmse_bin'] if b['rmse_bin'] else 0:.1f} | {b['pct_sse']:.2f}% |")
        md.append("\n敏感性(长滑行>1800 档 RMSE 调整):")
        md.append("| 目标 RMSE | 整盘 RMSE | Δ |")
        md.append("|---|---|---|")
        for s in r["sensitivity"]:
            md.append(f"| {s['long_bin_target_rmse']:.0f} | {s['overall_rmse']:.2f} | "
                      f"{s['delta_vs_current']:+.2f} |")
        md.append("")
    md.append("## 任务2: AOBT 打标\n")
    md.append(f"- n_long={t2['n_long']},分布: {t2['counts']}({t2['pct']})")
    md.append(f"- 跨天±86400 命中: {t2['cross_day_86400_hits']}")
    md.append(f"- MVT 日期差>30d 命中: {t2['date_diff_gt_30d_hits']}")
    md.append(f"- 可建模样本(real)确切行数: **{out['modelable_real_count']}**\n")
    md.append("## 任务4: 信号强弱排序\n")
    md.append(f"- stand_rwy_dist 匹配率: {t4_meta['stand_rwy_dist_match_rate']:.1%}(dist 文件仅 2025-01 月,跨月未命中)")
    md.append("| 信号 | 判定 | d(分层) | ρ_holdout | ρ_allyear |")
    md.append("|---|---|---|---|---|")
    for r in t4:
        d_s = r.get("cohens_d_strat")
        rho_h = r.get("spearman_rho_holdout")
        rho_y = r.get("spearman_rho_allyear")
        md.append(f"| {r['signal']} | {r['verdict']} | "
                  f"{f'{d_s:+.3f}' if d_s is not None else 'nan'} | "
                  f"{f'{rho_h:+.3f}' if rho_h is not None else 'nan'} | "
                  f"{f'{rho_y:+.3f}' if rho_y is not None else 'nan'} |")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"写入 {OUT_MD}")


if __name__ == "__main__":
    main()
