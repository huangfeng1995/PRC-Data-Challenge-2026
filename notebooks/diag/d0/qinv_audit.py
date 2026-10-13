"""
qinv_audit.py — step1: 排队反演量与 FEATS71 现状核对 + 反演合理性审计 (只读)

产出:
  1. 现有 mvt_* 排队列到底算了什么 (源码口径)
  2. 反演量清单: 哪些已存在 / 哪些真正新增
  3. 反演 k 的分布 + EGLL 退化根因诊断 + 自适应阈值修复效果
  4. 新增特征的建议清单
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "qinv_audit.json"
NS = 1_000_000_000
CACHE = D0 / "_holdout_allcols_s.parquet"

def log(*a): print(" ".join(str(x) for x in a), flush=True)

# ================= 1. 现有列口径 (源码) =================
sys.path.insert(0, str(D0))
from oof_calibrate import FEATS71

EXISTING_QUEUE = [
    "mvt_n_ahead", "mvt_pos_rw", "mvt_mu_rwy_5", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
    "mvt_w_actual_5", "mvt_w_actual_15", "mvt_w_actual_30",
    "mvt_na_J", "mvt_na_H", "mvt_na_M", "mvt_na_L", "mvt_na_heavy_frac",
    "mvt_na_heavy_x_pos", "mvt_q_peak", "mvt_q_mean",
    "dep_mvt_cnt_runway_5m", "dep_mvt_cnt_runway_10m", "dep_mvt_cnt_runway_15m",
    "dep_mvt_cnt_airport_5m", "dep_mvt_cnt_airport_10m", "dep_mvt_cnt_airport_15m",
    "dep_cnt_m_5", "dep_cnt_m_15", "dep_cnt_m_30",
    "dep_queue_n_runway", "dep_queue_n_airport",
    "ap_month_hourly_density", "rwy_month_share",
]
log("=" * 100)
log("1. FEATS71 中已有的排队/密度类列 (源码口径)")
log("=" * 100)
SRC = {
    "mvt_n_ahead":      "_r7_lib.py:420  enter(AOBT+t_unimp)事件在 MVT 时刻的瞬时队列长度",
    "mvt_pos_rw":       "_r7_lib.py:446  = n_ahead + 1 (队列位置 1-based)",
    "mvt_mu_rwy_*":     "_r7_lib.py:456  [MVT-w, MVT] 内同跑道起飞数 / w  → 架次/分钟",
    "mvt_w_actual_*":   "_r7_lib.py:466  = n_ahead / mu_rwy  (分钟)",
    "mvt_na_*":         "_r7_lib.py:426  队列内 J/H/M/L 尾流类别计数",
    "mvt_na_heavy_*":   "_r7_lib.py:442  (J+H)/n 及 (J+H)*pos_rw",
    "mvt_q_peak/mean":  "_r7_lib.py:469+ 机场级 [AOBT, MVT] 窗口内队列峰值/均值",
    "dep_mvt_cnt_runway_*": "features_queue.py:38-47  以 AOBT 为中心的前 w 分钟同跑道起飞数",
    "dep_cnt_m_*":      "_r7_lib.py:148  以 MVT 为查询点的起飞计数",
    "ap_month_hourly_density": "ap×月×小时 起飞密度",
}
for k, v in SRC.items():
    log(f"  {k:<26} {v}")
log(f"\n  合计已有排队/密度类列: {len(EXISTING_QUEUE)} / 71")

# ================= 2. 反演量清单 =================
log("\n" + "=" * 100)
log("2. 排队反演量清单: 已存在 vs 真正新增")
log("=" * 100)
CAND = [
    ("q_k_burst",        "序列反演队列位置: 从 MVT 往回走, 遇相邻起飞间隔>自适应阈值即停",
     "新增", "mvt_n_ahead 是 enter 事件瞬时快照, 非序列清空点反演"),
    ("q_k_x_s",          "反演等待时间 = k_burst × 跑道服务间隔", "新增", "mvt_w_actual 是 n_ahead/mu, 口径不同"),
    ("q_svc_gap",        "跑道服务间隔 = 相邻起飞间隔中位数(仅取紧邻对)", "新增", "无"),
    ("q_svc_p75",        "跑道服务间隔 p75", "新增", "无"),
    ("q_svc_ratio",      "k_burst / mu_rwy 归一化等待", "新增", "无"),
    ("q_idle_since_dep", "距上一架起飞的时间 (跑道空闲度)", "新增", "mu_rwy 只给均值密度, 不给距上次多久"),
    ("q_burst_len_min",  "当前连续起飞段的持续时长(分钟)", "新增", "无"),
    ("q_dep_cnt_60m",    "MVT 前 60 分钟同跑道起飞数", "新增", "现有最长窗口 30min"),
    ("q_dep_cnt_120m",   "MVT 前 120 分钟同跑道起飞数", "新增", "无"),
    ("q_arr_cnt_30m",    "MVT 前 30 分钟同跑道计划到港数", "新增", "现有全是离港量, 无到港"),
    ("q_arr_cnt_60m",    "MVT 前 60 分钟同跑道计划到港数", "新增", "无"),
    ("q_rwy_share_dep",  "该跑道占机场近 30min 起飞比例", "新增", "rwy_month_share 是月度静态占比"),
]
log(f"{'候选量':<18} {'状态':<6} 说明")
NEW_FEATS = []
for name, desc, st, why in CAND:
    log(f"{name:<18} {st:<6} {desc}")
    if st == "新增":
        NEW_FEATS.append(name)
log(f"\n  真正新增: {len(NEW_FEATS)} 个 (其余 {len(CAND)-len(NEW_FEATS)} 个已存在)")

# ================= 3. 反演合理性审计 =================
log("\n" + "=" * 100)
log("3. 反演 k 分布审计 (固定阈值 vs 自适应阈值)")
log("=" * 100)

h = pd.read_parquet(CACHE)
dep = h[h["PHASE_mvt"] == "DEP"].reset_index(drop=True)
log(f"holdout DEP {len(dep):,} 行")

mvt = dep["MVT_TIME_UTC_mvt_s"].to_numpy(np.float64)
rwy = dep["RUNWAY_mvt"].astype(str).to_numpy()
adep = dep["ADEP_mvt"].astype(str).to_numpy()
ok = np.isfinite(mvt)
mvt_s = np.where(ok, mvt, np.inf)

def invert(rwy_arr, mvt_arr, gap_mode="fixed", gap_s=300.0, cap=400):
    """对每条跑道按 MVT 排序, 从每个 MVT 往回连续计数(间隔<阈值), 得 k"""
    k = np.zeros(len(mvt_arr), dtype=np.int64)
    svc = np.full(len(mvt_arr), np.nan)
    for rw in np.unique(rwy_arr):
        m = rwy_arr == rw
        idx = np.where(m)[0]
        v = mvt_arr[idx]
        fin = np.isfinite(v)
        if fin.sum() < 10:
            continue
        order = idx[fin][np.argsort(v[fin], kind="stable")]
        vo = mvt_arr[order]
        gaps = np.diff(vo)
        tight = gaps[gaps < gap_s]
        s = float(np.median(tight)) if len(tight) else gap_s
        # 服务间隔按逐架广播
        sc = np.full(len(order), s)
        for i in range(1, len(vo)):
            # 局部服务间隔: 前面 20 个紧邻间隔的中位数 (自适应)
            lo = max(0, i - 20)
            loc = gaps[lo:i]
            loc_t = loc[loc < max(gap_s, 3 * s)]
            if gap_mode == "adaptive" and len(loc_t) >= 3:
                sc[i] = float(np.median(loc_t))
        thr = gap_s if gap_mode == "fixed" else max(120.0, 3.0 * s)
        cur = 0
        for i in range(1, len(vo)):
            cur = cur + 1 if (gaps[i - 1] < thr) else 0
            k[order[i]] = min(cur, cap)
        svc[order] = sc
    return k, svc

log(f"\n{'机场':>6} {'跑道数':>6} | {'固定300s: k中位':>15} {'p95':>7} {'max':>6} | "
    f"{'自适应: k中位':>13} {'p95':>7} {'max':>6} {'s中位':>7}")
R = {"feats71_queue_like": EXISTING_QUEUE, "new_candidates": NEW_FEATS,
     "source_map": SRC, "ap": {}}
for a in ["EGLL", "LTFM", "LIRF", "EHAM", "LFPG"]:
    m = adep == a
    if m.sum() < 100:
        continue
    kf, _ = invert(rwy[m], mvt_s[m], "fixed", 300.0)
    ka, sa = invert(rwy[m], mvt_s[m], "adaptive", 300.0)
    rwy_n = len(np.unique(rwy[m][np.isfinite(mvt_s[m])]))
    log(f"{a:>6} {rwy_n:>6} | {np.median(kf):>15.0f} {np.percentile(kf,95):>7.0f} {kf.max():>6} | "
        f"{np.median(ka):>13.0f} {np.percentile(ka,95):>7.0f} {ka.max():>6} {np.nanmedian(sa):>7.1f}")
    R["ap"][a] = {
        "n_rwy": rwy_n,
        "fixed300": {"k_med": float(np.median(kf)), "k_p95": float(np.percentile(kf, 95)),
                     "k_max": int(kf.max())},
        "adaptive": {"k_med": float(np.median(ka)), "k_p95": float(np.percentile(ka, 95)),
                     "k_max": int(ka.max()), "s_med": float(np.nanmedian(sa))},
    }

# ================= 4. 反演量与真值/已有特征的相关性 =================
log("\n" + "=" * 100)
log("4. 新增反演量 与 真值 / 已有排队特征 的相关性 (EGLL 举例)")
log("=" * 100)
taxi = dep["TAXITIME_SEC_mvt"].to_numpy(np.float64)
m = adep == "EGLL"
ka, sa = invert(rwy[m], mvt_s[m], "adaptive", 300.0)
q = pd.DataFrame({"k": ka, "s": sa, "taxi": taxi[m],
                  "since_dep": np.nan, "cnt60": np.nan})
# 距上一架起飞
for rw in np.unique(rwy[m]):
    mm = (rwy[m] == rw)
    v = np.where(np.isfinite(mvt_s[m]), mvt_s[m], np.inf)
    idx = np.where(mm)[0]
    vo = v[idx]
    o = idx[np.argsort(vo, kind="stable")]
    vv = mvt_s[m][o]
    q.loc[o, "since_dep"] = np.concatenate([[np.nan], np.diff(vv)])
# 近 60/120min 起飞数
sv = np.where(np.isfinite(mvt_s[m]), mvt_s[m], np.inf)
for w, nm in [(3600, "cnt60"), (7200, "cnt120")]:
    c = np.zeros(len(sv))
    for rw in np.unique(rwy[m]):
        mm = (rwy[m] == rw)
        vv = np.sort(sv[mm & np.isfinite(sv)])
        idx = np.where(mm)[0]
        c[idx] = np.searchsorted(vv, sv[idx] - w, side="left") * -1 + np.searchsorted(vv, sv[idx], side="right")
    q[nm] = c
log(f"\n{'量':<12} {'corr(·,TAXI)':>14} {'corr(·,k)':>12}")
for c in ["s", "since_dep", "cnt60", "cnt120"]:
    v = q[c].to_numpy(np.float64)
    msk = np.isfinite(v) & np.isfinite(q["taxi"].to_numpy())
    log(f"{c:<12} {np.corrcoef(v[msk], q['taxi'].to_numpy()[msk])[0,1]:>14.4f} "
        f"{np.corrcoef(v[msk], q['k'].to_numpy()[msk])[0,1]:>12.4f}")
log(f"  k 自身与 TAXI: {np.corrcoef(q['k'], q['taxi'])[0,1]:.4f}   "
    f"k 中位={q['k'].median():.0f} p95={q['k'].quantile(.95):.0f} max={q['k'].max()}")
R["corr_egll"] = {"k_vs_taxi": float(np.corrcoef(q["k"], q["taxi"])[0, 1])}
for c in ["s", "since_dep", "cnt60", "cnt120"]:
    v = q[c].to_numpy(np.float64)
    msk = np.isfinite(v) & np.isfinite(q["taxi"].to_numpy())
    R["corr_egll"][f"{c}_vs_taxi"] = float(np.corrcoef(v[msk], q["taxi"].to_numpy()[msk])[0, 1])

json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
log(f"\n→ {OUT}")
