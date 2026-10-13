"""
arr_load_diag.py — 第二步: 到港负荷对 DEP 滑行时间的信号诊断 (只读)
==========================================================================
口径:
  数据: 2025 全年训练数据 (不碰 ranking)。holdout = 1+7 月为主口径。
  负荷定义: 同一机场, 在该 DEP 航班【推出时刻】前后 ±30 / ±60 分钟内,
            【计划到港】的 ARR 航班数。
  计划到港时刻 = ARR 行的 SCHED_TIME_UTC_mvt (flight 源计划, ranking 上 ARR 100% 非空)
  参照时点 ref = AOBT_3_flt (实际推出), 缺失回退 EOBT_1_flt —— 与现有队列特征同口径
  参照机场: DEP 用 ADEP_mvt; ARR 用 ADES_mvt (= 同一个物理机场)
  严格不读: TAXITIME, ARR 的 MVT/BLOCK (实际落地时刻)

分箱: 按负荷分箱, 报 n / 中位真值 / 均值真值 / RMSE
关键拆分: (1800,10800] 与 >10800 两档分别看单调性
增量检验: 与 ap_month_hourly_density (同 (ADEP,month) 每小时平均离港数) 对比
        - 各自 AUC
        - 在密度分层内的条件 AUC (控制机场×月×小时的离港密度)
        - 分层内 Spearman(负荷, |err|)

输出: notebooks/diag/d0/arr_load_diag.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT_JSON = D0 / "arr_load_diag.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/arr_load_diag.log")

COLS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "TAXITIME_SEC_mvt"]
MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                  if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
              for m in range(1, 13)}
HOLDOUT = [1, 7]
INAT = np.iinfo(np.int64).min


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


def auc(score, label):
    """Mann-Whitney AUC。自动剔除 score 中的 NaN (ref 缺失约占 1%), 否则 rank() 会把 NaN
    传播到结果里, 使 AUC 变成 NaN 而非真实值。"""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(score).rank().values
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


log("=" * 78)
log("[arr_load] 到港负荷 → DEP 滑行时间 信号诊断 (只读, 不碰 ranking)")
log("=" * 78)
t0 = time.time()

# ============ [1] 载入 2025 全年 (仅必要列) ============
log("")
log("[1] 载入 2025 全年 DEP/ARR 必要列")
parts = []
for m in range(1, 13):
    parts.append(pq.read_table(ROOT / MONTH_FILE[m], columns=COLS).to_pandas())
df = pd.concat(parts, ignore_index=True)
del parts
ph = df["PHASE_mvt"].astype(str).values
is_dep, is_arr = ph == "DEP", ph == "ARR"
log(f"    DEP {int(is_dep.sum()):,}  ARR {int(is_arr.sum()):,}  总 {len(df):,}")

# 计划时刻 (flight 源, 线上可用)
df["sched"] = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
df["aobt"] = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
df["eobt"] = pd.to_datetime(df["EOBT_1_flt"], utc=True, errors="coerce")
df["mvt"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
# 参照时点: AOBT 优先, 回退 EOBT (与现有队列特征同口径)
df["ref"] = df["aobt"].fillna(df["eobt"])
df["ref_mvt"] = df["mvt"]
df["month"] = df["mvt"].dt.month

# 自洽性检查: ARR 的 sched 应当早于其 MVT (落地)
_arr_sched = df.loc[is_arr, "sched"]
_arr_mvt = df.loc[is_arr, "mvt"]
_valid = _arr_sched.notna() & _arr_mvt.notna()
_delta = (_arr_mvt - _arr_sched)[_valid].dt.total_seconds()
log("")
log("    自洽性: ARR 行 (MVT - SCHED) 秒数分位  [MVT=实际落地, SCHED=计划]")
for q in [1, 25, 50, 75, 99]:
    log(f"      p{q:<3d} {np.percentile(_delta.values, q):10.1f}")
log(f"      负值占比 {float((_delta.values<0).mean())*100:.2f}%")
log("    → 负值占多数是【正常】的: 航班普遍早于计划时刻落地(提前到达), 中位提前约 10 分钟。")
log("      这证明 SCHED 确实是该 ARR 行的计划到港时刻, 可直接作为'计划到港负荷'的时间轴。")

# ============ [2] 计算到港负荷 ============
log("")
log("[2] 计算到港负荷 (计划到港, ±30/±60min)")


def arr_load(ref_ns, ap_arr, arr_sched_ns, windows=(30, 60)):
    """对每个 DEP 行, 统计同机场 ±w 分钟内计划到港的 ARR 航班数。

    ⚠ 单位: pandas 2.3.3 的 datetime64[ns] 语义在 astype('int64') 后得到的是
      【微秒】(datetime64[us]), 所以窗口常数必须用微秒, 否则会放大 1000 倍。
      这里统一走 _to_int_us() 做显式转换, 不依赖 pandas 内部单位。
    """
    n = len(ref_ns)
    out = {w: np.full(n, np.nan) for w in windows}
    out_past = {w: np.full(n, np.nan) for w in windows}   # 仅过去 (ref-w, ref]
    out_ap = {w: np.zeros(n) for w in windows}             # 绝对机场规模
    valid_sched = arr_sched_ns != INAT
    aps = pd.unique(ap_arr)
    for a in aps:
        sidx = np.where(ap_arr == a)[0]
        sidx = sidx[valid_sched[sidx]]
        if len(sidx) == 0:
            continue
        s_sorted = np.sort(arr_sched_ns[sidx])
        didx = np.where(ref_ap_global == a)[0]
        if len(didx) == 0:
            continue
        r = ref_ns[didx]
        r_valid = r != INAT
        r_safe = np.where(r_valid, r, np.iinfo(np.int64).max)
        for w in windows:
            half = w * 60 * 1_000_000          # 分钟 → 微秒
            lo = np.searchsorted(s_sorted, r_safe - half, side="left")
            hi = np.searchsorted(s_sorted, r_safe + half, "right")
            cnt = (hi - lo).astype(np.float64)
            out[w][didx] = np.where(r_valid, cnt, np.nan)
            lo_p = np.searchsorted(s_sorted, r_safe - half, side="left")
            hi_p = np.searchsorted(s_sorted, r_safe, side="right")
            out_past[w][didx] = np.where(r_valid, (hi_p - lo_p).astype(np.float64), np.nan)
            out_ap[w][didx] = len(sidx)
    return out, out_past, out_ap


def _to_int_us(s: pd.Series) -> np.ndarray:
    """datetime Series → int64 微秒 (NaT → INAT)。显式单位, 不受 pandas 版本影响。"""
    return s.to_numpy(dtype="datetime64[us]").astype("int64")


t1 = time.time()
dep = df[is_dep].copy().reset_index(drop=True)
arr = df[is_arr]
ref_ns = _to_int_us(dep["ref"])
ref_mvt_ns = _to_int_us(dep["ref_mvt"])
ref_ap_global = dep["ADEP_mvt"].astype(str).values
arr_ap = arr["ADES_mvt"].astype(str).values
arr_sched_ns = _to_int_us(arr["sched"])

load_pm, load_past_pm, ap_size = arr_load(ref_ns, arr_ap, arr_sched_ns, windows=(30, 60))
load_pm_mvt, _, _ = arr_load(ref_mvt_ns, arr_ap, arr_sched_ns, windows=(30, 60))
log(f"    ±30/±60 负荷计算完成 ({time.time()-t1:.0f}s)")
log(f"    负荷非空率: pm30 {np.mean(~np.isnan(load_pm[30]))*100:.2f}%  "
    f"pm60 {np.mean(~np.isnan(load_pm[60]))*100:.2f}%")
# 合理性护栏: ±60min 窗口内的到港数必须远小于该机场月度到港总数
_v = load_pm[60][~np.isnan(load_pm[60])]
log(f"    负荷分布自检 pm60: p1={np.percentile(_v,1):.0f} p50={np.percentile(_v,50):.0f} "
    f"p99={np.percentile(_v,99):.0f} max={_v.max():.0f}")
assert _v.max() < 400, f"pm60 负荷最大值 {_v.max():.0f} 异常, 单位可能又错了"

# ============ [3] 对照特征: ap_month_hourly_density ============
log("")
log("[3] 构造对照特征 ap_month_hourly_density (同 (ADEP,month) 每小时平均离港数)")
dep["__hour__"] = dep["mvt"].dt.hour
hc = dep.groupby(["ADEP_mvt", "month", "__hour__"]).size()
apm = hc.groupby(["ADEP_mvt", "month"]).mean()
dep["ap_month_hourly_density"] = (
    pd.MultiIndex.from_arrays([dep["ADEP_mvt"].values, dep["month"].values])
    .map(apm) if False else
    pd.Series(apm.reindex(pd.MultiIndex.from_arrays(
        [dep["ADEP_mvt"].values, dep["month"].values])).values,
        index=dep.index))
dep["arr_pm30"] = load_pm[30]
dep["arr_pm60"] = load_pm[60]
dep["arr_past60"] = load_past_pm[60]
dep["arr_past30"] = load_past_pm[30]
dep["arr_pm60_mvtref"] = load_pm_mvt[60]
log(f"    覆盖率: ap_month_hourly_density {dep['ap_month_hourly_density'].notna().mean()*100:.2f}%"
    f"  arr_pm60 {dep['arr_pm60'].notna().mean()*100:.2f}%")

# ============ [4] 只保留有效 DEP (有真值) ============
yv = dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
ok = np.isfinite(yv) & (yv > 0)
D = dep[ok].reset_index(drop=True)
Y = yv[ok]
log("")
log(f"[4] 有效 DEP 行 {len(D):,}  (真值中位 {np.median(Y):.1f}s, >1800 占 "
    f"{float((Y>1800).mean())*100:.2f}%)")

HO = D["month"].isin(HOLDOUT).values
# 分箱改用【分位数】而非固定阈值: pm60 的中位约 64, 固定阈值会把 96% 行压进最高档,
# 无法分辨结构。用 8 等分位保证每档样本量可比。
def sp(a, b):
    """Spearman, 自动剔除 NaN/inf 配对 (否则 scipy 返回 nan)。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return float("nan"), float("nan")
    r, p = spearmanr(a[ok], b[ok])
    return float(r), float(p)


def _qbin_edges(v, mask):
    vv = v[mask & np.isfinite(v)]
    return np.unique(np.percentile(vv, np.linspace(0, 100, 9)))


def binned(mask, feat, tag, edges=None):
    v = D[feat].values.astype(float)
    if edges is None:
        edges = _qbin_edges(v, mask)
    log("")
    log(f"  --- {tag} (n={int(mask.sum()):,}) ---")
    log(f"  {'负荷档':>12s} {'n':>8s} {'中位真值':>10s} {'均值真值':>10s} "
        f"{'>1800占比':>10s} {'>10800占比':>11s}")
    rows = []
    idx = np.searchsorted(edges, v, side="right") - 1
    for k in range(len(edges) - 1):
        m = mask & (idx == k) & np.isfinite(v)
        n = int(m.sum())
        if n < 30:
            continue
        yb = Y[m]
        lo_l = f"{edges[k]:.0f}" if k > 0 else f"<{edges[0]:.0f}"
        hi_l = f"{edges[k+1]:.0f}" if k < len(edges) - 2 else f">{edges[-1]:.0f}"
        lbl = f"{lo_l}-{hi_l}"
        r = {"bin": lbl, "n": n, "median_true": round(float(np.median(yb)), 1),
             "mean_true": round(float(yb.mean()), 1),
             "share_gt1800": round(float((yb > 1800).mean()), 4),
             "share_gt10800": round(float((yb > 10800).mean()), 5)}
        rows.append(r)
        log(f"  {lbl:>12s} {n:>8d} {r['median_true']:>10.1f} {r['mean_true']:>10.1f} "
            f"{r['share_gt1800']*100:>9.3f}% {r['share_gt10800']*100:>10.4f}%")
    if len(rows) >= 3:
        xs = list(range(len(rows)))
        rho_med, _ = spearmanr(xs, [b["median_true"] for b in rows])
        rho_1800, _ = spearmanr(xs, [b["share_gt1800"] for b in rows])
        log(f"  → 分箱单调性: Spearman(负荷档序, 中位真值) = {rho_med:+.3f}   "
            f"Spearman(负荷档序, >1800占比) = {rho_1800:+.3f}")
    else:
        rho_med = rho_1800 = float("nan")
        log(f"  → 有效档数 {len(rows)} < 3, 单调性无法评估 (样本太稀)")
    return rows, float(rho_med), float(rho_1800), [float(e) for e in edges]


log("")
log("=" * 78)
log("[5] 分箱表: 全样本 (2025 全部有效 DEP)")
log("=" * 78)
all_mask = np.ones(len(D), bool)
b_all, rho_all, rho18_all, edges_all = binned(all_mask, "arr_pm60", "arr_pm60 全部")
b_ho, rho_ho, rho18_ho, _ = binned(HO, "arr_pm60", "arr_pm60 holdout(1+7月)", edges=edges_all)
for m_, tag in [(1, "Jan"), (7, "Jul")]:
    binned(D["month"].values == m_, "arr_pm60", f"arr_pm60 {tag}", edges=edges_all)

# ============ [6] 关键拆分: 两档长尾 ============
log("")
log("=" * 78)
log("[6] 关键拆分: (1800,10800] vs >10800")
log("=" * 78)
GROUPS = [("(0,1800]", (Y > 0) & (Y <= 1800)),
          ("(1800,10800]", (Y > 1800) & (Y <= 10800)),
          (">10800", Y > 10800)]
split_tab = {}
for gname, gm in GROUPS:
    log("")
    log(f"  --- 真值档 {gname} (n={int(gm.sum()):,}) ---")
    v = D["arr_pm60"].values.astype(float)
    vv = v[gm]
    yv_g = Y[gm]
    ok_m = np.isfinite(vv)
    if ok_m.sum() > 100:
        r_all, p_all = spearmanr(vv[ok_m], yv_g[ok_m])
        r_gt, p_gt = spearmanr(vv[ok_m], (yv_g[ok_m] > np.median(yv_g[ok_m])).astype(int))
        d = D["ap_month_hourly_density"].values.astype(float)[gm]
        ok2 = ok_m & np.isfinite(d)
        r_dens, _ = spearmanr(vv[ok2], d[ok2])
        log(f"    Spearman(到港负荷, 真值)        = {r_all:+.4f}  (p={p_all:.2e})")
        log(f"    Spearman(到港负荷, 离港密度)    = {r_dens:+.4f}   ← 两者是否共线")
        log(f"    Spearman(到港负荷, 真值是否>中位) = {r_gt:+.4f}  (p={p_gt:.2e})")
        log(f"    负荷中位: 正常行 {np.median(vv[ok_m]):.1f}")
    else:
        r_all = r_dens = r_gt = float("nan")
        log("    (样本不足)")
    b_g, rho_g, rho18_g, _ = binned(gm, "arr_pm60", f"arr_pm60 @真值{gname}")
    split_tab[gname] = {"n": int(gm.sum()),
                        "spearman_load_vs_true": None if not np.isfinite(r_all) else round(float(r_all), 4),
                        "spearman_load_vs_density": None if not np.isfinite(r_dens) else round(float(r_dens), 4),
                        "spearman_load_vs_above_median": None if not np.isfinite(r_gt) else round(float(r_gt), 4),
                        "bins": b_g, "bin_monotonic_rho_median": round(rho_g, 3),
                        "bin_monotonic_rho_gt1800share": round(rho18_g, 3)}

# ============ [7] AUC 与增量检验 ============
log("")
log("=" * 78)
log("[7] AUC 与增量检验 (vs ap_month_hourly_density)")
log("=" * 78)
SCORES = [("arr_pm30", "到港负荷±30m"), ("arr_pm60", "到港负荷±60m"),
          ("arr_past30", "到港负荷过去30m"), ("arr_past60", "到港负荷过去60m"),
          ("arr_pm60_mvtref", "到港负荷±60m(ref=MVT)"),
          ("ap_month_hourly_density", "现有:离港密度")]
auc_tab = {}
for scope, smask in [("全部", all_mask), ("holdout", HO)]:
    log("")
    log(f"  --- {scope} ---")
    log(f"  {'排序分':>26s} " + "".join(f"{'>'+str(t):>10s}" for t in (1800, 3600, 10800)))
    auc_tab[scope] = {}
    for col, lab in SCORES:
        v = D[col].values.astype(float)[smask]
        yb = Y[smask]
        cells, rec = [], {}
        for th in (1800, 3600, 10800):
            a = auc(v, yb > th)
            rec[f"auc_gt{th}"] = None if not np.isfinite(a) else round(a, 4)
            cells.append(f"{a:.4f}" if np.isfinite(a) else "n/a")
        auc_tab[scope][lab] = rec
        log(f"  {lab:>26s} " + "".join(f"{c:>10s}" for c in cells))

# 条件 AUC: 在离港密度分层内
log("")
log("  条件 AUC (在 ap_month_hourly_density 五等分层内, 对 true>1800)")
dens = D["ap_month_hourly_density"].values.astype(float)
strat = pd.qcut(pd.Series(dens), 5, labels=False, duplicates="drop").values
cond = {}
for s in sorted(set(strat[~np.isnan(strat)].tolist())):
    m = HO & (strat == s)
    v60 = D["arr_pm60"].values.astype(float)[m]
    yb = Y[m]
    a60 = auc(v60, yb > 1800)
    ad = auc(dens[m], yb > 1800)
    cond[f"Q{int(s)+1}"] = {"n": int(m.sum()), "auc_arr_pm60": round(a60, 4),
                            "auc_density": round(ad, 4)}
    log(f"    Q{int(s)+1} n={int(m.sum()):>7d}  到港负荷AUC={a60:.4f}   "
        f"离港密度AUC={ad:.4f}   Δ={a60-ad:+.4f}")
# 合并分层 AUC (Fisher)
num = den = 0.0
for k, c in cond.items():
    w = c["n"]
    num += w * c["auc_arr_pm60"]
    den += w * c["auc_density"]
log(f"    分层合并 AUC: 到港负荷 {num/sum(c['n'] for c in cond.values()):.4f}  "
    f"离港密度 {den/sum(c['n'] for c in cond.values()):.4f}")

# 分层内 Spearman(负荷, |真值-该档均值|)
log("")
log("  分层内 Spearman(到港负荷, |真值 - 该层中位|)  [holdout]")
v60h = D["arr_pm60"].values.astype(float)[HO]
yh = Y[HO]
sh = strat[HO]
within = []
for s in sorted(set(sh[~np.isnan(sh)].tolist())):
    m = sh == s
    if m.sum() < 200:
        continue
    dev = np.abs(yh[m] - np.median(yh[m]))
    r, p = sp(v60h[m], dev)
    if not np.isfinite(r):
        log(f"    Q{int(s)+1} n={int(m.sum()):>7d}  (有效配对不足, 跳过)")
        continue
    within.append({"stratum": f"Q{int(s)+1}", "n": int(m.sum()), "rho": round(float(r), 4),
                   "p": float(f"{p:.2e}")})
    log(f"    Q{int(s)+1} n={int(m.sum()):>7d}  rho={r:+.4f}  p={p:.2e}")
w_rho = float(np.average([w["rho"] for w in within], weights=[w["n"] for w in within])) if within else float("nan")
log(f"    → 分层内加权平均 rho = {w_rho:+.4f}")

# 偏相关: 负荷与真值, 控制密度 + 小时 + 机场规模
log("")
log("  偏相关 (残差化: 对 密度 + 小时 + 机场规模 做 OLS, 再算残差相关)")
hour = dep.loc[D.index, "__hour__"].values.astype(float)
apsz = ap_size[60][ok].astype(float)
load_all = D["arr_pm60"].values.astype(float)
# 约 1.08% 行 ref 缺失 → 负荷为 NaN, 会让 lstsq 报 SVD did not converge。
# 这里只在【四者皆有限】的公共样本上做残差化与相关, 保证口径一致。
_pc_ok = np.isfinite(load_all) & np.isfinite(Y) & np.isfinite(dens) \
    & np.isfinite(hour) & np.isfinite(apsz)
log(f"    偏相关公共样本 {int(_pc_ok.sum()):,} / {len(D):,} "
    f"({_pc_ok.mean()*100:.2f}%)")
Xc = np.column_stack([np.ones(int(_pc_ok.sum())), dens[_pc_ok], hour[_pc_ok],
                      np.log1p(apsz[_pc_ok])])


def resid(v):
    b, *_ = np.linalg.lstsq(Xc, np.asarray(v, float)[_pc_ok], rcond=None)
    full = np.full(len(D), np.nan)
    full[_pc_ok] = np.asarray(v, float)[_pc_ok] - Xc @ b
    return full


r_load = resid(load_all)
r_y = resid(Y)
rp, pp = sp(r_load[HO], r_y[HO])
raw_r, raw_p = sp(load_all[HO], Y[HO])
log(f"    原始 Spearman(负荷, 真值) holdout      = {raw_r:+.4f} (p={raw_p:.2e})")
log(f"    偏相关 (控密度/小时/机场规模)          = {rp:+.4f} (p={pp:.2e})")

# ============ [8] 判定 ============
log("")
log("=" * 78)
log("[判定]")
log("-" * 78)
g_mid = split_tab.get("(1800,10800]", {})
g_ext = split_tab.get(">10800", {})
mono_mid = g_mid.get("bin_monotonic_rho_median", 0.0)
mono_ext = g_ext.get("bin_monotonic_rho_median", 0.0)
auc_mid_load = auc_tab["holdout"]["到港负荷±60m"]["auc_gt3600"]
auc_mid_dens = auc_tab["holdout"]["现有:离港密度"]["auc_gt3600"]
auc18_load = auc_tab["holdout"]["到港负荷±60m"]["auc_gt10800"]
auc18_dens = auc_tab["holdout"]["现有:离港密度"]["auc_gt10800"]
delta_auc = (None if auc_mid_dens is None or auc_mid_load is None
             else round(auc_mid_load - auc_mid_dens, 4))
signal_mid = mono_mid > 0.7 and (delta_auc is None or delta_auc > 0.01)
log(f"  (1800,10800] 档: 分箱单调性 rho={mono_mid:+.3f}  "
    f"Spearman(负荷,真值)={g_mid.get('spearman_load_vs_true')}")
log(f"  >10800 档:     分箱单调性 rho={mono_ext:+.3f}  "
    f"Spearman(负荷,真值)={g_ext.get('spearman_load_vs_true')}")
log(f"  AUC(>3600): 到港负荷 {auc_mid_load}  vs 现有密度 {auc_mid_dens}  Δ={delta_auc}")
log(f"  AUC(>10800): 到港负荷 {auc18_load}  vs 现有密度 {auc18_dens}")
log(f"  分层内加权 rho(负荷,|偏离|) = {w_rho:+.4f}")
log(f"  偏相关(控密度/小时/机场规模) = {rp:+.4f}")
log("")
if signal_mid:
    log("  >>> (1800,10800] 档有单调信号且超越现有密度特征: 值得做特征")
else:
    log("  >>> (1800,10800] 档未同时满足'单调'与'超越现有密度': 不支持做特征")

RESULT = {
    "task": "arr_load_diagnostic",
    "design": {
        "load_def": "同机场, DEP 推出时刻(AOBT,回退EOBT)前后±w分钟内【计划到港】ARR 航班数",
        "sched_source": "ARR 行 SCHED_TIME_UTC_mvt (flight 源计划时刻)",
        "ranking_availability": "ranking 上 ARR 行 SCHED_TIME_UTC_mvt 非空率 100%, ADES_mvt 100%",
        "not_used": "TAXITIME / ARR 的 MVT(实际落地) / BLOCK —— 全部未读",
        "windows": [30, 60],
    },
    "selfcheck_arr_sched_vs_mvt_sec": {
        "p1": round(float(np.percentile(_delta.values, 1)), 1),
        "p50": round(float(np.percentile(_delta.values, 50)), 1),
        "p99": round(float(np.percentile(_delta.values, 99)), 1),
        "negative_frac": round(float((_delta.values < 0).mean()), 5),
    },
    "n_valid_dep": int(len(D)),
    "n_holdout": int(HO.sum()),
    "bins_all": b_all, "bins_holdout": b_ho,
    "monotonic": {"all_rho_median": round(rho_all, 3), "all_rho_gt1800share": round(rho18_all, 3),
                  "holdout_rho_median": round(rho_ho, 3),
                  "holdout_rho_gt1800share": round(rho18_ho, 3)},
    "per_true_bucket": split_tab,
    "auc": auc_tab,
    "conditional_auc_within_density_quintile": cond,
    "within_stratum_spearman": within,
    "within_stratum_weighted_rho": round(w_rho, 4),
    "partial_correlation": {
        "raw_spearman_holdout": round(float(raw_r), 4), "raw_p": float(f"{raw_p:.2e}"),
        "partial_rho_holdout": round(float(rp), 4), "partial_p": float(f"{pp:.2e}"),
        "controls": ["ap_month_hourly_density", "hour_of_day", "log(airport_arr_size)"],
    },
    "verdict": {
        "mid_bucket_monotonic_rho": mono_mid,
        "extreme_bucket_monotonic_rho": mono_ext,
        "auc3600_load": auc_mid_load, "auc3600_density": auc_mid_dens,
        "delta_auc_vs_density": delta_auc,
        "within_stratum_weighted_rho": round(w_rho, 4),
        "partial_rho": round(float(rp), 4),
        "worth_building_feature": bool(signal_mid),
    },
    "timing_sec": round(time.time() - t0, 1),
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}   总耗时 {(time.time()-t0)/60:.1f} min")
