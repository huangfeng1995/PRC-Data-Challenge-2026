"""
disagreement_signal.py — 多模型分歧度能否识别长尾/大误差 (只读统计)
==========================================================================
输入: notebooks/diag/d0/multi_preds.parquet (train_multi.py 产出)
      列: y_true, month, ADEP, t_unimp_r7, pred_xgb, pred_lgb42, pred_lgb123,
          pred_lgb2024, pred_cat

核心问题: 上一轮已证明「只靠单 pred 做单调后处理」天花板 ≈ 0 (条件方差被压缩)。
本轮检验: 多基模型的分歧度 ( epistemic uncertainty) 是否携带额外信号?

分析:
  A. 三种分歧度定义
     - 算法分歧 (algo): std over {xgb, lgb42, cat}      ← 真正不同的归纳偏置
     - 种子分歧 (seed): std over {lgb42, lgb123, lgb2024} ← 同算法换种子 (阴性对照)
     - 全部分歧 (all):  std over 全部 5 个
  B. Spearman(分歧度, |true - 集成pred|)
  C. 分歧度识别长尾的 AUC: true>1800 / >3600 / >10800
  D. 对照基线: 单 pred (集成均值) 自身的 AUC —— 已知"用 pred 排序"能到什么程度
  E. 极端行 (>10800) 的分歧度分布 vs 正常行
  F. Jan / Jul 分别算, 确认方向一致
  G. 增量检验: 在 pred 分档内部分歧度是否仍有区分度 (排除"分歧度只是 pred 的代理")

输出:
  notebooks/diag/d0/disagreement_signal.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT_JSON = D0 / "disagreement_signal.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/disagreement.log")

IN_PQ = D0 / "multi_preds.parquet"
ALGO = ["pred_xgb", "pred_lgb42", "pred_cat"]
SEEDS = ["pred_lgb42", "pred_lgb123", "pred_lgb2024"]
ALLC = ALGO + ["pred_lgb123", "pred_lgb2024"]


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def auc(score, label):
    """Mann-Whitney U 形式的 AUC, 无需 sklearn。返回 (auc, n_pos)。"""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan"), npos
    r = pd.Series(score).rank().values
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg)), npos


def auc_ci(score, label, n_boot=400, seed=0):
    """bootstrap AUC 95% CI。"""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(score))
    out = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        lb = label[s]
        if lb.sum() == 0 or (~lb).sum() == 0:
            continue
        a, _ = auc(score[s], lb)
        if np.isfinite(a):
            out.append(a)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


log("=" * 78)
log("[disagreement_signal] 多模型分歧度 → 长尾/大误差 信号检验 (只读)")
log("=" * 78)

df = pd.read_parquet(IN_PQ)
y = df["y_true"].values.astype(np.float64)
month = df["month"].values
log(f"载入 {IN_PQ.name}  {df.shape}")
log(f"  y 范围 [{y.min():.0f}, {y.max():.0f}]  >10800: {int((y>10800).sum())} 行")

# ---------- 集成与分歧度 ----------
P = df[ALLC].values.astype(np.float64)          # (n, 5)
df["ens_algo"] = df[ALGO].mean(axis=1)
df["ens_seed"] = df[SEEDS].mean(axis=1)
df["ens_all"] = df[ALLC].mean(axis=1)
df["d_algo"] = df[ALGO].std(axis=1, ddof=1)     # 算法分歧
df["d_seed"] = df[SEEDS].std(axis=1, ddof=1)    # 种子分歧 (对照)
df["d_all"] = df[ALLC].std(axis=1, ddof=1)      # 全部
df["r_algo"] = df[ALGO].max(axis=1) - df[ALGO].min(axis=1)   # 极差
df["abs_err"] = (df["ens_all"] - y).abs()

CORE = y <= 10800
log("")
log("[1] 集成基线 RMSE")
log(f"  {'模型':16s} {'core':>10s} {'full':>10s}")
for nm, col in [("XGB", "pred_xgb"), ("LGB42", "pred_lgb42"), ("LGB123", "pred_lgb123"),
                ("LGB2024", "pred_lgb2024"), ("CAT", "pred_cat"),
                ("集成(算法)", "ens_algo"), ("集成(种子)", "ens_seed"),
                ("集成(全部)", "ens_all")]:
    p = df[col].values
    log(f"  {nm:16s} {rmse(p[CORE], y[CORE]):>10.2f} {rmse(p, y):>10.2f}")

# 模型两两相关 (看多样性是否真实存在)
log("")
log("[2] 模型预测两两相关 (Pearson) —— 算法分歧是否真的存在")
cols = ["pred_xgb", "pred_lgb42", "pred_lgb123", "pred_lgb2024", "pred_cat"]
C = df[cols].corr().values
log("        " + "".join(f"{c.replace('pred_',''):>10s}" for c in cols))
for i, c in enumerate(cols):
    log(f"  {c.replace('pred_',''):>6s}" + "".join(f"{C[i,j]:>10.4f}" for j in range(len(cols))))

log("")
log("[3] 两两 RMSE 差 / 预测极差 (秒)")
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        a, b = df[cols[i]].values, df[cols[j]].values
        d = np.abs(a - b)
        log(f"  {cols[i].replace('pred_',''):>7s} vs {cols[j].replace('pred_',''):<7s} "
            f"mean|Δ|={d.mean():7.2f}  p95={np.percentile(d,95):8.2f}  max={d.max():9.2f}")

# ---------- 信号检验 ----------
THRESH = [1800, 3600, 10800]
SCORE_COLS = [("d_algo", "算法分歧"), ("d_seed", "种子分歧(对照)"),
              ("d_all", "全部分歧"), ("r_algo", "算法极差"), ("ens_all", "集成pred(基线)")]


def analyze(mask, tag):
    """在给定行子集上做全部信号检验。"""
    sub = df[mask]
    yy = sub["y_true"].values.astype(np.float64)
    n = len(sub)
    res = {"n": int(n), "n_gt10800": int((yy > 10800).sum()),
           "rmse_ens": round(rmse(sub["ens_all"].values, yy), 2)}
    # B. Spearman(分歧度, |误差|)
    sp = {}
    ae = sub["abs_err"].values
    for c, lab in SCORE_COLS:
        r, pv = spearmanr(sub[c].values, ae)
        sp[lab] = {"rho": round(float(r), 4), "p": float(f"{pv:.3g}")}
    res["spearman_disp_vs_abserr"] = sp
    # C/D. AUC
    au = {}
    for th in THRESH:
        lb = yy > th
        row = {"n_pos": int(lb.sum())}
        for c, lab in SCORE_COLS:
            a, npos = auc(sub[c].values, lb)
            lo, hi = auc_ci(sub[c].values, lb, n_boot=300) if lb.sum() >= 5 else (np.nan, np.nan)
            row[lab] = {"auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)] if np.isfinite(lo) else None}
        au[f"true_gt{th}"] = row
    res["auc"] = au
    return res


log("")
log("=" * 78)
log("[4] Spearman(分歧度, |true - 集成pred|)")
log("=" * 78)
log(f"  {'口径':10s} {'n':>9s} " + "".join(f"{lab:>18s}" for _, lab in SCORE_COLS))
sp_all = {}
for tag, mask in [("全部", np.ones(len(df), bool)),
                  ("Jan", month == 1), ("Jul", month == 7),
                  ("核心", CORE)]:
    sub = df[mask]
    ae = sub["abs_err"].values
    cells = []
    for c, lab in SCORE_COLS:
        r, pv = spearmanr(sub[c].values, ae)
        sp_all.setdefault(tag, {})[lab] = {"rho": round(float(r), 4), "p": float(f"{pv:.3g}")}
        cells.append(f"{r:+.4f}({pv:.1e})")
    log(f"  {tag:10s} {len(sub):>9,d} " + "".join(f"{c:>18s}" for c in cells))

log("")
log("=" * 78)
log("[5] 分歧度识别长尾的 AUC (对照: 集成pred 自身作排序分)")
log("=" * 78)
for th in THRESH:
    log("")
    log(f"  --- true > {th}s ---")
    log(f"  {'口径':8s} {'n_pos':>7s} " + "".join(f"{lab:>16s}" for _, lab in SCORE_COLS))
    for tag, mask in [("全部", np.ones(len(df), bool)), ("Jan", month == 1), ("Jul", month == 7)]:
        sub = df[mask]
        yy = sub["y_true"].values
        lb = yy > th
        cells = []
        for c, lab in SCORE_COLS:
            a, _ = auc(sub[c].values, lb)
            cells.append(f"{a:.4f}" if np.isfinite(a) else "n/a")
        log(f"  {tag:8s} {int(lb.sum()):>7d} " + "".join(f"{c:>16s}" for c in cells))

# ---------- 极端行 vs 正常行 ----------
log("")
log("=" * 78)
log("[6] 极端行 (>10800s) 分歧度分布 vs 正常行")
log("=" * 78)
ext = y > 10800
norm = (y > 0) & (y <= 1800)
mid = (y > 1800) & (y <= 10800)
log(f"  {'组':18s} {'n':>7s} " + "".join(f"{lab:>14s}" for _, lab in SCORE_COLS[:3]))
grp_stats = {}
for gname, m in [("正常 (0,1800]", norm), ("中长 (1800,10800]", mid), ("极端 (>10800]", ext)]:
    sub = df[m]
    cells = []
    st = {}
    for c, lab in SCORE_COLS[:3]:
        v = sub[c].values
        st[lab] = {"median": round(float(np.median(v)), 2),
                   "p75": round(float(np.percentile(v, 75)), 2),
                   "mean": round(float(v.mean()), 2)}
        cells.append(f"{np.median(v):>14.2f}")
    grp_stats[gname] = {"n": int(m.sum()), **st}
    log(f"  {gname:18s} {int(m.sum()):>7d} " + "".join(cells))

# Mann-Whitney: 极端 vs 正常
from scipy.stats import mannwhitneyu
mw = mannwhitneyu(df.loc[ext, "d_algo"].values, df.loc[norm, "d_algo"].values,
                  alternative="greater")
log("")
log(f"  极端 vs 正常 (算法分歧, 单侧 Mann-Whitney): U={mw.statistic:.0f} p={mw.pvalue:.3g}")
log(f"  极端行分歧度中位 / 正常行 = "
    f"{np.median(df.loc[ext,'d_algo'])/np.median(df.loc[norm,'d_algo']):.2f}x")

# ---------- 增量检验: 控制 pred 后 ----------
log("")
log("=" * 78)
log("[7] 增量检验: 在集成pred 分档内, 分歧度是否仍有区分度")
log("=" * 78)
log("  (若分歧度只是 pred 的代理, 档内 AUC 应塌到 0.5)")
sub = df[CORE].copy()
sub["dec"] = pd.qcut(sub["ens_all"], 10, labels=False, duplicates="drop")
log(f"  {'pred十分位':10s} {'n':>8s} {'AUC(分档内)':>12s} {'Spearman':>10s} {'AUC(不分档)':>12s}")
inc = {}
for d in sorted(sub["dec"].unique()):
    m = sub[sub["dec"] == d]
    a, _ = auc(m["d_algo"].values, m["y_true"].values > 1800)
    r, _ = spearmanr(m["d_algo"].values, (m["ens_all"] - m["y_true"]).abs())
    a_glob, _ = auc(sub["d_algo"].values, sub["y_true"].values > 1800)
    inc[f"D{int(d)+1}"] = {"n": int(len(m)), "auc_within": round(a, 4),
                           "rho_within": round(float(r), 4), "auc_global": round(a_glob, 4)}
    log(f"  D{int(d)+1:<9d} {len(m):>8d} {a:>12.4f} {r:>10.4f} {a_glob:>12.4f}")

# ---------- 判定 ----------
res_all = analyze(np.ones(len(df), bool), "全部")
d_jan = analyze(month == 1, "Jan")
d_jul = analyze(month == 7, "Jul")
rho_jan = sp_all["Jan"]["算法分歧"]["rho"]
rho_jul = sp_all["Jul"]["算法分歧"]["rho"]
auc_jan = d_jan["auc"]["true_gt3600"]["算法分歧"]["auc"]
auc_jul = d_jul["auc"]["true_gt3600"]["算法分歧"]["auc"]
auc_base_jan = d_jan["auc"]["true_gt3600"]["集成pred(基线)"]["auc"]
auc_base_jul = d_jul["auc"]["true_gt3600"]["集成pred(基线)"]["auc"]

# 门槛: |rho|>=0.05 视为有意义; AUC>=0.60 视为有识别力
sig_rho = abs(rho_jan) >= 0.05 and abs(rho_jul) >= 0.05 and (rho_jan > 0) == (rho_jul > 0)
sig_auc = auc_jan >= 0.60 and auc_jul >= 0.60
beats_base = (auc_jan + auc_jul) / 2 > (auc_base_jan + auc_base_jul) / 2
verdict = {
    "rho_jan": rho_jan, "rho_jul": rho_jul, "rho_same_sign": (rho_jan > 0) == (rho_jul > 0),
    "auc3600_jan": auc_jan, "auc3600_jul": auc_jul,
    "auc3600_base_jan": auc_base_jan, "auc3600_base_jul": auc_base_jul,
    "disp_beats_single_pred": bool(beats_base),
    "signal_significant": bool(sig_rho and sig_auc),
}

log("")
log("=" * 78)
log("[判定]")
log("-" * 78)
log(f"  Spearman(算法分歧, |err|):  Jan={rho_jan:+.4f}  Jul={rho_jul:+.4f}  "
    f"同向={(rho_jan>0)==(rho_jul>0)}  达标(|rho|>=0.05)={sig_rho}")
log(f"  AUC(识别 true>3600):       Jan={auc_jan:.4f}  Jul={auc_jul:.4f}  达标(>=0.60)={sig_auc}")
log(f"  单 pred 基线 AUC:          Jan={auc_base_jan:.4f}  Jul={auc_base_jul:.4f}")
log(f"  分歧度是否强于单 pred:      {beats_base}")
log("")
if verdict["signal_significant"]:
    log("  >>> 分歧度信号显著: 下一步可做多模型组合的二阶 stacking")
elif sig_rho or sig_auc:
    log("  >>> 分歧度有弱信号但未达门槛: 可作为二阶模型的辅助特征, 不宜单独立项")
else:
    log("  >>> 分歧度信号弱: 长尾条件方差真不可约, 这条路关闭")

RESULT = {
    "task": "multi_model_disagreement_signal",
    "input": IN_PQ.name,
    "integration_baseline": {
        "per_model_rmse": {c.replace("pred_", ""): {
            "core": round(rmse(df[c].values[CORE], y[CORE]), 2),
            "full": round(rmse(df[c].values, y), 2)} for c in cols},
        "ens_algo_core": round(rmse(df["ens_algo"].values[CORE], y[CORE]), 2),
        "ens_all_core": round(rmse(df["ens_all"].values[CORE], y[CORE]), 2),
        "ens_all_full": round(rmse(df["ens_all"].values, y), 2),
    },
    "prediction_corr": {c.replace("pred_", ""): {c2.replace("pred_", ""): round(float(C[i, j]), 4)
                                                 for j, c2 in enumerate(cols)}
                        for i, c in enumerate(cols)},
    "spearman_disp_vs_abserr": sp_all,
    "auc": {"all": res_all["auc"], "Jan": d_jan["auc"], "Jul": d_jul["auc"]},
    "extreme_vs_normal": {"groups": grp_stats,
                          "mw_u_algo_extreme_gt_normal": float(mw.statistic),
                          "mw_p": float(f"{mw.pvalue:.3g}"),
                          "median_ratio": round(float(np.median(df.loc[ext, "d_algo"]) /
                                                      np.median(df.loc[norm, "d_algo"])), 2)},
    "incremental_within_decile": inc,
    "verdict": verdict,
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}")
