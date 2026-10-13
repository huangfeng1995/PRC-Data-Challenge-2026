"""
disagreement_refine.py — 剔除弱 XGB 后的分歧度复检 + 增量价值判定
==========================================================================
disagreement_signal.py 的两个发现需要跟进:

1. XGB 明显偏弱 (holdout core 393 vs LGB 311 / CAT 317), 且 xgb 与其它库
   的 mean|Δ| ≈ 160s, 而 lgb-lgb ≈ 34s、lgb-cat ≈ 65s。
   → "算法分歧" 很可能主要在度量"XGB 有多差", 而不是" epistemic 不确定性"。
   证据: 种子分歧 (纯 3 个 LGB) 的 AUC 反而高于算法分歧。

2. 单 pred 作排序分的 AUC 已经很�� (>3600 达 0.96), 高于任何分歧度。
   → 必须看"增量": 在 pred 分档内部, 分歧度还有没有区分度。

本脚本:
  A. 定义 4 种分歧度, 全部剔除 xgb:
     d_lgbs   = std(lgb42, lgb123, lgb2024)              种子分歧
     d_lgbcat = std(lgb42, lgb123, lgb2024, cat)         去 xgb 的类分歧
     d_all    = std(全部 5 个)
     d_algo   = std(xgb, lgb42, cat)                     含 xgb (对照)
  B. 各分歧度 vs 单 pred 的 AUC 逐档对比
  C. 增量价值: pred 十分位内的 AUC + Spearman, 以及"偏残差相关"
     (控制 pred 后, 分歧度与 |err| 的相关是否仍在)
  D. 关键判据: 分歧度 AUC 与 pred AUC 的差 ΔAUC, 以及在 pred 分档内
     是否稳定 > 0.5
  E. Jan / Jul 同向性

输出: notebooks/diag/d0/disagreement_refine.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT_JSON = D0 / "disagreement_refine.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/disagreement_refine.log")

IN_PQ = D0 / "multi_preds.parquet"
LGBS = ["pred_lgb42", "pred_lgb123", "pred_lgb2024"]
NOX = LGBS + ["pred_cat"]
ALLC = LGBS + ["pred_xgb", "pred_cat"]


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def auc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(score).rank().values
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


log("=" * 78)
log("[disagreement_refine] 剔除弱 XGB 后的分歧度复检")
log("=" * 78)

df = pd.read_parquet(IN_PQ)
y = df["y_true"].values.astype(np.float64)
mo = df["month"].values
CORE = y <= 10800

# 只用 3 个 LGB + CAT 的集成作为"最佳可用单 pred"参照
df["ens_nox"] = df[NOX].mean(axis=1)
df["d_lgbs"] = df[LGBS].std(axis=1, ddof=1)
df["d_lgbcat"] = df[NOX].std(axis=1, ddof=1)
df["d_all"] = df[ALLC].std(axis=1, ddof=1)
df["d_algo"] = df[["pred_xgb", "pred_lgb42", "pred_cat"]].std(axis=1, ddof=1)
df["abs_err"] = (df["ens_nox"] - y).abs()

DISP = [("d_algo", "算法分歧(含XGB)"), ("d_lgbs", "种子分歧(3×LGB)"),
        ("d_lgbcat", "类分歧(LGB×3+CAT, 无XGB)"), ("d_all", "全部分歧(5模型)")]

log("")
log("[1] 参照集成 (剔除 XGB) RMSE")
log(f"  ens_nox  core={rmse(df['ens_nox'].values[CORE], y[CORE]):.2f}  "
    f"full={rmse(df['ens_nox'].values, y):.2f}")
log(f"  ens_all  core={rmse(df[ALLC].mean(axis=1).values[CORE], y[CORE]):.2f}  "
    f"full={rmse(df[ALLC].mean(axis=1).values, y):.2f}")
log("  各模型 core RMSE:")
for c in LGBS + ["pred_cat", "pred_xgb"]:
    log(f"    {c.replace('pred_',''):>8s} {rmse(df[c].values[CORE], y[CORE]):>8.2f}")

log("")
log("=" * 78)
log("[2] Spearman(分歧度, |true - ens_nox|)")
log("=" * 78)
log(f"  {'口径':8s} {'n':>9s} " + "".join(f"{lab:>26s}" for _, lab in DISP))
sp_tab = {}
for tag, mask in [("全部", np.ones(len(df), bool)), ("Jan", mo == 1), ("Jul", mo == 7)]:
    sub = df[mask]
    ae = sub["abs_err"].values
    cells, rec = [], {}
    for c, lab in DISP:
        r, p = spearmanr(sub[c].values, ae)
        rec[lab] = {"rho": round(float(r), 4), "p": float(f"{p:.3g}")}
        cells.append(f"{r:+.4f}")
    sp_tab[tag] = rec
    log(f"  {tag:8s} {len(sub):>9,d} " + "".join(f"{c:>26s}" for c in cells))
# 参照: 单 pred 自身
sub = df
r_pred, _ = spearmanr(df["ens_nox"].values, df["abs_err"].values)
log(f"  (参照) Spearman(ens_nox, |err|) 全部 = {r_pred:+.4f}  ← 单 pred 自身远高于任何分歧度")

log("")
log("=" * 78)
log("[3] 长尾识别 AUC: 分歧度 vs 单 pred (同一目标, 直接可比)")
log("=" * 78)
auc_tab = {}
for th in [1800, 3600, 10800]:
    log("")
    log(f"  --- true > {th}s ---")
    log(f"  {'口径':6s} {'n_pos':>7s} " + "".join(f"{lab:>26s}" for _, lab in DISP)
        + f"{'ens_nox(基线)':>16s}")
    auc_tab[th] = {}
    for tag, mask in [("全部", np.ones(len(df), bool)), ("Jan", mo == 1), ("Jul", mo == 7)]:
        s = df[mask]
        lb = s["y_true"].values > th
        cells, rec = [], {}
        for c, lab in DISP:
            a = auc(s[c].values, lb)
            rec[lab] = round(a, 4)
            cells.append(f"{a:.4f}")
        a_pred = auc(s["ens_nox"].values, lb)
        rec["ens_nox(基线)"] = round(a_pred, 4)
        auc_tab[th][tag] = rec
        log(f"  {tag:6s} {int(lb.sum()):>7d} " + "".join(f"{c:>26s}" for c in cells)
            + f"{a_pred:>16.4f}")

log("")
log("=" * 78)
log("[4] 增量价值: pred 十分位内部分歧度的区分度")
log("=" * 78)
log("  若分歧度只是 pred 的代理, 档内 AUC 会塌向 0.5")
log("")
inc = {}
for th in [1800, 3600]:
    log(f"  --- 目标 true > {th}s, 按 ens_nox 十分位分层 ---")
    log(f"  {'D':>4s} {'n':>8s} " + "".join(f"{lab:>26s}" for _, lab in DISP)
        + f"{'档内Spearman(无XGB)':>20s}")
    s = df[CORE].copy()
    s["dec"] = pd.qcut(s["ens_nox"], 10, labels=False, duplicates="drop")
    inc[th] = {}
    for d in sorted(s["dec"].unique()):
        m = s[s["dec"] == d]
        lb = m["y_true"].values > th
        cells, rec = [], {}
        for c, lab in DISP:
            a = auc(m[c].values, lb)
            rec[lab] = round(a, 4)
            cells.append(f"{a:.4f}" if np.isfinite(a) else "n/a")
        rr, _ = spearmanr(m["d_lgbcat"].values,
                          (m["ens_nox"] - m["y_true"]).abs())
        rec["spearman_d_lgbcat"] = round(float(rr), 4)
        rec["n"] = int(len(m))
        inc[th][f"D{int(d)+1}"] = rec
        log(f"  D{int(d)+1:<3d} {len(m):>8d} " + "".join(f"{c:>26s}" for c in cells)
            + f"{rr:>+20.4f}")
    # 档内 AUC 的加权平均 (按 n)
    dec_keys = [k for k in inc[th] if not k.startswith("_")]
    for lab in [l for _, l in DISP]:
        vals = [(inc[th][k][lab], inc[th][k]["n"]) for k in dec_keys]
        ok = [(a, n) for a, n in vals if np.isfinite(a)]
        tot = sum(n for _, n in ok)
        w = sum(a * n for a, n in ok) / tot if tot else float("nan")
        inc[th].setdefault("_weighted_auc", {})[lab] = round(w, 4)
    log(f"  → 档内加权平均 AUC: " + "  ".join(
        f"{l}={inc[th]['_weighted_auc'][l]:.4f}" for _, l in DISP))

log("")
log("=" * 78)
log("[5] 极端行 vs 正常行 (无 XGB 分歧度)")
log("=" * 78)
ext, norm, mid = y > 10800, (y > 0) & (y <= 1800), (y > 1800) & (y <= 10800)
grp = {}
log(f"  {'组':18s} {'n':>7s} " + "".join(f"{lab:>26s}" for _, lab in DISP))
for nm, m in [("正常 (0,1800]", norm), ("中长 (1800,10800]", mid), ("极端 (>10800]", ext)]:
    s = df[m]
    cells, rec = [], {}
    for c, lab in DISP:
        v = s[c].values
        rec[lab] = {"median": round(float(np.median(v)), 2),
                    "p75": round(float(np.percentile(v, 75)), 2)}
        cells.append(f"{np.median(v):>26.2f}")
    rec["n"] = int(m.sum())
    grp[nm] = rec
    log(f"  {nm:18s} {int(m.sum()):>7d} " + "".join(cells))

mws = {}
for c, lab in DISP:
    mw = mannwhitneyu(df.loc[ext, c].values, df.loc[norm, c].values, alternative="greater")
    ratio = float(np.median(df.loc[ext, c]) / np.median(df.loc[norm, c]))
    mws[lab] = {"U": float(mw.statistic), "p": float(f"{mw.pvalue:.3g}"),
                "median_ratio": round(ratio, 2)}
    log(f"  {lab:26s} 极端/正常 中位比 = {ratio:6.2f}x  (MWU p={mw.pvalue:.2e})")

# ---------- 判定 ----------
best_disp = "类分歧(LGB×3+CAT, 无XGB)"
a36_jan = auc_tab[3600]["Jan"][best_disp]
a36_jul = auc_tab[3600]["Jul"][best_disp]
p36_jan = auc_tab[3600]["Jan"]["ens_nox(基线)"]
p36_jul = auc_tab[3600]["Jul"]["ens_nox(基线)"]
w36 = inc[3600]["_weighted_auc"][best_disp]
rho_j = sp_tab["Jan"][best_disp]["rho"]
rho_l = sp_tab["Jul"][best_disp]["rho"]
mean_disp_auc = (a36_jan + a36_jul) / 2
mean_pred_auc = (p36_jan + p36_jul) / 2

verdict = {
    "best_dispersion": best_disp,
    "spearman_jan": rho_j, "spearman_jul": rho_l,
    "spearman_same_sign": (rho_j > 0) == (rho_l > 0),
    "auc3600_disp_jan": a36_jan, "auc3600_disp_jul": a36_jul,
    "auc3600_pred_jan": p36_jan, "auc3600_pred_jul": p36_jul,
    "disp_minus_pred_auc": round(mean_disp_auc - mean_pred_auc, 4),
    "within_decile_weighted_auc_3600": w36,
    "disp_above_chance_within_decile": bool(w36 > 0.55),
    "disp_beats_single_pred": bool(mean_disp_auc > mean_pred_auc),
}
log("")
log("=" * 78)
log("[判定]")
log("-" * 78)
log(f"  最优分歧度定义: {best_disp}")
log(f"  Spearman vs |err|: Jan={rho_j:+.4f} Jul={rho_l:+.4f} 同向={(rho_j>0)==(rho_l>0)}")
log(f"  AUC(>3600) 分歧度: Jan={a36_jan:.4f} Jul={a36_jul:.4f} (均值 {mean_disp_auc:.4f})")
log(f"  AUC(>3600) 单pred: Jan={p36_jan:.4f} Jul={p36_jul:.4f} (均值 {mean_pred_auc:.4f})")
log(f"  ΔAUC (分歧度 - 单pred) = {verdict['disp_minus_pred_auc']:+.4f}")
log(f"  pred 分档内加权 AUC(>3600) = {w36:.4f}  (>0.55 = 有增量信息)")
log("")
if verdict["disp_beats_single_pred"]:
    log("  >>> 分歧度比单 pred 更强: 值得做二阶 stacking")
elif verdict["disp_above_chance_within_decile"]:
    log("  >>> 分歧度弱于单 pred, 但在 pred 分档内仍有增量信息 (AUC>0.55):")
    log("      可作二阶模型的辅助特征, 不能替代 pred; 预期收益有限")
else:
    log("  >>> 分歧度在控制 pred 后无增量: 这条路关闭")

RESULT = {
    "task": "disagreement_refine_no_xgb",
    "motivation": "XGB core 393 显著弱于 LGB 311/CAT 317, 且 xgb 与他库 mean|Δ|≈160s "
                  "远大于 lgb-lgb 34s / lgb-cat 65s, 怀疑'算法分歧'主要在度量 XGB 有多差",
    "reference_ens": "ens_nox = mean(LGB42,LGB123,LGB2024,CAT)",
    "reference_rmse": {
        "ens_nox_core": round(rmse(df["ens_nox"].values[CORE], y[CORE]), 2),
        "ens_nox_full": round(rmse(df["ens_nox"].values, y), 2),
        "ens_all5_core": round(rmse(df[ALLC].mean(axis=1).values[CORE], y[CORE]), 2),
    },
    "per_model_core_rmse": {c.replace("pred_", ""): round(rmse(df[c].values[CORE], y[CORE]), 2)
                            for c in LGBS + ["pred_cat", "pred_xgb"]},
    "spearman": sp_tab,
    "spearman_single_pred_reference": round(float(r_pred), 4),
    "auc": auc_tab,
    "incremental_within_decile": inc,
    "extreme_vs_normal": {"groups": grp, "mannwhitney": mws},
    "verdict": verdict,
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}")
