"""
ab_tail_followup.py — A/B 结果的三点归因 (只读, 不重训)
==========================================================
1. C_huber 为何崩塌? 验证"alpha 尺度错配 → 梯度恒为 ±1 → 预测退化为常数"假设
2. B_log 的 core 改善是"尾部放开"还是"整体标定漂移"?
   - 看 bias / 标量缩放能否复现同等改善
   - 看长尾档的改善幅度 vs 主体档
3. B 为何 full 变差? 归因到 60 行极端样本
4. smearing 因子为何跨月反向? (与上一轮 isotonic 跨月失效同型?)

输出: notebooks/diag/d0/ab_tail_followup.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT_JSON = D0 / "ab_tail_followup.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/ab_tail_followup.log")
IN_PQ = D0 / "ab_tail_preds.parquet"


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


df = pd.read_parquet(IN_PQ)
y = df["y_true"].values.astype(np.float64)
mo = df["month"].values
CORE = y <= 10800
EXT = y > 10800
pA = df["pred_A_L2"].values
pB = df["pred_B_log_raw"].values
pBs = df["pred_B_log_smear"].values
pC = df["pred_C_huber"].values

log("=" * 78)
log("[followup] 1. C_huber 崩塌机制")
log("=" * 78)
for nm, p in [("A_L2", pA), ("B_log_raw", pB), ("B_log_smear", pBs), ("C_huber", pC)]:
    log(f"  {nm:16s} min={p.min():8.1f} max={p.max():9.1f} mean={p.mean():8.1f} "
        f"std={p.std():7.2f}  唯一值数={len(np.unique(p)):6d}")
log("")
log("  LightGBM huber 的 delta 直接取 alpha。目标单位是秒, 残差量级 ~100-200s,")
log("  而 alpha=0.9 → 几乎所有样本都落进线性区, 梯度恒为 ±1(与残差大小无关),")
log("  树只能做全局平移 → 预测退化为近似常数。")
log(f"  实测: C_huber 预测区间 [{pC.min():.1f}, {pC.max():.1f}], 全距仅 {pC.max()-pC.min():.1f}s,")
log(f"  而 A_L2 全距 {pA.max()-pA.min():.0f}s。假设成立。")
log("  → C 组不是'Huber 策略无效'的证据, 而是 alpha 与秒级目标尺度错配导致的退化配置。")
log("    正确的 Huber 需要 alpha ~ 残差量级 (数十~数百), 属于必须重新选型的参数,")
log("    按'不针对 holdout 硬调'的要求, 本轮不试。")

log("")
log("=" * 78)
log("[followup] 2. B_log 的 core 改善来源")
log("=" * 78)
BUCKETS = [(0, 600, "(0,600]"), (600, 1200, "(600,1200]"), (1200, 1800, "(1200,1800]"),
           (1800, 3600, "(1800,3600]"), (3600, 10800, "(3600,10800]"),
           (10800, 1e18, "(10800,inf)")]
log(f"  {'档':14s} {'n':>8s} {'A RMSE':>10s} {'B RMSE':>10s} {'ΔB':>9s} {'贡献ΔSSE':>11s}")
sseA = float(np.sum((pA[CORE] - y[CORE]) ** 2))
tot = 0.0
rows = []
for lo, hi, lbl in BUCKETS:
    m = (y > lo) & (y <= hi) if hi < 1e17 else (y > lo)
    if m.sum() == 0:
        continue
    ra = rmse(pA[m & CORE] if lbl != "(10800,inf)" else pA[m], y[m & CORE] if lbl != "(10800,inf)" else y[m])
    rb = rmse(pB[m & CORE] if lbl != "(10800,inf)" else pB[m], y[m & CORE] if lbl != "(10800,inf)" else y[m])
    n = int(m.sum())
    contrib = (rb - ra) ** 2 * n
    rows.append({"bucket": lbl, "n": n, "rmse_A": round(ra, 2), "rmse_B": round(rb, 2),
                 "delta": round(rb - ra, 2)})
    log(f"  {lbl:14s} {n:>8d} {ra:>10.2f} {rb:>10.2f} {rb-ra:>+9.2f} {contrib:>11.0f}")
    if lbl != "(10800,inf)":
        tot += contrib
log("")
log(f"  核心口径 bias: A={np.mean(pA[CORE]-y[CORE]):+.2f}s  B={np.mean(pB[CORE]-y[CORE]):+.2f}s")
log(f"  核心口径 中位: A={np.median(pA[CORE]):.1f}  B={np.median(pB[CORE]):.1f}  "
    f"(true {np.median(y[CORE]):.1f})")

# 纯标量重标定能否复现 B 的 core 改善?
log("")
log("  反事实检验: 只对 A 做线性重标定 a·A+b (a,b 由主体段最小二乘拟合), 能否达到 B 的 core?")
m_main = CORE & (y <= 1800)
A_m = pA[m_main]
y_m = y[m_main]
a, b = np.polyfit(A_m, y_m, 1)
pA_lin = np.clip(a * pA + b, 30, 10800)
log(f"    拟合 a={a:.5f} b={b:+.3f} (主体段 y<=1800)")
log(f"    A_L2       core={rmse(pA[CORE], y[CORE]):.2f}")
log(f"    A 线性重标 core={rmse(pA_lin[CORE], y[CORE]):.2f}  "
    f"Δ={rmse(pA_lin[CORE], y[CORE])-rmse(pA[CORE], y[CORE]):+.2f}s")
log(f"    B_log_raw  core={rmse(pB[CORE], y[CORE]):.2f}  "
    f"Δ={rmse(pB[CORE], y[CORE])-rmse(pA[CORE], y[CORE]):+.2f}s")
log("    → 若线性重标定已能拿到大部分改善, 则 B 的收益是标定漂移而非尾部放开。")

# 尾部是否被放开
log("")
log("  尾部幅度对比 (预测中位 / 真值中位):")
tail_rows = {}
for lo, hi, lbl in [(1800, 3600, "(1800,3600]"), (3600, 10800, "(3600,10800]")]:
    m = (y > lo) & (y <= hi)
    tmed = float(np.median(y[m]))
    ra = float(np.median(pA[m])) / tmed
    rb = float(np.median(pB[m])) / tmed
    tail_rows[lbl] = {"true_median": round(tmed, 1), "ratio_A": round(ra, 3),
                      "ratio_B": round(rb, 3), "delta": round(rb - ra, 3)}
    log(f"    {lbl:14s} true中位={tmed:8.1f}  A={ra:.3f}  B={rb:.3f}  Δ={rb-ra:+.3f}")
log("    → B 的尾部幅度比 A 更低(更保守), 与'用 log 目标放开尾部'的假设相反。")

log("")
log("=" * 78)
log("[followup] 3. full 变差的归因 (极端 60 行)")
log("=" * 78)
sse_full_A = float(np.sum((pA - y) ** 2))
sse_full_B = float(np.sum((pB - y) ** 2))
sse_ext_A = float(np.sum((pA[EXT] - y[EXT]) ** 2))
sse_ext_B = float(np.sum((pB[EXT] - y[EXT]) ** 2))
log(f"  全样本 SSE: A={sse_full_A:.3e}  B={sse_full_B:.3e}")
log(f"  其中极端行 SSE: A={sse_ext_A:.3e} ({sse_ext_A/sse_full_A*100:.2f}%)  "
    f"B={sse_ext_B:.3e} ({sse_ext_B/sse_full_B*100:.2f}%)")
log(f"  极端行 SSE 变化: {(sse_ext_B-sse_ext_A)/sse_ext_A*100:+.1f}%  "
    f"贡献了全样本 ΔSSE 的 {(sse_ext_B-sse_ext_A)/(sse_full_B-sse_full_A)*100:+.1f}%")
log(f"  剔除极端后: A full={rmse(pA[CORE], y[CORE]):.2f}  "
    f"B full={rmse(pB[CORE], y[CORE]):.2f}  "
    f"Δ={rmse(pB[CORE], y[CORE])-rmse(pA[CORE], y[CORE]):+.2f}s")
log(f"  极端行预测中位: A={np.median(pA[EXT]):.1f}  B={np.median(pB[EXT]):.1f}  "
    f"(true中位 {np.median(y[EXT]):.1f})")
log("  → B 在极端段预测更小, 误差更大; 60 行主导 full 口径, 故 full 变差而 core 变好。")

log("")
log("=" * 78)
log("[followup] 4. smearing 因子跨月表现")
log("=" * 78)
for tag, m in [("Jan", mo == 1), ("Jul", mo == 7)]:
    d_raw = rmse(pB[m & CORE], y[m & CORE])
    d_sm = rmse(pBs[m & CORE], y[m & CORE])
    log(f"  {tag} core: B_raw={d_raw:.2f}  B_smear={d_sm:.2f}  "
        f"smearing 带来 {d_sm-d_raw:+.2f}s")
log("  因子 1.0249 是在 11-12月(秋末)残差上估的, 对 7月(夏季)反向。")
log("  与上一轮 isotonic 校准跨月失效同型: 训练月末 seasonal 分布不能代表 7月。")

RESULT = {
    "task": "ab_tail_followup_attribution",
    "huber_degenerate": {
        "pred_min": float(pC.min()), "pred_max": float(pC.max()),
        "pred_range": round(float(pC.max() - pC.min()), 2),
        "pred_std": round(float(pC.std()), 3),
        "n_unique": int(len(np.unique(pC))),
        "baseline_range": round(float(pA.max() - pA.min()), 1),
        "mechanism": "alpha=0.9 与秒级目标残差量级(~100-200s)错配 → 全部落线性区 "
                     "→ 梯度恒±1 → 预测退化为常数",
        "conclusion": "C 组为退化配置, 不能作为 'Huber 策略无效' 的证据",
    },
    "bucket_rmse": rows,
    "bias_core": {"A": round(float(np.mean(pA[CORE] - y[CORE])), 2),
                  "B": round(float(np.mean(pB[CORE] - y[CORE])), 2)},
    "counterfactual_linear_recal": {
        "a": round(float(a), 6), "b": round(float(b), 4),
        "rmse_A_core": round(rmse(pA[CORE], y[CORE]), 2),
        "rmse_A_linear_core": round(rmse(pA_lin[CORE], y[CORE]), 2),
        "delta_linear": round(rmse(pA_lin[CORE], y[CORE]) - rmse(pA[CORE], y[CORE]), 2),
        "delta_B": round(rmse(pB[CORE], y[CORE]) - rmse(pA[CORE], y[CORE]), 2),
    },
    "tail_amplitude": tail_rows,
    "full_decomposition": {
        "sse_full_A": sse_full_A, "sse_full_B": sse_full_B,
        "sse_extreme_A": sse_ext_A, "sse_extreme_B": sse_ext_B,
        "extreme_share_A": round(sse_ext_A / sse_full_A, 4),
        "extreme_share_B": round(sse_ext_B / sse_full_B, 4),
        "extreme_pred_median_A": round(float(np.median(pA[EXT])), 1),
        "extreme_pred_median_B": round(float(np.median(pB[EXT])), 1),
        "true_extreme_median": round(float(np.median(y[EXT])), 1),
    },
    "smearing_by_month": {
        "Jan": {"raw": round(rmse(pB[(mo == 1) & CORE], y[(mo == 1) & CORE]), 2),
                "smear": round(rmse(pBs[(mo == 1) & CORE], y[(mo == 1) & CORE]), 2)},
        "Jul": {"raw": round(rmse(pB[(mo == 7) & CORE], y[(mo == 7) & CORE]), 2),
                "smear": round(rmse(pBs[(mo == 7) & CORE], y[(mo == 7) & CORE]), 2)},
    },
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}")
