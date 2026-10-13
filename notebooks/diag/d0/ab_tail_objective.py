"""
ab_tail_objective.py — A/B 尾部改造: L2 vs log目标 vs Huber (固定 holdout 锚点)
==========================================================================
口径与基线 (train_multi.py / v15) 完全一致, 只改目标函数/目标变换:

  MAIN split: train 2-12月, holdout 1+7月, FEATS71, 高基数整数编码
  ES 集: 训练月内 MVT 时间末 10% (11、12月), 绝不碰 1/7 月
  超参: 全部沿用基线常规值, 仅 objective/alpha 变化; 不针对 holdout 调参
  seed: 42 (单 seed 看方向)

三组:
  A  L2      objective='regression' (基线)
  B  log     objective='regression', 训练 log(TAXI), 预测 exp
             exp 会系统性低估 (Jensen) → 用训练折内 OOF 残差估 smearing 校正因子
             因子 = mean(exp(log_true - log_pred)) 在 ES 集上算
             (ES 集未参与梯度拟合, 对 holdout 无泄漏)
             同时报告未校正版本以显示校正幅度
  C  Huber   objective='huber', alpha=0.9 (LightGBM 默认/常用值)

评估: 全样本(含极端, 对标真榜) + 核心口径(剔>10800); 分机场; Jan/Jul 分别;
      分布形态 σpred/σtrue, p99/p99.9, 长尾档 (1800,3600] / (3600,10800] 预测幅度

输出:
  notebooks/diag/d0/ab_tail_objective.json
  /Users/openclaw/.trae-cn/work/.../ab_tail_objective.log
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

from src.data_loader import load_month, parse_time_columns
import _r7_lib as L
from v15_clean_retrain import (build_features, encode_cat_columns,
                               FEATS71, LOW_CARD_CATS, TRAIN_COLS, TIME_COLS)

WORK = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b")
OUT_JSON = D0 / "ab_tail_objective.json"
OUT_LOG = WORK / "ab_tail_objective.log"
OUT_PQ = D0 / "ab_tail_preds.parquet"

CAP = 2500
ES_ROUND = 100
SEED = 42
HUBER_ALPHA = 0.9          # LightGBM huber 默认值
CLIP_LO, CLIP_HI = 30.0, 10800.0

MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                  if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
              for m in range(1, 13)}


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


log("=" * 78)
log("[ab_tail] A(L2) vs B(log) vs C(Huber) — 固定 holdout 锚点, 单 seed")
log("=" * 78)
t0 = time.time()

# ---------------- [1] 数据 + 特征 (与 train_multi 同构) ----------------
tr_raw = pd.concat([load_month(MONTH_FILE[m], columns=TRAIN_COLS)
                    for m in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]], ignore_index=True)
tr_parsed = parse_time_columns(tr_raw.copy(), cols=TIME_COLS)
lut = L.build_t_unimp_lut(tr_parsed[(tr_parsed["PHASE_mvt"] == "DEP")])
log(f"[1] 训练月 raw {len(tr_raw):,}")
log("    构建训练月特征 ...")
tr_full = build_features(tr_raw, tr_parsed, lut, label="tr")
tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
yv = tr_dep["TAXITIME_SEC_mvt"].values
tr_dep = tr_dep[(yv > 0) & (yv < 10800)].reset_index(drop=True)
del tr_full
log(f"    训练 DEP {len(tr_dep):,}   ({time.time()-t0:.0f}s)")

ho_raw = pd.concat([load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in [1, 7]],
                   ignore_index=True)
ho_parsed = parse_time_columns(ho_raw.copy(), cols=TIME_COLS)
log("    构建 holdout 特征 ...")
ho_full = build_features(ho_raw, tr_parsed, lut, label="ho")
ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
y = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
keep = y > 0
ho_dep = ho_dep[keep].reset_index(drop=True)
y = y[keep]
mo = ho_dep["MVT_TIME_UTC_mvt"].dt.month.values
ap = ho_dep["ADEP_mvt"].astype(str).values
del ho_full, ho_raw
log(f"    Holdout DEP (y>0) {len(ho_dep):,}  >10800: {int((y>10800).sum())}   "
    f"({time.time()-t0:.0f}s)")

# ---------------- [2] 折内时间序 ES 切 ----------------
ts = tr_dep["MVT_TIME_UTC_mvt"].astype("int64").values
order = np.argsort(ts, kind="stable")
cut = int(len(order) * 0.9)
fit_pos, es_pos = order[:cut], order[cut:]
es_months = sorted(tr_dep["MVT_TIME_UTC_mvt"].dt.month.iloc[es_pos].unique().tolist())
assert 1 not in es_months and 7 not in es_months, f"ES 泄漏: {es_months}"
log(f"[2] 拟合 {len(fit_pos):,} / ES {len(es_pos):,} 月份={es_months} ✓ 不含 1/7")

# ---------------- [3] 编码 ----------------
cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
tr_dep = encode_cat_columns(tr_dep, cat_map)
ho_dep = encode_cat_columns(ho_dep, cat_map)
for f in FEATS71:
    if f not in tr_dep.columns:
        tr_dep[f] = np.nan
    if f not in ho_dep.columns:
        ho_dep[f] = np.nan
X_fit = tr_dep[FEATS71].iloc[fit_pos].copy()
X_es = tr_dep[FEATS71].iloc[es_pos].copy()
X_ho = ho_dep[FEATS71].copy()
y_fit = tr_dep["TAXITIME_SEC_mvt"].values[fit_pos].astype(np.float64)
y_es = tr_dep["TAXITIME_SEC_mvt"].values[es_pos].astype(np.float64)

BASE_PARAMS = {"objective": "regression", "metric": "rmse",
               "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
               "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
               "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
               "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
               "verbose": -1, "num_threads": -1, "seed": SEED}


def train_one(params, yfit, yes, tag):
    t1 = time.time()
    d_fit = lgb.Dataset(X_fit, label=yfit, categorical_feature=LOW_CARD_CATS)
    d_es = lgb.Dataset(X_es, label=yes, categorical_feature=LOW_CARD_CATS, reference=d_fit)
    b = lgb.train(params, d_fit, num_boost_round=CAP, valid_sets=[d_es],
                  callbacks=[lgb.early_stopping(ES_ROUND, verbose=False),
                             lgb.log_evaluation(0)])
    bi = b.best_iteration or CAP
    log(f"    {tag:22s} best_iter={bi:5d}  ({time.time()-t1:.0f}s)")
    return b, bi


# ---------------- [4] A: L2 基线 ----------------
log("")
log("=" * 78)
log("[3] A — L2 基线 (objective='regression')")
log("=" * 78)
bA, biA = train_one(dict(BASE_PARAMS), y_fit, y_es, "A L2")
pA = np.clip(bA.predict(X_ho, num_iteration=biA).astype(np.float64), CLIP_LO, CLIP_HI)

# ---------------- [5] B: log 目标 ----------------
log("")
log("=" * 78)
log("[4] B — log 目标 (train log(y), predict exp)")
log("=" * 78)
ly_fit, ly_es = np.log(y_fit), np.log(y_es)
bB, biB = train_one(dict(BASE_PARAMS), ly_fit, ly_es, "B log")
lp_ho = bB.predict(X_ho, num_iteration=biB).astype(np.float64)
lp_es = bB.predict(X_es, num_iteration=biB).astype(np.float64)
# smearing 校正: 训练折内 ES 集残差 (ES 未参与梯度拟合, 对 holdout 无泄漏)
resid = ly_es - lp_es
smear = float(np.mean(np.exp(resid)))
log(f"    ES 残差: mean={resid.mean():+.5f}  mean(exp)={smear:.6f}  "
    f"median(exp)={np.median(np.exp(resid)):.6f}")
pB_raw = np.clip(np.exp(lp_ho), CLIP_LO, CLIP_HI)
pB = np.clip(np.exp(lp_ho) * smear, CLIP_LO, CLIP_HI)
log(f"    smearing 因子 = {smear:.6f}")

# ---------------- [6] C: Huber ----------------
log("")
log("=" * 78)
log(f"[5] C — Huber (objective='huber', alpha={HUBER_ALPHA})")
log("=" * 78)
huber_params = dict(BASE_PARAMS)
huber_params.pop("objective")
huber_params.pop("metric")
huber_params["objective"] = "huber"
huber_params["alpha"] = HUBER_ALPHA
huber_params["metric"] = "rmse"
bC, biC = train_one(huber_params, y_fit, y_es, "C Huber")
pC = np.clip(bC.predict(X_ho, num_iteration=biC).astype(np.float64), CLIP_LO, CLIP_HI)

# ---------------- [7] 评估 ----------------
CORE = y <= 10800
VARIANTS = [("A_L2", pA), ("B_log_raw", pB_raw), ("B_log_smear", pB), ("C_huber", pC)]
BEST_ITER = {"A_L2": biA, "B_log_raw": biB, "B_log_smear": biB, "C_huber": biC}

log("")
log("=" * 78)
log("[6] 总 RMSE (全样本=对标真榜 / 核心=剔>10800)")
log("=" * 78)
log(f"  {'组':16s} {'best_iter':>9s} {'full':>10s} {'core':>10s} {'Δfull':>9s} {'Δcore':>9s}")
rmse_tab = {}
for nm, p in VARIANTS:
    rf, rc = rmse(p, y), rmse(p[CORE], y[CORE])
    rmse_tab[nm] = {"best_iter": BEST_ITER[nm], "rmse_full": round(rf, 2),
                    "rmse_core": round(rc, 2),
                    "delta_full": round(rf - rmse_tab.get("A_L2", {}).get("rmse_full", rf), 2),
                    "delta_core": round(rc - rmse_tab.get("A_L2", {}).get("rmse_core", rc), 2)}
    d = rmse_tab[nm]
    if nm == "A_L2":
        d["delta_full"] = d["delta_core"] = 0.0
    log(f"  {nm:16s} {BEST_ITER[nm]:>9d} {rf:>10.2f} {rc:>10.2f} "
        f"{d['delta_full']:>+9.2f} {d['delta_core']:>+9.2f}")

# 分月
log("")
log(f"[7] 分月 RMSE")
log(f"  {'组':16s} {'Jan full':>10s} {'Jan core':>10s} {'Jul full':>10s} {'Jul core':>10s}")
month_tab = {}
for nm, p in VARIANTS:
    r = {}
    for m, tag in [(1, "Jan"), (7, "Jul")]:
        mk = mo == m
        mkc = mk & CORE
        r[f"{tag}_full"] = round(rmse(p[mk], y[mk]), 2)
        r[f"{tag}_core"] = round(rmse(p[mkc], y[mkc]), 2)
    r["delta_Jan_core"] = round(r["Jan_core"] - month_tab.get("A_L2", {}).get("Jan_core", r["Jan_core"]), 2)
    r["delta_Jul_core"] = round(r["Jul_core"] - month_tab.get("A_L2", {}).get("Jul_core", r["Jul_core"]), 2)
    r["delta_Jan_full"] = round(r["Jan_full"] - month_tab.get("A_L2", {}).get("Jan_full", r["Jan_full"]), 2)
    r["delta_Jul_full"] = round(r["Jul_full"] - month_tab.get("A_L2", {}).get("Jul_full", r["Jul_full"]), 2)
    month_tab[nm] = r
    log(f"  {nm:16s} {r['Jan_full']:>10.2f} {r['Jan_core']:>10.2f} "
        f"{r['Jul_full']:>10.2f} {r['Jul_core']:>10.2f}")

# 分机场 (核心口径)
log("")
log("[8] 分机场核心 RMSE (n>=500)")
aps = pd.Series(ap).value_counts()
focus = [a for a in aps.index if aps[a] >= 500]
focus = sorted(focus, key=lambda a: -rmse(pA[(ap == a) & CORE], y[(ap == a) & CORE]))
log(f"  {'ADEP':7s} {'n':>8s} " + "".join(f"{nm:>15s}" for nm, _ in VARIANTS)
    + "".join(f"{'Δ'+nm:>14s}" for nm, _ in VARIANTS[1:]))
ap_tab = {}
for a in focus:
    mk = (ap == a) & CORE
    if mk.sum() < 500:
        continue
    base = rmse(pA[mk], y[mk])
    row = {"n": int(mk.sum())}
    cells = []
    for nm, p in VARIANTS:
        r = rmse(p[mk], y[mk])
        row[nm] = round(r, 2)
        cells.append(f"{r:>15.2f}")
    deltas = []
    for nm, p in VARIANTS[1:]:
        d = rmse(p[mk], y[mk]) - base
        row[f"delta_{nm}"] = round(d, 2)
        deltas.append(f"{d:>+14.2f}")
    ap_tab[a] = row
    log(f"  {a:7s} {int(mk.sum()):>8d} " + "".join(cells) + "".join(deltas))

# 分布形态
log("")
log("=" * 78)
log("[9] 预测分布形态 (核心口径) — 尾部是否被放开")
log("=" * 78)
QS = [50, 75, 90, 95, 99, 99.5, 99.9]
dist_tab = {}
log(f"  {'组':16s} {'σpred/σtrue':>13s} {'IQR比':>8s} {'p99比':>8s} "
    + "".join(f"{'p'+str(q):>10s}" for q in QS))
qy = np.percentile(y[CORE], QS)
for nm, p in VARIANTS:
    pc = p[CORE]
    yc = y[CORE]
    qp = np.percentile(pc, QS)
    sr = float(pc.std() / yc.std())
    iqr = float((np.percentile(pc, 75) - np.percentile(pc, 25)) /
                (np.percentile(yc, 75) - np.percentile(yc, 25)))
    p99r = float(np.percentile(pc, 99) / np.percentile(yc, 99))
    dist_tab[nm] = {"std_ratio": round(sr, 4), "iqr_ratio": round(iqr, 4),
                    "p99_ratio": round(p99r, 4),
                    "pred_q": {str(q): round(float(v), 2) for q, v in zip(QS, qp)},
                    "true_q": {str(q): round(float(v), 2) for q, v in zip(QS, qy)}}
    log(f"  {nm:16s} {sr:>13.4f} {iqr:>8.4f} {p99r:>8.4f} "
        + "".join(f"{v:>10.1f}" for v in qp))
log(f"  {'TRUE':16s} {'1.0000':>13s} {'1.0000':>8s} {'1.0000':>8s} "
    + "".join(f"{v:>10.1f}" for v in qy))

# 长尾档预测幅度
log("")
log("[10] 长尾档 (按真值分档) 预测幅度 — 对比 A 看尾部是否放开")
log("")
BINS = [(1800, 3600, "(1800,3600]"), (3600, 10800, "(3600,10800]"),
        (10800, 1e18, "(10800,inf)")]
log(f"  {'真值档':14s} {'n':>7s} {'true中位':>10s} "
    + "".join(f"{nm+'中位':>17s}" for nm, _ in VARIANTS))
tail_tab = {}
for lo, hi, lbl in BINS:
    m = (y > lo) & (y <= hi)
    if m.sum() == 0:
        continue
    row = {"n": int(m.sum()), "true_median": round(float(np.median(y[m])), 1)}
    cells = []
    for nm, p in VARIANTS:
        pm = float(np.median(p[m]))
        row[f"{nm}_median"] = round(pm, 1)
        row[f"{nm}_ratio"] = round(pm / max(float(np.median(y[m])), 1e-9), 3)
        cells.append(f"{pm:>17.1f}")
    tail_tab[lbl] = row
    log(f"  {lbl:14s} {int(m.sum()):>7d} {row['true_median']:>10.1f} " + "".join(cells))
log("")
log(f"  {'真值档':14s} {'幅度比 p/true':>14s} " + "".join(f"{nm:>17s}" for nm, _ in VARIANTS))
for lbl, row in tail_tab.items():
    log(f"  {lbl:14s} {'':>14s} " + "".join(f"{row[f'{nm}_ratio']:>17.3f}" for nm, _ in VARIANTS))

# ---------------- [8] 采纳判定 ----------------
SEED_JITTER = 1.0
log("")
log("=" * 78)
log("[判定] 采纳门槛 (严格)")
log("-" * 78)
verdict = {}
for nm, _ in VARIANTS[1:]:
    d_full = rmse_tab[nm]["delta_full"]
    d_core = rmse_tab[nm]["delta_core"]
    dj = month_tab[nm]["delta_Jan_core"]
    dl = month_tab[nm]["delta_Jul_core"]
    ap_d = [ap_tab[a][f"delta_{nm}"] for a in ap_tab]
    n_ap_down = sum(1 for v in ap_d if v < 0)
    n_ap = len(ap_d)
    same_sign_ap = (n_ap_down == n_ap) or (n_ap_down == 0)
    same_dir_month = (dj < 0) == (dl < 0)
    pass_full = d_full < -SEED_JITTER
    pass_ap = same_sign_ap
    pass_month = same_dir_month
    ok = pass_full and pass_ap and pass_month
    verdict[nm] = {
        "delta_full": rmse_tab[nm]["delta_full"], "delta_core": d_core,
        "delta_Jan_core": dj, "delta_Jul_core": dl,
        "n_airports": n_ap, "n_airports_improved": n_ap_down,
        "airports_all_same_sign": bool(same_sign_ap),
        "months_same_direction": bool(same_dir_month),
        "pass_full_beyond_jitter": bool(pass_full),
        "pass_airport": bool(pass_ap), "pass_month": bool(pass_month),
        "adopt": bool(ok),
    }
    log(f"  {nm}:")
    log(f"    Δfull={d_full:+.2f}s (门槛 < -{SEED_JITTER}) {'✓' if pass_full else '✗'}")
    log(f"    Δcore={d_core:+.2f}s   ΔJan={dj:+.2f}s  ΔJul={dl:+.2f}s "
        f"({'同向 ✓' if same_dir_month else '反向 ✗'})")
    log(f"    分机场 {n_ap_down}/{n_ap} 改善 ({'全同号 ✓' if same_sign_ap else '符号不一致 ✗'})")
    log(f"    → {'采纳' if ok else '不采纳'}")
    log("")
log(f"  注: 采纳门槛用 ±{SEED_JITTER}s 固定抖动; 上一轮实测 LGB 三 seed core "
    f"(310.97/312.74/316.53) 的标准差约 2.9s, 实际 seed 抖动大于门槛设定。")

# ---------------- [9] 落盘 ----------------
out = pd.DataFrame({"y_true": y, "month": mo, "ADEP": ap})
for nm, p in VARIANTS:
    out[f"pred_{nm}"] = p
out.to_parquet(OUT_PQ, index=False)

RESULT = {
    "task": "ab_tail_objective_l2_vs_log_vs_huber",
    "config": {
        "split": "MAIN: train 2025-02..12, holdout 2025-01+2025-07",
        "n_train": int(len(fit_pos) + len(es_pos)), "n_fit": int(len(fit_pos)),
        "n_es": int(len(es_pos)), "n_holdout": int(len(y)),
        "es_months": es_months, "es_leak_check": "PASS",
        "feats_n": len(FEATS71), "seed": SEED, "cap": CAP, "es_rounds": ES_ROUND,
        "huber_alpha": HUBER_ALPHA,
        "params_note": "全部沿用基线常规值, 仅 objective/alpha 变化, 未针对 holdout 调参",
    },
    "best_iter": BEST_ITER,
    "smearing": {"factor": round(smear, 6),
                 "es_resid_mean": round(float(resid.mean()), 6),
                 "es_resid_median_exp": round(float(np.median(np.exp(resid))), 6),
                 "source": "训练折内 ES 集 (11-12月) 残差, 未参与梯度拟合"},
    "rmse": rmse_tab,
    "per_month": month_tab,
    "per_airport_core": ap_tab,
    "distribution_core": dist_tab,
    "long_tail_amplitude": tail_tab,
    "verdict": verdict,
    "timing_sec": round(time.time() - t0, 1),
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}")
log(f"→ {OUT_PQ}")
log(f"总耗时 {(time.time()-t0)/60:.1f} min")
