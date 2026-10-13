"""
calib_ceiling.py — 分布诊断 + 保序回归(calibration)天花板
============================================================
只读 + 轻量统计, 不重训, 不发版.

模型: 直接载入已落盘的 oof_lgb_fold03_seed42.txt (2999 棵树, 73 特征),
      零重训. 该模型训练月 = {1,2,3,4,8,9,10,11,12}, 7 月完全未见 → 真 holdout.
      其折内 ES val (训练月时间末 20%) 从未参与梯度拟合 → 合法内部 val,
      用于拟合单调映射 g. 严禁在 7 月上拟合.

特征: 完整复刻 oof_calibrate.py 的构建顺序 / LUT / offset / 类别编码,
      保证与模型训练时的输入分布一致.

输出:
  notebooks/diag/d0/calib_ceiling.json
  /Users/openclaw/.trae-cn/work/.../calib_ceiling.log
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
import os
os.chdir(ROOT)

import oof_calibrate as OC
from src.data_loader import load_month
import _r7_lib as L

MODEL_PATH = D0 / "oof_lgb_fold03_seed42.txt"
OUT_JSON = D0 / "calib_ceiling.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/calib_ceiling.log")

QS = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
T_BINS = [(0, 600), (600, 1200), (1200, 1800), (1800, 3600), (3600, 10800), (10800, 1e18)]
P_EDGES = [0, 600, 1200, 1800, 3000, 10800.1]

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


log("=" * 76)
log("[calib_ceiling] 只读诊断: 分布错配 / 长尾识别 / 保序校准天花板")
log(f"  模型: {MODEL_PATH.name} (载入, 不重训)")
log("=" * 76)
t0 = time.time()

# ============= [1] 复刻 oof_calibrate 的 12 月特征 =============
tr_dfs = [load_month(MONTH_FILE[m], columns=OC.TRAIN_COLS)
          for m in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]]
df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
del tr_dfs
df_tr_dep_all = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                          (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                          (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)].copy()
lut_main = L.build_t_unimp_lut(df_tr_dep_all)
log(f"[1] 训练月 raw {len(df_tr_raw):,}  LUT 源 {len(df_tr_dep_all):,}")

log("    构建训练月特征 (2-12) ...")
tr_full = OC.build_features(df_tr_raw, df_tr_raw, lut_main, label="tr_main")
tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
tt = tr_dep["TAXITIME_SEC_mvt"].values
tr_dep = tr_dep[(tt > 0) & (tt <= 10800)].reset_index(drop=True)
del tr_full
log(f"    训练月 DEP (0,10800] {len(tr_dep):,}   ({time.time()-t0:.0f}s)")

ho_raw = pd.concat([load_month(MONTH_FILE[m], columns=OC.TRAIN_COLS) for m in [1, 7]],
                   ignore_index=True)
log("    构建 holdout (1+7) 特征 ...")
# ref 与 oof_calibrate 完全一致: 传训练月 raw (内部只取其 DEP 行)
ho_full = OC.build_features(ho_raw, df_tr_raw, lut_main, label="ho")
ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
y_ho_raw = ho_dep["TAXITIME_SEC_mvt"].values.astype(float)
ho_dep = ho_dep[y_ho_raw > 0].reset_index(drop=True)   # 全样本口径保留 >10800 极端
del ho_full, ho_raw, df_tr_raw, df_tr_dep_all, lut_main
n_ext = int((ho_dep["TAXITIME_SEC_mvt"].values > 10800).sum())
log(f"    holdout DEP (y>0) {len(ho_dep):,}  其中 y>10800: {n_ext:,}   "
    f"({time.time()-t0:.0f}s)")

# ============= [2] offset 特征 (与 oof_calibrate 一致) =============
global_off = float(np.median(tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)) -
                   np.median(tr_dep["t_unimp_r7"].values.astype(np.float64)))
tr_off, tr_known, full_map = OC.fit_offset_map(tr_dep, global_off)
tr_dep["t_unimp_bias_offset"] = tr_off
tr_dep["t_unimp_bias_offset_known"] = tr_known
rk_off, rk_known = OC.apply_offset_map(pd.DataFrame({
    "ADEP_mvt": ho_dep["ADEP_mvt"].astype(str).values,
    "RUNWAY_mvt": ho_dep["RUNWAY_mvt"].astype(str).values}), full_map, global_off)
ho_dep["t_unimp_bias_offset"] = rk_off
ho_dep["t_unimp_bias_offset_known"] = rk_known
log(f"[2] offset global={global_off:+.1f}s  训练覆盖={tr_known.mean()*100:.1f}%  "
    f"holdout覆盖={rk_known.mean()*100:.1f}%")

# ============= [3] 拼 12 月 + 编码 =============
all_dep = pd.concat([tr_dep, ho_dep], ignore_index=True)
y_all = all_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
month_all = pd.to_datetime(all_dep["MVT_TIME_UTC_mvt"], utc=True,
                           errors="coerce").dt.month.values
n_all = len(all_dep)
core_rows = y_all <= 10800
n_ext_total = n_all - int(core_rows.sum())
# 关键: X_all 必须与 all_dep 完全同序, 否则 iloc 会取到错行 (上一版就踩了这个坑)。
# 类别字典复刻 oof_calibrate: 只由核心行 (0,10800] 决定, 极端行沿用同一套码。
_cat_levels = {c: sorted(pd.unique(all_dep.loc[core_rows, c].astype(str)).tolist())
               for c in OC.LOW_CARD_CATS}
X_all = OC.encode_X(all_dep, OC.FEATS_FULL)                  # 原序, 含极端行
for c in OC.LOW_CARD_CATS:
    X_all[c] = pd.Categorical(X_all[c].astype(str), categories=_cat_levels[c])
assert len(X_all) == n_all, f"X_all 行数 {len(X_all)} != all_dep {n_all}"
# 行序自检: 抽样比对 ADEP 编码值与 runway, 错位会立刻暴露
_chk = np.random.RandomState(0).choice(n_all, 3000, replace=False)
for _c in ("ADEP_mvt", "RUNWAY_mvt"):
    _ref = (all_dep[_c].astype(str).map(OC.CAT_MAP.get(_c, {})).fillna(-1)
            .astype("int32").values)
    assert (_ref[_chk] == X_all[_c].values[_chk]).all(), f"X_all 行序与 all_dep 错位 ({_c})"
log(f"[3] all_dep {n_all:,}  X {X_all.shape}  (含极端行 {n_ext_total:,}, 行序自检通过)")

# ============= [4] 复刻 fold-3 折内时间序切 =============
FOLD3_PRED_MONTHS = [5, 6, 7]
train_mask = ~np.isin(month_all, FOLD3_PRED_MONTHS)
train_idx_in = train_mask.nonzero()[0]
t_fold = pd.to_datetime(all_dep["MVT_TIME_UTC_mvt"].iloc[train_idx_in],
                        utc=True, errors="coerce").astype("int64").values
order = np.argsort(t_fold, kind="stable")
cut = int(len(order) * 0.8)
fit_pos = train_idx_in[order[:cut]]
es_pos = train_idx_in[order[cut:]]          # 原 ES val: 未参与梯度拟合 → 内部 val
assert len(np.intersect1d(fit_pos, es_pos)) == 0
es_months = sorted(set(month_all[es_pos].tolist()))
log(f"[4] 内部 val (原 ES val, 训练月时间末 20%) {len(es_pos):,}  月份={es_months}")
log(f"    梯度拟合侧 {len(fit_pos):,} 行, 与内部 val 零重叠 ✓")

# ============= [5] 载入已训好模型并预测 (零重训) =============
b = lgb.Booster(model_file=str(MODEL_PATH))
log(f"[5] 载入 {MODEL_PATH.name}: {b.num_trees()} 棵树 / {b.num_feature()} 特征")
t_p = time.time()
p_es = np.clip(b.predict(X_all.iloc[es_pos]).astype(np.float64), 30.0, 10800.0)
jul_pos = np.where(month_all == 7)[0]
p_ho = np.clip(b.predict(X_all.iloc[jul_pos]).astype(np.float64), 30.0, 10800.0)
y_es = y_all[es_pos]
y_ho = y_all[jul_pos]
ap_ho = all_dep["ADEP_mvt"].iloc[jul_pos].astype(str).values
log(f"    预测完成 ({time.time()-t_p:.0f}s)  内部val {len(p_es):,}  7月 {len(p_ho):,}")

CORE = y_ho <= 10800
FULL = np.ones(len(y_ho), bool)
r_base_core = rmse(p_ho[CORE], y_ho[CORE])
r_base_full = rmse(p_ho[FULL], y_ho[FULL])
r_es = rmse(p_es, y_es)
# 合理性护栏: 该模型 fold-3 在 7 月的 RMSE 历史上为 ~296s (learning_curve k=3000)。
# 若显著偏离, 基本可判定为行序错位或特征不一致, 必须停下而不是输出报告。
assert 200.0 < r_base_core < 400.0, (
    f"7 月核心 RMSE={r_base_core:.2f} 超出合理区间 (期望 ~296), "
    f"疑似行序错位或特征构建不一致")
log(f"    基线 RMSE: 内部val={r_es:.2f}  7月核心={r_base_core:.2f}  7月全样本={r_base_full:.2f}")
log(f"    (核心 n={int(CORE.sum()):,} / 全样本 n={int(FULL.sum()):,})")

# ============= [6] 分布错配: 分位数对照 =============
log("")
log("=" * 76)
log("[6] 预测分布 vs 真值分布 (7 月 holdout)")
log("=" * 76)
qtab = {}
for cname, m in [("core", CORE), ("full", FULL)]:
    p, y = p_ho[m], y_ho[m]
    qp, qy = np.percentile(p, QS), np.percentile(y, QS)
    slope, intercept = np.polyfit(p, y, 1)
    iqr_p = np.percentile(p, 75) - np.percentile(p, 25)
    iqr_y = np.percentile(y, 75) - np.percentile(y, 25)
    qtab[cname] = {
        "n": int(m.sum()), "q": QS,
        "pred": qp.round(2).tolist(), "true": qy.round(2).tolist(),
        "gap": (qp - qy).round(2).tolist(),
        "mean_pred": float(p.mean()), "mean_true": float(y.mean()),
        "std_pred": float(p.std()), "std_true": float(y.std()),
        "std_ratio": float(p.std() / y.std()), "iqr_ratio": float(iqr_p / iqr_y),
        "p99_9_ratio": float(np.percentile(p, 99.9) / max(np.percentile(y, 99.9), 1e-9)),
        "bias_mean": float(np.mean(p - y)), "bias_median": float(np.median(p - y)),
        "calib_slope": float(slope), "calib_intercept": float(intercept),
        "corr": float(np.corrcoef(p, y)[0, 1]),
    }
    log(f"  --- 口径 {cname} (n={int(m.sum()):,}) ---")
    log(f"  {'分位':>8s} {'pred':>11s} {'true':>11s} {'gap':>11s} {'gap%':>9s}")
    for q, a, c_ in zip(QS, qp, qy):
        log(f"  p{q:<7g} {a:>11.2f} {c_:>11.2f} {a-c_:>+11.2f} "
            f"{(a-c_)/max(c_,1e-9)*100:>+8.2f}%")
    d = qtab[cname]
    log(f"  [{cname}] mean_bias={d['bias_mean']:+.2f}s  median_bias={d['bias_median']:+.2f}s")
    log(f"  [{cname}] σpred/σtrue={d['std_ratio']:.4f}  IQR比={d['iqr_ratio']:.4f}  "
        f"p99.9比={d['p99_9_ratio']:.4f}  corr={d['corr']:.4f}")
    log(f"  [{cname}] 校准斜率 true~pred={d['calib_slope']:.4f} (截距 {d['calib_intercept']:+.2f})")

rel = []
p_c, y_c = p_ho[CORE], y_ho[CORE]
dec = pd.qcut(p_c, 10, labels=False, duplicates="drop")
for dd in sorted(set(dec.tolist())):
    s = dec == dd
    rel.append({"decile": int(dd) + 1, "n": int(s.sum()),
                "pred_mean": float(p_c[s].mean()), "true_mean": float(y_c[s].mean()),
                "bias": float(p_c[s].mean() - y_c[s].mean()), "rmse": rmse(p_c[s], y_c[s])})
log("")
log("  可靠性曲线 (核心口径, 按 pred 十分位):")
log(f"  {'D':>3s} {'n':>8s} {'pred均值':>11s} {'true均值':>11s} {'bias':>10s} {'RMSE':>10s}")
for r in rel:
    log(f"  {r['decile']:>3d} {r['n']:>8d} {r['pred_mean']:>11.2f} {r['true_mean']:>11.2f} "
        f"{r['bias']:>+10.2f} {r['rmse']:>10.2f}")

# ============= [7] 长尾幅度分析 =============
log("")
log("=" * 76)
log("[7] 长尾分析 (全样本口径)")
log("=" * 76)
sse_all = float(np.sum((p_ho - y_ho) ** 2))
by_true = []
for lo, hi in T_BINS:
    m = (y_ho > lo) & (y_ho <= hi)
    n = int(m.sum())
    if n == 0:
        continue
    p, y = p_ho[m], y_ho[m]
    sse = float(np.sum((p - y) ** 2))
    by_true.append({"lo": lo, "hi": (None if hi > 1e17 else hi), "n": n,
                    "share": n / len(y_ho), "pred_median": float(np.median(p)),
                    "pred_mean": float(p.mean()), "true_median": float(np.median(y)),
                    "bias_median": float(np.median(p - y)),
                    "bias_mean": float(np.mean(p - y)),
                    "rmse": float(np.sqrt(sse / n)), "sse_share": sse / sse_all})
log(f"  {'真值档(秒)':>18s} {'n':>8s} {'占比':>8s} {'pred中位':>10s} {'pred均值':>10s} "
    f"{'bias中位':>10s} {'RMSE':>10s} {'SSE占比':>9s}")
for r in by_true:
    lbl = f"({r['lo']:.0f},>10800]" if r["hi"] is None else f"({r['lo']:.0f},{r['hi']:.0f}]"
    log(f"  {lbl:>18s} {r['n']:>8d} {r['share']*100:>7.3f}% {r['pred_median']:>10.1f} "
        f"{r['pred_mean']:>10.1f} {r['bias_median']:>+10.1f} {r['rmse']:>10.1f} "
        f"{r['sse_share']*100:>8.2f}%")

by_pred = []
for i in range(len(P_EDGES) - 1):
    lo, hi = P_EDGES[i], P_EDGES[i + 1]
    m = (p_ho >= lo) & (p_ho < hi)
    n = int(m.sum())
    if n == 0:
        continue
    p, y = p_ho[m], y_ho[m]
    by_pred.append({
        "pred_lo": lo, "pred_hi": (None if hi > 1e8 else hi), "n": n,
        "share": n / len(p_ho),
        "true_median": float(np.median(y)), "true_p25": float(np.percentile(y, 25)),
        "true_p75": float(np.percentile(y, 75)), "true_p90": float(np.percentile(y, 90)),
        "true_p99": float(np.percentile(y, 99)), "true_mean": float(y.mean()),
        "precision_gt1800": float(np.mean(y > 1800)),
        "precision_gt3600": float(np.mean(y > 3600)),
        "recall_true_gt1800": float(np.sum((y > 1800) & (p > 1800)) / max(np.sum(y > 1800), 1)),
        "recall_true_gt3600": float(np.sum((y > 3600) & (p > 3600)) / max(np.sum(y > 3600), 1)),
        "median_pred": float(np.median(p)), "median_true": float(np.median(y)),
        "median_ratio": float(np.median(p) / max(np.median(y), 1e-9)),
    })
log("")
log(f"  {'预测档(秒)':>18s} {'n':>8s} {'占比':>8s} {'true中位':>10s} {'true均值':>10s} "
    f"{'P(y>1800)':>10s} {'P(y>3600)':>10s} {'中位p/y':>9s}")
for r in by_pred:
    lbl = f"[{r['pred_lo']:.0f},10800]" if r["pred_hi"] is None else f"[{r['pred_lo']:.0f},{r['pred_hi']:.0f})"
    log(f"  {lbl:>18s} {r['n']:>8d} {r['share']*100:>7.3f}% {r['true_median']:>10.1f} "
        f"{r['true_mean']:>10.1f} {r['precision_gt1800']*100:>9.2f}% "
        f"{r['precision_gt3600']*100:>9.2f}% {r['median_ratio']:>9.3f}")

m_t36, m_p36 = y_ho > 3600, p_ho > 3600
rec36 = float((m_t36 & m_p36).sum() / max(m_t36.sum(), 1))
log("")
log(f"  长尾真值>3600s: n={int(m_t36.sum()):,}  被识别(pred>3600) {int((m_t36&m_p36).sum()):,} "
    f"→ recall={rec36*100:.2f}%")
if m_t36.sum():
    log(f"    该子集 pred 中位={np.median(p_ho[m_t36]):.1f}s  true 中位={np.median(y_ho[m_t36]):.1f}s  "
        f"幅度比={float(np.median(p_ho[m_t36])/max(np.median(y_ho[m_t36]),1e-9)):.3f}")
m_t18, m_p18 = y_ho > 1800, p_ho > 1800
log(f"  长尾真值>1800s: n={int(m_t18.sum()):,}  recall={float((m_t18&m_p18).sum()/max(m_t18.sum(),1))*100:.2f}%")
if n_ext:
    log(f"  极端真值>10800s: n={n_ext:,}  pred 中位={np.median(p_ho[y_ho>10800]):.1f}s  "
        f"pred 最大={p_ho[y_ho>10800].max():.1f}s  true 最大={y_ho[y_ho>10800].max():.1f}s")
sse_ext = float(np.sum((p_ho[y_ho > 10800] - y_ho[y_ho > 10800]) ** 2))
log(f"  极端行 SSE 占全样本 {sse_ext/sse_all*100:.2f}%  "
    f"({'主导' if sse_ext/sse_all > 0.3 else '非主导'})")
# 若剔除极端, 全样本 RMSE 变成多少 (说明极端的杠杆)
r_full_noext = rmse(p_ho[y_ho <= 10800], y_ho[y_ho <= 10800])
log(f"  杠杆校验: 全样本 {r_base_full:.2f} vs 剔极端 {r_full_noext:.2f} "
    f"(极端仅 {n_ext} 行却把 RMSE 抬高 {r_base_full-r_full_noext:+.2f}s)")

# ============= [8] 保序回归: 校准天花板 =============
log("")
log("=" * 76)
log("[8] 保序回归 (isotonic) 校准天花板")
log("=" * 76)
log("  拟合集 = 内部 val (训练月时间末 20%, 不含 7 月, 未参与梯度拟合)")
log("  评估集 = 7 月 holdout  ← 严禁在 7 月上拟合, 已严格遵守")
log("")

A = np.column_stack([p_es, np.ones_like(p_es)])
coef, *_ = np.linalg.lstsq(A, y_es, rcond=None)
lin_ho = np.clip(coef[0] * p_ho + coef[1], 30.0, 10800.0)
r_lin_core = rmse(lin_ho[CORE], y_ho[CORE])
r_lin_full = rmse(lin_ho[FULL], y_ho[FULL])
log(f"  ②线性重标定 g(p) = {coef[0]:.4f}·p {coef[1]:+.2f}   [内部val拟合]")
log(f"     7月核心 {r_base_core:.2f} → {r_lin_core:.2f}  ({r_lin_core-r_base_core:+.2f}s)")
log(f"     7月全样 {r_base_full:.2f} → {r_lin_full:.2f}  ({r_lin_full-r_base_full:+.2f}s)")

iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p_es, y_es)
p_iso_ho = np.clip(iso.predict(p_ho), 30.0, 10800.0)
r_iso_es_in = rmse(iso.predict(p_es), y_es)
r_iso_core = rmse(p_iso_ho[CORE], y_ho[CORE])
r_iso_full = rmse(p_iso_ho[FULL], y_ho[FULL])
log(f"  ③保序回归 (内部val拟合, {len(iso.X_thresholds_)} 个阈值)")
log(f"     内部val in-sample {r_es:.2f} → {r_iso_es_in:.2f}  ({r_iso_es_in-r_es:+.2f}s, 乐观参照)")
log(f"     7月核心 {r_base_core:.2f} → {r_iso_core:.2f}  ({r_iso_core-r_base_core:+.2f}s)")
log(f"     7月全样 {r_base_full:.2f} → {r_iso_full:.2f}  ({r_iso_full-r_base_full:+.2f}s)")

half = len(p_es) // 2
f1 = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p_es[:half], y_es[:half])
f2 = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p_es[half:], y_es[half:])
cv_a = rmse(f2.predict(p_es[:half]), y_es[:half])
cv_b = rmse(f1.predict(p_es[half:]), y_es[half:])
r_cv = float(np.sqrt((cv_a**2 * half + cv_b**2 * (len(p_es) - half)) / len(p_es)))
log(f"  内部val 2折交叉 (前段拟合→后段评估, 完全不碰 7 月): {cv_a:.2f} / {cv_b:.2f} "
    f"→ 合并 {r_cv:.2f} vs 原始 {r_es:.2f}  ({r_cv-r_es:+.2f}s)")
log(f"  迁移衰减: in-sample {r_es-r_iso_es_in:+.2f}s → 同月交叉 {r_cv-r_es:+.2f}s "
    f"→ 跨月(7月) {r_iso_core-r_base_core:+.2f}s")

m_hi, m_hi_es = p_ho > 1800, p_es > 1800
iso_hi = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p_es[m_hi_es], y_es[m_hi_es])
tail_ho = p_ho.copy()
tail_ho[m_hi] = np.clip(iso_hi.predict(p_ho[m_hi]), 30.0, 10800.0)
r_tail_core = rmse(tail_ho[CORE], y_ho[CORE])
r_tail_full = rmse(tail_ho[FULL], y_ho[FULL])
log(f"  ④仅长尾段 (p>1800, n={int(m_hi.sum()):,}) 单独保序, 其余不动:")
log(f"     7月核心 {r_base_core:.2f} → {r_tail_core:.2f}  ({r_tail_core-r_base_core:+.2f}s)   "
    f"全样 {r_base_full:.2f} → {r_tail_full:.2f}  ({r_tail_full-r_base_full:+.2f}s)")

knots, kv = iso.X_thresholds_, iso.y_thresholds_
knot_idx = np.unique(np.linspace(0, len(knots) - 1, 13).astype(int))
knot_tab = [{"p": float(knots[i]), "g": float(kv[i]), "delta": float(kv[i] - knots[i])}
            for i in knot_idx]
log("")
log("  保序映射 g 关键节点 (g-p = 该 pred 段的平移量):")
log(f"  {'pred':>11s} {'g(pred)':>11s} {'g-p':>11s}")
for r in knot_tab:
    log(f"  {r['p']:>11.1f} {r['g']:>11.1f} {r['delta']:>+11.1f}")

# ============= [9] 重点机场 =============
log("")
log("=" * 76)
log("[9] 重点机场 (全样本口径 RMSE)")
log("=" * 76)
ap_per = {}
log(f"  {'机场':6s} {'n':>8s} {'基线':>10s} {'线性':>10s} {'保序':>10s} {'Δ保序':>9s}")
for ap in OC.FOCUS_AP:
    mk = (ap_ho == ap) & FULL
    if mk.sum() < 100:
        continue
    rb, rl, ri = rmse(p_ho[mk], y_ho[mk]), rmse(lin_ho[mk], y_ho[mk]), rmse(p_iso_ho[mk], y_ho[mk])
    ap_per[ap] = {"n": int(mk.sum()), "base": rb, "linear": rl, "isotonic": ri, "delta": ri - rb}
    log(f"  {ap:6s} {mk.sum():>8d} {rb:>10.2f} {rl:>10.2f} {ri:>10.2f} {ri-rb:>+9.2f}")

# ============= [10] 判定 =============
d_core = r_iso_core - r_base_core
d_full = r_iso_full - r_base_full
d_lin = r_lin_core - r_base_core
log("")
log("=" * 76)
log("[判定]")
log("-" * 76)
log(f"  位置偏+尺度可修 (线性)         Δcore = {d_lin:+.2f}s")
log(f"  形状错配额外部分 (保序-线性)   Δcore = {d_core - d_lin:+.2f}s")
log(f"  保序总天花板                   Δcore = {d_core:+.2f}s   Δfull = {d_full:+.2f}s")
log(f"  长尾段专项                     Δcore = {r_tail_core - r_base_core:+.2f}s   "
    f"Δfull = {r_tail_full - r_base_full:+.2f}s")
if d_core < -3.0:
    log("  >>> 天花板降幅大: 软校准 / 残差 stacking 值得做")
elif d_core < -1.0:
    log("  >>> 天花板中等 (1-3s): 可做但非主杠杆, 优先补轮数/特征")
else:
    log("  >>> 天花板只降几秒: 长尾是信息硬上限, 普通段已到顶")

RESULT = {
    "task": "prediction_distribution_calibration_ceiling",
    "provenance": {
        "model_file": MODEL_PATH.name, "n_trees": int(b.num_trees()),
        "n_features": int(b.num_feature()),
        "trained_months": sorted(set(month_all[train_mask].tolist())),
        "holdout_month": 7, "inner_val_months": es_months,
        "inner_val_rows": int(len(es_pos)),
        "note": ("learning_curve.py 的 8000 轮 booster 未落盘(只落 RMSE 曲线), "
                 "本次用已落盘 fold-3 2999 树模型; 学习曲线显示 3000→8000 轮 "
                 "7 月 RMSE 仅 -3.13s, 分布形态与校准天花板对该轮数差不敏感。"),
        "leakage_guard": "isotonic 仅在内部 val(训练月, 未参与梯度拟合)拟合; 7 月只做评估",
    },
    "counts": {"n_jul_full": int(FULL.sum()), "n_jul_core": int(CORE.sum()),
               "n_jul_extreme_gt10800": n_ext},
    "baseline_rmse": {"inner_val": round(r_es, 2), "jul_core": round(r_base_core, 2),
                      "jul_full": round(r_base_full, 2),
                      "jul_full_excl_extreme": round(r_full_noext, 2)},
    "distribution": qtab,
    "reliability_decile_core": rel,
    "long_tail_by_true": by_true,
    "long_tail_by_pred": by_pred,
    "tail_recall": {
        "true_gt1800_n": int(m_t18.sum()),
        "true_gt1800_recall": round(float((m_t18 & m_p18).sum() / max(m_t18.sum(), 1)), 4),
        "true_gt3600_n": int(m_t36.sum()), "true_gt3600_recall": round(rec36, 4),
        "extreme_sse_share_full": round(sse_ext / sse_all, 4),
    },
    "recalibration": {
        "linear": {"a": float(coef[0]), "b": float(coef[1]),
                   "rmse_core": round(r_lin_core, 2), "rmse_full": round(r_lin_full, 2),
                   "delta_core": round(d_lin, 2),
                   "delta_full": round(r_lin_full - r_base_full, 2)},
        "isotonic": {"n_thresholds": int(len(knots)),
                     "rmse_inner_in_sample": round(r_iso_es_in, 2),
                     "rmse_inner_xval": round(r_cv, 2),
                     "rmse_core": round(r_iso_core, 2), "rmse_full": round(r_iso_full, 2),
                     "delta_core": round(d_core, 2), "delta_full": round(d_full, 2),
                     "knots": knot_tab},
        "tail_only": {"rmse_core": round(r_tail_core, 2), "rmse_full": round(r_tail_full, 2),
                      "delta_core": round(r_tail_core - r_base_core, 2),
                      "delta_full": round(r_tail_full - r_base_full, 2)},
    },
    "per_airport": ap_per,
    "verdict": {
        "baseline_core": round(r_base_core, 2), "baseline_full": round(r_base_full, 2),
        "linear_delta_core": round(d_lin, 2),
        "shape_extra_delta_core": round(d_core - d_lin, 2),
        "isotonic_delta_core": round(d_core, 2), "isotonic_delta_full": round(d_full, 2),
        "isotonic_in_sample_delta_inner": round(r_iso_es_in - r_es, 2),
        "isotonic_xval_delta_inner": round(r_cv - r_es, 2),
        "tail_only_delta_core": round(r_tail_core - r_base_core, 2),
        "ceiling_big": bool(d_core < -3.0),
    },
    "timing_sec": round(time.time() - t0, 1),
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")
log("")
log(f"→ {OUT_JSON}   总耗时 {(time.time()-t0)/60:.1f} min")
