"""
learning_curve.py — 学习曲线判定 (fold3 / holdout=7月)
=====================================================
判断 best_iter 触顶 3000 是欠拟合还是 ES 验证集问题:
- 训 8000 轮, 不开 ES
- 在 k=200,400,...,8000 处 predict, 记录两条曲线
  (内部 val=训练月时间末 10%, 真 holdout=7 月)
- 不发版, 单 seed, 后台 nohup
输出: notebooks/diag/d0/learning_curve.json
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
import warnings; warnings.filterwarnings("ignore")

# 直接复用 oof_calibrate 已验证过的配置 (单一事实来源)
from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D
import oof_calibrate as OC

OUT_JSON = D0 / "learning_curve.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/learning_curve.log")

TRAIN_COLS = OC.TRAIN_COLS
TIME_COLS = OC.TIME_COLS
FEATS71 = OC.FEATS71  # 73? 是 OC.FEATS71 (基线 71)
LGB_PARAMS = dict(OC.LGB_PARAMS)
HIGH_CARD_CATS = OC.HIGH_CARD_CATS
LOW_CARD_CATS = OC.LOW_CARD_CATS

CAP = int(os.environ.get("CAP", 8000))
SAMPLE_K = [200, 400, 800, 1200, 1600, 2000, 2400, 2800,
            3000, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600, 8000]
HOLDOUT_MONTHS = [7]
TRAIN_EXCLUDE = [5, 6, 7]    # fold3 排除月


def month_file(m):
    return (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet" if m < 12
            else "data/raw/training_2025-12-01_2026-01-01.parquet")


def build_features(df_raw, ref_pool, lut, label):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    b = build_basic(df)
    f = add_queue_features(b.copy(), include_a=True, include_b=True,
                           include_c=True, include_d=True, include_e=True)
    f["ref_was_estimated"] = f["AOBT_3_flt"].isna().astype("int8")
    f["is_diverted"] = (f["ADES_flt"] != f["ADES_FILED_flt"]).astype("int8")
    for k, v in D.compute_drift_features(f, ref_dep=ref_pool, label=label).items():
        f[k] = v
    t, _ = L.lookup_t_unimp(f, lut)
    f["t_unimp_r7"] = t
    m = (f["PHASE_mvt"] == "DEP").values
    for k, v in L.compute_extra_features(f[m], label=label).items():
        f[k] = np.nan; f.loc[m, k] = v
    for k, v in L.compute_mvt_anchor_features(f[m], t_unimp_vals=t[m], label=label).items():
        f[k] = np.nan; f.loc[m, k] = v
    return f


def log(*a):
    print(*a, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(" ".join(str(x) for x in a) + "\n")


log("=" * 70)
log(f"[learning_curve] CAP={CAP}  SEEDS=1  "
    f"holdout={HOLDOUT_MONTHS}  train_exclude={TRAIN_EXCLUDE}")
log("=" * 70)

# ============= 加载 + 切 =============
t0 = time.time()
all_parts = []
# 训练池: 全 12 月, 我们后面再切
# 加载 10 个月(2,3,4,8,9,10,11,12 + 5,6)做训练池候选, 只加载 7 月做真 holdout
need = sorted(set(range(2, 13)) | {1})  # 训练池月 + 1 月 (不入训练但参与 ref_pool)
log(f"  加载月份 {need} ...")
dfs = [load_month(month_file(m), columns=TRAIN_COLS) for m in need]
raw = pd.concat(dfs, ignore_index=True)
del dfs
raw = parse_time_columns(raw, cols=TIME_COLS)
month_raw = raw["MVT_TIME_UTC_mvt"].dt.month.values
log(f"  raw {len(raw):,}  {time.time()-t0:.0f}s")

# ref_pool for drift 特征 (DEP+ARR 全量, 跟 oof_calibrate 一致)
ref_pool = raw[(raw["PHASE_mvt"] == "DEP")][
    ["MVT_TIME_UTC_mvt", "ADEP_mvt", "RUNWAY_mvt", "PHASE_mvt"]].copy()
log(f"  ref_pool {len(ref_pool):,}")

# LUT
lut_src = raw[(month_raw == 2) & (raw["PHASE_mvt"] == "DEP") &
              (raw["TAXITIME_SEC_mvt"] > 0) & (raw["TAXITIME_SEC_mvt"] < 10800)]
lut = L.build_t_unimp_lut(lut_src)
del lut_src

log("  build_features ...")
all_dep_full = build_features(raw, ref_pool, lut, "all")
del raw, ref_pool
all_dep_full = all_dep_full[(all_dep_full["PHASE_mvt"] == "DEP") &
                            (all_dep_full["TAXITIME_SEC_mvt"] > 0) &
                            (all_dep_full["TAXITIME_SEC_mvt"] < 10800)].copy()
all_dep_full = all_dep_full.sort_values("MVT_TIME_UTC_mvt").reset_index(drop=True)
log(f"  all_dep_full {len(all_dep_full):,}  {time.time()-t0:.0f}s")
del lut

month_all = all_dep_full["MVT_TIME_UTC_mvt"].dt.month.values

# 切训练池 / 内部 val / 真 holdout
train_mask = (~np.isin(month_all, TRAIN_EXCLUDE))
ho_mask = np.isin(month_all, HOLDOUT_MONTHS)
# 内部 val: 训练月份内按时间末 10%
ts = all_dep_full["MVT_TIME_UTC_mvt"].astype("int64").values
train_idx = np.where(train_mask)[0]
order = np.argsort(ts[train_idx], kind="stable")
train_ord_idx = train_idx[order]
cut = int(len(order) * 0.9)
inner_tr_idx = train_ord_idx[:cut]
inner_va_idx = train_ord_idx[cut:]
log(f"  训练 {len(inner_tr_idx):,} (训练月 2,3,4,8,9,10,11,12 去掉 5,6)")
log(f"  内部 val {len(inner_va_idx):,} (训练月时间末 10%)")
log(f"  真 holdout (7 月) {ho_mask.sum():,}")

Xtr_full = all_dep_full[FEATS71].copy()
# 类别编码 (沿用 oof_calibrate.encode_X 的逻辑, 但不要它的 OFFSET 特征)
# 直接读 OC.CAT_MAP 是从 _rebuild_12m_cat.json 加载的, 71 特征子集都在
for col in HIGH_CARD_CATS:
    if col in Xtr_full.columns:
        m = OC.CAT_MAP.get(col, {})
        Xtr_full[col] = Xtr_full[col].astype(str).map(m).fillna(-1).astype("int32")
for col in LOW_CARD_CATS:
    if col in Xtr_full.columns:
        Xtr_full[col] = Xtr_full[col].astype("category")

Xtr = Xtr_full.iloc[inner_tr_idx]
ytr = all_dep_full["TAXITIME_SEC_mvt"].values[inner_tr_idx].astype(np.float32)
Xva = Xtr_full.iloc[inner_va_idx]
yva = all_dep_full["TAXITIME_SEC_mvt"].values[inner_va_idx].astype(np.float32)
Xho = Xtr_full.iloc[ho_mask]
yho = all_dep_full["TAXITIME_SEC_mvt"].values[ho_mask].astype(np.float32)
log(f"  Xtr {Xtr.shape}  Xva {Xva.shape}  Xho {Xho.shape}")

# ============= 训练 (8000 轮, 无 ES) =============
log("")
log(f"[训练] LGB 8000 轮, 无 ES, learning_rate=0.0148")
log("=" * 70)
ds = lgb.Dataset(Xtr, label=ytr, categorical_feature=LOW_CARD_CATS)
t_train = time.time()
params = dict(LGB_PARAMS); params["seed"] = 42
b = lgb.train(params, ds, num_boost_round=CAP,
              callbacks=[lgb.log_evaluation(0)])
log(f"  训练耗时 {time.time()-t_train:.0f}s ({CAP} 轮)")
del ds

# ============= 采样预测 =============
log("")
log(f"[采点预测] k={SAMPLE_K}")
log("=" * 70)
curve_va, curve_ho = [], []
t_pred = time.time()
for k in SAMPLE_K:
    pv = np.clip(b.predict(Xva, num_iteration=k).astype(np.float64), 30.0, 10800.0)
    ph = np.clip(b.predict(Xho, num_iteration=k).astype(np.float64), 30.0, 10800.0)
    rmse_va = float(np.sqrt(np.mean((pv - yva) ** 2)))
    rmse_ho = float(np.sqrt(np.mean((ph - yho) ** 2)))
    curve_va.append((k, rmse_va))
    curve_ho.append((k, rmse_ho))
    log(f"  k={k:>5d}  RMSE_inner_val={rmse_va:.2f}  RMSE_holdout_7月={rmse_ho:.2f}")
log(f"  全部采点耗时 {time.time()-t_pred:.1f}s")

# ============= 关键分析 =============
log("")
log("=" * 70)
log("[判定]")
log("-" * 70)
cur = dict(curve_va); cur2 = dict(curve_ho)


def at(k):
    return cur[k], cur2[k]


r3000va, r3000ho = at(3000)
r8000va, r8000ho = at(8000)
log(f"  k=3000:  inner_val={r3000va:.2f}  holdout_7={r3000ho:.2f}")
log(f"  k=8000:  inner_val={r8000va:.2f}  holdout_7={r8000ho:.2f}")
log(f"  k=8000-k=3000: Δinner_val={r8000va-r3000va:+.2f}s  "
    f"Δholdout_7={r8000ho-r3000ho:+.2f}s")
# 找 holdout 最低点 (按 |k-5000|<30 偏移后)
best_k_ho, best_rmse_ho = min(curve_ho, key=lambda x: x[1])
best_k_va, best_rmse_va = min(curve_va, key=lambda x: x[1])
log(f"  holdout 最低点: k={best_k_ho}  RMSE={best_rmse_ho:.2f}")
log(f"  inner  最低点: k={best_k_va}  RMSE={best_rmse_va:.2f}")

# 3000 后每 k 段 Δ (holdout 曲线)
post_3000_ho = [(k, cur2[k]) for k in SAMPLE_K if k > 3000]
delta_post = [b - cur2[3000] for _, b in post_3000_ho]
log(f"  holdout_7 3000 后 Δ vs k=3000 (min/median/max):")
log(f"    min={min(delta_post):+.2f}  median={float(np.median(delta_post)):+.2f}  "
    f"max={max(delta_post):+.2f}")
# 单调性: 3000 后 7 月曲线是否整体下降 (单调或近单调)
# 简单指标: 3000 后所有 k 较 3000 是否同号
n_neg = sum(1 for d in delta_post if d < -1)
n_pos = sum(1 for d in delta_post if d > 1)
n_zero = sum(1 for d in delta_post if -1 <= d <= 1)
log(f"    |Δ|>1s 桶: 负 {n_neg} / 平 {n_zero} / 正 {n_pos}")
# 是否"持续显著下降"
sustained_drop = (best_rmse_ho < r3000ho - 1.0)
plateau_or_rise = (best_rmse_ho >= r3000ho - 1.0)

log("")
log("  → 结论:")
if sustained_drop:
    log(f"    3000 后 holdout_7 最低 RMSE={best_rmse_ho:.2f} "
        f"({best_rmse_ho - r3000ho:+.2f}s vs 3000 轮)")
    log("    >>> 判定: 持续下降, 证实欠拟合 (大杠杆: 重定轮数/学习率)")
else:
    log(f"    3000 后 holdout_7 最低 RMSE={best_rmse_ho:.2f} "
        f"({best_rmse_ho - r3000ho:+.2f}s vs 3000 轮)")
    log("    >>> 判定: 平台或回升, 3000 轮合理, ES 验证集问题或 stacking 方向要改")

# ============= 落盘 =============
RESULT = {
    "task": "learning_curve_fold3_holdout7",
    "config": {
        "CAP": CAP,
        "sample_k": SAMPLE_K,
        "holdout_months": HOLDOUT_MONTHS,
        "train_exclude": TRAIN_EXCLUDE,
        "n_inner_train": int(len(inner_tr_idx)),
        "n_inner_val": int(len(inner_va_idx)),
        "n_holdout_7": int(ho_mask.sum()),
        "seed": 42,
        "lr": params["learning_rate"],
        "feats_n": len(FEATS71),
        "no_ES": True,
    },
    "curve_inner_val": {k: v for k, v in curve_va},
    "curve_holdout_7": {k: v for k, v in curve_ho},
    "key_points": {
        "k3000": {"inner_val": r3000va, "holdout_7": r3000ho},
        "k8000": {"inner_val": r8000va, "holdout_7": r8000ho},
        "best_inner_val": {"k": best_k_va, "rmse": best_rmse_va},
        "best_holdout_7": {"k": best_k_ho, "rmse": best_rmse_ho},
        "delta_8000_minus_3000": {
            "inner_val": r8000va - r3000va,
            "holdout_7": r8000ho - r3000ho,
        },
    },
    "post_3000_delta_holdout": [{"k": k, "rmse": v, "delta_vs_3000": v - r3000ho}
                                 for k, v in post_3000_ho],
    "verdict": {
        "sustained_drop": sustained_drop,
        "best_k_holdout": best_k_ho,
        "best_rmse_holdout": round(best_rmse_ho, 2),
        "best_rmse_minus_3000": round(best_rmse_ho - r3000ho, 2),
    },
    "timing": {
        "train_sec": round(time.time() - t_train, 1),
        "predict_all_k_sec": round(time.time() - t_pred, 1),
        "total_sec": round(time.time() - t0, 1),
    },
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}")