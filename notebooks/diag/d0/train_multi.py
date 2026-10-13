"""
train_multi.py — 轻量补训多基模型 (XGB / LGB / CAT), 产出 holdout 逐行预测
=========================================================================
目的: 为「多模型分歧度能否识别长尾/大误差」提供干净的逐行预测。

为什么不能直接用 models/v15_clean_retrain/:
  v15_clean_retrain.py 的三个库都把 holdout(1+7) 当 ES 验证集:
    XGB: evals=[(dholdout, "holdout")], early_stopping_rounds=100
    LGB: valid_sets=[lgb_holdout], early_stopping(100)
    CAT: eval_set=(df_ho_cat, y_ho), early_stopping_rounds=100
  → best_iter 是用 holdout 标签挑出来的, holdout 预测带乐观偏差,
    不能作为"折外模型"用于 holdout 上的信号检验。故按用户许可轻量补训。
  另: CAT seed123 因 >20min 保护在 save_model 之前 break, 磁盘上只有 cat_seed42,
     所以 v15 实际是 "xgb+lgb 有 3 seed, cat 只有 1 seed", 不是完整三库 3seed。

本脚本口径 (与 v15 对齐, 仅修 ES 泄漏):
  训练月: 2025-02..12 (剔 1+7), 训练标签 0 < TAXITIME < 10800
  ES 集: 训练月内按 MVT 时间末 10% (≈11-12 月), 绝不碰 1/7 月
  Holdout: 2025-01 + 2025-07, 全样本口径保留 y>0 (含 >10800 极端)
  特征: FEATS71 (与 v15 完全同一份列表, 便于与历史结果对照)
  模型: XGB(42) / LGB(42) / CAT(42) 各 1 seed  → 算法多样性
        + LGB(123) / LGB(2024)                  → 种子多样性对照

输出:
  notebooks/diag/d0/multi_preds.parquet   逐行 y / 各库 pred
  notebooks/diag/d0/train_multi.json     元信息 + 各自 RMSE
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

from src.data_loader import load_month, parse_time_columns
import _r7_lib as L
import _r8_drift_lib as D
from v15_clean_retrain import (build_features, encode_cat_columns, encode_for_catboost,
                               FEATS71, LOW_CARD_CATS, HIGH_CARD_CATS, TRAIN_COLS, TIME_COLS)

WORK = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b")
OUT_PQ = D0 / "multi_preds.parquet"
OUT_JSON = D0 / "train_multi.json"
OUT_LOG = WORK / "train_multi.log"
MODEL_DIR = D0 / "multi_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CAT_ITER = int(os.environ.get("CAT_ITER", 2500))
CAP = int(os.environ.get("CAP", 2500))
ES_ROUND = 100
SEED = 42

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
log("[train_multi] 轻量补训 XGB/LGB/CAT (ES 在训练月内, 不碰 holdout)")
log("=" * 76)
t0 = time.time()

# ---------------- [1] 数据 + 特征 (与 v15 同构) ----------------
tr_raw = pd.concat([load_month(MONTH_FILE[m], columns=TRAIN_COLS)
                    for m in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]], ignore_index=True)
tr_parsed = parse_time_columns(tr_raw.copy(), cols=TIME_COLS)
lut = L.build_t_unimp_lut(tr_parsed[(tr_parsed["PHASE_mvt"] == "DEP")])
log(f"[1] 训练月 raw {len(tr_raw):,}  LUT L1={len(lut['L1_q'])}")

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
y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
keep = y_ho > 0                       # 全样本口径: 保留 >10800 极端
ho_dep = ho_dep[keep].reset_index(drop=True)
y_ho = y_ho[keep]
del ho_full, ho_raw
log(f"    Holdout DEP (y>0) {len(ho_dep):,}  其中 >10800: {int((y_ho>10800).sum()):,}"
    f"   ({time.time()-t0:.0f}s)")

# ---------------- [2] 折内时间序 ES 切 (只用训练月) ----------------
ts = tr_dep["MVT_TIME_UTC_mvt"].astype("int64").values
order = np.argsort(ts, kind="stable")
cut = int(len(order) * 0.9)
fit_pos, es_pos = order[:cut], order[cut:]
es_months = sorted(tr_dep["MVT_TIME_UTC_mvt"].dt.month.iloc[es_pos].unique().tolist())
assert 1 not in es_months and 7 not in es_months, f"ES 集泄漏到 holdout 月: {es_months}"
log(f"[2] 梯度拟合 {len(fit_pos):,} / ES {len(es_pos):,} (训练月末 10%, 月份={es_months}) ✓ 不含 1/7 月")

# ---------------- [3] 编码三套输入 ----------------
cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
tr_dep = encode_cat_columns(tr_dep, cat_map)
ho_dep = encode_cat_columns(ho_dep, cat_map)
for f in FEATS71:
    if f not in tr_dep.columns:
        tr_dep[f] = np.nan
    if f not in ho_dep.columns:
        ho_dep[f] = np.nan

# LGB: 低基数保留 category
Xtr_lgb = tr_dep[FEATS71].copy()
Xes_lgb = tr_dep[FEATS71].iloc[es_pos].copy()
Xho_lgb = ho_dep[FEATS71].copy()
# XGB: category → int
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = {k: dict(v) for k, v in cat_map.items()}
Xtr_xgb = tx._encode_categoricals(Xtr_lgb.copy(), fit=False).values.astype(np.float32)
Xes_xgb = tx._encode_categoricals(Xes_lgb.copy(), fit=False).values.astype(np.float32)
Xho_xgb = tx._encode_categoricals(Xho_lgb.copy(), fit=False).values.astype(np.float32)
# CAT: 全数值
Xtr_cat = encode_for_catboost(Xtr_lgb.copy()).values.astype(np.float32)
Xes_cat = encode_for_catboost(Xes_lgb.copy()).values.astype(np.float32)
Xho_cat = encode_for_catboost(Xho_lgb.copy()).values.astype(np.float32)

y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)
y_es = tr_dep["TAXITIME_SEC_mvt"].values[es_pos].astype(np.float32)
y_fit = y_tr[fit_pos]
log(f"    X shapes: lgb {Xtr_lgb.shape} xgb {Xtr_xgb.shape} cat {Xtr_cat.shape}")

preds = {}
meta = {}

# ---------------- [4] XGB ----------------
log("")
log("=" * 76)
log("[3] XGB (hist, depth 11)")
log("=" * 76)
xgb_params = {"objective": "reg:squarederror", "eval_metric": "rmse",
              "learning_rate": 0.0147555939449363, "max_depth": 11,
              "min_child_weight": 5, "subsample": 0.9813515130184908,
              "colsample_bytree": 0.6720708449966734,
              "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
              "tree_method": "hist", "nthread": -1, "verbosity": 0, "seed": SEED}
t1 = time.time()
d_fit = xgb.DMatrix(Xtr_xgb[fit_pos], label=y_fit, feature_names=FEATS71)
d_es = xgb.DMatrix(Xes_xgb, label=y_es, feature_names=FEATS71)
d_ho = xgb.DMatrix(Xho_xgb, feature_names=FEATS71)
bst = xgb.train(xgb_params, d_fit, num_boost_round=CAP, evals=[(d_es, "es")],
                early_stopping_rounds=ES_ROUND, verbose_eval=500)
bi = bst.best_iteration + 1
p = np.clip(bst.predict(d_ho, iteration_range=(0, bi)), 30, 10800)
preds["xgb"] = p
bst.save_model(str(MODEL_DIR / f"xgb_seed{SEED}.json"))
meta["xgb"] = {"best_iter": int(bi), "sec": round(time.time() - t1, 1),
               "rmse_ho_core": round(rmse(p[y_ho <= 10800], y_ho[y_ho <= 10800]), 2),
               "rmse_ho_full": round(rmse(p, y_ho), 2)}
log(f"  best_iter={bi}  holdout core={meta['xgb']['rmse_ho_core']}  "
    f"full={meta['xgb']['rmse_ho_full']}  ({time.time()-t1:.0f}s)")

# ---------------- [5] LGB (3 seed: 算法多样性 1 个 + 种子多样性对照) ----------------
log("")
log("=" * 76)
log("[4] LGB (3 seed: 42 主用, 123/2024 作种子多样性对照)")
log("=" * 76)
lgb_params = {"objective": "regression", "metric": "rmse",
              "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
              "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
              "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
              "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
              "verbose": -1, "num_threads": -1}
for sd in [42, 123, 2024]:
    t1 = time.time()
    d_fit = lgb.Dataset(Xtr_lgb.iloc[fit_pos], label=y_fit, categorical_feature=LOW_CARD_CATS)
    d_es = lgb.Dataset(Xes_lgb, label=y_es, categorical_feature=LOW_CARD_CATS, reference=d_fit)
    b = lgb.train(dict(lgb_params, seed=sd), d_fit, num_boost_round=CAP,
                  valid_sets=[d_es],
                  callbacks=[lgb.early_stopping(ES_ROUND, verbose=False),
                             lgb.log_evaluation(0)])
    bi = b.best_iteration or CAP
    p = np.clip(b.predict(Xho_lgb, num_iteration=bi).astype(np.float64), 30, 10800)
    preds[f"lgb{sd}"] = p
    b.save_model(str(MODEL_DIR / f"lgb_seed{sd}.txt"))
    meta[f"lgb{sd}"] = {"best_iter": int(bi), "sec": round(time.time() - t1, 1),
                        "rmse_ho_core": round(rmse(p[y_ho <= 10800], y_ho[y_ho <= 10800]), 2),
                        "rmse_ho_full": round(rmse(p, y_ho), 2)}
    log(f"  seed={sd} best_iter={bi}  holdout core={meta[f'lgb{sd}']['rmse_ho_core']}  "
        f"full={meta[f'lgb{sd}']['rmse_ho_full']}  ({time.time()-t1:.0f}s)")

# ---------------- [6] CAT ----------------
log("")
log("=" * 76)
log(f"[5] CAT (depth 11, cap {CAT_ITER})")
log("=" * 76)
t1 = time.time()
m = CatBoostRegressor(iterations=CAT_ITER, learning_rate=0.0147555939449363,
                      depth=11, l2_leaf_reg=5, random_seed=SEED, verbose=500,
                      early_stopping_rounds=ES_ROUND, thread_count=-1,
                      eval_metric="RMSE", cat_features=[])
m.fit(Xtr_cat[fit_pos], y_fit, eval_set=(Xes_cat, y_es))
p = np.clip(m.predict(Xho_cat).astype(np.float64), 30, 10800)
preds["cat"] = p
m.save_model(str(MODEL_DIR / f"cat_seed{SEED}"))
meta["cat"] = {"best_iter": int(m.tree_count_), "sec": round(time.time() - t1, 1),
               "rmse_ho_core": round(rmse(p[y_ho <= 10800], y_ho[y_ho <= 10800]), 2),
               "rmse_ho_full": round(rmse(p, y_ho), 2)}
log(f"  best_iter={m.tree_count_}  holdout core={meta['cat']['rmse_ho_core']}  "
    f"full={meta['cat']['rmse_ho_full']}  ({time.time()-t1:.0f}s)")

# ---------------- [7] 落盘 ----------------
out = pd.DataFrame({
    "y_true": y_ho,
    "month": ho_dep["MVT_TIME_UTC_mvt"].dt.month.values,
    "ADEP": ho_dep["ADEP_mvt"].astype(str).values,
    "t_unimp_r7": ho_dep["t_unimp_r7"].values,
})
for k, v in preds.items():
    out[f"pred_{k}"] = v
out.to_parquet(OUT_PQ, index=False)
log("")
log(f"[6] 逐行预测落盘: {OUT_PQ}  {out.shape}")
log(f"    列: {[c for c in out.columns]}")

core = y_ho <= 10800
summary = {
    "task": "train_multi_xgb_lgb_cat_clean_es",
    "config": {
        "train_months": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        "holdout_months": [1, 7],
        "es_split": "训练月内 MVT 时间末 10%",
        "es_months": es_months,
        "es_leak_check": "PASS (ES 集不含 1/7 月)",
        "n_fit": int(len(fit_pos)), "n_es": int(len(es_pos)), "n_holdout": int(len(y_ho)),
        "feats_n": len(FEATS71), "cap": CAP, "cat_iter": CAT_ITER, "es_rounds": ES_ROUND,
        "seeds": {"xgb": SEED, "lgb": [42, 123, 2024], "cat": SEED},
    },
    "v15_reuse_rejected": {
        "reason": "v15 三库均以 holdout(1+7) 作 ES 验证集, best_iter 由 holdout 标签选出, "
                  "holdout 预测带乐观偏差, 不可用于 holdout 上的信号检验",
        "xgb_es_on": "evals=[(dholdout,'holdout')], early_stopping_rounds=100",
        "lgb_es_on": "valid_sets=[lgb_holdout], early_stopping(100)",
        "cat_es_on": "eval_set=(df_ho_cat,y_ho), early_stopping_rounds=100",
        "cat_seed_count": "磁盘仅 cat_seed42; seed123 在 save_model 前被 >20min 保护 break",
    },
    "per_model": meta,
    "timing_sec": round(time.time() - t0, 1),
}
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"→ {OUT_JSON}   总耗时 {(time.time()-t0)/60:.1f} min")
