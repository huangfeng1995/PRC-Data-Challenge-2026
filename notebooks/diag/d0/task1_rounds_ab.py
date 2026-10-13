"""
task1_rounds_ab.py — 隔离"轮数效应"
同一 split / 同一 FEATS71 / 同一编码 / 同一 seed42, 只变 num_boost_round。
用一次 cap8000 训练 + 多点 num_iteration 预测, 彻底隔离轮数因素。
同时对比 P0 parquet 的行数与标签, 排查 split/特征差异。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "task1_rounds_ab.json"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
from v15_clean_retrain import FEATS71, LOW_CARD_CATS

TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
CAP = 8000
ES = 80
LP = {"objective": "regression", "metric": "rmse",
      "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
      "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
      "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
      "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
      "verbose": -1, "num_threads": -1}
_t0 = time.time()
def log(*a): print(f"[{time.time()-_t0:6.0f}s]", " ".join(str(x) for x in a), flush=True)

dep = pd.read_pickle(D0 / "_corr_feat_cache.pkl")
mo = dep["month"].to_numpy(); y = dep["TAXITIME_SEC_mvt"].to_numpy(float)
X = dep[FEATS71]
tr = np.where(np.isin(mo, TRAIN_MONTHS))[0]
ho = np.where(np.isin(mo, HOLD_MONTHS))[0]
log(f"训练 {len(tr):,} 行 | holdout {len(ho):,} 行")

# ---------- 0. 与 P0 parquet 对齐检查 (行数 / 标签) ----------
p0 = pd.read_parquet(D0 / "p0_holdout_predictions.parquet",
                     columns=["MVT_ID_mvt", "TAXI_TRUE_s", "TAXI_PRED_s_ENS"])
mine = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy()[ho],
                     "TAXI_TRUE_mine": y[ho]})
mm = p0.merge(mine, on="MVT_ID_mvt", how="outer", indicator=True)
log(f"P0 parquet n={len(p0):,}  我的 holdout n={len(ho):,}")
log(f"  MVT_ID 匹配: {mm._merge.value_counts().to_dict()}")
both = mm[mm._merge == "both"]
if len(both):
    d = (both["TAXI_TRUE_s"].to_numpy(float) - both["TAXI_TRUE_mine"].to_numpy(float))
    log(f"  标签差异: max|Δ|={np.abs(d).max():.1f}s  非零行={int((np.abs(d)>0.5).sum()):,}")

# ---------- 1. ES 折: 训练月内时间末 10% ----------
ts = dep["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()[tr]
order = np.argsort(ts, kind="stable"); cut = int(len(order) * 0.9)
fit_idx, es_idx = tr[order[:cut]], tr[order[cut:]]
es_m = sorted(set(mo[es_idx].tolist()))
assert 1 not in es_m and 7 not in es_m
log(f"ES: n={len(es_idx):,} 月份={es_m} ✓")

# ---------- 2. 一次 cap8000 训练 ----------
dtr = lgb.Dataset(X.iloc[fit_idx], label=y[fit_idx], categorical_feature=LOW_CARD_CATS)
dev = lgb.Dataset(X.iloc[es_idx], label=y[es_idx], categorical_feature=LOW_CARD_CATS, reference=dtr)
log("训练 cap=8000 seed42 ...")
b = lgb.train(dict(LP, seed=42, bagging_seed=42, feature_fraction_seed=42, data_random_seed=42),
              dtr, num_boost_round=CAP, valid_sets=[dev],
              callbacks=[lgb.early_stopping(ES, verbose=False), lgb.log_evaluation(0)])
bi = b.best_iteration or CAP
log(f"best_iter={bi}  (cap={CAP})")

# ---------- 3. 多点预测: 隔离轮数 ----------
t = y[ho]; c = (t > 0) & (t <= 10800)
pts = sorted({2000, 4000, 6000, 8000, bi, CAP} - {0})
res = []
for k in pts:
    p = np.clip(b.predict(X.iloc[ho], num_iteration=k).astype(np.float64), 30, 10800)
    r = {"k": int(k), "full": round(float(np.sqrt(np.mean((p-t)**2))), 2),
         "core": round(float(np.sqrt(np.mean((p[c]-t[c])**2))), 2),
         "jan": round(float(np.sqrt(np.mean((p[mo[ho]==1]-t[mo[ho]==1])**2))), 2),
         "jul": round(float(np.sqrt(np.mean((p[mo[ho]==7]-t[mo[ho]==7])**2))), 2)}
    log(f"  k={k:>5d}  full={r['full']:7.2f}  core={r['core']:7.2f}  Jan={r['jan']:7.2f}  Jul={r['jul']:7.2f}")
    res.append(r)

log("")
log(f"轮数效应: k=2000 → k={max(pts)}   core {res[0]['core']:.2f} → "
    f"{[r for r in res if r['k']==max(pts)][0]['core']:.2f}")
log(f"P0 parquet 报 LGB cap8000 core≈294-305; 本实验同口径 k=8000 core={[r for r in res if r['k']==8000][0]['core'] if any(r['k']==8000 for r in res) else 'n/a'}")

json.dump({"best_iter": int(bi), "cap": CAP, "es_months": es_m,
           "n_train": int(len(fit_idx)), "n_es": int(len(es_idx)),
           "n_holdout": int(len(ho)),
           "p0_parquet_n": int(len(p0)),
           "label_match": {"n_both": int(len(both)),
                           "max_abs_diff": float(np.abs(
                               (both["TAXI_TRUE_s"].to_numpy(float) - both["TAXI_TRUE_mine"].to_numpy(float))).max()) if len(both) else None},
           "curve": res}, open(OUT, "w"), ensure_ascii=False, indent=2)
log(f"→ {OUT}")
