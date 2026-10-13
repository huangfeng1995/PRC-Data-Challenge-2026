"""
v23_ablation.py — 拆解 v23 真榜退化 (392.58 vs v22 360.86) 的两个混淆因素

v23 (P50) 与 v15/v22 (L2) 的差异不止目标函数:
  (A) 目标函数: quantile(alpha=0.5)  vs  regression(L2)
  (B) 轮数:     CAP=2000 (撞 cap)   vs  3000 (撞 cap)
  (C) 训练标签: 未清洗 (含 y<=0 / y>10800) vs 清洗 (0 < y <= 10800)

在**完全相同的 ES 划分**上评估四个组合, 用 ES full / core RMSE 判定主因。
ES = 全年按 MVT 排序最后 10% (与 v23 训练脚本一致)。
"""
from __future__ import annotations
import sys, json, time, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
import os; os.chdir(ROOT)

OUT = D0 / "v23_ablation.json"
BASE_PARAMS = {
    "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
SEED = 42
_t0 = time.time()
def lap(m=""): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

lap("加载全年特征缓存")
d = pickle.load(open(D0 / "_p50_full_X.pkl", "rb"))
X, y, mvt = d["X"], d["y"], d["mvt"]
lap(f"  X={X.shape}  y 中位={np.median(y):.0f}")

n_all = len(y)
n_le0 = int((y <= 0).sum())
n_gt = int((y > 10800).sum())
lap(f"  未清洗标签: y<=0 {n_le0:,}  y>10800 {n_gt:,}  合计剔 {n_le0+n_gt:,} ({100*(n_le0+n_gt)/n_all:.3f}%)")

# ES 划分: 与 v23 脚本一致
order = np.argsort(mvt, kind="stable")
cut = int(n_all * 0.9)
fit_idx, es_idx = order[:cut], order[cut:]
lap(f"fit {len(fit_idx):,} / ES {len(es_idx):,}")

y_es = y[es_idx]
core = (y_es > 0) & (y_es <= 10800)
lap(f"ES core {int(core.sum()):,} / 全量 {len(y_es):,}")
lap(f"ES 真值: >10800 有 {int((y_es>10800).sum())} 行, 其对 full SSE 贡献 "
    f"{100*((y_es[~core]-y_es[~core].mean())**2).sum()/max(1e-9,((y_es-y_es.mean())**2).sum()):.1f}% (粗估)")

def evaluate(p, name):
    p = np.clip(p, 30.0, 10800.0)
    full = float(np.sqrt(np.mean((p - y_es) ** 2)))
    c = float(np.sqrt(np.mean((p[core] - y_es[core]) ** 2)))
    # 分档
    bins = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, 10**9)]
    seg = {}
    for lo, hi in bins:
        m = (y_es >= lo) & (y_es < hi)
        seg[f"[{lo},{hi})"] = {
            "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean((p[m] - y_es[m]) ** 2))) if m.sum() else None,
            "bias": float(np.mean(p[m] - y_es[m])) if m.sum() else None,
        }
    lap(f"  {name:<34} full={full:8.2f}  core={c:8.2f}  med_p={np.median(p):7.1f}  max_p={p.max():8.1f}")
    return {"full": full, "core": c, "median_pred": float(np.median(p)),
            "max_pred": float(p.max()), "segments": seg}

def train(obj, rounds, clean, name, use_early_stop=True):
    p = dict(BASE_PARAMS); p["objective"] = obj
    if obj == "quantile":
        p["alpha"] = 0.5; p["metric"] = "quantile"
    else:
        p["metric"] = "rmse"
    p["seed"] = SEED; p["bagging_seed"] = SEED + 1; p["feature_fraction_seed"] = SEED + 2
    idx = fit_idx[clean.to_numpy()[fit_idx]] if clean is not None else fit_idx
    dtr = lgb.Dataset(X.iloc[idx], label=y[idx], categorical_feature="auto", free_raw_data=False)
    dva = lgb.Dataset(X.iloc[es_idx], label=y_es, reference=dtr)
    kw = dict(num_boost_round=rounds, valid_sets=[dva])
    if use_early_stop:
        kw["callbacks"] = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    b = lgb.train(p, dtr, **kw)
    lap(f"  [{name}] n_train={len(idx):,} best_iter={b.best_iteration}")
    pr = b.predict(X.iloc[es_idx], num_iteration=b.best_iteration)
    res = evaluate(pr, name)
    res.update({"objective": obj, "rounds": rounds, "clean": clean is not None,
                "best_iter": int(b.best_iteration), "n_train": int(len(idx))})
    return res

results = {}
lap("\n=== 组 1: 未清洗标签 (v23 口径) ===")
results["P50_r2000_dirty"] = train("quantile", 2000, None, "P50 r2000 dirty (== v23)")
results["L2_r2000_dirty"]  = train("regression", 2000, None, "L2  r2000 dirty")

lap("\n=== 组 2: 清洗标签 (v15 口径) ===")
clean_mask = (y > 0) & (y <= 10800)
results["L2_r3000_clean"]  = train("regression", 3000, clean_mask, "L2  r3000 clean (== v15)")
results["P50_r3000_clean"] = train("quantile", 3000, clean_mask, "P50 r3000 clean")

lap("\n=== 组 3: 隔离轮数 (清洗标签, 2000 轮) ===")
results["L2_r2000_clean"]  = train("regression", 2000, clean_mask, "L2  r2000 clean")
results["P50_r2000_clean"] = train("quantile", 2000, clean_mask, "P50 r2000 clean")

lap("\n=== 组 4: 拉满轮数 (清洗标签, 6000 轮) ===")
results["L2_r6000_clean"]  = train("regression", 6000, clean_mask, "L2  r6000 clean")
results["P50_r6000_clean"] = train("quantile", 6000, clean_mask, "P50 r6000 clean")

print("\n" + "=" * 104)
print(f"{'组合':<32} {'full':>9} {'core':>9} {'Δfull vs v15口径':>18}")
print("=" * 104)
base = results["L2_r3000_clean"]["full"]
for k, v in results.items():
    print(f"{k:<32} {v['full']:>9.2f} {v['core']:>9.2f} {v['full']-base:>+18.2f}")
print("=" * 104)

json.dump({"es_rows": int(len(es_idx)), "fit_rows": int(len(fit_idx)),
           "n_label_dirty": {"le0": n_le0, "gt10800": n_gt},
           "results": results}, open(OUT, "w"), ensure_ascii=False, indent=2)
lap(f"→ {OUT}")
