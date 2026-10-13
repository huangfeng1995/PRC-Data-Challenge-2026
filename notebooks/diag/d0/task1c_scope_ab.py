"""
task1c_scope_ab.py — 定位特征"构建范围"是否为根因
P0: 训练集与 holdout 【分开】调用 build_features(各自只含本批月份)
本次: 12 个月【一次性】调用 build_features  ← 队列/漂移特征的分母与邻域不同
本脚本用 P0 的分开口径重建 holdout 特征, 与缓存逐特征对比。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "task1c_scope_ab.json"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
from v15_clean_retrain import (build_features, encode_cat_columns,
                              FEATS71, TRAIN_COLS, TIME_COLS)
from src.data_loader import load_month
import _r7_lib as L

_t0 = time.time()
def log(*a): print(f"[{time.time()-_t0:6.0f}s]", " ".join(str(x) for x in a), flush=True)

TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
def _mf(m: int) -> str:
    """月份文件: 2025-01..11 用 (m+1), 12 月跨年到 2026-01"""
    ny, nm2 = (2025, m + 1) if m < 12 else (2026, 1)
    return str(ROOT / "data/raw" / f"training_2025-{m:02d}-01_{ny}-{nm2:02d}-01.parquet")

MF = {m: _mf(m) for m in range(1, 13)}
assert all(Path(p).exists() for p in MF.values()), "月份文件缺失"

# ---- P0 口径: 训练与 holdout 分开构建 ----
log("按 P0 口径分别构建 训练 / holdout 特征 ...")
tr_raw = pd.concat([load_month(MF[m], columns=TRAIN_COLS) for m in TRAIN_MONTHS],
                   ignore_index=True)
tr_parsed = L.__dict__ and tr_raw  # 占位, 真正 parse 在 build_features 内
from src.data_loader import parse_time_columns
tr_parsed = parse_time_columns(tr_raw.copy(), cols=TIME_COLS)
lut = L.build_t_unimp_lut(tr_parsed[(tr_parsed["PHASE_mvt"] == "DEP")])
tr_full = build_features(tr_raw, tr_parsed, lut, label="tr")
tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
del tr_full, tr_raw
log(f"  训练 DEP {len(tr_dep):,}")

ho_raw = pd.concat([load_month(MF[m], columns=TRAIN_COLS) for m in HOLD_MONTHS],
                   ignore_index=True)
ho_full = build_features(ho_raw, tr_parsed, lut, label="ho")
ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
y_p0 = ho_dep["TAXITIME_SEC_mvt"].to_numpy(np.float64)
keep = y_p0 > 0
ho_dep = ho_dep[keep].reset_index(drop=True)
y_p0 = y_p0[keep]
log(f"  holdout DEP {len(ho_dep):,}  >10800={int((y_p0>10800).sum())}")

# ---- 我方口径: 一次性 12 个月 ----
log("加载我方一次性构建的缓存 ...")
mine = pd.read_pickle(D0 / "_corr_feat_cache.pkl")
mo = mine["month"].to_numpy()
m = np.isin(mo, HOLD_MONTHS)
cm = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
mine_ho = mine[m].copy().reset_index(drop=True)
y_mine = mine_ho["TAXITIME_SEC_mvt"].to_numpy(np.float64)
km = y_mine > 0
mine_ho = mine_ho[km].reset_index(drop=True); y_mine = y_mine[km]
log(f"  我方 holdout DEP {len(mine_ho):,}")

ho_dep = encode_cat_columns(ho_dep, cm)
for f in FEATS71:
    if f not in ho_dep.columns:
        ho_dep[f] = np.nan
    if f not in mine_ho.columns:
        mine_ho[f] = np.nan

# ---- 按 MVT_ID 对齐后逐特征对比 ----
key = "MVT_ID_mvt"
A = ho_dep[[key] + FEATS71].copy()
B = mine_ho[[key] + FEATS71].copy()
mg = A.merge(B, on=key, suffixes=("_p0", "_mine"), how="inner")
log(f"  MVT_ID 对齐: {len(mg):,} (P0 {len(A):,} / 我方 {len(B):,})")
assert len(mg) > 300000, "对齐行数异常"

rows = []
for f in FEATS71:
    x = pd.to_numeric(mg[f + "_p0"], errors="coerce")
    z = pd.to_numeric(mg[f + "_mine"], errors="coerce")
    both = x.notna() & z.notna()
    d = (x - z).abs()
    denom = x.abs().clip(lower=1.0)
    rows.append({"feat": f,
                 "nan_p0": int(x.isna().sum()), "nan_mine": int(z.isna().sum()),
                 "n_cmp": int(both.sum()),
                 "max_abs_diff": float(d[both].max()) if both.any() else None,
                 "mean_abs_diff": float(d[both].mean()) if both.any() else None,
                 "pct_rows_differ_gt1s": float((d[both] > 1.0).mean() * 100) if both.any() else None,
                 "median_p0": float(x[both].median()) if both.any() else None,
                 "median_mine": float(z[both].median()) if both.any() else None})
rows.sort(key=lambda r: -(r["pct_rows_differ_gt1s"] or 0))
print("\n" + "=" * 108)
print("逐特征对比 (P0 分开构建 vs 我方一次性构建) —— 按差异行占比排序")
print("=" * 108)
print(f"{'特征':>30} {'差异>1s行%':>11} {'max|Δ|':>14} {'mean|Δ|':>12} {'med(P0)':>12} {'med(我方)':>12}")
for r in rows[:28]:
    print(f"{r['feat']:>30} {r['pct_rows_differ_gt1s']:>10.2f}% {r['max_abs_diff']:>14.2f} "
          f"{r['mean_abs_diff']:>12.4f} {r['median_p0']:>12.2f} {r['median_mine']:>12.2f}")
same = [r for r in rows if (r["pct_rows_differ_gt1s"] or 0) < 0.01]
print(f"\n完全一致(<0.01% 行有差异)的特征: {len(same)}/71")
print("  " + ", ".join(r["feat"] for r in same[:30]))
json.dump({"n_aligned": int(len(mg)), "n_p0": int(len(A)), "n_mine": int(len(B)),
           "n_identical": len(same), "rows": rows},
          open(OUT, "w"), ensure_ascii=False, indent=2)
log(f"→ {OUT}")
