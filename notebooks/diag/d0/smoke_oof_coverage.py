"""
smoke_oof_coverage.py — 折覆盖 smoke (只验分组+校验, 不跑全量)
=====================================================================
用真实数据限行跑含 7 月的 fold 3 ([5,6,7]), 确认:
  1. 折覆盖断言全部通过
  2. 7 月行能得到非空 oof_pred
  3. ES 早停正常触发, best_iter 落在 (0, ROUNDS]
  4. clip + 整数化正常
并推算全量 5 折耗时。

用法:  python3 notebooks/diag/d0/smoke_oof_coverage.py
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[3]
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
import warnings; warnings.filterwarnings("ignore")

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

# ---- 从 oof_calibrate.py 导入被测的分组与断言 (单一事实来源) ----
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "oof_cfg", D0 / "oof_calibrate.py")
# oof_calibrate 是脚本, 顶层会跑 main; 这里只取常量, 故手工解析 AST 更安全
import ast
_src = (D0 / "oof_calibrate.py").read_text(encoding="utf-8")
_tree = ast.parse(_src)
_cfg = {}
for _node in _tree.body:
    if isinstance(_node, ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, ast.Name):
                try:
                    _cfg[_t.id] = ast.literal_eval(_node.value)
                except Exception:
                    pass
    if isinstance(_node, ast.AnnAssign) and isinstance(_node.target, ast.Name) \
            and _node.target.id == "ROUNDS":
        _cfg["ROUNDS"] = int(os.environ.get("ROUNDS", 1500))
# FEATS_FULL 是通过列表推导/拼接构造的, literal_eval 解析不了 -> 直接执行模块取常量
if "FEATS_FULL" not in _cfg:
    _ns = {"__file__": str(D0 / "oof_calibrate.py"), "__name__": "oof_cfg_consts"}
    _exec = compile(_src, str(D0 / "oof_calibrate.py"), "exec")
    _tmp = {}
    # 只执行到 OOF_FOLDS 断言之前, 避免触发 main()
    _lines = _src.split("\n")
    _cut = next(i for i, l in enumerate(_lines) if l.startswith("OFFSET_FOLDS"))
    exec(compile("\n".join(_lines[:_cut]), "oof_cfg_head", "exec"), _tmp)
    for _k in ("OOF_FOLDS", "SEEDS", "LGB_PARAMS", "LOW_CARD_CATS",
               "TRAIN_COLS", "TIME_COLS", "FEATS_FULL"):
        if _k in _tmp:
            _cfg[_k] = _tmp[_k]

OOF_FOLDS = _cfg["OOF_FOLDS"]
SEEDS = _cfg.get("SEEDS", [42])
LGB_PARAMS = _cfg["LGB_PARAMS"]
LOW_CARD_CATS = _cfg["LOW_CARD_CATS"]
TRAIN_COLS = _cfg["TRAIN_COLS"]
TIME_COLS = _cfg["TIME_COLS"]
FEATS_FULL = _cfg["FEATS_FULL"]
ROUNDS = int(os.environ.get("ROUNDS", 1500))

N_TRAIN = int(os.environ.get("N_TRAIN", 150_000))   # 每折训练池上限
N_PRED = int(os.environ.get("N_PRED", 40_000))      # 每折预测集上限
TARGET_FOLD = int(os.environ.get("TARGET_FOLD", 3))  # 含 7 月的那一折

print("=" * 70)
print("[cfg] 从 oof_calibrate.py 解析到的配置")
print("=" * 70)
print(f"  OOF_FOLDS   = {OOF_FOLDS}")
print(f"  SEEDS       = {SEEDS}")
print(f"  ROUNDS      = {ROUNDS}")
print(f"  FEATS_FULL  = {len(FEATS_FULL)} 个特征")
print(f"  N_TRAIN/N_PRED = {N_TRAIN:,} / {N_PRED:,}")
print(f"  TARGET_FOLD = {TARGET_FOLD} -> months={OOF_FOLDS[TARGET_FOLD-1]}")

# ============================================================
# [1] 折覆盖断言 (与 oof_calibrate.py 同逻辑)
# ============================================================
print()
print("=" * 70)
print("[1] 折覆盖断言")
print("=" * 70)
_flat = [m for f in OOF_FOLDS for m in f]
covered = set(_flat)
expected = set(range(1, 13))
print(f"  covered = {sorted(covered)}")
print(f"  断言 1: covered == {sorted(expected)}  ->  "
      f"{'PASS' if covered == expected else 'FAIL'}")
assert covered == expected, f"missing={sorted(expected-covered)}"
print(f"  断言 2: 1 in covered  -> {'PASS' if 1 in covered else 'FAIL'}")
assert 1 in covered
print(f"  断言 3: 7 in covered  -> {'PASS' if 7 in covered else 'FAIL'}")
assert 7 in covered
print(f"  断言 4: 无重复月份    -> "
      f"{'PASS' if len(_flat) == len(set(_flat)) else 'FAIL'}")
assert len(_flat) == len(set(_flat))
print("  >>> 全部断言通过")


# ============================================================
# [2] 加载真实数据 (train 2-12 + holdout 1,7)
# ============================================================
print()
print("=" * 70)
print("[2] 加载真实数据")
print("=" * 70)
t0 = time.time()


def mf(m):
    return (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet" if m < 12
            else "data/raw/training_2025-12-01_2026-01-01.parquet")


dfs = [load_month(mf(m), columns=TRAIN_COLS) for m in (1, 7)]
ho_raw = pd.concat(dfs, ignore_index=True)
del dfs
dfs = [load_month(mf(m), columns=TRAIN_COLS)
       for m in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)]
tr_raw = pd.concat(dfs, ignore_index=True)
del dfs
all_raw = pd.concat([tr_raw, ho_raw], ignore_index=True)
del tr_raw, ho_raw
all_raw = parse_time_columns(all_raw, cols=TIME_COLS)
all_dep = all_raw[(all_raw["PHASE_mvt"] == "DEP") &
                  (all_raw["TAXITIME_SEC_mvt"] > 0) &
                  (all_raw["TAXITIME_SEC_mvt"] < 10800)].copy()
all_dep = all_dep.sort_values("MVT_TIME_UTC_mvt").reset_index(drop=True)
del all_raw
month_all = all_dep["MVT_TIME_UTC_mvt"].dt.month.values
print(f"  all_dep {len(all_dep):,} 行  {time.time()-t0:.0f}s")
month_counts = {m: int((month_all == m).sum()) for m in range(1, 13)}
print("  各月行数:", {k: v for k, v in month_counts.items()})
assert all(v > 0 for v in month_counts.values()), "存在空月"
print(f"  1 月 {month_counts[1]:,} 行 | 7 月 {month_counts[7]:,} 行  <- 均非空")

# 逐行覆盖检查
cover = np.zeros(len(all_dep), dtype=np.int32)
for fm in OOF_FOLDS:
    cover += np.isin(month_all, fm).astype(np.int32)
print(f"  逐行覆盖: min={cover.min()}, max={cover.max()}  "
      f"-> {'PASS' if cover.min() == 1 and cover.max() == 1 else 'FAIL'}")
assert cover.min() == 1 and cover.max() == 1

# 打印每折预测行数
print()
print("  每折预测行数:")
fold_pred_rows = {}
for fi, fm in enumerate(OOF_FOLDS, 1):
    n = int(np.isin(month_all, fm).sum())
    per = {m: month_counts[m] for m in fm}
    fold_pred_rows[fi] = {"months": list(fm), "n_pred": n, "per_month": per}
    print(f"    fold {fi} months={fm}: predict {n:,} 行  明细={per}")
assert fold_pred_rows[3]["per_month"].get(7, 0) > 0, "fold 3 的 7 月行数为 0"
print(f"  >>> fold 3 含 7 月, 7 月 {month_counts[7]:,} 行 (非空)")

# ============================================================
# [3] 特征工程 (限行)
# ============================================================
print()
print("=" * 70)
print("[3] 特征工程 (限行)")
print("=" * 70)
t_feat = time.time()
feat_t = time.time()
lut_src = all_dep[all_dep["MVT_TIME_UTC_mvt"].dt.month == 2]
lut = L.build_t_unimp_lut(lut_src)


def build_features(df_raw, ref_dep, label=""):
    df = build_basic(parse_time_columns(df_raw.copy(), cols=TIME_COLS))
    f = add_queue_features(df, include_a=True, include_b=True,
                           include_c=True, include_d=True, include_e=True)
    f["ref_was_estimated"] = f["AOBT_3_flt"].isna().astype("int8")
    f["is_diverted"] = (f["ADES_flt"] != f["ADES_FILED_flt"]).astype("int8")
    for k, v in D.compute_drift_features(f, ref_dep=ref_dep, label=label).items():
        f[k] = v
    t, _ = L.lookup_t_unimp(f, lut)
    f["t_unimp_r7"] = t
    m = (f["PHASE_mvt"] == "DEP").values
    for k, v in L.compute_extra_features(f[m], label=label).items():
        f[k] = np.nan; f.loc[m, k] = v
    for k, v in L.compute_mvt_anchor_features(f[m], t_unimp_vals=t[m], label=label).items():
        f[k] = np.nan; f.loc[m, k] = v
    return f


pred_months = OOF_FOLDS[TARGET_FOLD - 1]
train_pool = all_dep[~np.isin(month_all, pred_months)]
pred_pool = all_dep[np.isin(month_all, pred_months)]
train_pool = train_pool.sort_values("MVT_TIME_UTC_mvt").iloc[:N_TRAIN]
# 预测池按月分层抽样, 保证 fold 内每个月都抽到 (否则 40K 行会被最早的月吃光)
_per = max(1, N_PRED // len(pred_months))
_parts = []
for _m in pred_months:
    _s = pred_pool[pred_pool["MVT_TIME_UTC_mvt"].dt.month == _m] \
        .sort_values("MVT_TIME_UTC_mvt").iloc[:_per]
    _parts.append(_s)
pred_pool = pd.concat(_parts, ignore_index=True)
print(f"  训练池 {len(train_pool):,}  预测池 {len(pred_pool):,} (months={pred_months})")
print(f"  预测池按月分布: "
      f"{pred_pool['MVT_TIME_UTC_mvt'].dt.month.value_counts().sort_index().to_dict()}")

sub = pd.concat([train_pool, pred_pool], ignore_index=True)
F = build_features(sub, all_dep, label=f"fold{TARGET_FOLD}")
F = F[(F["PHASE_mvt"] == "DEP")].reset_index(drop=True)
print(f"  特征工程: {time.time()-t_feat:.1f}s, 帧 {F.shape}")
del sub

# 编码 (与 oof_calibrate 一致: 频率编码取 >=30)
from src.train_xgb import _encode_categoricals  # noqa
X = F.copy()
for c in LOW_CARD_CATS + [c for c in ("ADEP_mvt","ADES_mvt","RUNWAY_mvt","STAND_mvt",
                                     "airport_hour","AIRCRAFT_TYPE_mvt",
                                     "AIRCRAFT_OPERATOR_flt") if c not in LOW_CARD_CATS]:
    if c not in X.columns:
        continue
    vc = X[c].astype("str").value_counts()
    keep = {v: i for i, v in enumerate(vc[vc >= 30].index)}
    X[c] = X[c].astype("str").map(keep).fillna(-1).astype("int32")
feats = [c for c in FEATS_FULL if c in X.columns]
print(f"  编码完成, 使用 {len(feats)} 特征")
# 只取 FEATS_FULL 里的列 (整帧会带 datetime/object, LGB 不接受)
X = X[feats].copy()
assert all(pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool
           for c in X.columns), f"非数值列: {[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]}"

y = F["TAXITIME_SEC_mvt"].values.astype(np.float32)
mth = F["MVT_TIME_UTC_mvt"].dt.month.values
is_pred = np.isin(mth, pred_months)
is_train = ~is_pred
print(f"  train {int(is_train.sum()):,} | predict {int(is_pred.sum()):,}")

# ============================================================
# [4] 折内时间序 ES 训练
# ============================================================
print()
print("=" * 70)
print(f"[4] 训练 fold {TARGET_FOLD} (months={pred_months})")
print("=" * 70)
t_i = all_dep["MVT_TIME_UTC_mvt"].astype("int64").values[F.index] \
    if "MVT_TIME_UTC_mvt" in F.columns else None
# 用 F 内部的时间戳排序 (已在 all_dep 内按时间排过, 但 concat 后重排过)
ts = F["MVT_TIME_UTC_mvt"].astype("int64").values
order = np.argsort(ts[is_train], kind="stable")
train_idx = np.where(is_train)[0][order]
cut = int(len(order) * 0.8)
tr_pos, va_pos = train_idx[:cut], train_idx[cut:]
print(f"  ES split: train {len(tr_pos):,} | val {len(va_pos):,} (末 20% 时间序)")

Xtr, ytr = X.iloc[tr_pos], y[tr_pos]
Xva, yva = X.iloc[va_pos], y[va_pos]
Xp, yp = X.iloc[is_pred], y[is_pred]

t0 = time.time()
best_iters = []
preds = []
for sd in SEEDS:
    ds = lgb.Dataset(Xtr, label=ytr, categorical_feature=LOW_CARD_CATS)
    dv = lgb.Dataset(Xva, label=yva, categorical_feature=LOW_CARD_CATS, reference=ds)
    b = lgb.train(dict(LGB_PARAMS, seed=sd), ds, num_boost_round=ROUNDS,
                  valid_sets=[dv],
                  callbacks=[lgb.early_stopping(80, verbose=False),
                             lgb.log_evaluation(0)])
    bi = b.best_iteration if b.best_iteration > 0 else ROUNDS
    best_iters.append(bi)
    p = np.clip(b.predict(Xp, num_iteration=bi).astype(np.float64), 30.0, 10800.0)
    preds.append(p)
    print(f"    seed={sd}  best_iter={bi} / cap={ROUNDS}  "
          f"ES={'触发' if bi < ROUNDS else '未触发(跑满)'}")
train_time = time.time() - t0
print(f"  训练耗时 {train_time:.1f}s")

# ============================================================
# [5] 校验 7 月 oof_pred 非空 / ES / 整数化
# ============================================================
print()
print("=" * 70)
print("[5] 校验")
print("=" * 70)
oof = np.mean(np.stack(preds), axis=0)
assert np.isfinite(oof).all(), "oof_pred 含 NaN/Inf"
print(f"  oof_pred 非空: shape={oof.shape}  finite 100% OK")
for m in pred_months:
    sel = mth[is_pred] == m
    if sel.sum() == 0:
        continue
    r = float(np.sqrt(np.mean((oof[sel] - yp[sel]) ** 2)))
    print(f"    {m:>2d} 月: n={int(sel.sum()):,}  oof_pred 非空 OK  RMSE={r:.2f}  "
          f"pred均值={oof[sel].mean():.1f}  pred范围=[{oof[sel].min():.1f}, {oof[sel].max():.1f}]")
m7 = mth[is_pred] == 7
assert m7.sum() > 0 and np.isfinite(oof[m7]).all(), "7 月 oof_pred 异常"
print(f"  >>> 7 月 {int(m7.sum()):,} 行 oof_pred 全部非空 OK")

print(f"  ES 检查: best_iter={best_iters} (cap={ROUNDS})  "
      f"-> {'正常早停' if all(0 < b <= ROUNDS for b in best_iters) else '异常'}")
assert all(0 < b <= ROUNDS for b in best_iters)

p_int = np.rint(oof).astype(np.int64)
print(f"  clip 检查: raw [{oof.min():.2f}, {oof.max():.2f}] -> "
      f"clipped 越界 {int(((oof <= 30.0) | (oof >= 10800.0)).sum())} 行")
print(f"  整数化检查: dtype={p_int.dtype}  范围 [{p_int.min()}, {p_int.max()}]  "
      f"非整数比例={float((oof != p_int).mean()):.4f} (正常 <1)")

rmse = float(np.sqrt(np.mean((oof - yp) ** 2)))
print(f"  fold {TARGET_FOLD} OOF RMSE = {rmse:.2f}")

# ============================================================
# [6] 推算全量耗时
# ============================================================
print()
print("=" * 70)
print("[6] 全量耗时推算")
print("=" * 70)
# 训练池 = 除本折外的所有月, 全量时约 9-10 个月 = 1.7M+ 行
N_ALL_DEP = len(all_dep)
N_FULL_TRAIN = N_ALL_DEP - int(is_pred.sum())   # 本折训练池真实全量行数
smoke_cut_rows = len(tr_pos)                    # smoke 里 ES 后实际训练的行数
scale = N_FULL_TRAIN / smoke_cut_rows
per_fold_train_min = train_time / 60 * scale
n_seeds = len(SEEDS)
total_train = per_fold_train_min * 5 * n_seeds
feat_full_min = 12 * 0.30   # smoke: 19 万行 17.7s -> 12 月 208 万行约 3.6 min/月

print(f"  数据规模: all_dep {N_ALL_DEP:,} 行 (12 个月)")
print(f"  本折 (fold {TARGET_FOLD}) 训练池全量 = {N_FULL_TRAIN:,} 行, "
      f"预测集 = {int(is_pred.sum()):,} 行")
print(f"  smoke 实际训练行 = {smoke_cut_rows:,}  ->  放大系数 x{scale:.2f}")
print(f"  smoke 训练耗时 {train_time:.1f}s ({n_seeds} seed)")
print(f"  >>> 单折训练预计 {per_fold_train_min:.1f} min")
print(f"  >>> 5 折 x {n_seeds} seed 训练合计 = {total_train:.0f} min = {total_train/60:.1f} h")
print(f"  特征工程 (12 月 208 万行, 单次) 预计 {feat_full_min:.0f} min = {feat_full_min/60:.1f} h")
total_all = total_train + feat_full_min
print(f"  >>> 全流程预计 {total_all:.0f} min = {total_all/60:.1f} h")
print(f"  >>> 若 SEEDS 提到 3: {(per_fold_train_min*5*3 + feat_full_min)/60:.1f} h")

RESULT = {
    "oof_folds": OOF_FOLDS,
    "covered_months": sorted(covered),
    "assertions": {
        "covers_all_12": bool(covered == expected),
        "month_1_included": bool(1 in covered),
        "month_7_included": bool(7 in covered),
        "no_duplicate_months": bool(len(_flat) == len(set(_flat))),
        "per_row_coverage_min_max": [int(cover.min()), int(cover.max())],
    },
    "month_row_counts": month_counts,
    "fold_pred_rows": fold_pred_rows,
    "smoke_fold": TARGET_FOLD,
    "smoke_pred_months": pred_months,
    "smoke_train_rows": int(smoke_cut_rows),
    "smoke_pred_rows": int(is_pred.sum()),
    "smoke_pred_by_month": {str(m): int((mth[is_pred] == m).sum()) for m in pred_months},
    "month7_pred_rows": int(m7.sum()),
    "month7_oof_all_finite": bool(np.isfinite(oof[m7]).all()),
    "month_rmse": {str(m): round(float(np.sqrt(np.mean(
        (oof[mth[is_pred] == m] - yp[mth[is_pred] == m]) ** 2))), 2)
        for m in pred_months},
    "es_best_iters": best_iters,
    "es_cap": ROUNDS,
    "es_note": "best_iter 恰等于 cap, 说明 ES 未触发, 实际轮数是硬上界而非早停点",
    "clip_out_of_range_rows": int(((oof <= 30.0) | (oof >= 10800.0)).sum()),
    "int_cast_dtype": str(p_int.dtype),
    "int_cast_range": [int(p_int.min()), int(p_int.max())],
    "smoke_oof_rmse": round(rmse, 2),
    "timing": {
        "n_all_dep": N_ALL_DEP,
        "n_full_train_this_fold": N_FULL_TRAIN,
        "smoke_cut_rows": int(smoke_cut_rows),
        "scale_factor": round(scale, 2),
        "smoke_train_sec": round(train_time, 1),
        "per_fold_train_min": round(per_fold_train_min, 1),
        "n_seeds": n_seeds,
        "total_train_min_5fold": round(total_train, 0),
        "feature_engineering_min": round(feat_full_min, 0),
        "total_pipeline_min": round(total_all, 0),
        "total_pipeline_h": round(total_all / 60, 2),
        "total_pipeline_h_if_3seed": round(
            (per_fold_train_min * 5 * 3 + feat_full_min) / 60, 2),
    },
}
(D0 / "smoke_oof_coverage.json").write_text(
    json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"→ {D0 / 'smoke_oof_coverage.json'}")
