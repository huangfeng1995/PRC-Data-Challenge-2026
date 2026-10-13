"""
(ADEP, STAND, RUNWAY) TE add-one v3
====================================
路径安全: 写到 /Users/openclaw/Documents/PRC_data_challenge/ 而不是 notebooks/diag/d0/
"""
import sys
import json
import time
import gc
import subprocess
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0  # 写到 notebooks/diag/d0/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

spec = importlib.util.spec_from_file_location(
    "ab_module", ROOT / "notebooks/diag/holdout_v1_queue_ab.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
import src.train_xgb as tx

# 动态 import d0 目录下的模块
_r7_lib_spec = importlib.util.spec_from_file_location(
    "_r7_lib", D0 / "_r7_lib.py")
L = importlib.util.module_from_spec(_r7_lib_spec)
_r7_lib_spec.loader.exec_module(L)

_r8_drift_lib_spec = importlib.util.spec_from_file_location(
    "_r8_drift_lib", D0 / "_r8_drift_lib.py")
Dlib = importlib.util.module_from_spec(_r8_drift_lib_spec)
_r8_drift_lib_spec.loader.exec_module(Dlib)

LOG_PATH = OUT / "_te_stand_rwy.log"
JSON_PATH = OUT / "te_stand_rwy.json"

print("=" * 70)
print("Task TE (ADEP, STAND, RUNWAY) add-one v3 (安全路径)")
print("=" * 70)
t0 = time.time()


def load_months(months):
    dfs = []
    for m in months:
        if m < 12:
            fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        else:
            fname = "training_2025-12-01_2026-01-01.parquet"
        dfs.append(ab.load_month(str(ROOT / "data/raw" / fname),
                                  columns=ab.TRAIN_COLS))
    df = pd.concat(dfs, ignore_index=True)
    df = ab.parse_time_columns(df, cols=ab.TIME_COLS)
    return df


def build_v15_71(df_raw, label):
    gd_cols = ab.get_feat_cols_for_group(df_raw, ab.GROUPS["Gd_ABD_runway"])
    FEATS69 = list(dict.fromkeys(
        gd_cols + L.MVT_FEATS_R5 + L.G2_FEATS + L.G3_FEATS + L.G5_FEATS + L.G7_FEATS))
    df_tr_dep_all = df_raw[df_raw["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    df_tr_arr_all = df_raw[df_raw["PHASE_mvt"] == "ARR"].reset_index(drop=True)
    D_tr = Dlib.compute_drift_features(df_tr_dep_all, ref_dep=None, ref_all=None,
                                       label=label)
    df_full = ab.build_full_features(df_raw.copy())
    mask = (df_full["TAXITIME_SEC_mvt"] > 0) & (df_full["TAXITIME_SEC_mvt"] < 10800)
    df_full = df_full[mask].reset_index(drop=True)
    D_tr["MVT_ID_mvt"] = df_tr_dep_all["MVT_ID_mvt"].values
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    for c in Dlib.D_FEATS:
        if c not in df_full.columns:
            df_full[c] = np.nan
        mapping = dict(zip(D_tr["MVT_ID_mvt"], D_tr[c]))
        df_full.loc[dep_mask, c] = df_full.loc[dep_mask, "MVT_ID_mvt"].map(mapping)
    del D_tr, df_tr_dep_all, df_tr_arr_all; gc.collect()
    FEATS71 = list(dict.fromkeys(FEATS69 + Dlib.D_FEATS))
    # 兜底: 缺失列补 NaN (修 bug: build_v15_71 未 join MVT 锚定 / G2-G7 列)
    for c in FEATS71:
        if c not in df_full.columns:
            df_full[c] = np.nan
    return df_full, FEATS71


def te_oof_train(df_full):
    """全局 5 折 KFold OOF, 4 层中位数, 严格防泄漏"""
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    dep_idx = np.where(dep_mask)[0]
    n_dep = dep_idx.size
    n_full = len(df_full)
    print(f"  TE OOF: DEP 行 {n_dep:,}, full 行 {n_full:,}", flush=True)

    L1 = np.full(n_dep, np.nan)
    L2 = np.full(n_dep, np.nan)
    L3 = np.full(n_dep, np.nan)
    L4 = np.full(n_dep, np.nan)

    dep_df = df_full.iloc[dep_idx][
        ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
         "TAXITIME_SEC_mvt"]].copy().reset_index(drop=True)

    t = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    g_A = dep_df["ADEP_mvt"].values
    g_S = dep_df["STAND_mvt"].values
    g_R = dep_df["RUNWAY_mvt"].values
    g_y = dep_df["TAXITIME_SEC_mvt"].values

    for fold_i, (tr_pos, va_pos) in enumerate(kf.split(np.arange(n_dep))):
        tr_y = g_y[tr_pos]
        tr_a = g_A[tr_pos]
        tr_s = g_S[tr_pos]
        tr_r = g_R[tr_pos]
        df_l1 = pd.DataFrame({"y": tr_y, "a": tr_a, "s": tr_s, "r": tr_r})
        m1 = df_l1.groupby(["a", "s", "r"])["y"].median()
        m2 = df_l1.groupby(["a", "r"])["y"].median()
        m3 = df_l1.groupby(["a", "s"])["y"].median()
        m4 = df_l1.groupby("a")["y"].median()
        va_a = g_A[va_pos]
        va_s = g_S[va_pos]
        va_r = g_R[va_pos]
        idx_l1 = pd.MultiIndex.from_arrays([va_a, va_s, va_r])
        idx_l2 = pd.MultiIndex.from_arrays([va_a, va_r])
        idx_l3 = pd.MultiIndex.from_arrays([va_a, va_s])
        L1[va_pos] = m1.reindex(idx_l1).values
        L2[va_pos] = m2.reindex(idx_l2).values
        L3[va_pos] = m3.reindex(idx_l3).values
        L4[va_pos] = m4.reindex(va_a).values
        print(f"  TE OOF fold{fold_i}: 训练 {len(tr_pos):,}, 验证 {len(va_pos):,}",
              flush=True)
    print(f"  TE OOF 耗时 {time.time()-t:.1f}s", flush=True)

    new_feats = {
        "te_L1_median": np.full(n_full, np.nan),
        "te_L2_median": np.full(n_full, np.nan),
        "te_L3_median": np.full(n_full, np.nan),
        "te_L4_median": np.full(n_full, np.nan),
    }
    new_feats["te_L1_median"][dep_idx] = L1
    new_feats["te_L2_median"][dep_idx] = L2
    new_feats["te_L3_median"][dep_idx] = L3
    new_feats["te_L4_median"][dep_idx] = L4
    return new_feats


def te_lut_apply(df_target, df_train_full):
    dep_mask = (df_target["PHASE_mvt"] == "DEP").values
    dep_idx = np.where(dep_mask)[0]
    n_dep = dep_idx.size
    tr_dep = df_train_full[df_train_full["PHASE_mvt"] == "DEP"]
    m1 = tr_dep.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])[
        "TAXITIME_SEC_mvt"].median()
    m2 = tr_dep.groupby(["ADEP_mvt", "RUNWAY_mvt"])[
        "TAXITIME_SEC_mvt"].median()
    m3 = tr_dep.groupby(["ADEP_mvt", "STAND_mvt"])[
        "TAXITIME_SEC_mvt"].median()
    m4 = tr_dep.groupby(["ADEP_mvt"])["TAXITIME_SEC_mvt"].median()
    sub = df_target.iloc[dep_idx][
        ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]].copy()
    vk1 = pd.MultiIndex.from_arrays([sub["ADEP_mvt"].values,
                                      sub["STAND_mvt"].values,
                                      sub["RUNWAY_mvt"].values])
    vk2 = pd.MultiIndex.from_arrays([sub["ADEP_mvt"].values,
                                      sub["RUNWAY_mvt"].values])
    vk3 = pd.MultiIndex.from_arrays([sub["ADEP_mvt"].values,
                                      sub["STAND_mvt"].values])
    L1 = m1.reindex(vk1).values
    L2 = m2.reindex(vk2).values
    L3 = m3.reindex(vk3).values
    L4 = m4.reindex(sub["ADEP_mvt"].values).values
    n_l1 = int(np.sum(~np.isnan(L1)))
    n_l2 = int(np.sum(np.isnan(L1) & ~np.isnan(L2)))
    n_l3 = int(np.sum(np.isnan(L1) & np.isnan(L2) & ~np.isnan(L3)))
    n_l4 = int(np.sum(np.isnan(L1) & np.isnan(L2) & np.isnan(L3) & ~np.isnan(L4)))
    hit = {"L1_hit": n_l1, "L1_pct": n_l1 / n_dep,
           "L2_hit": n_l2, "L2_pct": n_l2 / n_dep,
           "L3_hit": n_l3, "L3_pct": n_l3 / n_dep,
           "L4_hit": n_l4, "L4_pct": n_l4 / n_dep,
           "total_dep": int(n_dep)}
    new_feats = {
        "te_L1_median": np.full(len(df_target), np.nan),
        "te_L2_median": np.full(len(df_target), np.nan),
        "te_L3_median": np.full(len(df_target), np.nan),
        "te_L4_median": np.full(len(df_target), np.nan),
    }
    new_feats["te_L1_median"][dep_idx] = L1
    new_feats["te_L2_median"][dep_idx] = L2
    new_feats["te_L3_median"][dep_idx] = L3
    new_feats["te_L4_median"][dep_idx] = L4
    return new_feats, hit


def run_fold(name, train_months, val_months):
    print(f"\n{'=' * 70}\n[{name}] train={train_months} val={val_months}\n{'=' * 70}",
          flush=True)
    t_f = time.time()
    df_raw_tr = load_months(train_months)
    df_raw_va = load_months(val_months)
    print(f"  raw: train {len(df_raw_tr):,}, val {len(df_raw_va):,}", flush=True)

    df_tr, FEATS71 = build_v15_71(df_raw_tr, f"{name}-tr")
    df_va, _ = build_v15_71(df_raw_va, f"{name}-va")

    new_feats_tr = te_oof_train(df_tr)
    for c, arr in new_feats_tr.items():
        df_tr[c] = arr
    new_feats_va, val_hit = te_lut_apply(df_va, df_tr)
    for c, arr in new_feats_va.items():
        df_va[c] = arr

    del df_raw_tr, df_raw_va; gc.collect()

    df_tr_d = df_tr[df_tr["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    df_va_d = df_va[df_va["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    print(f"  DEP: train {len(df_tr_d):,}, val {len(df_va_d):,}", flush=True)
    y_tr = df_tr_d["TAXITIME_SEC_mvt"].values
    y_va = df_va_d["TAXITIME_SEC_mvt"].values

    te_feats = list(new_feats_tr.keys())
    FEATS75 = FEATS71 + te_feats

    cat_path = OUT / f"te_{name}_v3_cat.json"
    if cat_path.exists():
        cat_path.unlink()
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = cat_path

    X_tr_v15 = tx._encode_categoricals(df_tr_d[FEATS71].copy(), fit=True)
    X_va_v15 = tx._encode_categoricals(df_va_d[FEATS71].copy(), fit=False)
    X_tr_v75 = tx._encode_categoricals(df_tr_d[FEATS75].copy(), fit=False)
    X_va_v75 = tx._encode_categoricals(df_va_d[FEATS75].copy(), fit=False)

    base_params_xgb = ab.DEFAULT_PARAMS
    if not hasattr(ab, "DEFAULT_PARAMS_LGB"):
        base_params_lgb = {
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
            "min_data_in_leaf": 50, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
        }
    else:
        base_params_lgb = ab.DEFAULT_PARAMS_LGB

    seed_v15 = []
    seed_v75 = []
    feat_imp_te_xgb = []
    feat_imp_te_lgb = []

    X_tr_v15_arr = X_tr_v15.values.astype(np.float32)
    X_va_v15_arr = X_va_v15.values.astype(np.float32)
    X_tr_v75_arr = X_tr_v75.values.astype(np.float32)
    X_va_v75_arr = X_va_v75.values.astype(np.float32)

    for seed in [0, 1, 2]:
        params_xgb = {**base_params_xgb, "seed": seed}
        dtr_v15 = xgb.DMatrix(X_tr_v15_arr, label=y_tr)
        dva_v15 = xgb.DMatrix(X_va_v15_arr, label=y_va)
        dtr_v75 = xgb.DMatrix(X_tr_v75_arr, label=y_tr)
        dva_v75 = xgb.DMatrix(X_va_v75_arr, label=y_va)
        bst_v15 = xgb.train(params_xgb, dtr_v15, num_boost_round=1200,
                             verbose_eval=False)
        bst_v75 = xgb.train(params_xgb, dtr_v75, num_boost_round=1200,
                             verbose_eval=False)
        p_v15 = bst_v15.predict(dva_v15)
        p_v75 = bst_v75.predict(dva_v75)
        seed_v15.append(float(np.sqrt(np.mean((y_va - p_v15) ** 2))))
        seed_v75.append(float(np.sqrt(np.mean((y_va - p_v75) ** 2))))
        s_v75 = bst_v75.get_score(importance_type="gain")
        for i, c in enumerate(FEATS75):
            k = f"f{i}"
            if k in s_v75 and c in te_feats:
                feat_imp_te_xgb.append({"seed": seed, "feat": c, "gain": s_v75[k]})
        del bst_v15, bst_v75, dtr_v15, dva_v15, dtr_v75, dva_v75; gc.collect()

        params_lgb = {**base_params_lgb, "seed": seed}
        dtr_v15_lgb = lgb.Dataset(X_tr_v15_arr, label=y_tr)
        dva_v15_lgb = lgb.Dataset(X_va_v15_arr, label=y_va)
        dtr_v75_lgb = lgb.Dataset(X_tr_v75_arr, label=y_tr)
        dva_v75_lgb = lgb.Dataset(X_va_v75_arr, label=y_va)
        m_v15 = lgb.train(params_lgb, dtr_v15_lgb, num_boost_round=1200,
                           valid_sets=[dva_v15_lgb],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
        m_v75 = lgb.train(params_lgb, dtr_v75_lgb, num_boost_round=1200,
                           valid_sets=[dva_v75_lgb],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
        p_v15_lgb = m_v15.predict(X_va_v15_arr)
        p_v75_lgb = m_v75.predict(X_va_v75_arr)
        seed_v15.append(float(np.sqrt(np.mean((y_va - p_v15_lgb) ** 2))))
        seed_v75.append(float(np.sqrt(np.mean((y_va - p_v75_lgb) ** 2))))
        imp_v75 = m_v75.feature_importance(importance_type="gain")
        for i, c in enumerate(FEATS75):
            if c in te_feats:
                feat_imp_te_lgb.append({"seed": seed, "feat": c, "gain": float(imp_v75[i])})
        del m_v15, m_v75, dtr_v15_lgb, dva_v15_lgb, dtr_v75_lgb, dva_v75_lgb; gc.collect()

    rmse_v15 = float(np.mean(seed_v15))
    rmse_v75 = float(np.mean(seed_v75))
    print(f"  [{name}] v15 6 模型集成 RMSE = {rmse_v15:.4f}", flush=True)
    print(f"  [{name}] +TE 6 模型集成 RMSE = {rmse_v75:.4f}", flush=True)
    print(f"  [{name}] Δ = {rmse_v75 - rmse_v15:+.4f}", flush=True)

    # 分 val 月份集成预测
    p_v15_ens = np.zeros(len(y_va))
    p_v75_ens = np.zeros(len(y_va))
    for seed in [0, 1, 2]:
        params_xgb = {**base_params_xgb, "seed": seed}
        bst_v15 = xgb.train(params_xgb, xgb.DMatrix(X_tr_v15_arr, label=y_tr),
                             num_boost_round=1200, verbose_eval=False)
        bst_v75 = xgb.train(params_xgb, xgb.DMatrix(X_tr_v75_arr, label=y_tr),
                             num_boost_round=1200, verbose_eval=False)
        p_v15_ens += bst_v15.predict(xgb.DMatrix(X_va_v15_arr))
        p_v75_ens += bst_v75.predict(xgb.DMatrix(X_va_v75_arr))
        del bst_v15, bst_v75; gc.collect()
    for seed in [0, 1, 2]:
        params_lgb = {**base_params_lgb, "seed": seed}
        m_v15 = lgb.train(params_lgb, lgb.Dataset(X_tr_v15_arr, label=y_tr),
                           num_boost_round=1200,
                           valid_sets=[lgb.Dataset(X_va_v15_arr, label=y_va)],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
        m_v75 = lgb.train(params_lgb, lgb.Dataset(X_tr_v75_arr, label=y_tr),
                           num_boost_round=1200,
                           valid_sets=[lgb.Dataset(X_va_v75_arr, label=y_va)],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
        p_v15_ens += m_v15.predict(X_va_v15_arr)
        p_v75_ens += m_v75.predict(X_va_v75_arr)
        del m_v15, m_v75; gc.collect()
    p_v15_ens /= 6
    p_v75_ens /= 6

    month_rmse = {}
    val_month_col = df_va_d["MVT_TIME_UTC_mvt"].dt.month.values
    for m in sorted(set(val_month_col)):
        mask = val_month_col == m
        if mask.sum() == 0:
            continue
        rmse_v15_m = float(np.sqrt(np.mean((y_va[mask] - p_v15_ens[mask]) ** 2)))
        rmse_v75_m = float(np.sqrt(np.mean((y_va[mask] - p_v75_ens[mask]) ** 2)))
        month_rmse[f"month_{int(m):02d}"] = {
            "n": int(mask.sum()),
            "v15": rmse_v15_m,
            "+TE": rmse_v75_m,
            "Δ": rmse_v75_m - rmse_v15_m,
        }
    print(f"  [{name}] 分月: {month_rmse}", flush=True)
    print(f"  [{name}] val TE 命中率: {val_hit}", flush=True)
    print(f"  [{name}] 折耗时 {time.time()-t_f:.1f}s", flush=True)

    return {
        "rmse_v15_ens": rmse_v15,
        "rmse_v75_ens": rmse_v75,
        "delta": rmse_v75 - rmse_v15,
        "per_month": month_rmse,
        "val_te_hit": val_hit,
        "feat_imp_te_xgb": feat_imp_te_xgb,
        "feat_imp_te_lgb": feat_imp_te_lgb,
        "wallclock_sec": time.time() - t_f,
    }


fold_results = {}
for name, tr_m, va_m in [
    ("holdout", [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], [1, 7]),
    ("long_span", [1, 2, 3], [7]),
    ("fold2", [1, 2, 3, 4, 5], [7]),
]:
    fold_results[name] = run_fold(name, tr_m, va_m)

print(f"\n{'=' * 70}\n[ranking TE 命中率]\n{'=' * 70}", flush=True)
df_rank_raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
df_rank_raw = ab.parse_time_columns(df_rank_raw, cols=ab.TIME_COLS)
df_rank_full = ab.build_full_features(df_rank_raw.copy())
# ranking 不应用 TAXITIME mask (ranking DEP 行 TAXITIME 为 NaN 是待预测目标, 不是要过滤的脏数据)
df_rank_full = df_rank_full.reset_index(drop=True)
df_2025_raw = load_months(list(range(1, 13)))
df_2025_full, _ = build_v15_71(df_2025_raw, "rank-2025")
rank_feats, rank_hit = te_lut_apply(df_rank_full, df_2025_full)
print(f"  ranking TE 命中率: {rank_hit}", flush=True)

print(f"\n{'=' * 70}\n[gate_check]\n{'=' * 70}", flush=True)
gc_result = {}
try:
    out = subprocess.run(
        ["python3", str(ROOT / "scripts/gate_check.py")],
        capture_output=True, text=True, timeout=300)
    gc_result["returncode"] = out.returncode
    gc_result["stdout_tail"] = out.stdout[-3000:]
    gc_result["stderr_tail"] = out.stderr[-1000:]
except Exception as e:
    gc_result["error"] = str(e)

final = {
    "task": "(ADEP, STAND, RUNWAY) TE add-one v3, XGB+LGBM × 3 seed = 6 模型",
    "folds": fold_results,
    "ranking_te_hit": rank_hit,
    "gate_check": gc_result,
    "wallclock_total_sec": time.time() - t0,
}
with open(JSON_PATH, "w") as f:
    json.dump(final, f, indent=2, ensure_ascii=False, default=str)
print(f"\n✅ saved -> {JSON_PATH}", flush=True)
print(f"总耗时 {time.time()-t0:.1f}s", flush=True)
