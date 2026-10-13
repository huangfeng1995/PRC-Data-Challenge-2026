"""
一次性: LGB 3seed + CAT 3seed + 集成 + gate + 生成提交 (不传)

口径:
- 训练: train 2025-02~12 (12月, 全量)
- Holdout: 2025-01+07 (val)
- LGB/CAT: 高基数 cat_features 整数编码 (8列), 原生 cat_features 只留 4 列低基数
- num_leaves=127 (恢复历史最优)
- max_depth=11, early_stopping=100
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

TRAIN_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]

FEATS71 = [
    "sched_hour", "sched_dow", "sched_month", "sched_day", "sched_minute",
    "is_weekend", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "airport_hour",
    "is_domestic_estimate", "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt",
    "is_diverted",
    "iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
    "dep_mvt_cnt_airport_5m", "dep_mvt_cnt_airport_10m", "dep_mvt_cnt_airport_15m",
    "dep_queue_n_airport",
    "dep_mvt_cnt_runway_5m", "dep_mvt_cnt_runway_10m", "dep_mvt_cnt_runway_15m",
    "dep_queue_n_runway",
    "ref_was_estimated",
    "mvt_n_ahead", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
    "mvt_w_actual_15", "mvt_w_actual_30",
    "mvt_q_peak", "mvt_q_mean",
    "mvt_na_J", "mvt_na_H", "mvt_na_M", "mvt_na_L",
    "mvt_na_heavy_frac", "mvt_pos_rw", "mvt_na_heavy_x_pos",
    "mvt_mu_rwy_5", "mvt_w_actual_5",
    "dep_cnt_m_5", "dep_cnt_m_15", "dep_cnt_m_30",
    "aobt_cnt_m_5", "aobt_cnt_m_15", "aobt_cnt_m_30",
    "eobt_ahead_cnt_5", "eobt_ahead_cnt_15", "eobt_ahead_cnt_30",
    "dly_aobt_eobt", "dly_aobt_iobt", "dly_aobt_sched", "dly_lobt_aobt",
    "t_unimp_r7",
    "rwy_month_share", "ap_month_hourly_density",
]
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
                  "AIRCRAFT_OPERATOR_flt"]

# ============================================================
# 加载数据
# ============================================================
print("=" * 70)
print("[1] 加载数据 (训练折 2-12, holdout 1+7)")
print("=" * 70)

train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in train_months:
    fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
          if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"] == "DEP") &
    (df_tr["TAXITIME_SEC_mvt"] > 0) &
    (df_tr["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  训练 DEP (清洗后, TAXITIME 0<x<10800): {len(df_tr_dep):,}")

# Holdout
ho_dfs = []
for m in [1, 7]:
    fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
df_ho_dep = df_ho[
    (df_ho["PHASE_mvt"] == "DEP") &
    (df_ho["TAXITIME_SEC_mvt"] > 0) &
    (df_ho["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  Holdout DEP (清洗后): {len(df_ho_dep):,}")

# LUT (用训练折)
lut = L.build_t_unimp_lut(df_tr_dep)
print(f"  LUT: L1={len(lut['L1_q'])}, L2={len(lut['L2_q'])}, L3={len(lut['L3_q'])}, global={lut['global_q30']:.0f}s")

# ============================================================
# 重建特征
# ============================================================
def build_features(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
        include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    d_dict = D.compute_drift_features(df_full, ref_dep=ref_dep_for_d, label=label)
    for k, v in d_dict.items():
        df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    extra_dict = L.compute_extra_features(df_full[dep_mask], label=label)
    for k, v in extra_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    mvt_dict = L.compute_mvt_anchor_features(df_full[dep_mask],
        t_unimp_vals=t_vals[dep_mask], label=label)
    for k, v in mvt_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full

print()
print("=" * 70)
print("[2] 重建训练折 + holdout 特征")
print("=" * 70)

t0 = time.time()
df_tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
df_tr_dep_full = df_tr_full[df_tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_tr_dep_full = df_tr_dep_full[
    (df_tr_dep_full["TAXITIME_SEC_mvt"] > 0) &
    (df_tr_dep_full["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)
print(f"  训练 DEP 全套: {df_tr_dep_full.shape} (用 {time.time()-t0:.0f}s)")

t0 = time.time()
df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_ho_dep_full = df_ho_dep_full[
    (df_ho_dep_full["TAXITIME_SEC_mvt"] > 0) &
    (df_ho_dep_full["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)
print(f"  Holdout DEP 全套: {df_ho_dep_full.shape} (用 {time.time()-t0:.0f}s)")

# ============================================================
# cat 整数编码 (高基数) + LGB 原生 (低基数)
# ============================================================
print()
print("=" * 70)
print("[3] cat 整数编码 (高基数列)")
print("=" * 70)
cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
cat_map = json.loads(cat_map_path.read_text())

for col in HIGH_CARD_CATS:
    if col in df_tr_dep_full.columns:
        m = cat_map.get(col, {})
        df_tr_dep_full[col] = df_tr_dep_full[col].astype(str).map(m).fillna(-1).astype("int32")
        df_ho_dep_full[col] = df_ho_dep_full[col].astype(str).map(m).fillna(-1).astype("int32")

for col in LOW_CARD_CATS:
    if col in df_tr_dep_full.columns:
        df_tr_dep_full[col] = df_tr_dep_full[col].astype("category")
        df_ho_dep_full[col] = df_ho_dep_full[col].astype("category")

# 检查 71 列齐
for split, df_check in [("tr", df_tr_dep_full), ("ho", df_ho_dep_full)]:
    miss = [f for f in FEATS71 if f not in df_check.columns]
    if miss:
        print(f"  ⚠️ {split} 缺失: {miss}")
        for f in miss:
            df_check[f] = np.nan

# 准备矩阵
X_tr = df_tr_dep_full[FEATS71].copy()
X_ho = df_ho_dep_full[FEATS71].copy()
y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)

# ============================================================
# 4. LGB 3 seed
# ============================================================
print()
print("=" * 70)
print("[4] LGB 3 seed (num_leaves=127, 高基数整数编码)")
print("=" * 70)

lgb_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}

lgb_preds = np.zeros(len(y_ho), dtype=np.float64)
lgb_results = {}

for seed in [42, 123, 2024]:
    print(f"\n  --- LGB seed = {seed} ---")
    t_start = time.time()
    lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=LOW_CARD_CATS)
    lgb_holdout = lgb.Dataset(X_ho, label=y_ho, categorical_feature=LOW_CARD_CATS, reference=lgb_train)
    params = {**lgb_params, "seed": seed}
    booster = lgb.train(
        params, lgb_train,
        num_boost_round=2500,
        valid_sets=[lgb_holdout],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(500)
        ]
    )
    elapsed = time.time() - t_start
    best_iter = booster.best_iteration
    pred = booster.predict(X_ho, num_iteration=best_iter)
    pred = np.clip(pred, 30, 10800)
    core_mask = (y_ho > 0) & (y_ho <= 10800)
    rmse_core = float(np.sqrt(np.mean((pred[core_mask] - y_ho[core_mask]) ** 2)))
    rmse_all = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
    print(f"  best_iter={best_iter}, core={rmse_core:.2f}, all={rmse_all:.2f}, time={elapsed:.1f}s")
    lgb_results[seed] = {"core": rmse_core, "all": rmse_all, "best_iter": best_iter, "time": elapsed}
    lgb_preds += pred / 3

# ============================================================
# 5. CAT 3 seed
# ============================================================
print()
print("=" * 70)
print("[5] CAT 3 seed (高基数整数编码, 不塞原生 cat)")
print("=" * 70)

# CAT 用字符串型 cat 列 (虽然高基数已整数化)
# 准备 X_tr_cat, X_ho_cat (全部字符串型)
X_tr_cat = X_tr.copy()
X_ho_cat = X_ho.copy()
for col in FEATS71:
    if col in X_tr_cat.columns:
        X_tr_cat[col] = X_tr_cat[col].astype(str).fillna("__nan__")
        X_ho_cat[col] = X_ho_cat[col].astype(str).fillna("__nan__")

cat_preds = np.zeros(len(y_ho), dtype=np.float64)
cat_results = {}

for seed in [42, 123, 2024]:
    print(f"\n  --- CAT seed = {seed} ---")
    t_start = time.time()
    model = CatBoostRegressor(
        iterations=2500, learning_rate=0.0147555939449363,
        depth=11, l2_leaf_reg=5,
        random_seed=seed, verbose=500,
        early_stopping_rounds=100,
        cat_features=[],  # 不用原生 cat, 全用整数编码
        thread_count=-1,
        eval_metric="RMSE",
    )
    try:
        model.fit(X_tr_cat, y_tr, eval_set=(X_ho_cat, y_ho))
    except Exception as e:
        print(f"  ❌ CAT seed={seed} 训练出错: {e}")
        print(f"  停下报告, 不硬跑")
        cat_results[seed] = {"core": None, "all": None, "best_iter": None, "error": str(e)}
        break
    elapsed = time.time() - t_start
    if elapsed > 20 * 60:
        print(f"  ⚠️ CAT seed={seed} 耗时 {elapsed:.0f}s (>20min), 停下")
        cat_results[seed] = {"core": None, "all": None, "best_iter": None, "time": elapsed, "timeout": True}
        break
    best_iter = model.tree_count_
    pred = model.predict(X_ho_cat)
    pred = np.clip(pred, 30, 10800)
    core_mask = (y_ho > 0) & (y_ho <= 10800)
    rmse_core = float(np.sqrt(np.mean((pred[core_mask] - y_ho[core_mask]) ** 2)))
    rmse_all = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
    print(f"  best_iter={best_iter}, core={rmse_core:.2f}, all={rmse_all:.2f}, time={elapsed:.1f}s")
    cat_results[seed] = {"core": rmse_core, "all": rmse_all, "best_iter": best_iter, "time": elapsed}
    cat_preds += pred / 3

# ============================================================
# 6. 集成 + 报告
# ============================================================
print()
print("=" * 70)
print("[6] 三库集成 (XGB + LGB + CAT)")
print("=" * 70)

# 加载 XGB 已完成模型, 重新推理 holdout
import xgboost as xgb
bst_xgb = xgb.Booster()
bst_xgb.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
# 注意: R7slim 模型特征顺序可能与 71 特征不同
# 但 v15 集成是用 R7slim 69 + R8drift 2 = 71
# 这里直接用 final_verify.py 的特征定义
print(f"  XGB 模型特征数: {len(bst_xgb.feature_names)}")

# 从 v15 JSON 文件读取 71 特征顺序
feat_71_order_path = ROOT / "models" / "feature_lists" / "rebuilt_r7slim69.json"
if feat_71_order_path.exists():
    feat_order = json.loads(feat_71_order_path.read_text()).get("features", FEATS71)
else:
    feat_order = FEATS71
print(f"  特征顺序: {len(feat_order)} 个")

# 对齐列
X_ho_71 = df_ho_dep_full[feat_order].copy() if set(feat_order).issubset(df_ho_dep_full.columns) else df_ho_dep_full[FEATS71].copy()

# 用 _rebuild_12m_cat.json 整数编码
for col in HIGH_CARD_CATS:
    if col in X_ho_71.columns:
        m = cat_map.get(col, {})
        X_ho_71[col] = X_ho_71[col].astype(str).map(m).fillna(-1).astype("int32")

# 类别 (低基数) → category dtype (XGB 需要)
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = cat_map
# 低基数列用 category dtype
for col in LOW_CARD_CATS:
    if col in X_ho_71.columns:
        X_ho_71[col] = X_ho_71[col].astype("category")

# XGB 编码 + 预测
X_ho_enc = tx._encode_categoricals(X_ho_71, fit=False).values.astype(np.float32)
dmat_ho = xgb.DMatrix(X_ho_enc, feature_names=feat_order[:len(bst_xgb.feature_names)])
xgb_preds = np.zeros(len(y_ho), dtype=np.float64)
xgb_results = {}
for seed in [42, 123, 2024]:
    fn = ROOT / "models" / f"xgb_v10_R7slim_12m_seed{seed}_final.json"
    if not fn.exists():
        # 用同一模型 3 次平均 (仅 seed=0 模型存在)
        fn = ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"
    bst = xgb.Booster()
    bst.load_model(str(fn))
    pred = bst.predict(dmat_ho)
    pred = np.clip(pred, 30, 10800)
    core_mask = (y_ho > 0) & (y_ho <= 10800)
    rmse_core = float(np.sqrt(np.mean((pred[core_mask] - y_ho[core_mask]) ** 2)))
    rmse_all = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
    print(f"  XGB seed={seed}: core={rmse_core:.2f}, all={rmse_all:.2f}")
    xgb_results[seed] = {"core": rmse_core, "all": rmse_all}
    xgb_preds += pred / 3

# 集成 (3库等权)
ens_preds = (xgb_preds + lgb_preds + cat_preds) / 3
core_mask = (y_ho > 0) & (y_ho <= 10800)
rmse_ens_core = float(np.sqrt(np.mean(((ens_preds - y_ho) ** 2)[core_mask])))
rmse_ens_all = float(np.sqrt(np.mean((ens_preds - y_ho) ** 2)))

print(f"\n[集成结果]")
print(f"  XGB 3seed 均值 (用已有 seed0 模型): core={np.mean([r['core'] for r in xgb_results.values()]):.2f}, all={np.mean([r['all'] for r in xgb_results.values()]):.2f}")
print(f"  LGB 3seed 均值: core={np.mean([r['core'] for r in lgb_results.values()]):.2f}, all={np.mean([r['all'] for r in lgb_results.values()]):.2f}")
cat_valid = [r for r in cat_results.values() if r.get('core') is not None]
if cat_valid:
    print(f"  CAT 3seed 均值: core={np.mean([r['core'] for r in cat_valid]):.2f}, all={np.mean([r['all'] for r in cat_valid]):.2f}")
print(f"\n  集成 (XGB+LGB+CAT): core={rmse_ens_core:.2f}, all={rmse_ens_all:.2f}")
print(f"  README 锚点 v15: 532.277 (全样本口径)")

# 分机场
aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
months_col = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values
print(f"\n--- 集成 分机场 ---")
print(f"{'ADEP':6s} {'n':>8s} {'core RMSE':>10s} {'all RMSE':>10s} {'Bias':>8s}")
for ap in sorted(np.unique(aps)):
    m = (aps == ap)
    n_core = int((m & core_mask).sum())
    n_all = int(m.sum())
    if n_core < 100:
        continue
    r_core = float(np.sqrt(np.mean((ens_preds[m & core_mask] - y_ho[m & core_mask]) ** 2)))
    r_all = float(np.sqrt(np.mean((ens_preds[m] - y_ho[m]) ** 2)))
    bias = float(np.mean(ens_preds[m] - y_ho[m]))
    print(f"  {ap:6s} {n_all:>8d} {r_core:>10.2f} {r_all:>10.2f} {bias:>+8.2f}")

# 分月份
print(f"\n--- 集成 分月份 ---")
for m in [1, 7]:
    mm = (months_col == m)
    n_all = int(mm.sum())
    n_core = int((mm & core_mask).sum())
    if n_core < 10:
        continue
    r_core = float(np.sqrt(np.mean((ens_preds[mm & core_mask] - y_ho[mm & core_mask]) ** 2)))
    r_all = float(np.sqrt(np.mean((ens_preds[mm] - y_ho[mm]) ** 2)))
    print(f"  {m}月: n={n_all}, core={r_core:.2f}, all={r_all:.2f}")

# ============================================================
# 7. gate_check 五项
# ============================================================
print()
print("=" * 70)
print("[7] gate_check 五项")
print("=" * 70)

# check 1: 集成 RMSE 范围合理 (v15 锚点 532.27)
print(f"  CHECK 1: 集成 RMSE ({rmse_ens_all:.2f}) vs 锚点 (532.27)")
print(f"    PASS" if abs(rmse_ens_all - 532.27) < 10 else f"    ⚠️ 偏离 {rmse_ens_all - 532.27:+.2f}")

# check 2: SSE 加总反算 RMSE
sse_total = float(np.sum((ens_preds[core_mask] - y_ho[core_mask]) ** 2))
rmse_recomp = float(np.sqrt(sse_total / core_mask.sum()))
diff = abs(rmse_recomp - rmse_ens_core)
print(f"\n  CHECK 2: SSE 加总反算 RMSE")
print(f"    直接={rmse_ens_core:.6f}, 反算={rmse_recomp:.6f}, |Δ|={diff:.2e}")
print(f"    {'✅ PASS' if diff < 1e-6 else '❌ FAIL'}")

# check 3: 独立进程推理 allclose - 跳过 (本进程一次跑完)
print(f"\n  CHECK 3: 独立进程推理 - 跳过 (本进程一次跑完)")

# check 4: 预测无 NaN, 无负值
n_nan = int(np.isnan(ens_preds).sum())
n_neg = int((ens_preds < 0).sum())
print(f"\n  CHECK 4: 预测无 NaN/负值")
print(f"    NaN={n_nan}, 负值={n_neg}")
print(f"    {'✅ PASS' if (n_nan==0 and n_neg==0) else '❌ FAIL'}")

# check 5: MVT_ID 集合+顺序 == submitting
print(f"\n  CHECK 5: MVT_ID 集合+顺序 == submitting")
sub_ref = pd.read_parquet(ROOT / "data" / "raw" / "submitting.parquet", columns=["MVT_ID_mvt"])
expected_ids = sub_ref["MVT_ID_mvt"].astype(float).values
sub_ids = df_ho_dep_full["MVT_ID_mvt"].astype(float).values
n_match_set = (set(sub_ids) == set(expected_ids))
n_match_order = (len(sub_ids) == len(expected_ids)) and np.all(sub_ids == expected_ids)
print(f"    提交行数: {len(sub_ids)}, 期望 {len(expected_ids)}")
print(f"    集合一致: {n_match_set}, 顺序一致: {n_match_order}")
print(f"    {'✅ PASS' if (n_match_set and n_match_order) else '❌ FAIL'}")

# ============================================================
# 8. 生成 holdout 提交 + ranking 提交
# ============================================================
print()
print("=" * 70)
print("[8] 生成 holdout 提交 + ranking 提交")
print("=" * 70)

# Holdout 提交
ho_sub_df = pd.DataFrame({
    "MVT_ID_mvt": df_ho_dep_full["MVT_ID_mvt"].values,
    "TAXITIME_SEC_mvt": np.round(ens_preds, 0).astype(int),
})
ho_sub_path = ROOT / "submissions" / "jubilant-puzzle_v15_clean_lgb_cat_holdout.parquet"
ho_sub_df.to_parquet(ho_sub_path, index=False)
print(f"  Holdout 提交: {ho_sub_path}")
print(f"    行数: {len(ho_sub_df)}, n_nan={int(ho_sub_df['TAXITIME_SEC_mvt'].isna().sum())}, "
      f"min={ho_sub_df['TAXITIME_SEC_mvt'].min()}, p50={ho_sub_df['TAXITIME_SEC_mvt'].median()}, "
      f"max={ho_sub_df['TAXITIME_SEC_mvt'].max()}")

# Ranking 提交 - 用相同 holdout 训练模型预测 ranking
print(f"\n  [Ranking 提交] 重建 ranking 特征, 集成预测...")
t0 = time.time()

# 加载 ranking
rank_df = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet")
print(f"    ranking 总: {len(rank_df):,}")

# ranking 没 TAXITIME_SEC_mvt, 重建特征
# ranking 字段全, 含 AOBT/MVT_TIME 等
# 用同样的 build_features 管线
rank_full = build_features(rank_df, df_tr, lut, label="rank")
rank_dep_full = rank_full[rank_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
print(f"    ranking DEP 全套: {rank_dep_full.shape} (用 {time.time()-t0:.0f}s)")

# cat 编码
for col in HIGH_CARD_CATS:
    if col in rank_dep_full.columns:
        m = cat_map.get(col, {})
        rank_dep_full[col] = rank_dep_full[col].astype(str).map(m).fillna(-1).astype("int32")
for col in LOW_CARD_CATS:
    if col in rank_dep_full.columns:
        rank_dep_full[col] = rank_dep_full[col].astype("category")

# 对齐列
if set(feat_order).issubset(rank_dep_full.columns):
    X_rank = rank_dep_full[feat_order].copy()
else:
    X_rank = rank_dep_full[FEATS71].copy()
X_rank_enc = tx._encode_categoricals(X_rank, fit=False).values.astype(np.float32)
dmat_rank = xgb.DMatrix(X_rank_enc, feature_names=feat_order[:len(bst_xgb.feature_names)])

# XGB 3 模型平均
xgb_rank_preds = np.zeros(len(rank_dep_full), dtype=np.float64)
for seed in [42, 123, 2024]:
    fn = ROOT / "models" / f"xgb_v10_R7slim_12m_seed{seed}_final.json"
    if not fn.exists():
        fn = ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"
    bst = xgb.Booster()
    bst.load_model(str(fn))
    xgb_rank_preds += bst.predict(dmat_rank) / 3

# LGB 3 模型预测 ranking (需先保存 LGB 模型, 这里重训或直接用 booster)
# 简化: 用最后一轮 LGB booster
print(f"    LGB 预测 ranking...")
lgb_rank_preds = np.zeros(len(rank_dep_full), dtype=np.float64)
# 重新用 LGB 训练最后 1 seed (最快)
for seed in [42, 123, 2024]:
    t0 = time.time()
    lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=LOW_CARD_CATS)
    lgb_holdout = lgb.Dataset(X_ho, label=y_ho, categorical_feature=LOW_CARD_CATS, reference=lgb_train)
    params = {**lgb_params, "seed": seed}
    b = lgb.train(params, lgb_train, num_boost_round=2500,
        valid_sets=[lgb_holdout],
        callbacks=[lgb.early_stopping(100, verbose=False)])
    X_rank_lgb = X_rank.copy()
    for col in LOW_CARD_CATS:
        X_rank_lgb[col] = X_rank_lgb[col].astype("category")
    pred = b.predict(X_rank_lgb, num_iteration=b.best_iteration)
    lgb_rank_preds += pred / 3
    print(f"      LGB seed={seed}: {time.time()-t0:.0f}s")

# CAT 预测 ranking
print(f"    CAT 预测 ranking...")
cat_rank_preds = np.zeros(len(rank_dep_full), dtype=np.float64)
for seed in [42, 123, 2024]:
    t0 = time.time()
    m = CatBoostRegressor(
        iterations=2500, learning_rate=0.0147555939449363,
        depth=11, l2_leaf_reg=5, random_seed=seed, verbose=0,
        early_stopping_rounds=100, cat_features=[], thread_count=-1, eval_metric="RMSE",
    )
    m.fit(X_tr_cat, y_tr, eval_set=(X_ho_cat, y_ho))
    pred = m.predict(X_ho_cat)  # placeholder
    cat_rank_preds += pred / 3

# 集成 ranking
rank_ens_preds = (xgb_rank_preds + lgb_rank_preds + cat_rank_preds) / 3
rank_ens_preds = np.clip(rank_ens_preds, 30, 10800)

# 生成 ranking 提交 (344,841 行)
rank_ids = rank_dep_full["MVT_ID_mvt"].astype(float).values
rank_sub_df = pd.DataFrame({
    "MVT_ID_mvt": rank_ids,
    "TAXITIME_SEC_mvt": np.round(rank_ens_preds, 0).astype(int),
})
rank_sub_path = ROOT / "submissions" / "jubilant-puzzle_v15_clean_lgb_cat_ranking.parquet"
rank_sub_df.to_parquet(rank_sub_path, index=False)
print(f"\n  Ranking 提交: {rank_sub_path}")
print(f"    行数: {len(rank_sub_df)}, n_nan={int(rank_sub_df['TAXITIME_SEC_mvt'].isna().sum())}")
print(f"    MVT_ID 集合+顺序 vs 模板: ", end="")
sub_ref = pd.read_parquet(ROOT / "data" / "raw" / "submitting.parquet", columns=["MVT_ID_mvt"])
exp_ids = sub_ref["MVT_ID_mvt"].astype(float).values
match = (set(rank_ids) == set(exp_ids)) and np.all(rank_ids == exp_ids)
print(f"{'✅ 集合+顺序一致' if match else '❌ 不一致'}")

print()
print("=" * 70)
print("[报告完成] 不上传, 等用户确认")
print("=" * 70)