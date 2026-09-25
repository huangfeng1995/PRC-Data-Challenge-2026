"""
data/sample 上天气 add-one 方向性验证
======================================

切分:
  训练: 2025-02 / 03 / 04 / 05 / 06 / 08 / 09 / 10 / 11 / 12 (共 10 月)
  留出: 2025-01 (冬季除冰/风暴) 和 2025-07 (夏季雷暴,archive 无能见度/cape) 单独验

两臂 (同数据、同切分):
  基线臂: build_full_features(feature_set="all", add_weather=False)
  天气臂: 同上 + add_weather=True (追加 11 列天气)

模型: XGBoost / LightGBM (CatBoost 能跑则加) × seed 42 / 123 / 456

报告:
  - 每折每库每 seed 的基线 / 天气 / Δ
  - 3 seed 的 Δ 均值和范围,方向一致性
  - 天气 11 特征 gain (重点: dewpoint_deficit / snowfall / wind_gust)

局限:
  sample 每机场每月仅 40 DEP,Δ 只判方向、不当增益
  基线口径 = build_full_features (all ≈ 77 列),不是 v15 71 列
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_month, filter_departures, parse_time_columns  # noqa: E402
from src.train_ensemble import build_full_features  # noqa: E402

SAMPLE_DIR = Path("data/sample")
SAMPLE_MONTHS = [
    "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01",
    "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01",
    "2025-11-01", "2025-12-01",
]

TRAIN_MONTHS = [m for m in SAMPLE_MONTHS if m[:7] not in ("2025-01", "2025-07")]
HOLDOUT_MONTHS = ["2025-01-01", "2025-07-01"]

TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]
SEEDS = [42, 123, 456]


def load_dep_from_sample(start_dates: list[str]) -> pd.DataFrame:
    dfs = []
    for sd in start_dates:
        path = SAMPLE_DIR / f"training_{sd}_{_next_month(sd)}.parquet"
        df = load_month(str(path))
        df = filter_departures(parse_time_columns(df, cols=TIME_COLS))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _next_month(ymd: str) -> str:
    y, m, d = ymd.split("-")
    m = int(m)
    y = int(y)
    if m == 12:
        return f"{y+1:04d}-01-01"
    return f"{y:04d}-{m+1:02d}-01"


def load_arm(months: list[str], add_weather: bool) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Load + build features, return (X, feat_cols, y)."""
    df = load_dep_from_sample(months)
    df_feat, feat_cols = build_full_features(df, feature_set="all", add_weather=add_weather)
    # extract y, filter NaN
    y = df_feat["TAXITIME_SEC_mvt"]
    mask = y.notna()
    df_feat = df_feat[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    # ensure all feat_cols in df_feat
    feat_cols = [c for c in feat_cols if c in df_feat.columns]
    return df_feat, feat_cols, y


def _encode_shared(X_tr, X_te):
    """Encode categorical/object/string cols to int32, sharing train+te mappings."""
    cat_cols = list(X_tr.select_dtypes(include=["category", "object", "string"]).columns)
    if not cat_cols:
        return X_tr, X_te
    X_tr_out = X_tr.copy()
    X_te_out = X_te.copy()
    for col in cat_cols:
        # build union mapping (train+te)
        all_vals = set(X_tr[col].astype(str).fillna("__nan__").tolist())
        all_vals |= set(X_te[col].astype(str).fillna("__nan__").tolist())
        mapping = {v: i for i, v in enumerate(sorted(all_vals))}
        X_tr_out[col] = X_tr[col].astype(str).fillna("__nan__").map(mapping).astype("int32")
        X_te_out[col] = X_te[col].astype(str).fillna("__nan__").map(mapping).astype("int32").fillna(-1).astype("int32")
    return X_tr_out, X_te_out


def train_xgb(X_tr, y_tr, X_te, seed=42, n_est=300):
    import xgboost as xgb
    X_tr, X_te = _encode_shared(X_tr, X_te)
    model = xgb.XGBRegressor(
        n_estimators=n_est, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.01, reg_lambda=0.1,
        random_state=seed, objective="reg:squarederror", n_jobs=1, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = float(np.sqrt(np.mean((pred - y_te_values) ** 2)))
    return rmse, model


def train_lgb(X_tr, y_tr, X_te, seed=42, n_est=300):
    try:
        import lightgbm as lgb
    except ImportError:
        return None, None
    X_tr, X_te = _encode_shared(X_tr, X_te)
    model = lgb.LGBMRegressor(
        n_estimators=n_est, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.01, reg_lambda=0.1,
        random_state=seed, objective="regression", n_jobs=1, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = float(np.sqrt(np.mean((pred - y_te_values) ** 2)))
    return rmse, model


def train_cat(X_tr, y_tr, X_te, seed=42, n_est=300):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return None, None
    X_tr, X_te = _encode_shared(X_tr, X_te)
    model = CatBoostRegressor(
        iterations=n_est, depth=4, learning_rate=0.05,
        random_seed=seed, loss_function="RMSE", verbose=0,
        task_type="CPU", thread_count=2,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = float(np.sqrt(np.mean((pred - y_te_values) ** 2)))
    return rmse, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== data/sample 天气 add-one 方向性验证 ===")
print(f"训练月份 (10 月): {TRAIN_MONTHS}")
print(f"留出月份 (2 折) : {HOLDOUT_MONTHS}")
print(f"seeds           : {SEEDS}")
print()

# Preload arms (only once per fold to keep features identical between seeds)
results = []  # (fold, model, seed, arm, rmse, n_feat)

for fold_month in HOLDOUT_MONTHS:
    fold_label = fold_month[:7]
    print(f"\n>>> Fold {fold_label} (holdout)")
    print("  Loading train (non-{}/7) + holdout ({})...".format(fold_label[5:7], fold_label))

    # Train arms
    print("  Building baseline arm (train)...")
    Xb_tr, fb, yb_tr = load_arm(TRAIN_MONTHS, add_weather=False)
    print(f"    train: {Xb_tr.shape}, target non-null: {yb_tr.notna().sum()}")

    print("  Building weather arm (train)...")
    Xw_tr, fw, yw_tr = load_arm(TRAIN_MONTHS, add_weather=True)
    print(f"    train: {Xw_tr.shape}, target non-null: {yw_tr.notna().sum()}, +{len(fw)-len(fb)} weather feats")

    # Holdout
    print("  Building baseline arm (holdout)...")
    Xb_te, _, yb_te = load_arm([fold_month], add_weather=False)
    print(f"    holdout: {Xb_te.shape}, target non-null: {yb_te.notna().sum()}")

    print("  Building weather arm (holdout)...")
    Xw_te, _, yw_te = load_arm([fold_month], add_weather=True)

    # Align columns (train vs test shouldn't differ; but just in case)
    Xb_tr = Xb_tr[fb]
    Xb_te = Xb_te[[c for c in fb if c in Xb_te.columns]]
    fb_final = list(Xb_te.columns)
    Xb_tr = Xb_tr[fb_final]

    Xw_tr = Xw_tr[fw]
    Xw_te = Xw_te[[c for c in fw if c in Xw_te.columns]]
    fw_final = list(Xw_te.columns)
    Xw_tr = Xw_tr[fw_final]

    # Keep y consistent (baseline weather targets identical)
    global y_te_values  # hack for closures
    # Ensure targets refer to same rows (already aligned via same DEP filter + TAXITIME_SEC_mvt)
    y_tr_v = yb_tr.values
    y_te_values = yb_te.values

    for model_name, train_fn in [("xgb", train_xgb), ("lgb", train_lgb), ("cat", train_cat)]:
        if model_name == "cat":
            # cat needs int-safe features; skip if heavy
            pass
        for seed in SEEDS:
            # Baseline
            rmse_b, model_b = train_fn(Xb_tr, y_tr_v, Xb_te, seed=seed)
            if rmse_b is None:
                continue
            # Weather
            y_te_values = yw_te.values  # same rows, identical DEPs
            rmse_w, model_w = train_fn(Xw_tr, y_tr_v, Xw_te, seed=seed)

            results.append({
                "fold": fold_label, "model": model_name, "seed": seed,
                "baseline": rmse_b, "weather": rmse_w,
                "delta": rmse_w - rmse_b,
                "n_feat_base": len(fb_final), "n_feat_weather": len(fw_final),
                "model_w": model_w,
                "model_b": model_b,
            })

    print()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
df_res = pd.DataFrame(results)
print("\n" + "=" * 80)
print("每折 × 每库 × 每 seed 结果 (Δ = weather - baseline, 负更好)")
print("=" * 80)

for fold in df_res["fold"].unique():
    print(f"\n### Fold {fold}")
    sub = df_res[df_res["fold"] == fold]
    for model in ["xgb", "lgb", "cat"]:
        ms = sub[sub["model"] == model]
        if len(ms) == 0:
            continue
        print(f"\n  [{model}]")
        print(f"  {'seed':>6}  {'base':>8}  {'weather':>8}  {'Δ':>9}  {'direction':>9}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")
        for _, r in ms.iterrows():
            sign = "↓ Δ<0" if r["delta"] < 0 else "↑ Δ>0" if r["delta"] > 0 else "=="
            print(f"  {r['seed']:>6}  {r['baseline']:>8.2f}  {r['weather']:>8.2f}  {r['delta']:>+9.2f}  {sign:>9}")
        # Summary
        deltas = ms["delta"].values
        print(f"  ─────────────────────────────")
        print(f"  mean Δ = {deltas.mean():+.2f}  min = {deltas.min():+.2f}  max = {deltas.max():+.2f}  同号 {'✅' if len(set(np.sign(deltas.tolist()))) == 1 else '❌'}")

# ---------------------------------------------------------------------------
# Weather feature gain analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("天气特征 Gain 分析 (取 xgb seed=42 的天气臂)")
print("=" * 80)

for fold in df_res["fold"].unique():
    print(f"\n### Fold {fold} — XGB seed=42")
    row = df_res[(df_res["fold"] == fold) & (df_res["model"] == "xgb") & (df_res["seed"] == 42)]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    model_w = row["model_w"]
    model_b = row["model_b"]

    # 天气臂 gain
    feat_names_w = fw_final
    try:
        try:
            # XGB: gain = raw gain
            imp_dict = model_w.get_booster().get_score(importance_type="gain")
            imp_w = pd.Series(imp_dict)
        except AttributeError:
            # LGB / CatBoost fallback
            imp_w = pd.Series(model_w.feature_importances_, index=feat_names_w)
        imp_w = imp_w.sort_values(ascending=False)
    except Exception as e:
        print(f"  weather importances failed: {e}")
        continue

    total_gain = imp_w.sum()
    weather_gain = imp_w[[c for c in imp_w.index if c.startswith("weather_")]].sum()
    n_nonzero = int((imp_w > 0).sum())
    print(f"  总特征数: {len(feat_names_w)}, 非零 gain: {n_nonzero}")
    print(f"  总 gain: {total_gain:.0f}, weather_* 总 gain: {weather_gain:.0f}  占比: {weather_gain/total_gain*100:.2f}%")

    # Top 11 weather features
    weather_feats_in_model = imp_w[[c for c in imp_w.index if c.startswith("weather_")]]
    weather_feats_in_model = weather_feats_in_model.sort_values(ascending=False)
    print(f"\n  {'特征':<28} {'gain':>8} {'占 weather%':>12} {'占 total%':>10}")
    print(f"  {'-'*28}  {'-'*8} {'-'*12} {'-'*10}")
    for c, v in weather_feats_in_model.items():
        print(f"  {c:<28} gain={v:>10.1f}  占weather={v/weather_gain*100:>5.1f}%  占total={v/total_gain*100:>5.2f}%")

    # 关键特征重点看
    for key in ["weather_dewpoint_deficit", "weather_snowfall", "weather_wind_gust",
                "weather_snow_depth", "weather_wind_speed", "weather_temperature", "weather_pressure"]:
        if key in weather_feats_in_model.index:
            v = weather_feats_in_model[key]
            print(f"  ⭐ {key:<24} gain={v:>10.1f}  (关键)  [非零, 被模型用到]")
        else:
            print(f"  ✗ {key:<24} 不在 imp_w 中 = 0 gain (未被模型用到)")

    # 对比基线的所有特征数
    print(f"\n  基线 arm: {len(fb_final)} 特征; 天气 arm: {len(fw_final)} 特征 (+{len(fw_final)-len(fb_final)})")

# ---------------------------------------------------------------------------
# Overall summary: direction consistency
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("方向一致性汇总 (3 seed 的 Δ 是否同号)")
print("=" * 80)
for fold in df_res["fold"].unique():
    print(f"\n  Fold {fold}:")
    sub = df_res[df_res["fold"] == fold]
    for model in ["xgb", "lgb", "cat"]:
        ms = sub[sub["model"] == model]
        if len(ms) == 0:
            continue
        deltas = ms["delta"].values
        signs = np.sign(deltas).tolist()
        if len(set(signs)) == 1:
            sign_desc = "全部同号 ✅"
        else:
            sign_desc = f"同号 {signs.count(signs[0])}/{len(signs)} ⚠️"
        print(f"    {model}: Δ ∈ [{deltas.min():+.2f}, {deltas.max():+.2f}], mean={deltas.mean():+.2f}  {sign_desc}")

print("\n=== 局限 ===")
print("• sample 每机场每月仅 40 DEP; Δ 只判方向, 不当增益")
print("• 基线 = build_full_features(all ≈ 77 列), 不是 v15 71 列")
print("• CatBoost 未跑(CPU 慢); 全量 add-one 需等 Mac 地基")
print("• archive 无 visibility/cape/cloud_base, 雷暴刻画缺关键变量")
print("• Δ 置信区间很宽, 3 seed 同号才算方向有意义")
