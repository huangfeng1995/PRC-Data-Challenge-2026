"""
OOF 二阶校正器 — 验证信号是否存在
=====================================
- 基模型: LGB 71+offset (v15 同款超参), leave-one-month-out 12 fold
  每折: train 11 月 → predict 1 月 (target 月) → 该月每个航班得到 OOF 预测
- 二阶校正器 (轻量 LGB):
  输入 = [oof_pred, AOBT_NaT, hour, ADEP_int, t_unimp, P_long_proxy]
  目标 = 真值
  训练: 12 月 OOF 预测 + 真值; 评估: holdout 1+7 月上预测
- 红线: 不改 label / 不 clip 真值 / 不建单机场模型

v1 先用 LGB-only (单 seed, 12 fold), 验证校正器信号; 若显著, 扩展到三库.
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

TRAIN_COLS = [
    "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
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
OFFSET_FEATS = ["t_unimp_bias_offset", "t_unimp_bias_offset_known"]
FEATS_FULL = FEATS71 + OFFSET_FEATS

LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]

LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
SEEDS = [42]
# 改为 3000: smoke 中 best_iter=1500 触顶, 说明 1500 不够收敛;
# ES (patience 80) 会让训练在收敛后自然停, 高 cap 不浪费算力。
ROUNDS = int(os.environ.get("ROUNDS", 3000))
# 最小判定路径: 只跑含 holdout 月的 fold 1 (1月) 和 fold 3 (7月),
# 其余折跳过。后续若判定有效再补全 fold 2/4/5。
# 默认关闭 (跑全 5 折); 设 env TWO_FOLD_ONLY=1 启用。
TWO_FOLD_ONLY = bool(int(os.environ.get("TWO_FOLD_ONLY", "0")))
JUDGE_FOLDS = [0, 2]   # OOF_FOLDS 中需要跑的折索引
# 5 折: 月份分组, 覆盖全部 12 个月
# 修复: 旧版 [5,6]/[8,10]/[11,12] 漏掉 7 和 9 月, 这两月永远拿不到 OOF 预测,
#       oof_pred 恒为 0, 会污染全样本 RMSE 与二阶校正器。现改为全覆盖。
OOF_FOLDS = [
    [1, 2],        # 冬末
    [3, 4],        # 春
    [5, 6, 7],     # 夏 (含 7 月)
    [8, 9],        # 秋
    [10, 11, 12],  # 年末
]
EXPECTED_MONTHS = set(range(1, 13))
_FLAT = [m for f in OOF_FOLDS for m in f]
COVERED_MONTHS = set(_FLAT)
# 全量跑之前的自检: 无遗漏 + 无重复 + 1/7 月必须在
assert COVERED_MONTHS == EXPECTED_MONTHS, (
    f"OOF_FOLDS 未覆盖全部 12 个月: missing={sorted(EXPECTED_MONTHS - COVERED_MONTHS)}")
assert 1 in COVERED_MONTHS, "1 月必须被 OOF 预测覆盖"
assert 7 in COVERED_MONTHS, "7 月必须被 OOF 预测覆盖"
assert len(_FLAT) == len(set(_FLAT)), f"OOF_FOLDS 月份重复: {_FLAT}"
OFFSET_FOLDS = 5
OFFSET_SHRINK = 20.0
OFFSET_MIN_OBS = 20
OFFSET_CLIP = (-900.0, 2400.0)
FOCUS_AP = ["LIRF", "EGLL", "LTFM", "LFPG", "EHAM"]

CAT_MAP = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())


def build_features(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
                                 include_a=True, include_b=True, include_c=True,
                                 include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    for k, v in D.compute_drift_features(df_full, ref_dep=ref_dep_for_d, label=label).items():
        df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    for k, v in L.compute_extra_features(df_full[dep_mask], label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    for k, v in L.compute_mvt_anchor_features(df_full[dep_mask],
                                              t_unimp_vals=t_vals[dep_mask], label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full


def fit_offset_map(tr_df, global_off):
    rng = np.random.RandomState(0)
    fold = rng.randint(0, OFFSET_FOLDS, size=len(tr_df))
    base = pd.DataFrame({
        "ADEP_mvt": tr_df["ADEP_mvt"].astype(str).values,
        "RUNWAY_mvt": tr_df["RUNWAY_mvt"].astype(str).values,
        "y": tr_df["TAXITIME_SEC_mvt"].values.astype(np.float64),
        "t": tr_df["t_unimp_r7"].values.astype(np.float64),
        "fold": fold,
    })

    def _fit(sub):
        med = sub.groupby(["ADEP_mvt", "RUNWAY_mvt"]).agg(
            med_y=("y", "median"), med_t=("t", "median"), n=("y", "size"))
        med = med[med["n"] >= OFFSET_MIN_OBS]
        off = (med["med_y"] - med["med_t"]).clip(*OFFSET_CLIP)
        med["off"] = (off * med["n"] + global_off * OFFSET_SHRINK) / (med["n"] + OFFSET_SHRINK)
        return med[["off"]].reset_index()

    out = np.full(len(tr_df), global_off, dtype=np.float64)
    known = np.zeros(len(tr_df), dtype=np.int8)
    for f in range(OFFSET_FOLDS):
        med = _fit(base[base["fold"] != f])
        pos = np.where(fold == f)[0]
        v = base.iloc[pos][["ADEP_mvt", "RUNWAY_mvt"]].merge(
            med, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])["off"].to_numpy()
        out[pos] = np.where(np.isnan(v), global_off, np.nan_to_num(v))
        known[pos] = (~np.isnan(v)).astype(np.int8)
    return out.astype(np.float32), known.astype(np.int8), _fit(base)


def apply_offset_map(keys_df, fit_map, global_off):
    v = keys_df.merge(fit_map, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])["off"].to_numpy()
    return np.where(np.isnan(v), global_off, np.nan_to_num(v)).astype(np.float32), \
        (~np.isnan(v)).astype(np.int8)


def encode_X(df, feats):
    X = df[feats].copy()
    for col in HIGH_CARD_CATS:
        if col in X.columns:
            m = CAT_MAP.get(col, {})
            X[col] = X[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def main():
    t_start = time.time()
    print("=" * 70)
    print("[OOF 二阶校正器 v1] LGB-only × 12 月折")
    print("=" * 70)
    print(f"  seeds={SEEDS} rounds={ROUNDS}")

    MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                       if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
                  for m in range(1, 13)}

    # ============= 加载并构建特征 (全部 12 月) =============
    print("\n[1] 加载并构建 12 月特征")
    # 先构建训练折 (2-12) 特征
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr_dfs = []
    for m in train_months:
        tr_dfs.append(load_month(MONTH_FILE[m], columns=TRAIN_COLS))
    df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
    df_tr_dep_all = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                              (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                              (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)].copy()
    lut_main = L.build_t_unimp_lut(df_tr_dep_all)
    print(f"  训练 DEP (2-12, TX∈(0,10800]): {len(df_tr_dep_all):,}")
    print("  构建训练折 (2-12) 特征...")
    t0 = time.time()
    tr_full = build_features(df_tr_raw, df_tr_raw, lut_main, label="tr_main")
    tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt = tr_dep["TAXITIME_SEC_mvt"].values
    tr_dep = tr_dep[(tt > 0) & (tt <= 10800)].reset_index(drop=True)
    print(f"    {tr_dep.shape}  ({time.time()-t0:.0f}s)")

    # holdout 特征
    ho_dfs = [load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in [1, 7]]
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
    print("  构建 holdout (1+7) 特征...")
    t0 = time.time()
    ho_full = build_features(df_ho_raw, df_tr_raw, lut_main, label="ho")
    ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tth = ho_dep["TAXITIME_SEC_mvt"].values
    ho_dep = ho_dep[(tth > 0) & (tth <= 10800)].reset_index(drop=True)
    print(f"    {ho_dep.shape}  ({time.time()-t0:.0f}s)")

    # offset (在 2-12 上 OOF, 1+7 用 2-12 全量拟合)
    print("\n[2] t_unimp_bias_offset")
    global_off_main = float(np.median(tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)) -
                            np.median(tr_dep["t_unimp_r7"].values.astype(np.float64)))
    tr_off, tr_known, full_map = fit_offset_map(tr_dep, global_off_main)
    tr_dep["t_unimp_bias_offset"] = tr_off
    tr_dep["t_unimp_bias_offset_known"] = tr_known
    rk_off_vals, rk_known_vals = apply_offset_map(pd.DataFrame({
        "ADEP_mvt": ho_dep["ADEP_mvt"].astype(str).values,
        "RUNWAY_mvt": ho_dep["RUNWAY_mvt"].astype(str).values,
    }), full_map, global_off_main)
    ho_dep["t_unimp_bias_offset"] = rk_off_vals
    ho_dep["t_unimp_bias_offset_known"] = rk_known_vals
    print(f"  global_offset={global_off_main:+.1f}s  tr_cov={tr_known.mean()*100:.1f}%")

    y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)
    y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)

    # 加上 month 列用于 fold
    month_tr = pd.to_datetime(tr_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").dt.month.values
    month_ho = pd.to_datetime(ho_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").dt.month.values
    print(f"\n[3] OOF 训练 ({len(OOF_FOLDS)}-fold) × {len(SEEDS)} seed + 折内时间序 ES")

    # 重构: 把 holdout 的特征矩阵也拼到 tr_dep, 一起做 month fold
    print("  把 holdout (1+7) 加入 OOF 训练集")
    all_dep = pd.concat([tr_dep, ho_dep], ignore_index=True)
    # 一次性 encode_X, 全局共享
    X_all = encode_X(all_dep, FEATS_FULL)
    y_all = all_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)
    month_all = pd.to_datetime(all_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").dt.month.values
    n_all = len(all_dep)
    print(f"  all_dep rows: {n_all:,}  (tr 2-12 + ho 1+7 = 12 月全量)")

    # OOF 预测容器: 用 NaN 哨兵, 任何未被折覆盖的行会在 [3.5] 断言里暴露,
    # 不会像旧版 np.zeros 那样静默变成 0 污染 RMSE。
    oof_pred = np.full(n_all, np.nan, dtype=np.float64)
    fold_models = {}  # fold_idx -> dict

    # ---- [3.0] 折覆盖预检: 每个月都必须有行, 且 1/7 月非空 ----
    print("\n  [3.0] 折覆盖预检")
    month_counts = {m: int((month_all == m).sum()) for m in range(1, 13)}
    empty_months = [m for m, c in month_counts.items() if c == 0]
    assert not empty_months, f"以下月份在数据中无任何行: {empty_months}"
    fold_pred_rows = {}
    for fi, fm in enumerate(OOF_FOLDS):
        n_pred = int(np.isin(month_all, fm).sum())
        per_m = {m: int((month_all == m).sum()) for m in fm}
        fold_pred_rows[fi + 1] = {"months": list(fm), "n_pred": n_pred, "per_month": per_m}
        print(f"    fold {fi+1} months={fm}: predict {n_pred:,} 行  明细={per_m}")
    # 逐行覆盖检查: 每行恰好被一个折预测
    cover_mask = np.zeros(n_all, dtype=np.int32)
    for fm in OOF_FOLDS:
        cover_mask += np.isin(month_all, fm).astype(np.int32)
    assert cover_mask.min() == 1 and cover_mask.max() == 1, (
        f"折覆盖异常: min={cover_mask.min()} max={cover_mask.max()} "
        f"(要求每行恰好被 1 个折预测)")
    n_m1 = int((month_all == 1).sum()); n_m7 = int((month_all == 7).sum())
    assert n_m1 > 0, "1 月行数为 0"
    assert n_m7 > 0, "7 月行数为 0"
    print(f"    OK: 全部 12 月均有行; 1 月 {n_m1:,} 行, 7 月 {n_m7:,} 行")
    print(f"    OK: 每行恰好被 1 折预测 (min={cover_mask.min()}, max={cover_mask.max()})")

    for fold_idx, fold_months in enumerate(OOF_FOLDS):
        if TWO_FOLD_ONLY and fold_idx not in JUDGE_FOLDS:
            print(f"\n  === fold {fold_idx+1} (months={fold_months}): "
                  f"SKIP (TWO_FOLD_ONLY=1, 仅跑 fold {JUDGE_FOLDS}) ===")
            continue
        pred_mask = np.isin(month_all, fold_months)
        train_mask = ~pred_mask
        if pred_mask.sum() == 0:
            continue
        _t0_fold = time.time()
        _seed_times = []
        _seed_bi = []
        print(f"\n  === fold {fold_idx+1} (months={fold_months}): "
              f"train {train_mask.sum():,}, predict {pred_mask.sum():,} ===")
        # 折内时间序切: 训练月份里按 MVT_TIME 排序, 末 20% 作 ES val
        t_fold = pd.to_datetime(all_dep["MVT_TIME_UTC_mvt"].iloc[train_mask.nonzero()[0]],
                                utc=True, errors="coerce").astype("int64").values
        order = np.argsort(t_fold, kind="stable")
        train_idx_in = train_mask.nonzero()[0]
        cut = int(len(order) * 0.8)
        tr_pos = train_idx_in[order[:cut]]
        va_pos = train_idx_in[order[cut:]]
        print(f"    ES val: {len(va_pos):,} (末 20% 时间序) / train {len(tr_pos):,}")
        Xtr = X_all.iloc[tr_pos]
        ytr = y_all[tr_pos]
        Xva = X_all.iloc[va_pos]
        yva = y_all[va_pos]
        Xp  = X_all.iloc[pred_mask]
        yp  = y_all[pred_mask]
        preds = []
        for sd in SEEDS:
            ds_tr = lgb.Dataset(Xtr, label=ytr, categorical_feature=LOW_CARD_CATS)
            ds_va = lgb.Dataset(Xva, label=yva, categorical_feature=LOW_CARD_CATS, reference=ds_tr)
            _t0 = time.time()
            b = lgb.train(dict(LGB_PARAMS, seed=sd), ds_tr,
                          num_boost_round=ROUNDS,
                          valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(80, verbose=False),
                                     lgb.log_evaluation(0)])
            _train_sec = time.time() - _t0
            _bi = b.best_iteration if b.best_iteration > 0 else ROUNDS
            _es = "触发" if _bi < ROUNDS else "未触发(跑满)"
            print(f"    seed={sd} best_iter={_bi} / cap={ROUNDS}  "
                  f"ES={_es}  train={_train_sec:.1f}s")
            _t1 = time.time()
            p = b.predict(Xp, num_iteration=_bi).astype(np.float64)
            p = np.clip(p, 30.0, 10800.0)
            preds.append(p)
            b.save_model(str(D0 / f"oof_lgb_fold{fold_idx+1:02d}_seed{sd}.txt"))
            _seed_times.append(round(_train_sec, 1))
            _seed_bi.append(_bi)
        avg = np.mean(np.stack(preds), axis=0)
        oof_pred[pred_mask] = avg
        fold_rmse = float(np.sqrt(np.mean((avg - yp) ** 2)))
        _pred_sec = time.time() - _t1
        _fold_total_sec = time.time() - _t0_fold
        print(f"    fold {fold_idx+1} OOF RMSE = {fold_rmse:.2f}  "
              f"pred={_pred_sec:.1f}s  fold_total={_fold_total_sec:.1f}s  "
              f"({time.strftime('%H:%M:%S')})")
        fold_models[fold_idx+1] = {"months": list(fold_months),
                                   "n_train": int(train_mask.sum()),
                                   "n_pred": int(pred_mask.sum()),
                                   "oof_rmse": fold_rmse,
                                   "best_iters": _seed_bi,
                                   "es_triggered": [bi < ROUNDS for bi in _seed_bi],
                                   "train_sec_per_seed": _seed_times,
                                   "pred_sec": round(_pred_sec, 1),
                                   "fold_total_sec": round(_fold_total_sec, 1)}

    # ---- [3.5] OOF 完整性后置校验 ----
    n_nan = int(np.isnan(oof_pred).sum())
    n_uncovered_months = sorted(set(month_all[np.isnan(oof_pred)].tolist()))
    if n_nan:
        if TWO_FOLD_ONLY:
            # 最小判定路径: 未跑折的行允许为 NaN, 但已跑折 + holdout 月必须非空
            for m in (1, 7):
                sel = month_all == m
                if np.isnan(oof_pred[sel]).any():
                    raise AssertionError(f"TWO_FOLD_ONLY 路径下 {m} 月仍有 NaN, "
                                       "说明 fold 1 或 fold 3 没跑成功")
            print(f"\n  [3.5] TWO_FOLD_ONLY OOF 校验: PASS (1月/7月非空, "
                  f"未覆盖月={n_uncovered_months}, 共 {n_nan:,} 行 NaN)")
        else:
            raise AssertionError(
                f"oof_pred 有 {n_nan:,} 行未被预测 (月份={n_uncovered_months})")
    else:
        print("\n  [3.5] OOF 完整性校验: PASS (0 行缺失)")
    for m in (1, 7):
        sel = month_all == m
        if sel.sum() == 0 or np.isnan(oof_pred[sel]).all():
            continue
        r = float(np.sqrt(np.mean((oof_pred[sel] - y_all[sel]) ** 2)))
        print(f"    {m:>2d} 月: n={int(sel.sum()):,}  oof_pred 非空 ✓  RMSE={r:.2f}")
    print(f"    oof_pred 范围 [{oof_pred.min():.1f}, {oof_pred.max():.1f}]  "
          f"(整数化后 clip 生效)")

    oof_valid = ~np.isnan(oof_pred)
    print(f"\n[4] OOF 已覆盖样本 RMSE = "
          f"{float(np.sqrt(np.mean((oof_pred[oof_valid] - y_all[oof_valid]) ** 2))):.2f} "
          f"({int(oof_valid.sum()):,}/{n_all:,} 行有效)")

    # ============= 二阶校正器 =============
    print("\n[5] 二阶校正器 (轻量 LGB)")
    # 输入特征: oof_pred + AOBT_NaT + hour + ADEP_int + t_unimp + P_long_proxy
    # P_long_proxy: 用 oof_pred > 1800 简单估计
    aobt_nat_all = all_dep["AOBT_3_flt"].isna().values.astype(np.int8)
    hour_all = all_dep["SCHED_TIME_UTC_mvt"].dt.hour.values.astype(np.int32)
    adep_all = all_dep["ADEP_mvt"].astype(str).map(CAT_MAP.get("ADEP_mvt", {})).fillna(-1).astype("int32").values
    tunimp_all = all_dep["t_unimp_r7"].fillna(800).astype(np.float32).values
    plung_proxy_all = (oof_pred > 1800).astype(np.float32)

    # 校正器输入矩阵 (数值)
    Z = np.column_stack([
        oof_pred.astype(np.float32),
        aobt_nat_all.astype(np.float32),
        hour_all.astype(np.float32),
        adep_all.astype(np.float32),
        tunimp_all,
        plung_proxy_all,
        (oof_pred - tunimp_all).astype(np.float32),  # residual proxy
    ]).astype(np.float32)
    Z_cols = ["oof_pred", "aobt_nat", "hour", "ADEP_int", "t_unimp", "P_long_proxy",
              "oof_minus_tunimp"]
    print(f"  Z shape: {Z.shape}, cols: {Z_cols}")

    # 训练: 全 12 月 OOF 数据
    # 校正器对 holdout (1+7) 推理时, 用同样的 oof_pred 输入
    Z_train = Z
    y_calib = y_all

    # 校正器评估: 用 OOF 预测 + 真值, 训 LGB 然后用同分布的 holdout 推理
    # 为避免校正器对 holdout 区域过拟合, 用训练折 11 月 OOF 训校正器, 在 1+7 上评
    train_z_mask = month_all != 1
    train_z_mask = train_z_mask & (month_all != 7)  # 校正器训练只用 2-12 OOF
    holdout_z_mask = (month_all == 1) | (month_all == 7)

    print(f"  校正器训练: {train_z_mask.sum():,} (11 月 OOF), 校正器评估: "
          f"{holdout_z_mask.sum():,} (1+7 月)")

    # 两折路径下: 3/4/8/9/10/11/12 月的 oof_pred 是 NaN, 校正器训练时必须剔除
    if TWO_FOLD_ONLY:
        _oof_ok = ~np.isnan(oof_pred)
        train_z_mask = train_z_mask & _oof_ok
        # 校正器评估仍用 1+7 月 OOF, 这两月必须非空
        if np.isnan(oof_pred[holdout_z_mask]).any():
            raise AssertionError("holdout 1/7 月 oof_pred 仍有 NaN")
        print(f"  两折路径下: 校正器训练用 OOF 已覆盖的月 ({train_z_mask.sum():,} 行), "
              f"评估用 1+7 月 ({holdout_z_mask.sum():,} 行)")

    Z_tr_corr = Z_train[train_z_mask]
    y_tr_corr = y_calib[train_z_mask]
    Z_ho_corr = Z_train[holdout_z_mask]
    y_ho_corr = y_calib[holdout_z_mask]
    oof_pred_ho = oof_pred[holdout_z_mask]

    LGB_CORR_PARAMS = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.02, "num_leaves": 31, "max_depth": 5,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "reg_alpha": 1.0, "reg_lambda": 1.0,
        "verbose": -1, "num_threads": -1,
    }
    corr_preds = []
    corr_per_seed = []
    for sd in SEEDS:
        ds = lgb.Dataset(Z_tr_corr, label=y_tr_corr.astype(np.float32))
        b = lgb.train(dict(LGB_CORR_PARAMS, seed=sd), ds, num_boost_round=300)
        p = b.predict(Z_ho_corr).astype(np.float64)
        p = np.clip(p, 30.0, 10800.0)
        corr_preds.append(p)
        corr_per_seed.append(float(np.sqrt(np.mean((p - y_ho_corr) ** 2))))
        print(f"    校正器 seed={sd} RMSE={corr_per_seed[-1]:.2f}")
    corr_avg = np.mean(np.stack(corr_preds), axis=0)

    # 校正器训练集的 in-sample 表现 (参照)
    corr_in = b.predict(Z_tr_corr)
    print(f"  校正器 in-sample RMSE = {float(np.sqrt(np.mean((corr_in - y_tr_corr) ** 2))):.2f}")

    # ============= 评估 =============
    print("\n[6] 评估 holdout (1+7)")
    m_full = y_ho_corr > 0
    m_core = (y_ho_corr > 0) & (y_ho_corr <= 10800)
    m_short = (y_ho_corr > 0) & (y_ho_corr <= 1800)
    m_long = (y_ho_corr > 1800) & (y_ho_corr <= 10800)
    aps_ho = all_dep["ADEP_mvt"].iloc[np.where(holdout_z_mask)[0]].astype(str).values

    def ev(p, lbl):
        return {
            "label": lbl,
            "rmse_full": float(np.sqrt(np.mean((p[m_full] - y_ho_corr[m_full]) ** 2))),
            "rmse_core": float(np.sqrt(np.mean((p[m_core] - y_ho_corr[m_core]) ** 2))),
            "rmse_short": float(np.sqrt(np.mean((p[m_short] - y_ho_corr[m_short]) ** 2))),
            "rmse_long": float(np.sqrt(np.mean((p[m_long] - y_ho_corr[m_long]) ** 2))),
        }

    base = ev(oof_pred_ho, "base_oof_lgb")
    final = ev(corr_avg, "oof_+_corrector")
    print(f"  {'组':22s} {'短段':>9s} {'长段':>9s} {'核心':>9s} {'全样本':>9s}")
    for k, lbl in [("base_oof_lgb", "①基模型(OOF)"), ("oof_+_corrector", "②+二阶校正")]:
        r = {"base_oof_lgb": base, "oof_+_corrector": final}[k]
        print(f"  {lbl:22s} {r['rmse_short']:>9.2f} {r['rmse_long']:>9.2f} "
              f"{r['rmse_core']:>9.2f} {r['rmse_full']:>9.2f}")

    # 分机场
    print(f"\n  重点 5 机场 (全样本 RMSE)")
    print(f"  {'机场':6s} {'n':>8s} {'①基':>9s} {'②+校':>9s} {'Δ':>8s}")
    ap_per = {}
    for ap in FOCUS_AP:
        mk = (aps_ho == ap) & m_full
        if mk.sum() < 100:
            continue
        rb = float(np.sqrt(np.mean((oof_pred_ho[mk] - y_ho_corr[mk]) ** 2)))
        rc = float(np.sqrt(np.mean((corr_avg[mk] - y_ho_corr[mk]) ** 2)))
        ap_per[ap] = {"n": int(mk.sum()), "base": rb, "corr": rc, "delta": rc - rb}
        print(f"  {ap:6s} {mk.sum():>8d} {rb:>9.2f} {rc:>9.2f} {rc-rb:>+8.2f}")

    seed_jitter = float(np.std(corr_per_seed))
    delta_core = final["rmse_core"] - base["rmse_core"]
    same_sign = all(ap_per[ap]["delta"] < 0 for ap in ap_per)
    pass_step = (delta_core < -0.5) and same_sign

    print(f"\n[判断]")
    print(f"  核心 Δ(②-①) = {delta_core:+.2f}s,  seedσ={seed_jitter:.2f}s")
    print(f"  重点 5 机场 Δ 同号? {same_sign}")
    print(f"  通过? {pass_step}")

    print(f"\n[折耗时]")
    for fi, info in fold_models.items():
        print(f"  fold {fi} months={info['months']}  "
              f"best_iter={info['best_iters']}  "
              f"ES={'/'.join('是' if x else '否' for x in info['es_triggered'])}  "
              f"train/seed={info['train_sec_per_seed']}s  "
              f"pred={info['pred_sec']}s  "
              f"fold_total={info['fold_total_sec']}s")

    # JSON 落盘
    out = {
        "task": "oof_2nd_stage_calibrator_v1_lgb_only",
        "config": {
            "TWO_FOLD_ONLY": TWO_FOLD_ONLY,
            "JUDGE_FOLDS": JUDGE_FOLDS,
            "ROUNDS_cap": ROUNDS,
            "SEEDS": SEEDS,
        },
        "n_total": int(n_all),
        "n_holdout": int(holdout_z_mask.sum()),
        "fold_models": fold_models,
        "global_offset_s": global_off_main,
        "results": {
            "base_oof": {**base, "seed_per_corr_std": float(np.std(corr_per_seed))},
            "with_corrector": final,
            "corrector_per_seed_rmse": [round(v, 3) for v in corr_per_seed],
        },
        "per_airport": ap_per,
        "decision": {
            "delta_core_rmse": delta_core,
            "seed_jitter": seed_jitter,
            "same_sign_airports": same_sign,
            "pass": pass_step,
        },
        "total_time_sec": round(time.time() - t_start, 1),
    }
    fn = D0 / "oof_calibrate_step1.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()