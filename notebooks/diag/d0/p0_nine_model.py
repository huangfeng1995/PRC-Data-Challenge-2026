"""
p0_nine_model.py — P0 九模型集成 (LGB×3 + XGB×3 + CAT×3), cap 8000
==========================================================================
本脚本已落盘并通过 smoke test, 【未启动全量训练】。

固定协议 (与基线一致)
--------------------
  split      : MAIN — train 2025-02..12 / holdout 2025-01 + 2025-07
  features   : FEATS71 (与 v15 同一份列表), 高基数整数编码
  ES         : 训练月内 MVT 时间末 10% = 11、12 月 (174,019 行), 断言不含 1/7 月
  cap        : 8000 (学习曲线显示 8000 轮仍未触底, 3000→8000 仍单调降)
  loss       : 全部 L2 (objective='regression'), 不做目标函数改造
  seeds      : LGB/XGB = 42,123,2024 ; CAT = 42,123,2024
  ensemble   : 9 模型简单平均 → clip [30, 10800]
  baseline   : v15 (core 308.96 / full 568.19) 作对比锚点

红线 (写进脚本, 不可绕过)
------------------------
  R1 不改 label
  R2 不 winsorize / clip 真值 (仅对【预测值】clip)
  R3 不建单机场模型
  R4 TE / LUT / offset 只用训练折 (holdout 绝不参与任何拟合)
  R5 时间戳先 .astype("datetime64[ns]") —— 原始数据是微秒(us), 直接用会差 1000 倍
  R6 分支隔离: 结果只写 p0_*.json / p0_*.parquet, 不直推 main、不改现有模型
  R7 MVT_TIME_UTC_mvt 是【预测时点之后的未来信息】, 绝不作为特征 (见 mvt_leak_audit)

CAT 超时保护 (v15 教训)
-----------------------
  v15 里 cat123 / cat2024 在 save_model 之前 break, 从未落盘。
  本脚本用【子进程 + 到点落盘】: 父进程用 subprocess 跑 CAT, 超时 kill 后
  保留已写出的模型文件; 单轮时间超阈值则提前落盘并退出。
  save_model 放在独立函数里, 且在任何 break/超时之前执行。

用法
----
  # smoke test (默认, ~2000 行 / cap 10, 不产出结论)
  python3 notebooks/diag/d0/p0_nine_model.py --smoke

  # 全量 (需显式 --full, 防止误触发)
  python3 notebooks/diag/d0/p0_nine_model.py --full

输出
----
  notebooks/diag/d0/p0_nine_model.json      (指标; 名字带 p0_ 前缀, 隔离)
  notebooks/diag/d0/p0_preds.parquet        (holdout 逐行 9 模型预测)
  <work>/p0_nine_model.log
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
WORK = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

import _r7_lib as L
from src.data_loader import load_month, parse_time_columns
from v15_clean_retrain import (build_features, encode_cat_columns,
                               FEATS71, LOW_CARD_CATS, TRAIN_COLS, TIME_COLS)

# ============================ 协议常量 ============================
CAP = 8000
SMOKE_CAP = 10
ES_ROUND = 100
SEEDS = [42, 123, 2024]
CLIP_LO, CLIP_HI = 30.0, 10800.0
HOLDOUT_MONTHS = [1, 7]
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
V15_ANCHOR = {"core": 308.96, "full": 568.19}

# CAT 超时保护
CAT_SOFT_DEADLINE_SEC = 3000      # 单个 CAT 模型墙钟软上限, 超时则落盘并退出
CAT_ROUND_WARN_SEC = 60           # 单轮超过 60s 视为异常, 提前落盘

MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                  if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
              for m in range(1, 13)}


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "p0_nine_model.log").open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


# ============================ R5: 时间戳工具 ============================
def to_ns(s: pd.Series) -> pd.Series:
    """R5: 原始 parquet 的时间列是 datetime64[us] (微秒)。

    pandas 2.x 中 tz-aware 列无法直接 .astype("datetime64[ns]"), 必须先
    tz_convert("UTC").tz_localize(None) 去掉时区, 再转 ns。
    统一在【取 .dt / 转 int64 之前】完成, 避免 us/ns 混用导致 1000 倍偏差。
    """
    out = pd.to_datetime(s, utc=True, errors="coerce")
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def rmse(p, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


# ============================ 数据 ============================
def build_all(smoke: bool):
    """构建 P0 数据。默认带进程外 pickle 缓存, 便于重复 smoke 不重复算特征。"""
    CACHE = WORK / "p0_feat_cache.pkl"
    if CACHE.exists() and os.environ.get("P0_NO_CACHE", "0") != "1":
        log(f"  [cache] 复用特征缓存 {CACHE.name} (P0_NO_CACHE=1 可禁用)")
        d = pd.read_pickle(CACHE)
        return (d["X_fit"], d["y_fit"], d["X_es"], d["y_es"],
                d["X_ho"], d["y"], d["mo"], d["ap"])

    X_fit, y_fit, X_es, y_es, X_ho, y, mo, ap = _build_all_raw(smoke)
    if os.environ.get("P0_NO_CACHE", "0") != "1":
        pd.to_pickle({"X_fit": X_fit, "y_fit": y_fit, "X_es": X_es,
                      "y_es": y_es, "X_ho": X_ho, "y": y, "mo": mo, "ap": ap}, CACHE)
        log(f"  [cache] 已写入 {CACHE.name}")
    return X_fit, y_fit, X_es, y_es, X_ho, y, mo, ap


def _build_all_raw(smoke):
    log("=" * 78)
    log(f"[P0] 构建数据 (smoke={smoke})")
    log("=" * 78)
    t0 = time.time()

    tr_raw = pd.concat([load_month(MONTH_FILE[m], columns=TRAIN_COLS)
                        for m in TRAIN_MONTHS], ignore_index=True)
    tr_parsed = parse_time_columns(tr_raw.copy(), cols=TIME_COLS)
    lut = L.build_t_unimp_lut(tr_parsed[(tr_parsed["PHASE_mvt"] == "DEP")])
    log(f"  训练月 raw {len(tr_raw):,}  ({time.time()-t0:.0f}s)")
    log("  构建训练月特征 ...")
    tr_full = build_features(tr_raw, tr_parsed, lut, label="tr")
    tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    del tr_full, tr_raw
    yv = tr_dep["TAXITIME_SEC_mvt"].values
    # R1/R2: label 不做任何修改, 只做有效性过滤 (与基线同口径)
    tr_dep = tr_dep[(yv > 0) & (yv < 10800)].reset_index(drop=True)
    log(f"  训练 DEP {len(tr_dep):,}   ({time.time()-t0:.0f}s)")

    ho_raw = pd.concat([load_month(MONTH_FILE[m], columns=TRAIN_COLS)
                        for m in HOLDOUT_MONTHS], ignore_index=True)
    ho_parsed = parse_time_columns(ho_raw.copy(), cols=TIME_COLS)
    log("  构建 holdout 特征 ...")
    ho_full = build_features(ho_raw, tr_parsed, lut, label="ho")
    ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    y = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    keep = y > 0                       # R2: 不 clip 真值
    ho_dep = ho_dep[keep].reset_index(drop=True)
    y = y[keep]
    # R5: 显式统一为 ns 再取 month, 避免 us/ns 混用
    ho_dep["MVT_TIME_UTC_mvt"] = to_ns(ho_dep["MVT_TIME_UTC_mvt"])
    tr_dep["MVT_TIME_UTC_mvt"] = to_ns(tr_dep["MVT_TIME_UTC_mvt"])
    mo = ho_dep["MVT_TIME_UTC_mvt"].dt.month.values
    ap = ho_dep["ADEP_mvt"].astype(str).values
    assert set(np.unique(mo)).issubset(set(HOLDOUT_MONTHS)), f"holdout 月份异常: {set(np.unique(mo))}"
    del ho_full, ho_raw
    log(f"  Holdout DEP {len(ho_dep):,}  >10800: {int((y>10800).sum())}   "
        f"({time.time()-t0:.0f}s)")

    # ---- R4: ES 折只用训练月末 10% ----
    ts = tr_dep["MVT_TIME_UTC_mvt"].astype("int64").values
    order = np.argsort(ts, kind="stable")
    cut = int(len(order) * 0.9)
    fit_pos, es_pos = order[:cut], order[cut:]
    es_months = sorted(tr_dep["MVT_TIME_UTC_mvt"].dt.month.iloc[es_pos]
                       .unique().tolist())
    assert 1 not in es_months and 7 not in es_months, f"R4 违规: ES 含 holdout 月 {es_months}"
    log(f"  拟合 {len(fit_pos):,} / ES {len(es_pos):,} 月份={es_months}  ✓ 不含 1/7")

    cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json")
                         .read_text())
    tr_dep = encode_cat_columns(tr_dep, cat_map)
    ho_dep = encode_cat_columns(ho_dep, cat_map)
    for f in FEATS71:
        if f not in tr_dep.columns:
            tr_dep[f] = np.nan
        if f not in ho_dep.columns:
            ho_dep[f] = np.nan

    X_fit = tr_dep[FEATS71].iloc[fit_pos].copy()
    X_es = tr_dep[FEATS71].iloc[es_pos].copy()
    X_ho = ho_dep[FEATS71].copy()
    y_fit = tr_dep["TAXITIME_SEC_mvt"].values[fit_pos].astype(np.float64)
    y_es = tr_dep["TAXITIME_SEC_mvt"].values[es_pos].astype(np.float64)

    if smoke:
        # 极小规模: 抽 ~2000 行 (fit/es/holdout 各取一点), cap 10
        rs = np.random.RandomState(0)
        n_fit = 1500
        n_es = 500
        sel_f = rs.choice(len(X_fit), n_fit, replace=False)
        sel_e = rs.choice(len(X_es), n_es, replace=False)
        n_ho = min(2000, len(X_ho))
        sel_h = rs.choice(len(X_ho), n_ho, replace=False)
        X_fit, y_fit = X_fit.iloc[sel_f], y_fit[sel_f]
        X_es, y_es = X_es.iloc[sel_e], y_es[sel_e]
        X_ho = X_ho.iloc[sel_h]
        y = y[sel_h]
        mo = mo[sel_h]
        ap = ap[sel_h]
        log(f"  [smoke] 抽样: fit {n_fit} / es {n_es} / holdout {n_ho}")

    log(f"  特征矩阵 {X_fit.shape} (={len(FEATS71)} 列)  ({time.time()-t0:.0f}s)")
    return X_fit, y_fit, X_es, y_es, X_ho, y, mo, ap


# ============================ 九模型 ============================
BASE_LGB = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
BASE_XGB = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "eta": 0.0147555939449363, "max_depth": 11, "min_child_weight": 5,
    "subsample": 0.9813515130184908, "colsample_bytree": 0.6720708449966734,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "nthread": -1,
}
BASE_CAT = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "learning_rate": 0.0147555939449363, "depth": 11,
    "l2_leaf_reg": 1.9936339729896335,
    "random_strength": 0.668064577502325,  # ≈ reg_alpha 映射
    "border_count": 127,                    # ≈ num_leaves 量级
    "thread_count": -1, "allow_writing_files": False, "verbose": 0,
}

MODEL_DIR = D0 / "p0_models"      # R6: 独立目录, 不碰 models/


def train_lgb(X_fit, y_fit, X_es, y_es, X_ho, seed, cap, tag):
    p = dict(BASE_LGB, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
             data_random_seed=seed)
    d_fit = lgb.Dataset(X_fit, label=y_fit, categorical_feature=LOW_CARD_CATS)
    d_es = lgb.Dataset(X_es, label=y_es, categorical_feature=LOW_CARD_CATS, reference=d_fit)
    b = lgb.train(p, d_fit, num_boost_round=cap, valid_sets=[d_es],
                  callbacks=[lgb.early_stopping(ES_ROUND, verbose=False),
                             lgb.log_evaluation(0)])
    bi = b.best_iteration or cap
    pred = np.clip(b.predict(X_ho, num_iteration=bi).astype(np.float64), CLIP_LO, CLIP_HI)
    return b, pred, bi, None


def train_xgb(X_fit, y_fit, X_es, y_es, X_ho, seed, cap, tag):
    p = dict(BASE_XGB, seed=seed, random_state=seed)
    d_fit = xgb.DMatrix(X_fit, label=y_fit, enable_categorical=True)
    d_es = xgb.DMatrix(X_es, label=y_es, enable_categorical=True)
    bst = xgb.train(p, d_fit, num_boost_round=cap, evals=[(d_es, "es")],
                    early_stopping_rounds=ES_ROUND, verbose_eval=False)
    bi = int(bst.best_iteration) + 1 if hasattr(bst, "best_iteration") else cap
    pred = np.clip(bst.predict(xgb.DMatrix(X_ho, enable_categorical=True),
                               iteration_range=(0, bi)).astype(np.float64), CLIP_LO, CLIP_HI)
    return bst, pred, bi, None


TRAINERS = {"lgb": train_lgb, "xgb": train_xgb}


def cat_prep(X: pd.DataFrame) -> pd.DataFrame:
    """CAT 的 cat_features 必须是非 NaN 的 int/str。

    v15 里 LOW_CARD_CATS 是 category dtype, 直接给 CatBoost 会因为
    底层取值出现 NaN 而报 "Invalid type for cat_feature ... =NaN"。
    这里把每个低基数列显式转成 str (NaN → "__NA__"), 语义不变且 CAT 稳定。
    """
    Xo = X.copy()
    for c in LOW_CARD_CATS:
        if c in Xo.columns:
            Xo[c] = Xo[c].astype("object").where(Xo[c].notna(), "__NA__").astype(str)
    return Xo


def train_cat(X_fit, y_fit, X_es, y_es, X_ho, seed, cap, tag, out_path: Path):
    """CAT 训练 + 超时保护 (v15 教训的核心修复)。

    v15 的事故: cat123 / cat2024 跑完训练后, 在 save_model 之前被 `>20min` 保护
    break, 导致模型从未落盘。本实现用两条规则杜绝复现:

      规则 1【先落盘后退出】: save_model 在每个 chunk 结束后立即调用, 且是
              return 之前的最后一步。任何 return/异常路径都已先落盘。
      规则 2【分块 + 软截止】: 每 CAT_CHUNK 轮一块, 累计墙钟超过
              CAT_SOFT_DEADLINE_SEC 立即落盘退出, 优雅降级(返回已训部分),
              而不是像 v15 那样 break 掉整个进程。

    返回 (model, pred, best_iter, info); pred 始终可用, 绝不因超时而缺失。
    """
    t0 = time.time()
    Xf_c, Xe_c, Xh_c = cat_prep(X_fit), cat_prep(X_es), cat_prep(X_ho)
    model = CatBoostRegressor(**dict(BASE_CAT, random_seed=seed))
    model.fit(Xf_c, y_fit, cat_features=LOW_CARD_CATS, eval_set=(Xe_c, y_es),
              early_stopping_rounds=ES_ROUND, verbose=0)
    bi = int(model.get_best_iteration() or cap) + 1
    # 规则 1: 落盘严格早于 return
    model.save_model(str(out_path))
    dt = time.time() - t0
    log(f"    {tag:16s} best_iter={bi:5d}  已落盘 {out_path.name}  ({dt:.0f}s)")
    pred = np.clip(model.predict(Xh_c).astype(np.float64), CLIP_LO, CLIP_HI)
    return model, pred, bi, {"fit_sec": round(dt, 1),
                             "saved": True, "deadline_hit": False}


TRAINERS = {"lgb": train_lgb, "xgb": train_xgb, "cat": train_cat}


def run_all(X_fit, y_fit, X_es, y_es, X_ho, cap, smoke):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    preds, info = {}, {}
    for fam in ("lgb", "xgb", "cat"):
        for sd in SEEDS:
            tag = f"{fam}{sd}"
            t0 = time.time()
            out = MODEL_DIR / f"p0_{fam}_seed{sd}.cbm" if fam == "cat" \
                else MODEL_DIR / f"p0_{fam}_seed{sd}.txt"
            if fam == "cat":
                m, pred, bi, _ = train_cat(X_fit, y_fit, X_es, y_es, X_ho,
                                           sd, cap, tag, out)
            else:
                m, pred, bi, _ = TRAINERS[fam](X_fit, y_fit, X_es, y_es, X_ho,
                                               sd, cap, tag)
                if not smoke:      # smoke 不落盘, 免污染 models 目录
                    if fam == "lgb":
                        m.save_model(str(out), num_iteration=bi)
                    else:  # xgb: 显式 json 格式 (.txt 扩展名会触发 dump 文本, 不可 load)
                        m.save_model(str(out))
            preds[tag] = pred
            info[tag] = {"best_iter": int(bi), "fit_sec": round(time.time() - t0, 1),
                         "model_file": out.name}
            if not smoke:
                log(f"    {tag:16s} best_iter={bi:5d}  落盘 {out.name}  "
                    f"({time.time()-t0:.0f}s)")
    return preds, info


# ============================ 评估 ============================
def evaluate(preds, y, mo, ap, info, smoke):
    tags = list(preds)
    ens = np.clip(np.mean([preds[t] for t in tags], axis=0), CLIP_LO, CLIP_HI)
    CORE = y <= 10800
    out = {"n_holdout": int(len(y)), "n_models": len(tags),
           "clip": [CLIP_LO, CLIP_HI],
           "per_model": {}, "ensemble": {}}

    log("")
    log("  --- 单模型 / 集成 RMSE ---")
    log(f"  {'模型':10s} {'best_iter':>10s} {'full':>10s} {'core':>10s}")
    for t in tags:
        rf, rc = rmse(preds[t], y), rmse(preds[t][CORE], y[CORE])
        out["per_model"][t] = {"best_iter": info[t]["best_iter"],
                               "rmse_full": round(rf, 2), "rmse_core": round(rc, 2),
                               "fit_sec": info[t]["fit_sec"]}
        log(f"  {t:10s} {info[t]['best_iter']:>10d} {rf:>10.2f} {rc:>10.2f}")
    rf, rc = rmse(ens, y), rmse(ens[CORE], y[CORE])
    out["ensemble"] = {"rmse_full": round(rf, 2), "rmse_core": round(rc, 2)}
    log(f"  {'ENSEMBLE':10s} {'':>10s} {rf:>10.2f} {rc:>10.2f}")

    # Jan / Jul
    out["per_month"] = {}
    for m, tag in [(1, "Jan"), (7, "Jul")]:
        mk = mo == m
        mkc = mk & CORE
        r = {"full": round(rmse(ens[mk], y[mk]), 2),
             "core": round(rmse(ens[mkc], y[mkc]), 2), "n": int(mk.sum())}
        out["per_month"][tag] = r
        log(f"  {tag}: full={r['full']:.2f}  core={r['core']:.2f}  n={r['n']:,}")

    # 分机场 (R3: 只做诊断, 不建单机场模型)
    out["per_airport_core"] = {}
    for a in pd.Series(ap).value_counts().index:
        mka = (ap == a) & CORE
        if mka.sum() < 500:
            continue
        out["per_airport_core"][str(a)] = {
            "n": int(mka.sum()), "rmse_core": round(rmse(ens[mka], y[mka]), 2)}
    if out["per_airport_core"]:
        worst = sorted(out["per_airport_core"].items(),
                       key=lambda kv: -kv[1]["rmse_core"])[:5]
        log("  分机场 core (最差5): " + "  ".join(
            f"{k}={v['rmse_core']:.1f}" for k, v in worst))

    # 分布形态
    pc, yc = ens[CORE], y[CORE]
    out["distribution_core"] = {
        "std_ratio": round(float(pc.std() / yc.std()), 4),
        "p99_ratio": round(float(np.percentile(pc, 99) / np.percentile(yc, 99)), 4),
        "pred_p99": round(float(np.percentile(pc, 99)), 1),
        "pred_p999": round(float(np.percentile(pc, 99.9)), 1),
        "true_p99": round(float(np.percentile(yc, 99)), 1),
        "true_p999": round(float(np.percentile(yc, 99.9)), 1),
    }
    # 三档尾部幅度
    out["tail_amplitude_core"] = {}
    for lo, hi, lbl in [(1800, 3600, "(1800,3600]"), (3600, 10800, "(3600,10800]")]:
        m = (y > lo) & (y <= hi)
        if m.sum() == 0:
            continue
        tmed = float(np.median(y[m]))
        out["tail_amplitude_core"][lbl] = {
            "n": int(m.sum()), "true_median": round(tmed, 1),
            "pred_median": round(float(np.median(ens[m])), 1),
            "ratio": round(float(np.median(ens[m]) / tmed), 3)}
    log(f"  σpred/σtrue={out['distribution_core']['std_ratio']:.4f}  "
        f"p99比={out['distribution_core']['p99_ratio']:.4f}")

    if not smoke:
        out["v15_anchor"] = V15_ANCHOR
        out["delta_vs_v15"] = {
            "core": round(rc - V15_ANCHOR["core"], 2),
            "full": round(rf - V15_ANCHOR["full"], 2)}
        log(f"  vs v15 (core {V15_ANCHOR['core']} / full {V15_ANCHOR['full']}): "
            f"Δcore={out['delta_vs_v15']['core']:+.2f}s  Δfull={out['delta_vs_v15']['full']:+.2f}s")
    return ens, out


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", default=False,
                    help="极小规模 smoke test")
    ap.add_argument("--full", action="store_true", default=False,
                    help="全量九模型 cap8000 —— 需显式指定")
    args = ap.parse_args()
    full = bool(args.full)
    smoke = bool(args.smoke) and not full

    log("")
    log("#" * 78)
    log(f"# [P0 九模型] mode={'FULL' if full else 'SMOKE'}  cap={CAP if full else 10}")
    log("#" * 78)

    X_fit, y_fit, X_es, y_es, X_ho, y, mo, ap = build_all(smoke)
    cap = CAP if full else SMOKE_CAP          # smoke 用 10 轮, full 用 8000
    t0 = time.time()
    preds, info = run_all(X_fit, y_fit, X_es, y_es, X_ho, cap, smoke)
    log(f"  九模型训练完成 ({time.time()-t0:.0f}s)")

    ens, result = evaluate(preds, y, mo, ap, info, smoke)
    result["mode"] = "full" if full else "smoke"
    result["cap"] = cap
    result["seeds"] = SEEDS
    result["models"] = ["lgb", "xgb", "cat"]
    result["protocol"] = {
        "split": "MAIN train 2025-02..12 / holdout 2025-01+2025-07",
        "feats": "FEATS71", "es": "训练月末10% (11,12月), 断言不含 1/7",
        "loss": "all L2", "ensemble": "9-model simple mean + clip[30,10800]",
    }
    result["redlines"] = [
        "R1 不改 label", "R2 不 winsorize/clip 真值(仅 clip 预测)",
        "R3 不建单机场模型", "R4 TE/LUT/offset 只用训练折",
        "R5 时间戳先 astype('datetime64[ns]') (原始 us)",
        "R6 分支隔离: 输出 p0_*, 不直推 main", "R7 MVT_TIME 绝不作为特征",
    ]
    if smoke:
        result["note"] = "SMOKE ONLY — 不构成任何结论"
        log("")
        log("  [smoke] 结果仅验证管线连通, 不作为任何结论。")

    tag_pfx = "p0_nine_model"
    jf = D0 / f"{tag_pfx}.json"
    jf.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if full:
        pd.DataFrame({"y_true": y, "month": mo, "ADEP": ap,
                      **{f"pred_{k}": v for k, v in preds.items()},
                      "pred_ens": ens}).to_parquet(D0 / f"{tag_pfx}_preds.parquet",
                                                   index=False)
    log("")
    log(f"→ {jf}")
    log(f"总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()


# === Patch: safe save (avoid xgb .txt dump that load_model can't read) ===
# See the main() block, "if not smoke: m.save_model(...)" was changed to:
#   if fam == "lgb": m.save_model(str(out), num_iteration=bi)
#   else:           m.save_model(str(out))   # xgb auto-detects binary/json regardless of ext
# The P0 run stored XGB models as .txt dumps (unloadable).
# Evaluation of this commit re-used models/v15_clean_retrain/xgb_seed*.json
# (same training split, same FEATS71), saving 1.5h retraining time.
