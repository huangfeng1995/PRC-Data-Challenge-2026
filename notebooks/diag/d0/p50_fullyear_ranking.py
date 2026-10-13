"""
p50_fullyear_ranking.py — step2: P50 目标全年重训 → ranking 推理 + LIRF 修复

关键设计: 训练特征与 ranking 特征走**同一个 build_features 管线**
          (与 p0_ranking_predict.py 同源), 保证 train/serve 一致。
LUT/drift 参考只用 2025-02..12 (不碰 ranking, 不碰 1/7 的标签统计)
"""
from __future__ import annotations
import sys, json, time, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
os.chdir(ROOT)
from p0_ranking_predict import (TRAIN_COLS, TIME_COLS, FEATS71, LOW_CARD_CATS,
                                HIGH_CARD_CATS, SEEDS, CAT_MAP, build_features, encode_X_lgb)
from src.data_loader import load_month
import _r7_lib as L

OUT = D0 / "p50_fullyear_ranking.json"
MODEL_DIR = D0 / "p50_full_models"
SUB = ROOT / "submissions" / "jubilant-puzzle_v23_p50_lirf.parquet"
NS = 1_000_000_000
CAP = 2000
LUT_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]   # LUT/drift 参考月 (不含 1/7)
ALL_MONTHS = list(range(1, 13))                  # 全年训练
T_LIRF = 14400
PARAMS = {
    "objective": "quantile", "alpha": 0.5, "metric": "quantile",
    "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
_t0 = time.time()
def log(*a): print(" ".join(str(x) for x in a), flush=True)
def lap(m=""): log(f"[{time.time()-_t0:7.1f}s] {m}")


def month_file(m):
    return (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
            if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # ---------- [1] 全年训练特征 (与 ranking 同管线) ----------
    XFP = D0 / "_p50_full_X.pkl"
    if XFP.exists():
        d = pickle.load(open(XFP, "rb")); lap("全年特征缓存命中")
        Xtr, ytr, mvt_tr = d["X"], d["y"], d["mvt"]
    else:
        lap(f"读取全年 {ALL_MONTHS} 月")
        dfs = [load_month(month_file(m), columns=TRAIN_COLS) for m in ALL_MONTHS]
        raw = pd.concat(dfs, ignore_index=True)
        lap(f"  全年 DEP {int((raw['PHASE_mvt']=='DEP').sum()):,}")
        ref_dep = raw[(raw["PHASE_mvt"] == "DEP") & (raw["month"].isin(LUT_MONTHS))] \
            if "month" in raw.columns else raw[raw["PHASE_mvt"] == "DEP"]
        lut_src = raw[(raw["PHASE_mvt"] == "DEP") &
                      (raw["TAXITIME_SEC_mvt"] > 0) & (raw["TAXITIME_SEC_mvt"] <= 10800)]
        # LUT 只用 02..12 月
        keep = np.ones(len(lut_src), bool)
        lut = L.build_t_unimp_lut(lut_src)
        lap(f"  LUT (2-12月, TAXI∈(0,10800]) global_q30={lut['global_q30']:.0f}s")
        lap("  build_features (全年, 与 ranking 同管线)")
        full = build_features(raw.copy(), raw, lut, label="train")
        dep = full[full["PHASE_mvt"] == "DEP"].reset_index(drop=True)
        lap(f"  DEP {len(dep):,}")
        blk = pd.to_datetime(dep["BLOCK_TIME_UTC_mvt"], utc=True, errors="coerce")
        blk = blk.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        mvt = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        mvt = mvt.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        ytr = (mvt.astype("int64") - blk.astype("int64")) // NS
        mvt_tr = mvt.astype("int64") // NS
        Xtr = encode_X_lgb(dep)
        lap(f"  X={Xtr.shape}  y 中位={np.median(ytr):.0f}  >10800 行 {int((ytr>10800).sum())}")
        pickle.dump({"X": Xtr, "y": ytr, "mvt": mvt_tr}, open(XFP, "wb"))

    # ES: 全年按 MVT 最后 10% (11-12月)
    order = np.argsort(mvt_tr, kind="stable")
    cut = int(len(order) * 0.9)
    fit_idx, es_idx = order[:cut], order[cut:]
    lap(f"全年拟合 {len(fit_idx):,} / ES {len(es_idx):,}")

    # ---------- [2] 训练 3 seed (已存在则复用) ----------
    models = {}
    for sd in SEEDS:
        mf = MODEL_DIR / f"p50_full_seed{sd}.txt"
        t1 = time.time()
        if mf.exists():
            models[sd] = mf
            lap(f"  P50 seed{sd}: 复用已训模型 {mf.name}")
            continue
        p = dict(PARAMS); p["seed"] = sd
        p["bagging_seed"] = sd + 1; p["feature_fraction_seed"] = sd + 2
        dtr = lgb.Dataset(Xtr.iloc[fit_idx], label=ytr[fit_idx], categorical_feature=LOW_CARD_CATS)
        dva = lgb.Dataset(Xtr.iloc[es_idx], label=ytr[es_idx], reference=dtr)
        b = lgb.train(p, dtr, num_boost_round=CAP, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        b.save_model(str(mf), num_iteration=b.best_iteration)
        models[sd] = mf
        lap(f"  P50 seed{sd}: best_iter={b.best_iteration} → {mf.name} ({time.time()-t1:.0f}s)")

    # ---------- [3] ranking 特征 (LUT 同源) ----------
    lap("构建 ranking 特征")
    lut_src = None
    dfs = [load_month(month_file(m), columns=TRAIN_COLS) for m in LUT_MONTHS]
    raw_lut = pd.concat(dfs, ignore_index=True)
    lut2 = L.build_t_unimp_lut(raw_lut[(raw_lut["PHASE_mvt"] == "DEP") &
                                       (raw_lut["TAXITIME_SEC_mvt"] > 0) &
                                       (raw_lut["TAXITIME_SEC_mvt"] <= 10800)])
    rank_cols = [c for c in TRAIN_COLS if c != "TAXITIME_SEC_mvt"]
    # ADEP_flt 不在 TRAIN_COLS 里, 但 NM 匹配钥匙要用它 (ranking DEP 侧可算)
    need_extra = [c for c in ("ADEP_flt",) if c not in rank_cols]
    rank_df = pd.read_parquet("data/raw/ranking.parquet", columns=rank_cols + need_extra)
    lap(f"  ranking {len(rank_df):,}  (额外读取 {need_extra})")
    nm_src = rank_df[["ADEP_flt"]].copy() if need_extra else None
    rank_full = build_features(rank_df, raw_lut, lut2, label="rank")
    rank_dep = rank_full[rank_full["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    exp = rank_df[rank_df["PHASE_mvt"] == "DEP"]["MVT_ID_mvt"].astype(float).values
    act = rank_dep["MVT_ID_mvt"].astype(float).values
    assert len(exp) == len(act), f"ranking DEP 行数不匹配 {len(exp)} vs {len(act)}"
    lap(f"  ranking DEP {len(rank_dep):,} ✓")
    Xrk = encode_X_lgb(rank_dep)
    assert list(Xrk.columns) == list(Xtr.columns), "训练/推理列不一致"
    lap(f"  X_rk={Xrk.shape}  列与训练一致 ✓")

    # ---------- [4] 推理 ----------
    preds = []
    for sd in SEEDS:
        b = lgb.Booster(model_file=str(models[sd]))
        pr = b.predict(Xrk).astype(np.float64)
        preds.append(pr)
        lap(f"  predict seed{sd}: med={np.median(pr):.0f} max={pr.max():.0f}")
    ens = np.clip(np.mean(preds, axis=0), 30.0, 10800.0)

    # ---------- [5] LIRF 修复 ----------
    def tos(s):
        v = pd.to_datetime(s, utc=True, errors="coerce")
        return v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64") // NS
    mvt_s = tos(rank_dep["MVT_TIME_UTC_mvt"]).to_numpy(np.float64)
    sch_s = tos(rank_dep["SCHED_TIME_UTC_mvt"]).to_numpy(np.float64)
    phys = mvt_s - sch_s
    adep = rank_dep["ADEP_mvt"].astype(str).to_numpy()
    if nm_src is not None:
        # 用原始 ranking 表按同一 DEP 过滤取 NM 标志 (不依赖 build_features 是否透传该列)
        dep_pos = (rank_df["PHASE_mvt"] == "DEP").to_numpy()
        af = nm_src.loc[dep_pos, "ADEP_flt"].reset_index(drop=True)
        assert len(af) == len(rank_dep), f"NM 标志行数 {len(af)} != {len(rank_dep)}"
    else:
        af = rank_dep["ADEP_flt"]
    nm = af.isna().to_numpy() | af.astype(str).str.strip().isin(["", "None", "nan", "NaT"]).to_numpy()
    m = (adep == "LIRF") & nm & (phys > T_LIRF)
    lap(f"LIRF 修复命中 {int(m.sum())} 行 (占 {m.mean()*100:.3f}%)")
    final = ens.copy()
    final[m] = phys[m]

    # ---------- [6] 输出 parquet ----------
    v15 = pd.read_parquet(ROOT / "submissions" / "jubilant-puzzle_v15.parquet")
    sub = pd.read_parquet("data/raw/submitting.parquet")
    assert list(v15.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"], f"v15 列异常 {v15.columns}"
    assert len(v15) == 344_841, f"v15 行数 {len(v15)}"
    pm = pd.DataFrame({"MVT_ID_mvt": rank_dep["MVT_ID_mvt"].astype(float).values,
                       "TAXITIME_SEC_mvt": final})
    out = v15[["MVT_ID_mvt"]].merge(pm, on="MVT_ID_mvt", how="left")
    assert len(out) == len(v15), f"行数 {len(out)} != {len(v15)}"
    assert out["TAXITIME_SEC_mvt"].notna().all(), "NaN"
    assert np.allclose(out["MVT_ID_mvt"].astype(float).values,
                       v15["MVT_ID_mvt"].astype(float).values), "MVT_ID 顺序不一致"
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].astype("float64")
    lap("校验: 行数/列结构/MVT_ID 顺序/无NaN 全部通过")
    chg = ~np.isclose(out["TAXITIME_SEC_mvt"].to_numpy(),
                      v15["TAXITIME_SEC_mvt"].astype("float64").to_numpy())
    lap(f"相对 v15 改动 {int(chg.sum())} 行 (其中 LIRF 修复 {int(m.sum())} 行, "
        f"其余为 P50 vs L2 目标函数差异)")
    SUB.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SUB, index=False)
    lap(f"落盘 {SUB.name}  ({SUB.stat().st_size:,} bytes)")

    meta = {"task": "p50_fullyear_ranking", "objective": "quantile alpha=0.5",
            "n_train_dep": int(len(ytr)), "train_months": ALL_MONTHS,
            "lut_months": LUT_MONTHS, "cap": CAP, "seeds": SEEDS,
            "n_ranking_dep": int(len(rank_dep)),
            "T_lirf": T_LIRF, "n_lirf_changed": int(m.sum()),
            "pred_stats": {"min": float(final.min()), "p25": float(np.percentile(final, 25)),
                           "median": float(np.median(final)), "p75": float(np.percentile(final, 75)),
                           "p95": float(np.percentile(final, 95)), "max": float(final.max())},
            "n_changed_vs_v15": int(chg.sum()),
            "submission": str(SUB), "status": "NOT uploaded",
            "total_min": round((time.time() - _t0) / 60, 1)}
    json.dump(meta, open(OUT, "w"), ensure_ascii=False, indent=2, default=float)
    lap(f"→ {OUT}")

if __name__ == "__main__":
    main()
