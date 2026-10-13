"""
Stage 2: rebuild features on final_ranking (2026 Jan/Feb/Jun/Jul) and generate submission
==============================================================
Scheme D spec (verified in stage 1: no degradation at 10/10 airports):
  - LIRF rows        : expected mixture mu = p*(MVT-SCHED) + (1-p)*mu_N  (SCHED outside LOBT window -> p=0)
  - non-LIRF rows    : LGB baseline 3-seed average, no mixing
  - all LOBT-valid rows: project to [MVT-LOBT-3606, MVT-LOBT+3606]; skip when LOBT is missing
  - clip [30, 10800]

Feature rebuild:
  - use exactly the same build_features pipeline as 2025, rebuild FEATS71 on final_ranking DEP
  - all LUTs (high-cardinality encoding / t_unimp / cell stats) use only the 2025 training folds, no 2026 information
  - recompute all MVT-dependent features from final_ranking MVT (do not reuse the old ranking cache)
  - LOBT/SCHED/EOBT and other _flt columns are taken from final_ranking
  - train/serve consistency assertion: identical column names / order / dtype

Models:
  - LGB baseline: p0_models/p0_lgb_seed{42,123,2024}.txt (2025 training folds, 71 features)
  - copy classifier / normal regressor: cp_mixture_models/{clf,reg}.pkl

Output: submissions/jubilant-puzzle_final.parquet (MVT_ID_mvt, TAXITIME_SEC_mvt)

Hard constraints: no label changes, no external data, no hardcoded airports (except the existing LIRF rule), no auto upload
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

SUB_PATH = ROOT / "submissions" / "jubilant-puzzle_final.parquet"
META = D0 / "final_predict_meta.json"
NPY = D0 / "_final_d_raw.npz"

LIRF_AP = "LIRF"
T_LOBT = 3606.0
CLIP_LO, CLIP_HI = 30.0, 10800.0
SEEDS = [42, 123, 2024]
P0_DIR = D0 / "p0_models"
CP_DIR = D0 / "cp_mixture_models"

from src.data_loader import load_month, parse_time_columns  # noqa: E402
from src.features import build_features as build_basic  # noqa: E402
from src.features_queue import add_queue_features  # noqa: E402
import _r7_lib as L, _r8_drift_lib as D  # noqa: E402

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
assert len(FEATS71) == 71, len(FEATS71)

LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]
CAT_MAP = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def to_ns(s):
    s2 = pd.to_datetime(s)
    if hasattr(s2.dtype, "tz") and s2.dtype.tz is not None:
        s2 = s2.dt.tz_localize(None)
    return s2.astype("datetime64[ns]")


def to_sec(s):
    v = to_ns(s).to_numpy("int64") // 10**9
    v = v.astype(np.float64)
    v[v < 0] = np.nan
    return v


def build_features(df_raw, ref_dep, lut, label):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    dfb = build_basic(df)
    df_full = add_queue_features(dfb.copy(), include_a=True, include_b=True,
                                 include_c=True, include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    for k, v in D.compute_drift_features(df_full, ref_dep=ref_dep, label=label).items():
        df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    for k, v in L.compute_extra_features(df_full[dep_mask], label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    for k, v in L.compute_mvt_anchor_features(df_full[dep_mask],
                                              t_unimp_vals=t_vals[dep_mask],
                                              label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    df_full["MVT_TIME_UTC_mvt"] = to_ns(df_full["MVT_TIME_UTC_mvt"])
    return df_full


def encode_X(df):
    """Exactly consistent with P0 training (p0_nine_model.train_lgb):
    high-cardinality columns use integer encoding from CAT_MAP (CAT_MAP is derived from the 2025 training folds),
    the 4 low-cardinality columns keep category dtype (categorical_feature=LOW_CARD_CATS during training).
    NaN is uniformly filled with -999 (same as load_cache)."""
    X = df[FEATS71].copy()
    for c in HIGH_CARD_CATS:
        if c in X.columns:
            m = CAT_MAP.get(c, {})
            X[c] = X[c].astype(str).map(m).fillna(-1).astype("int32")
    for c in LOW_CARD_CATS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    # NaN in the remaining numeric columns -> -999
    num_cols = [c for c in X.columns if c not in LOW_CARD_CATS]
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(-999.0)
    assert list(X.columns) == FEATS71
    # LOW_CARD_CATS are category dtype, NaN is legal (same as training); only assert no NaN in numeric columns
    bad = [c for c in num_cols if X[c].isna().any()]
    assert not bad, f"numeric columns still have NaN: {bad}"
    return X


def encode_X_cp(df):
    """Training encoding for the copy classifier / normal regressor (copy_mixture_lobt reuses ngb_probe_reduced.load_cache):
    every category column takes cat.codes directly -> float32, NaN -> -999."""
    X = df[FEATS71].copy()
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    for c in cats:
        X[c] = X[c].cat.codes.astype(np.float32)
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    if nan_cols:
        X[nan_cols] = X[nan_cols].fillna(-999.0)
    assert list(X.columns) == FEATS71
    assert not X.isna().any().any(), "NaN still present after encoding"
    return X


def main():
    # ---------- [1] Training-fold LUT (2025-02..12 only, 2026 forbidden) ----------
    lap("[1] Building LUT (fit on 2025 months 2..12 only)")
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr = []
    for m in train_months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        tr.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(tr, ignore_index=True)
    del tr
    df_clean = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                         (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                         (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)].copy()
    lut = L.build_t_unimp_lut(df_clean)
    lap(f"  train DEP (2-12, TX in (0,10800]) = {len(df_clean):,}, global_q30={lut['global_q30']:.0f}s")

    # ---------- [2] Read final_ranking ----------
    lap("[2] Reading final_ranking.parquet (2026 Jan/Feb/Jun/Jul)")
    fin = pd.read_parquet("data/raw/final_ranking.parquet")
    assert "TAXITIME_SEC_mvt" in fin.columns
    n_all = len(fin)
    fin_dep_raw = fin[fin["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    lap(f"  all rows {n_all:,} → DEP {len(fin_dep_raw):,}")
    # do not touch label: TAXITIME / BLOCK of final DEP must be all null
    assert fin_dep_raw["TAXITIME_SEC_mvt"].isna().all(), "final DEP contains non-null TAXITIME!"
    assert fin_dep_raw["BLOCK_TIME_UTC_mvt"].isna().all(), "final DEP contains non-null BLOCK!"

    # ---------- [3] Rebuild features ----------
    lap("[3] Rebuilding FEATS71 (based on final MVT, LUT from 2025 only)")
    fin_full = build_features(fin_dep_raw, df_tr_raw, lut, label="final")
    fin_dep = fin_full[fin_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    lap(f"  fin_dep = {len(fin_dep):,}")

    # train/serve consistency assertion
    blob = pickle.load(open(D0 / "_qinv_feats.pkl", "rb"))
    tr71 = [c for c in blob["X"].columns if not c.startswith("q_")]
    assert tr71 == FEATS71, "FEATS71 is inconsistent with training"
    lap(f"  ✓ FEATS71 column names+order match training ({len(FEATS71)} columns)")

    # ---------- [4] Encoding ----------
    lap("[4] Integer encoding (two sets: P0 baseline / copy models)")
    X = encode_X(fin_dep)          # P0 LGB baseline
    Xcp = encode_X_cp(fin_dep)     # copy classifier + normal regressor
    assert list(X.columns) == FEATS71 and list(Xcp.columns) == FEATS71
    lap(f"  X (P0) {X.shape}   X_cp {Xcp.shape}")

    # ---------- [5] Inference ----------
    lap("[5] Inference: LGB baseline 3-seed + copy classifier + normal regressor")
    base_seeds = []
    for sd in SEEDS:
        b = lgb.Booster(model_file=str(P0_DIR / f"p0_lgb_seed{sd}.txt"))
        base_seeds.append(b.predict(X))
        lap(f"  lgb{sd} done")
    base = np.mean(np.stack(base_seeds), axis=0)

    clf = pickle.load(open(CP_DIR / "clf.pkl", "rb"))
    reg = pickle.load(open(CP_DIR / "reg.pkl", "rb"))
    p = np.clip(clf.predict(Xcp, num_iteration=clf.best_iteration), 0, 1)
    mu_N = reg.predict(Xcp, num_iteration=reg.best_iteration)
    lap(f"  p mean={p.mean():.4f}  median={np.median(p):.4f}  p>0.5={int((p>0.5).sum()):,}")

    # ---------- [6] Scheme D composition ----------
    lap("[6] Scheme D composition")
    mvt = to_sec(fin_dep["MVT_TIME_UTC_mvt"])
    sch = to_sec(fin_dep["SCHED_TIME_UTC_mvt"])
    lobt = to_sec(fin_dep["LOBT_flt"])
    lobt_ok = ~np.isnan(lobt)
    ap = fin_dep["ADEP_mvt"].astype(str).to_numpy()
    is_lirf = (ap == LIRF_AP)

    mu_C = mvt - sch
    ctr = mvt - lobt
    lo_w, hi_w = ctr - T_LOBT, ctr + T_LOBT
    sched_out = lobt_ok & (np.abs(sch - lobt) > T_LOBT)
    p_adj = p.copy()
    p_adj[sched_out] = 0.0
    mix = np.maximum(p_adj * mu_C + (1 - p_adj) * mu_N, 0.0)

    Dp = np.where(is_lirf, mix, base)
    Dp = np.clip(Dp, CLIP_LO, CLIP_HI)
    # LOBT projection is applied after clip, so it must be clipped once more to keep values in range
    m_ok = lobt_ok
    before = Dp[m_ok].copy()
    Dp[m_ok] = np.clip(before, lo_w[m_ok], hi_w[m_ok])
    n_proj_changed = int((np.abs(Dp[m_ok] - before) > 1e-9).sum())
    n_proj_reclip = int(((Dp < CLIP_LO) | (Dp > CLIP_HI)).sum())
    Dp = np.clip(Dp, CLIP_LO, CLIP_HI)
    lap(f"  LOBT valid {int(m_ok.sum()):,} / {len(Dp):,}  (missing skipped {int((~m_ok).sum()):,})")
    lap(f"  LOBT projection changed {n_proj_changed:,} rows  (need re-clip after projection {n_proj_reclip:,} rows)")
    lap(f"  SCHED outside window p=0: {int(sched_out.sum()):,}")
    lap(f"  LIRF rows {int(is_lirf.sum()):,}  (of which replaced by the mixture {int((is_lirf).sum()):,})")

    np.savez(NPY, pred_D=Dp, pred_base=base, p=p, mu_N=mu_N, mu_C=mu_C,
             mvt=mvt, sch=sch, lobt=lobt, lobt_ok=lobt_ok, is_lirf=is_lirf,
             mvt_id=fin_dep["MVT_ID_mvt"].astype(float).to_numpy(),
             ap=ap)

    # ---------- [7] Align with final_submitting and write out ----------
    lap("[7] Aligning final_submitting.parquet and writing to disk")
    submitting = pd.read_parquet("data/raw/final_submitting.parquet")
    assert set(submitting.columns) == {"MVT_ID_mvt", "TAXITIME_SEC_mvt"}, submitting.columns.tolist()
    lap(f"  final_submitting rows = {len(submitting):,}")

    pred_map = pd.DataFrame({"MVT_ID_mvt": fin_dep["MVT_ID_mvt"].astype(float).values,
                             "TAXITIME_SEC_mvt": np.round(Dp, 0).astype(np.int64)})
    assert pred_map["MVT_ID_mvt"].is_unique, "final DEP MVT_ID has duplicates"
    out = submitting[["MVT_ID_mvt"]].merge(pred_map, on="MVT_ID_mvt", how="left")
    assert len(out) == len(submitting), f"row count mismatch {len(out)} vs {len(submitting)}"
    assert out["TAXITIME_SEC_mvt"].notna().all(), \
        f"NaN {int(out['TAXITIME_SEC_mvt'].isna().sum())}"
    assert (out["TAXITIME_SEC_mvt"] >= CLIP_LO).all()
    assert (out["TAXITIME_SEC_mvt"] <= CLIP_HI).all()
    exp_ids = submitting["MVT_ID_mvt"].astype(float).values
    assert np.all(exp_ids == out["MVT_ID_mvt"].astype(float).values), "MVT_ID order mismatch"
    assert set(exp_ids) == set(out["MVT_ID_mvt"].astype(float).values), "MVT_ID set mismatch"
    lap(f"  ✓ rows {len(out):,} / MVT_ID set+order fully aligned / no NaN / range valid")

    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SUB_PATH, index=False)
    lap(f"  ✓ saved {SUB_PATH}  ({SUB_PATH.stat().st_size:,} bytes)")

    # ---------- [8] Distribution self-check ----------
    lap("[8] Distribution self-check + per-month breakdown")
    mo = pd.to_datetime(mvt, unit="s").dt.month.to_numpy()

    def q(s, p):
        return float(np.percentile(s, p))

    print("\n" + "=" * 74)
    print("[Distribution self-check] Scheme D final predictions")
    print("=" * 74)
    print(f"  n      = {len(Dp):,}")
    for nm, v in [("min", Dp.min()), ("p25", q(Dp, 25)), ("median", q(Dp, 50)),
                  ("p75", q(Dp, 75)), ("p95", q(Dp, 95)), ("max", Dp.max())]:
        print(f"  {nm:<7} = {v:>10.1f}")
    print(f"  >10800 : {int((Dp>10800).sum()):,}")
    print(f"  >3600  : {int((Dp>3600).sum()):,}")
    print(f"  >1800  : {int((Dp>1800).sum()):,}")
    print(f"  ==30   : {int((Dp==30).sum()):,}")
    # baseline comparison
    print(f"\n  base median={np.median(base):.0f}  D median={np.median(Dp):.0f} "
          f"(Δ{np.median(Dp)-np.median(base):+.0f})")
    print(f"  base p95  ={q(base,95):.0f}  D p95  ={q(Dp,95):.0f} (Δ{q(Dp,95)-q(base,95):+.0f})")

    print("\n  Per-month row counts and prediction distribution:")
    for mm in [1, 2, 6, 7]:
        m = mo == mm
        print(f"    mo={mm}: n={int(m.sum()):>7,}  median={np.median(Dp[m]):>7.0f} "
              f"p95={q(Dp[m],95):>8.0f}  max={Dp[m].max():>8.0f}  "
              f">3600={int((Dp[m]>3600).sum()):>6,}")

    print("\n  Per-airport (D vs baseline):")
    dfc = pd.DataFrame({"ap": ap, "D": Dp, "base": base})
    g = dfc.groupby("ap").agg(n=("D", "size"), med_D=("D", "median"),
                              med_b=("base", "median"), max_D=("D", "max"))
    g["Δmed"] = g["med_D"] - g["med_b"]
    print(g.to_string(float_format=lambda v: f"{v:,.0f}"))

    meta = {
        "scheme": "D (LIRF probability mixture + global LOBT window)",
        "stage1_gate": {"D_not_worse_per_airport": "10/10 PASS",
                        "delta_vs_C": 16.52, "note": "C(v22) is better than D on LIRF"},
        "n_final_dep": int(len(fin_dep)),
        "n_submitting": int(len(out)),
        "T_lobt": T_LOBT,
        "clip": [CLIP_LO, CLIP_HI],
        "models": {
            "baseline_lgb": [f"notebooks/diag/d0/p0_models/p0_lgb_seed{s}.txt" for s in SEEDS],
            "copy_classifier": "notebooks/diag/d0/cp_mixture_models/clf.pkl",
            "normal_regressor": "notebooks/diag/d0/cp_mixture_models/reg.pkl",
        },
        "lut_source": "2025 fit months 2-12 only (no 2026 info)",
        "n_lobt_valid": int(lobt_ok.sum()),
        "n_lobt_missing_skipped": int((~lobt_ok).sum()),
        "n_lobt_projection_changed": n_proj_changed,
        "n_sched_outside_window_p0": int(sched_out.sum()),
        "n_lirf_rows": int(is_lirf.sum()),
        "p_mean": round(float(p.mean()), 4), "p_median": round(float(np.median(p)), 4),
        "n_p_gt_0p5": int((p > 0.5).sum()),
        "distribution": {"min": float(Dp.min()), "p25": q(Dp, 25), "median": q(Dp, 50),
                         "p75": q(Dp, 75), "p95": q(Dp, 95), "max": float(Dp.max()),
                         "n_gt_1800": int((Dp > 1800).sum()),
                         "n_gt_3600": int((Dp > 3600).sum()),
                         "n_gt_10800": int((Dp > 10800).sum())},
        "per_month_rows": {f"mo{mm}": int((mo == mm).sum()) for mm in [1, 2, 6, 7]},
        "submission_path": str(SUB_PATH),
        "submission_status": "NOT uploaded (manual upload by the user)",
        "total_time_min": round((time.time() - _t0) / 60, 1),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    lap(f"\n→ {META}")
    lap(f"total time {(time.time() - _t0)/60:.1f} min")


if __name__ == "__main__":
    main()