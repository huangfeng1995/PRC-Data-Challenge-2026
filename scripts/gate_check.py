"""
D0-4: one-click gate check scripts/gate_check.py
=====================================
Imports the frozen entry point, automatically checks and prints PASS/FAIL (non-zero exit code on failure), writes d0_gate_report.json:
1. baseline RMSE ~= 357.14 (+/-2)
2. Per-airport SSE summed and back-computed into a global RMSE is allclose to the direct global RMSE (tolerance 1e-6)
3. After cat mappings are persisted, row-wise inference in a separate process is allclose, max|d|=0
4. Predictions contain no NaN and no negative values
5. Submission MVT_ID_mvt set and order == submitting.parquet (344,841 rows), no missing/extra rows

Usage:
    python3 scripts/gate_check.py
Exit code: 0=PASS, 1=FAIL
"""
import sys
import json
import importlib.util
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load the frozen entry point
spec = importlib.util.spec_from_file_location(
    "ab_module",
    ROOT / "notebooks/diag/holdout_v1_queue_ab.py",
)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

import src.train_xgb as tx

REPORT_PATH = ROOT / "notebooks/diag/d0/d0_gate_report.json"


def check1_baseline(tolerance=2.0):
    """check 1: baseline RMSE ~= 357.14 (+/-2)"""
    print("\n" + "=" * 70)
    print("CHECK 1: baseline RMSE = 357.14 (±2)")
    print("=" * 70)

    months_train = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    data_dir = globals().get("DATA_DIR", "data/raw")
    dfs_train = []
    for m in months_train:
        if m < 12:
            fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        else:
            fname = "training_2025-12-01_2026-01-01.parquet"
        dfs_train.append(ab.load_month(f"{data_dir}/{fname}", columns=ab.TRAIN_COLS))
    df_train = pd.concat(dfs_train, ignore_index=True)
    df_train = ab.parse_time_columns(df_train, cols=ab.TIME_COLS)
    df_train = ab.build_full_features(df_train.copy())
    y = df_train["TAXITIME_SEC_mvt"].values
    mask = (y > 0) & (y < 10800)
    df_train = df_train[mask].reset_index(drop=True)

    dfs_ho = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        dfs_ho.append(ab.load_month(f"{data_dir}/{fname}", columns=ab.TRAIN_COLS))
    df_ho = pd.concat(dfs_ho, ignore_index=True)
    df_ho = ab.parse_time_columns(df_ho, cols=ab.TIME_COLS)
    df_ho = ab.build_full_features(df_ho.copy())
    df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy()
    df_ho_dep = df_ho_dep[
        (df_ho_dep["TAXITIME_SEC_mvt"] > 0) & (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)

    feat_cols = ab.get_feat_cols_for_group(df_train, ab.GROUPS["G0_basic30"])

    cat_path = ROOT / "models" / "feature_lists" / "gate_check_cat.json"
    if cat_path.exists():
        cat_path.unlink()
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = cat_path

    X_tr = df_train[feat_cols].copy()
    X_tr_enc = tx._encode_categoricals(X_tr, fit=True)
    dtrain = xgb.DMatrix(X_tr_enc, label=df_train["TAXITIME_SEC_mvt"].values)
    params = {**ab.DEFAULT_PARAMS, "seed": 0}
    booster = xgb.train(params, dtrain, num_boost_round=1200, verbose_eval=False)

    X_ho = df_ho_dep[feat_cols].copy()
    X_ho_enc = tx._encode_categoricals(X_ho, fit=False)
    y_pred = booster.predict(xgb.DMatrix(X_ho_enc))
    y_true = df_ho_dep["TAXITIME_SEC_mvt"].values

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    print(f"  RMSE = {rmse:.4f} (expected 357.14 ± {tolerance})")
    pass_ = abs(rmse - 357.14) < tolerance
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, rmse, df_ho_dep, y_pred, y_true, booster, feat_cols, cat_path


def check2_sse_global(rmse_global, df_ho_dep, y_pred, y_true):
    """check 2: per-airport SSE summed and back-computed global RMSE vs direct RMSE allclose"""
    print("\n" + "=" * 70)
    print("CHECK 2: per-airport SSE back-computed global RMSE allclose (1e-6)")
    print("=" * 70)
    ap = df_ho_dep["ADEP_mvt"].astype(str).values
    sse_total = 0.0
    for a in np.unique(ap):
        m = ap == a
        sse_a = float(np.sum((y_true[m] - y_pred[m]) ** 2))
        sse_total += sse_a
    rmse_recomputed = float(np.sqrt(sse_total / len(y_true)))
    diff = abs(rmse_recomputed - rmse_global)
    print(f"  direct RMSE = {rmse_global:.6f}")
    print(f"  back-computed RMSE = {rmse_recomputed:.6f}")
    print(f"  |Δ| = {diff:.2e}")
    pass_ = diff < 1e-6
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, rmse_recomputed, diff


def check3_cross_process_inference(booster, df_ho_dep, feat_cols, cat_path):
    """check 3: after cat persistence, separate-process inference is row-wise allclose (max|d|=0)"""
    print("\n" + "=" * 70)
    print("CHECK 3: separate-process inference allclose (max|Δ|=0)")
    print("=" * 70)

    # Save the model first so the separate process can load it
    model_path = ROOT / "models" / "gate_check_v1_model.json"
    booster.save_model(str(model_path))

    # Main-process inference
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = cat_path
    X_ho_enc = tx._encode_categoricals(df_ho_dep[feat_cols].copy(), fit=False)
    y_pred_main = booster.predict(xgb.DMatrix(X_ho_enc))

    # Save the main predictions
    tmp_main = Path(tempfile.gettempdir()) / "gate_main_pred.npy"
    np.save(tmp_main, y_pred_main)

    # Generate the standalone inference script
    indep_script = f'''
import sys, json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("{ROOT}")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ab_module", ROOT / "notebooks/diag/holdout_v1_queue_ab.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
import src.train_xgb as tx

# Data directory (defaults to data/raw; on cloud use data/sample, injected via DATA_DIR)
DATA_DIR = "{globals().get("DATA_DIR", "data/raw")}"

# Load holdout (1+7) DEP
dfs_ho = []
for m in [1, 7]:
    fname = f"training_2025-{{m:02d}}-01_2025-{{m + 1:02d}}-01.parquet"
    dfs_ho.append(ab.load_month(f"{{DATA_DIR}}/{{fname}}", columns=ab.TRAIN_COLS))
df_ho = pd.concat(dfs_ho, ignore_index=True)
df_ho = ab.parse_time_columns(df_ho, cols=ab.TIME_COLS)
df_ho = ab.build_full_features(df_ho.copy())
df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy()
df_ho_dep = df_ho_dep[(df_ho_dep["TAXITIME_SEC_mvt"] > 0) & (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)

feat_cols = {feat_cols!r}

# Use the persisted cat mappings
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS_PATH = Path("{cat_path}")

X_ho_enc = tx._encode_categoricals(df_ho_dep[feat_cols].copy(), fit=False)
booster = xgb.Booster()
booster.load_model("{model_path}")
y_pred = booster.predict(xgb.DMatrix(X_ho_enc))
np.save("{tmp_main.with_suffix('.indep.npy')}", y_pred)
print(f"standalone inference done, n={{len(y_pred)}}")
'''

    # Save the model + cat mappings so the separate process can access them
    model_path_local = ROOT / "models" / "gate_check_v1_model.json"
    if not model_path_local.exists():
        booster.save_model(str(model_path_local))

    # Write the standalone script
    script_path = Path(tempfile.gettempdir()) / "gate_indep_infer.py"
    script_path.write_text(indep_script)

    # Run the separate process
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    print(f"  separate process stdout: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")

    # Load the separate-process result
    indep_path = tmp_main.with_suffix(".indep.npy")
    if not indep_path.exists():
        print(f"  ❌ FAIL: separate process did not produce a result file")
        return False, None, None
    y_pred_indep = np.load(indep_path)

    max_abs_diff = float(np.max(np.abs(y_pred_main - y_pred_indep)))
    print(f"  max|Δ| = {max_abs_diff:.6e}")
    pass_ = max_abs_diff == 0.0
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, max_abs_diff, y_pred_indep


def check4_no_nan_no_negative(y_pred_clipped=None):
    """check 4: predictions contain no NaN and no negative values
    Checks the latest submission file (should be the final submittable version, already clipped)
    """
    print("\n" + "=" * 70)
    print("CHECK 4: submission predictions contain no NaN and no negative values")
    print("=" * 70)

    sub_dir = ROOT / "submissions"
    sub_files = sorted([p for p in sub_dir.glob("*.parquet") if p.is_file()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not sub_files:
        print(f"  ⚠️ no submissions/*.parquet available to check")
        return True, 0, 0
    sub_latest = sub_files[0]
    df_sub = pd.read_parquet(sub_latest)
    pred_col = [c for c in df_sub.columns if c != "MVT_ID_mvt"][0]
    preds = df_sub[pred_col].astype(float).values
    n_nan = int(np.isnan(preds).sum())
    n_neg = int((preds < 0).sum())
    print(f"  file: {sub_latest.name}")
    print(f"  NaN count = {n_nan}")
    print(f"  negative count = {n_neg}")
    pass_ = (n_nan == 0) and (n_neg == 0)
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, n_nan, n_neg


def check5_submission_format():
    """check 5: submission MVT_ID_mvt set and order == submitting.parquet (344,841 rows)"""
    print("\n" + "=" * 70)
    print("CHECK 5: submission MVT_ID set+order == submitting.parquet")
    print("=" * 70)

    sub_path = ROOT / globals().get("DATA_DIR", "data/raw") / "submitting.parquet"
    sub = pd.read_parquet(sub_path)
    expected_ids = sub["MVT_ID_mvt"].astype(float).values
    expected_n = len(expected_ids)
    print(f"  submitting.parquet: n={expected_n}, cols={list(sub.columns)}")

    # Find the latest submission (v1 / v10 / e3probe) to check
    sub_dir = ROOT / "submissions"
    sub_files = sorted([p for p in sub_dir.glob("*.parquet") if p.is_file()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not sub_files:
        print(f"  ⚠️ no submissions/*.parquet available to check")
        return True, "no submission file to check (schema check only)"

    sub_latest = sub_files[0]
    print(f"  checking latest submission: {sub_latest.name}")
    df_sub = pd.read_parquet(sub_latest)
    if "MVT_ID_mvt" not in df_sub.columns:
        if "MVT_ID" in df_sub.columns:
            df_sub = df_sub.rename(columns={"MVT_ID": "MVT_ID_mvt"})
        else:
            print(f"  ❌ FAIL: submission is missing the MVT_ID column")
            return False, "missing MVT_ID column"
    sub_ids = df_sub["MVT_ID_mvt"].astype(float).values

    n_match_set = (set(sub_ids) == set(expected_ids))
    n_match_order = (len(sub_ids) == len(expected_ids)) and np.all(sub_ids == expected_ids)
    n_diff = abs(len(sub_ids) - expected_n)

    print(f"  submission rows: {len(sub_ids)}, expected {expected_n}")
    print(f"  set matches: {n_match_set}")
    print(f"  order matches: {n_match_order}")

    pass_ = bool(n_match_set and n_match_order)
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, f"set={n_match_set}, order={n_match_order}, n={len(sub_ids)}"


def main():
    import argparse
    print("=" * 70)
    print("D0-4: one-click gate check scripts/gate_check.py")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw",
                        help="Data directory (defaults to the full data/raw; on cloud you can point it at data/sample)")
    args, _ = parser.parse_known_args()
    # Inject into check1_baseline (via globals)
    globals()["DATA_DIR"] = args.data_dir
    print(f"  data directory: {args.data_dir}/", flush=True)

    results = {}

    # check 1: baseline RMSE
    c1, rmse_global, df_ho_dep, y_pred, y_true, booster, feat_cols, cat_path = check1_baseline()
    results["check1_baseline_rmse"] = {
        "rmse": rmse_global,
        "expected": 357.14,
        "tolerance": 2.0,
        "pass": bool(c1),
    }

    # check 2: global SSE self-consistency
    c2, rmse_recomp, diff = check2_sse_global(rmse_global, df_ho_dep, y_pred, y_true)
    results["check2_sse_global_recompute"] = {
        "rmse_direct": rmse_global,
        "rmse_recomputed_from_ap_sse": rmse_recomp,
        "abs_diff": diff,
        "tolerance": 1e-6,
        "pass": bool(c2),
    }

    # check 3: cross-process inference allclose
    c3, max_diff, _ = check3_cross_process_inference(booster, df_ho_dep, feat_cols, cat_path)
    results["check3_cross_process_inference"] = {
        "max_abs_diff": max_diff,
        "tolerance": 0.0,
        "pass": bool(c3),
    }

    # check 4: no NaN, no negative values (check the latest submission)
    c4, n_nan, n_neg = check4_no_nan_no_negative()
    results["check4_no_nan_no_negative"] = {
        "nan_count": n_nan,
        "negative_count": n_neg,
        "pass": bool(c4),
    }

    # check 5: submission schema
    c5, info = check5_submission_format()
    results["check5_submission_format"] = {
        "info": info,
        "pass": bool(c5),
    }

    # Summary
    all_pass = all(r["pass"] for r in results.values())
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, r in results.items():
        marker = "✅" if r["pass"] else "❌"
        print(f"  {marker} {name}: pass={r['pass']}")
    print(f"\n  overall verdict: {'✅ ALL PASS' if all_pass else '❌ FAIL'}")

    # Save the report
    final_report = {
        "gate_check_results": results,
        "all_pass": bool(all_pass),
        "timestamp": pd.Timestamp.now().isoformat(),
        "note": "any experiment that fails this gate is rejected outright",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(final_report, indent=2, default=str))
    print(f"\n✅ report saved to {REPORT_PATH}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())