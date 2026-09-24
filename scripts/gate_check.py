"""
D0-4: 一键门禁 scripts/gate_check.py
=====================================
import 冻结入口, 自动检查并输出 PASS/FAIL (失败返回非零退出码), 写 d0_gate_report.json:
1. baseline RMSE ≈ 357.14 (±2)
2. 分机场 SSE 加总反算全局 RMSE 与直接全局 RMSE allclose (容差 1e-6)
3. cat 映射持久化后, 独立进程推理逐行 allclose, max|Δ|=0
4. 预测无 NaN, 无负值
5. 提交 MVT_ID_mvt 集合与顺序 == submitting.parquet (344,841 行), 无缺行/多行

用法:
    python3 scripts/gate_check.py
退出码: 0=PASS, 1=FAIL
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

# 加载冻结入口
spec = importlib.util.spec_from_file_location(
    "ab_module",
    ROOT / "notebooks/diag/holdout_v1_queue_ab.py",
)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

import src.train_xgb as tx

REPORT_PATH = ROOT / "notebooks/diag/d0/d0_gate_report.json"


def check1_baseline(tolerance=2.0):
    """check 1: baseline RMSE ≈ 357.14 (±2)"""
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
    print(f"  RMSE = {rmse:.4f} (期望 357.14 ± {tolerance})")
    pass_ = abs(rmse - 357.14) < tolerance
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, rmse, df_ho_dep, y_pred, y_true, booster, feat_cols, cat_path


def check2_sse_global(rmse_global, df_ho_dep, y_pred, y_true):
    """check 2: 分机场 SSE 加总反算全局 RMSE vs 直接 RMSE allclose"""
    print("\n" + "=" * 70)
    print("CHECK 2: 分机场 SSE 加总反算全局 RMSE allclose (1e-6)")
    print("=" * 70)
    ap = df_ho_dep["ADEP_mvt"].astype(str).values
    sse_total = 0.0
    for a in np.unique(ap):
        m = ap == a
        sse_a = float(np.sum((y_true[m] - y_pred[m]) ** 2))
        sse_total += sse_a
    rmse_recomputed = float(np.sqrt(sse_total / len(y_true)))
    diff = abs(rmse_recomputed - rmse_global)
    print(f"  直接 RMSE = {rmse_global:.6f}")
    print(f"  反算 RMSE = {rmse_recomputed:.6f}")
    print(f"  |Δ| = {diff:.2e}")
    pass_ = diff < 1e-6
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, rmse_recomputed, diff


def check3_cross_process_inference(booster, df_ho_dep, feat_cols, cat_path):
    """check 3: cat 持久化后, 独立进程推理逐行 allclose (max|Δ|=0)"""
    print("\n" + "=" * 70)
    print("CHECK 3: 独立进程推理 allclose (max|Δ|=0)")
    print("=" * 70)

    # 先保存模型给独立进程加载
    model_path = ROOT / "models" / "gate_check_v1_model.json"
    booster.save_model(str(model_path))

    # 主进程推理
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = cat_path
    X_ho_enc = tx._encode_categoricals(df_ho_dep[feat_cols].copy(), fit=False)
    y_pred_main = booster.predict(xgb.DMatrix(X_ho_enc))

    # 保存 main 预测
    tmp_main = Path(tempfile.gettempdir()) / "gate_main_pred.npy"
    np.save(tmp_main, y_pred_main)

    # 生成独立推理脚本
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

# 数据目录(默认 data/raw, 云端用 data/sample 时通过 DATA_DIR 注入)
DATA_DIR = "{globals().get("DATA_DIR", "data/raw")}"

# 加载 holdout (1+7) DEP
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

# 用持久化的 cat mappings
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS_PATH = Path("{cat_path}")

X_ho_enc = tx._encode_categoricals(df_ho_dep[feat_cols].copy(), fit=False)
booster = xgb.Booster()
booster.load_model("{model_path}")
y_pred = booster.predict(xgb.DMatrix(X_ho_enc))
np.save("{tmp_main.with_suffix('.indep.npy')}", y_pred)
print(f"独立推理完成, n={{len(y_pred)}}")
'''

    # 保存模型 + cat mappings 让独立进程可访问
    model_path_local = ROOT / "models" / "gate_check_v1_model.json"
    if not model_path_local.exists():
        booster.save_model(str(model_path_local))

    # 写独立脚本
    script_path = Path(tempfile.gettempdir()) / "gate_indep_infer.py"
    script_path.write_text(indep_script)

    # 跑独立进程
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    print(f"  独立进程 stdout: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")

    # 加载独立进程结果
    indep_path = tmp_main.with_suffix(".indep.npy")
    if not indep_path.exists():
        print(f"  ❌ FAIL: 独立进程未生成结果文件")
        return False, None, None
    y_pred_indep = np.load(indep_path)

    max_abs_diff = float(np.max(np.abs(y_pred_main - y_pred_indep)))
    print(f"  max|Δ| = {max_abs_diff:.6e}")
    pass_ = max_abs_diff == 0.0
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, max_abs_diff, y_pred_indep


def check4_no_nan_no_negative(y_pred_clipped=None):
    """check 4: 预测无 NaN, 无负值
    检查最新提交文件 (应是最终可提交版本, 已 clip)
    """
    print("\n" + "=" * 70)
    print("CHECK 4: 提交预测无 NaN, 无负值")
    print("=" * 70)

    sub_dir = ROOT / "submissions"
    sub_files = sorted([p for p in sub_dir.glob("*.parquet") if p.is_file()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not sub_files:
        print(f"  ⚠️ 无 submissions/*.parquet 可检查")
        return True, 0, 0
    sub_latest = sub_files[0]
    df_sub = pd.read_parquet(sub_latest)
    pred_col = [c for c in df_sub.columns if c != "MVT_ID_mvt"][0]
    preds = df_sub[pred_col].astype(float).values
    n_nan = int(np.isnan(preds).sum())
    n_neg = int((preds < 0).sum())
    print(f"  文件: {sub_latest.name}")
    print(f"  NaN 数 = {n_nan}")
    print(f"  负值数 = {n_neg}")
    pass_ = (n_nan == 0) and (n_neg == 0)
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, n_nan, n_neg


def check5_submission_format():
    """check 5: 提交 MVT_ID_mvt 集合与顺序 == submitting.parquet (344,841 行)"""
    print("\n" + "=" * 70)
    print("CHECK 5: 提交 MVT_ID 集合+顺序 == submitting.parquet")
    print("=" * 70)

    sub_path = ROOT / globals().get("DATA_DIR", "data/raw") / "submitting.parquet"
    sub = pd.read_parquet(sub_path)
    expected_ids = sub["MVT_ID_mvt"].astype(float).values
    expected_n = len(expected_ids)
    print(f"  submitting.parquet: n={expected_n}, cols={list(sub.columns)}")

    # 找最新的提交 (v1 / v10 / e3probe) 用于检查
    sub_dir = ROOT / "submissions"
    sub_files = sorted([p for p in sub_dir.glob("*.parquet") if p.is_file()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not sub_files:
        print(f"  ⚠️ 无 submissions/*.parquet 可检查")
        return True, "无提交文件可检查 (仅 schema 检查)"

    sub_latest = sub_files[0]
    print(f"  检查最新提交: {sub_latest.name}")
    df_sub = pd.read_parquet(sub_latest)
    if "MVT_ID_mvt" not in df_sub.columns:
        if "MVT_ID" in df_sub.columns:
            df_sub = df_sub.rename(columns={"MVT_ID": "MVT_ID_mvt"})
        else:
            print(f"  ❌ FAIL: 提交缺 MVT_ID 列")
            return False, "缺 MVT_ID 列"
    sub_ids = df_sub["MVT_ID_mvt"].astype(float).values

    n_match_set = (set(sub_ids) == set(expected_ids))
    n_match_order = (len(sub_ids) == len(expected_ids)) and np.all(sub_ids == expected_ids)
    n_diff = abs(len(sub_ids) - expected_n)

    print(f"  提交行数: {len(sub_ids)}, 期望 {expected_n}")
    print(f"  集合一致: {n_match_set}")
    print(f"  顺序一致: {n_match_order}")

    pass_ = bool(n_match_set and n_match_order)
    print(f"  {'✅ PASS' if pass_ else '❌ FAIL'}")
    return pass_, f"set={n_match_set}, order={n_match_order}, n={len(sub_ids)}"


def main():
    import argparse
    print("=" * 70)
    print("D0-4: 一键门禁 scripts/gate_check.py")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw",
                        help="数据目录 (默认 data/raw 全量, 云端可指定 data/sample)")
    args, _ = parser.parse_known_args()
    # 注入到 check1_baseline (通过 globals)
    globals()["DATA_DIR"] = args.data_dir
    print(f"  数据目录: {args.data_dir}/", flush=True)

    results = {}

    # check 1: baseline RMSE
    c1, rmse_global, df_ho_dep, y_pred, y_true, booster, feat_cols, cat_path = check1_baseline()
    results["check1_baseline_rmse"] = {
        "rmse": rmse_global,
        "expected": 357.14,
        "tolerance": 2.0,
        "pass": bool(c1),
    }

    # check 2: SSE 全局自洽
    c2, rmse_recomp, diff = check2_sse_global(rmse_global, df_ho_dep, y_pred, y_true)
    results["check2_sse_global_recompute"] = {
        "rmse_direct": rmse_global,
        "rmse_recomputed_from_ap_sse": rmse_recomp,
        "abs_diff": diff,
        "tolerance": 1e-6,
        "pass": bool(c2),
    }

    # check 3: 跨进程推理 allclose
    c3, max_diff, _ = check3_cross_process_inference(booster, df_ho_dep, feat_cols, cat_path)
    results["check3_cross_process_inference"] = {
        "max_abs_diff": max_diff,
        "tolerance": 0.0,
        "pass": bool(c3),
    }

    # check 4: 无 NaN, 无负值 (检查最新提交)
    c4, n_nan, n_neg = check4_no_nan_no_negative()
    results["check4_no_nan_no_negative"] = {
        "nan_count": n_nan,
        "negative_count": n_neg,
        "pass": bool(c4),
    }

    # check 5: 提交 schema
    c5, info = check5_submission_format()
    results["check5_submission_format"] = {
        "info": info,
        "pass": bool(c5),
    }

    # 汇总
    all_pass = all(r["pass"] for r in results.values())
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    for name, r in results.items():
        marker = "✅" if r["pass"] else "❌"
        print(f"  {marker} {name}: pass={r['pass']}")
    print(f"\n  总判定: {'✅ ALL PASS' if all_pass else '❌ FAIL'}")

    # 保存 report
    final_report = {
        "gate_check_results": results,
        "all_pass": bool(all_pass),
        "timestamp": pd.Timestamp.now().isoformat(),
        "note": "任何实验没过此门禁直接拒收",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(final_report, indent=2, default=str))
    print(f"\n✅ 报告已保存到 {REPORT_PATH}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
