"""
阶段2 收尾: 复用 _final_d_raw.npz 完成 D 口径合成 + 落盘
=========================================================
final_predict_d.py 推理已完成并落盘 npz, 但在落盘 parquet 前因
"LOBT 投影把值推出 [30,10800]" 断言失败。修复: 投影后再 clip 一次。
本脚本复用 npz, 不重跑特征重建与模型推理。

红线同 final_predict_d.py: 不改 label、不用外部数据、不写死机场、不自动上传
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
NPY = D0 / "_final_d_raw.npz"
SUB_PATH = ROOT / "submissions" / "jubilant-puzzle_final.parquet"
META = D0 / "final_predict_meta.json"

LIRF_AP = "LIRF"
T_LOBT = 3606.0
CLIP_LO, CLIP_HI = 30.0, 10800.0

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:6.1f}s] {m}", flush=True)


def main():
    lap("加载 npz (复用已算好的模型输出)")
    z = np.load(NPY, allow_pickle=True)
    base, p, mu_N = z["pred_base"], z["p"], z["mu_N"]
    mvt, sch, lobt = z["mvt"], z["sch"], z["lobt"]
    lobt_ok, is_lirf = z["lobt_ok"], z["is_lirf"]
    mvt_id, ap = z["mvt_id"], z["ap"]
    lap(f"  n={len(base):,}  LIRF={int(is_lirf.sum()):,}  LOBT有效={int(lobt_ok.sum()):,}")

    mu_C = mvt - sch
    ctr = mvt - lobt
    lo_w, hi_w = ctr - T_LOBT, ctr + T_LOBT
    sched_out = lobt_ok & (np.abs(sch - lobt) > T_LOBT)
    p_adj = p.copy()
    p_adj[sched_out] = 0.0
    mix = np.maximum(p_adj * mu_C + (1 - p_adj) * mu_N, 0.0)

    Dp = np.where(is_lirf, mix, base)
    Dp = np.clip(Dp, CLIP_LO, CLIP_HI)
    before = Dp[lobt_ok].copy()
    Dp[lobt_ok] = np.clip(before, lo_w[lobt_ok], hi_w[lobt_ok])
    n_proj_changed = int((np.abs(Dp[lobt_ok] - before) > 1e-9).sum())
    n_proj_reclip = int(((Dp < CLIP_LO) | (Dp > CLIP_HI)).sum())
    Dp = np.clip(Dp, CLIP_LO, CLIP_HI)
    lap(f"  LOBT 投影改动 {n_proj_changed:,} 行 (投影后需再 clip {n_proj_reclip:,} 行)")
    lap(f"  SCHED 落窗外 p=0: {int(sched_out.sum()):,}")
    assert np.isfinite(Dp).all() and (Dp >= CLIP_LO).all() and (Dp <= CLIP_HI).all()

    # ---- 对齐 final_submitting ----
    lap("对齐 final_submitting.parquet")
    submitting = pd.read_parquet("data/raw/final_submitting.parquet")
    assert set(submitting.columns) == {"MVT_ID_mvt", "TAXITIME_SEC_mvt"}
    pred_map = pd.DataFrame({"MVT_ID_mvt": mvt_id.astype(float),
                             "TAXITIME_SEC_mvt": np.round(Dp, 0).astype(np.int64)})
    assert pred_map["MVT_ID_mvt"].is_unique, "final DEP MVT_ID 有重复"
    out = submitting[["MVT_ID_mvt"]].merge(pred_map, on="MVT_ID_mvt", how="left")
    assert len(out) == len(submitting), f"行数 {len(out)} vs {len(submitting)}"
    assert out["TAXITIME_SEC_mvt"].notna().all(), f"NaN {int(out['TAXITIME_SEC_mvt'].isna().sum())}"
    exp = submitting["MVT_ID_mvt"].astype(float).values
    got = out["MVT_ID_mvt"].astype(float).values
    assert np.all(exp == got), "MVT_ID 顺序不匹配"
    assert set(exp) == set(got), "MVT_ID 集合不匹配"
    lap(f"  ✓ {len(out):,} 行 / MVT_ID 集合+顺序完全对齐 / 无 NaN / 范围合法")

    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SUB_PATH, index=False)
    lap(f"  ✓ 落盘 {SUB_PATH} ({SUB_PATH.stat().st_size:,} bytes)")

    # ---- 分布自检 ----
    mo = pd.to_datetime(pd.Series(mvt), unit="s").dt.month.to_numpy()

    def q(s, p):
        return float(np.percentile(s, p))

    print("\n" + "=" * 72)
    print("[分布自检] D 方案 final 预测 (2026 Jan/Feb/Jun/Jul)")
    print("=" * 72)
    print(f"  n={len(Dp):,}   min={Dp.min():.0f}   p25={q(Dp,25):.0f}   "
          f"median={q(Dp,50):.0f}   p75={q(Dp,75):.0f}   p95={q(Dp,95):.0f}   max={Dp.max():.0f}")
    print(f"  >1800: {int((Dp>1800).sum()):,}   >3600: {int((Dp>3600).sum()):,}   "
          f">10800: {int((Dp>10800).sum()):,}   ==30: {int((Dp==30).sum()):,}")
    print(f"\n  基线 median={np.median(base):.0f} p95={q(base,95):.0f}  |  "
          f"D median={np.median(Dp):.0f} p95={q(Dp,95):.0f}")
    print(f"  Δmedian={np.median(Dp)-np.median(base):+.0f}  Δp95={q(Dp,95)-q(base,95):+.0f}")

    print("\n  分月:")
    for mm in [1, 2, 6, 7]:
        m = mo == mm
        print(f"    mo={mm}: n={int(m.sum()):>7,}  median={np.median(Dp[m]):>7.0f}  "
              f"p95={q(Dp[m],95):>8.0f}  max={Dp[m].max():>8.0f}  >3600={int((Dp[m]>3600).sum()):>6,}")

    print("\n  分机场 (D vs 基线):")
    g = (pd.DataFrame({"ap": ap, "D": Dp, "base": base})
         .groupby("ap").agg(n=("D", "size"), med_D=("D", "median"),
                            med_b=("base", "median"), max_D=("D", "max")))
    g["Δmed"] = g["med_D"] - g["med_b"]
    print(g.to_string(float_format=lambda v: f"{v:,.0f}"))

    meta = {
        "scheme": "D (LIRF 概率混合 + 全局 LOBT 窗口 + clip[30,10800])",
        "stage1_gate": {"D_not_worse_per_airport": "10/10 PASS",
                        "delta_vs_C": 16.52,
                        "note": "C(v22 T14400) 在 LIRF 上优于 D 77.12s; D 按用户指定口径落盘"},
        "n_final_dep": int(len(Dp)), "n_submitting": int(len(out)),
        "T_lobt": T_LOBT, "clip": [CLIP_LO, CLIP_HI],
        "models": {
            "baseline_lgb": [f"notebooks/diag/d0/p0_models/p0_lgb_seed{s}.txt" for s in [42, 123, 2024]],
            "copy_classifier": "notebooks/diag/d0/cp_mixture_models/clf.pkl",
            "normal_regressor": "notebooks/diag/d0/cp_mixture_models/reg.pkl"},
        "lut_source": "2025 fit months 2-12 only (no 2026 info)",
        "n_lobt_valid": int(lobt_ok.sum()),
        "n_lobt_missing_skipped": int((~lobt_ok).sum()),
        "n_lobt_projection_changed": n_proj_changed,
        "n_projection_reclip": n_proj_reclip,
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
        "submission_status": "NOT uploaded (用户手动上传)",
        "total_time_min": round((time.time() - _t0) / 60, 1),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    lap(f"\n→ {META}")


if __name__ == "__main__":
    main()
