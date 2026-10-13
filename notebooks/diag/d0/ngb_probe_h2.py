"""
NGBoost σ 信号 h2 泛化验证 (只预测, 不重训)
==========================================
- 加载已落盘 _ngb_reduced.pkl (30 万行训练的那版)
- holdout2 (mo ∈ {2,6}) 全量预测, 得 loc / scale
- 报: σ vs |残差| Spearman (>0.3 判定), σ 在 >1800 / <600 档中位与比值,
  σ max 与 >1e4 行数, loc full RMSE (基线 h2=398.40)
- 硬断言: loc/scale 非有限, scale<=0 才失败; σ max / 1e4 只记录

红线: 不重训, 不写 submissions/, 不上传, 不改 label
"""
from __future__ import annotations
import sys, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*ngboost.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="ngboost")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngb_probe_reduced import (  # noqa: E402
    D0, NGB_PKL, load_cache, spearman, rmse, per_band_report,
)

OUT = D0 / "ngb_probe_h2.json"
HOLD2 = [2, 6]
CLIP_LO, CLIP_HI = 30.0, 10800.0
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def main():
    lap("加载特征缓存 (与 h1 完全相同的 FEATS71 管线)")
    X_num, y, mo, mvt_s, adep = load_cache()

    i_h2 = np.where(np.isin(mo, HOLD2))[0]
    lap(f"holdout2 (Feb+Jun) {len(i_h2):,}")

    lap(f"加载模型 {NGB_PKL.name}")
    m = pickle.load(open(NGB_PKL, "rb"))
    best_iter = getattr(m, "best_val_loss_itr", None)

    lap("预测 holdout2 ...")
    t_pred = time.time()
    D = m.pred_dist(X_num.iloc[i_h2])
    loc = D.loc
    scale = D.scale
    pred_s = time.time() - t_pred
    lap(f"  预测耗时 {pred_s:.1f}s")

    # 硬断言
    assert np.isfinite(loc).all(), "loc 有非有限值"
    assert np.isfinite(scale).all(), "scale 有非有限值"
    assert (scale > 0).all(), "scale 有非正值"

    y_h2 = y[i_h2]
    P = np.clip(loc, CLIP_LO, CLIP_HI)
    resid = np.abs(P - y_h2)

    rho, pv = spearman(scale, resid)

    m_gt1800 = y_h2 > 1800
    m_lt600 = y_h2 < 600
    sig_gt = float(np.median(scale[m_gt1800])) if m_gt1800.any() else None
    sig_lt = float(np.median(scale[m_lt600])) if m_lt600.any() else None
    sig_ratio = (sig_gt / sig_lt) if (sig_gt and sig_lt) else None

    sigma_max = float(scale.max())
    n_sigma_gt1e4 = int((scale > 1e4).sum())

    rmse_full = rmse(P, y_h2)
    rmse_core = rmse(P, y_h2, (y_h2 > 0) & (y_h2 <= 10800))
    rmse_short = rmse(P, y_h2, (y_h2 > 0) & (y_h2 <= 1800))
    rmse_long = rmse(P, y_h2, (y_h2 > 1800) & (y_h2 <= 10800))
    band = per_band_report(P, y_h2)

    result = {
        "task": "ngb_probe_h2_generalization",
        "model_path": str(NGB_PKL),
        "model_reused": True,
        "best_iter": int(best_iter) if best_iter is not None else None,
        "n_holdout2": int(len(i_h2)),
        "loc_RMSE_full": round(rmse_full, 2),
        "loc_RMSE_core": round(rmse_core, 2),
        "loc_RMSE_short": round(rmse_short, 2),
        "loc_RMSE_long": round(rmse_long, 2),
        "sigma_vs_absresid_spearman": rho,
        "sigma_vs_absresid_p": pv,
        "sigma_median_gt1800": round(sig_gt, 3) if sig_gt is not None else None,
        "sigma_median_lt600": round(sig_lt, 3) if sig_lt is not None else None,
        "sigma_ratio_gt1800_over_lt600": round(sig_ratio, 3) if sig_ratio is not None else None,
        "sigma_max": round(sigma_max, 1),
        "n_sigma_gt_1e4": n_sigma_gt1e4,
        "sigma_median": round(float(np.median(scale)), 3),
        "loc_median": round(float(np.median(loc)), 1),
        "predict_seconds": round(pred_s, 1),
        "band_report": band,
        "baseline_lgb_full_h2_RMSE": 398.40,
        "signal_strong": bool(rho > 0.3),
        "ref_h1": {
            "sigma_vs_absresid_spearman": 0.4110,
            "sigma_ratio_gt1800_over_lt600": 2.290,
            "loc_RMSE_full": 603.41,
            "baseline_lgb_full_h1_RMSE": 568.49,
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("[NGBoost σ 信号 h2 泛化验证] holdout2 (Feb+Jun) — 模型复用, 未重训")
    print("=" * 70)
    print(f"  n_holdout2 = {len(i_h2):,}   预测耗时 {pred_s:.1f}s   best_iter={best_iter}")
    print()
    print(f"  --- 主指标: loc μ ---")
    print(f"  full RMSE = {rmse_full:.2f}  (LGB 基线 h2=398.40, {rmse_full-398.40:+.2f})")
    print(f"  core RMSE = {rmse_core:.2f}")
    print(f"  short RMSE = {rmse_short:.2f}")
    print(f"  long RMSE = {rmse_long:.2f}")
    print(f"  分档 loc RMSE: >1800段={band['rmse_gt1800']}, <600段={band['rmse_lt600']}")
    print()
    print(f"  --- σ 信号 (主目标) ---")
    mark = "✓ >0.3 信号强" if rho > 0.3 else "✗ ≤0.3 信号弱"
    print(f"  σ vs |残差| Spearman: {rho:+.4f}  (p={pv:.2e})  {mark}")
    print(f"  σ 中位数: >1800 段 {sig_gt:.3f}s vs <600 段 {sig_lt:.3f}s  → 比值 {sig_ratio:.2f}x")
    print(f"  σ max = {sigma_max:.1f}  (>1e4 行数: {n_sigma_gt1e4:,})")
    print(f"  σ 中位数 = {float(np.median(scale)):.3f}   loc 中位数 = {float(np.median(loc)):.1f}")
    print()
    print(f"  → {OUT}")
    print(f"  总耗时: {(time.time() - _t0)/60:.1f} min")


if __name__ == "__main__":
    main()
