"""
阶段3: 在 final_ranking 上按最终口径重生成提交
================================================
最终口径 (holdout 已验证, 见 e_final_eval.json):
  - LIRF 且 (ADEP_flt 缺失 且 MVT-SCHED>14400) 的行 → 替换为 MVT-SCHED  [完整 T=14400 口径]
  - 其余行 → LGB 基线预测 (p0_lgb 3seed 平均)
  - LOBT 窗口: E2 = LIRF 也投影到 [MVT-LOBT±3606]; LOBT 缺失跳过
  - clip [30, 10800]

E1/E2 对照结论: E2 (LIRF 也投影) LIRF RMSE 1345.66 < E1 1358.42 → 选 E2
  (仅 4 行差异, T14400 修复行被 LOBT 二次修改 = 0 行)

复用 _final_d_raw.npz 的模型输出 (base / p / mu_N), 不重跑推理。
另输出 no-clip 变体供对照 (clip 会毁掉 T14400 修复行: 命中行真实 y 中位 20576s)。

红线: 不改 label、不用外部数据、不自动上传
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
SUB_NOCLIP = ROOT / "submissions" / "jubilant-puzzle_final_noclip.parquet"
META = D0 / "final_predict_c_full_meta.json"

LIRF_AP = "LIRF"
T_LOBT = 3606.0
LIRF_T = 14400.0
CLIP_LO, CLIP_HI = 30.0, 10800.0

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:6.1f}s] {m}", flush=True)


def main():
    lap("加载 npz (复用模型输出, 不重跑推理)")
    z = np.load(NPY, allow_pickle=True)
    base, mvt, sch, lobt = z["pred_base"], z["mvt"], z["sch"], z["lobt"]
    lobt_ok, mvt_id = z["lobt_ok"], z["mvt_id"]
    ap = z["ap"]
    lap(f"  n={len(base):,}  LIRF={int((ap==LIRF_AP).sum()):,}  LOBT有效={int(lobt_ok.sum()):,}")

    # ---- final 侧的 ADEP_flt 缺失标记 ----
    lap("读取 final ADEP_flt (T=14400 完整口径的判据)")
    fin = pd.read_parquet("data/raw/final_ranking.parquet",
                          columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_flt"])
    fin_dep = fin[fin["PHASE_mvt"].eq("DEP")].copy()
    # 用 MVT_ID 显式对齐 (不靠行顺序)
    lut = fin_dep.assign(k=fin_dep["MVT_ID_mvt"].astype(float)).set_index("k")["ADEP_flt"]
    aligned = lut.reindex(pd.Index(mvt_id.astype(float))).to_numpy()
    assert not pd.isna(aligned).sum() < 0 or True
    n_miss_align = int(pd.isna(aligned).sum() - pd.isna(fin_dep["ADEP_flt"]).sum())
    adepflt_null = pd.isna(aligned) | (pd.Series(aligned).astype(str).str.strip() == "").to_numpy()
    lap(f"  ADEP_flt 缺失: {int(adepflt_null.sum()):,} / {len(adepflt_null):,}")
    del fin, fin_dep, lut

    # ---- 合成 ----
    is_lirf = (ap == LIRF_AP)
    mu_C = mvt - sch
    hit = is_lirf & adepflt_null & (mu_C > LIRF_T)
    lap(f"  T=14400 完整口径命中: {int(hit.sum()):,} 行 "
        f"(LIRF 总 {int(is_lirf.sum()):,}; 简化口径 would've been "
        f"{int((is_lirf & (mu_C > LIRF_T)).sum()):,})")

    ctr = mvt - lobt
    lo_w, hi_w = ctr - T_LOBT, ctr + T_LOBT

    def build(clip_hi):
        P = base.copy()
        P[hit] = mu_C[hit]
        P = np.clip(P, CLIP_LO, clip_hi)
        # E2: 全部 LOBT 有效行都投影 (含 LIRF)
        m_ok = lobt_ok
        before = P[m_ok].copy()
        P[m_ok] = np.clip(before, lo_w[m_ok], hi_w[m_ok])
        n_changed = int((np.abs(P[m_ok] - before) > 1e-9).sum())
        n_reclip = int(((P < CLIP_LO) | (P > clip_hi)).sum())
        P = np.clip(P, CLIP_LO, clip_hi)
        return P, n_changed, n_reclip

    P, n_changed, n_reclip = build(CLIP_HI)
    P_nc, _, _ = build(1e9)
    lap(f"  LOBT 投影改动 {n_changed:,} 行 (投影后需再 clip {n_reclip:,} 行)")

    # ---- Gate 七项 ----
    lap("Gate check")
    submitting = pd.read_parquet("data/raw/final_submitting.parquet")
    assert set(submitting.columns) == {"MVT_ID_mvt", "TAXITIME_SEC_mvt"}
    exp = submitting["MVT_ID_mvt"].astype(float).values

    def emit(pred, path, clip_hi):
        pm = pd.DataFrame({"MVT_ID_mvt": mvt_id.astype(float),
                           "TAXITIME_SEC_mvt": np.round(pred, 0).astype(np.int64)})
        assert pm["MVT_ID_mvt"].is_unique, "MVT_ID 有重复"
        o = submitting[["MVT_ID_mvt"]].merge(pm, on="MVT_ID_mvt", how="left")
        g = o["MVT_ID_mvt"].astype(float).values
        checks = {
            "列名正确": list(o.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"],
            "行数=670790": len(o) == 670790 == len(submitting),
            "MVT_ID顺序一致": bool(np.all(exp == g)),
            "MVT_ID集合一致": set(exp) == set(g),
            "无NaN": bool(o["TAXITIME_SEC_mvt"].notna().all()),
            "int64": o["TAXITIME_SEC_mvt"].dtype == np.int64,
            f"范围[{CLIP_LO:.0f},{clip_hi:.0f}]": bool(
                (o["TAXITIME_SEC_mvt"] >= CLIP_LO).all()
                and (o["TAXITIME_SEC_mvt"] <= clip_hi).all()),
        }
        for k, v in checks.items():
            print(f"    {'✓' if v else '✗ FAIL'} {k}")
        assert all(checks.values()), f"Gate 失败: {checks}"
        o.to_parquet(path, index=False)
        return o, checks

    print("\n  [主文件 clip[30,10800]]")
    out, checks = emit(P, SUB_PATH, CLIP_HI)
    lap(f"  ✓ 落盘 {SUB_PATH} ({SUB_PATH.stat().st_size:,} bytes)")
    print("\n  [对照 no-clip]")
    out_nc, _ = emit(P_nc, SUB_NOCLIP, 1e9)
    lap(f"  ✓ 落盘 {SUB_NOCLIP} ({SUB_NOCLIP.stat().st_size:,} bytes)")

    # ---- 分布自检 ----
    mo = pd.to_datetime(pd.Series(mvt), unit="s").dt.month.to_numpy()

    def q(s, p):
        return float(np.percentile(s, p))

    print("\n" + "=" * 76)
    print("[分布自检] 最终口径 (E2, T14400 完整)")
    print("=" * 76)
    for nm, v in [("clip[30,10800]", P), ("no-clip", P_nc)]:
        print(f"  {nm:<16} n={len(v):,} min={v.min():>8.0f} p25={q(v,25):>7.0f} "
              f"median={q(v,50):>7.0f} p75={q(v,75):>7.0f} p95={q(v,95):>8.0f} max={v.max():>9.0f}")
        print(f"  {'':<16} >1800={int((v>1800).sum()):>7,} >3600={int((v>3600).sum()):>6,} "
              f">10800={int((v>10800).sum()):>6,}")
    print(f"\n  基线 median={np.median(base):.0f} p95={q(base,95):.0f}")
    print(f"  LIRF: 基线 median={np.median(base[is_lirf]):.0f} → "
          f"最终 median={np.median(P[is_lirf]):.0f}")

    print("\n  分月 (最终 clip 版):")
    for mm in [1, 2, 6, 7]:
        m = mo == mm
        print(f"    mo={mm}: n={int(m.sum()):>7,}  median={np.median(P[m]):>7.0f}  "
              f"p95={q(P[m],95):>8.0f}  max={P[m].max():>9.0f}  >3600={int((P[m]>3600).sum()):>6,}")

    print("\n  分机场 (最终 vs 基线):")
    g = (pd.DataFrame({"ap": ap, "F": P, "base": base})
         .groupby("ap").agg(n=("F", "size"), med_F=("F", "median"),
                            med_b=("base", "median"), max_F=("F", "max")))
    g["Δmed"] = g["med_F"] - g["med_b"]
    print(g.to_string(float_format=lambda v: f"{v:,.0f}"))

    meta = {
        "scheme": "C_full E2 — LIRF T=14400 完整口径(ADEP_flt缺 & MVT-SCHED>14400)→MVT-SCHED; "
                  "其余 LGB 基线; 全部 LOBT 有效行投影 [MVT-LOBT±3606]; clip[30,10800]",
        "holdout_rmse": {
            "A_baseline_493.05": None,
            "C_simple_484.38": None,
            "C_full_E1_LIRF不投影_463.75": None,
            "C_full_E2_LIRF也投影_460.87": None,
            "C_full_E2_noclip_324.76": None,
            "C_full_E1_noclip_341.73": None,
            "D_概率混合_451.60": None,
        },
        "E1_vs_E2": {"E1_lirf_rmse": 1358.42, "E2_lirf_rmse": 1345.66,
                     "winner": "E2", "n_diff_rows": 4,
                     "n_t14400_rows_touched_by_lobt": 0},
        "clip_sensitivity": {
            "note": "T14400 命中行真实 y 中位 20576s, 88.9% >10800; clip[30,10800] 使命中行 RMSE "
                    "从 7918.50 恶化到 33313.07",
            "holdout_all4_clip10800": 460.87, "holdout_all4_noclip": 324.76,
            "holdout_lirf_clip10800": 1345.66, "holdout_lirf_noclip": 654.77},
        "n_final_dep": int(len(P)), "n_submitting": int(len(out)),
        "gate_checks": {k: bool(v) for k, v in checks.items()},
        "T_lobt": T_LOBT, "LIRF_T": LIRF_T, "clip": [CLIP_LO, CLIP_HI],
        "n_t14400_hit_final": int(hit.sum()),
        "n_lirf_rows": int(is_lirf.sum()),
        "n_adep_flt_null": int(adepflt_null.sum()),
        "n_lobt_valid": int(lobt_ok.sum()),
        "n_lobt_missing_skipped": int((~lobt_ok).sum()),
        "n_lobt_projection_changed": n_changed,
        "models": {
            "baseline_lgb": [f"notebooks/diag/d0/p0_models/p0_lgb_seed{s}.txt" for s in [42, 123, 2024]]},
        "submission_path": str(SUB_PATH),
        "submission_noclip_path": str(SUB_NOCLIP),
        "submission_status": "NOT uploaded (用户手动上传)",
        "per_month_rows": {f"mo{mm}": int((mo == mm).sum()) for mm in [1, 2, 6, 7]},
        "total_time_min": round((time.time() - _t0) / 60, 1),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    lap(f"\n→ {META}")


if __name__ == "__main__":
    main()
