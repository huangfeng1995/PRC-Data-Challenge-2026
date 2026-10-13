"""
noclip 正式提交文件 Gate 校验 (只读)
==================================
范围校验分两类 (不能用统一的 30~10800):
  · 非修复行         : 30 ~ 10800
  · 63 个 T=14400 完整口径命中行: 预测 == MVT-SCHED (允许 >10800, max ~111,666)

T=14400 完整口径 = (ADEP==LIRF) AND (ADEP_flt 缺失) AND (MVT-SCHED > 14400)
修复行标记用 final_ranking 按 MVT_ID 显式对齐 (不靠行顺序)。

只读: 不改文件、不重训、不上传。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
SUB = ROOT / "submissions" / "jubilant-puzzle_final.parquet"
BACKUP = ROOT / "submissions" / "jubilant-puzzle_final_clip_backup.parquet"
SUBMIT_TMPL = ROOT / "data/raw/final_submitting.parquet"
FINAL = ROOT / "data/raw/final_ranking.parquet"
OUT = D0 / "gate_noclip.json"

LIRF_AP = "LIRF"
LIRF_T = 14400.0
CLIP_LO, CLIP_HI = 30.0, 10800.0
N_FIX_EXPECT = 63


def sec(s: pd.Series) -> np.ndarray:
    if s.dtype == "datetime64[us, UTC]":
        s = s.dt.tz_localize(None)
    v = s.astype("datetime64[ns]").to_numpy("int64") // 10**9
    v = v.astype(np.float64)
    v[v < 0] = np.nan
    return v


def main():
    checks, info = {}, {}

    print("=" * 78)
    print("[Gate] noclip 正式提交文件校验")
    print("=" * 78)
    print(f"  文件: {SUB.name}  ({SUB.stat().st_size:,} bytes)")

    sub = pd.read_parquet(SUB)
    checks["列名正确"] = list(sub.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    checks["行数=670790"] = len(sub) == 670790
    checks["int64"] = sub["TAXITIME_SEC_mvt"].dtype == np.int64
    checks["无NaN"] = bool(sub["TAXITIME_SEC_mvt"].notna().all())
    checks["MVT_ID无重复"] = bool(sub["MVT_ID_mvt"].is_unique)

    tmpl = pd.read_parquet(SUBMIT_TMPL)
    exp = tmpl["MVT_ID_mvt"].astype(float).values
    got = sub["MVT_ID_mvt"].astype(float).values
    checks["MVT_ID顺序一致"] = bool(np.all(exp == got))
    checks["MVT_ID集合一致"] = set(exp) == set(got)
    checks["与模板行数一致"] = len(sub) == len(tmpl)

    for k, v in checks.items():
        print(f"    {'✓' if v else '✗ FAIL'}  {k}")
    assert all(checks.values()), f"基础 Gate 失败: {checks}"

    # ---- 修复行标记 (final_ranking 按 MVT_ID 对齐) ----
    fn = pq.read_table(FINAL, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt",
                                      "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                      "LOBT_flt"]).to_pandas()
    f = fn[fn["PHASE_mvt"].eq("DEP")].copy()
    f["mvt"] = sec(f["MVT_TIME_UTC_mvt"])
    f["sch"] = sec(f["SCHED_TIME_UTC_mvt"])
    f["lobt"] = sec(f["LOBT_flt"])
    muC = f["mvt"] - f["sch"]
    afn = f["ADEP_flt"].isna() | (f["ADEP_flt"].astype(str).str.strip() == "")
    hit = (f["ADEP_mvt"].eq(LIRF_AP).to_numpy() & afn.to_numpy()
           & (muC.to_numpy() > LIRF_T))
    key = pd.Index(f["MVT_ID_mvt"].astype(float))
    hit_s = pd.Series(hit, index=key)
    fix_flag = hit_s.reindex(pd.Index(got), fill_value=False).to_numpy(bool)
    muC_s = pd.Series(muC.to_numpy(), index=key)
    fix_muC = muC_s.reindex(pd.Index(got)).to_numpy()   # 与提交文件行序对齐
    mv_s = pd.Series(f["mvt"].to_numpy(), index=key)
    mvt_r = mv_s.reindex(pd.Index(got)).to_numpy()
    lo_s = pd.Series(f["lobt"].to_numpy(), index=key)
    lobt_r = lo_s.reindex(pd.Index(got)).to_numpy()
    lobt_ok_r = ~np.isnan(lobt_r)
    mo = pd.to_datetime(pd.Series(mvt_r), unit="s").dt.month.to_numpy()
    del f, fn

    n_fix = int(fix_flag.sum())
    pred = sub["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    is_fix = fix_flag
    non_fix = ~fix_flag

    print("\n  [分类范围校验]")
    checks[f"修复行数={N_FIX_EXPECT}"] = n_fix == N_FIX_EXPECT
    print(f"    {'✓' if checks[f'修复行数={N_FIX_EXPECT}'] else '✗ FAIL'}  "
          f"修复行数 = {n_fix} (期望 {N_FIX_EXPECT})")

    # 修复行: 预测必须 == MVT-SCHED (完全相等, 无 clip)
    d = pred[is_fix] - fix_muC[is_fix]
    n_eq = int((np.abs(d) < 0.5).sum())
    fix_ok = (n_eq == n_fix)
    checks["修复行预测==MVT-SCHED(无clip)"] = fix_ok
    print(f"    {'✓' if fix_ok else '✗ FAIL'}  63 个修复行预测 == MVT-SCHED: "
          f"{n_eq}/{n_fix}  (差值 max|d|={np.abs(d).max():.3f})")
    n_fix_gt = int((pred[is_fix] > CLIP_HI).sum())
    n_fix_at = int(np.isclose(pred[is_fix], CLIP_HI).sum())
    checks["修复行无clip痕迹"] = n_fix_at == 0
    print(f"    {'✓' if checks['修复行无clip痕迹'] else '✗ FAIL'}  "
          f"修复行: >10800 有 {n_fix_gt} 行, ==10800 有 {n_fix_at} 行 (应为 0)")
    print(f"    修复行预测: min={pred[is_fix].min():.0f} "
          f"p50={np.median(pred[is_fix]):.0f} max={pred[is_fix].max():.0f}")

    # 非修复行: 30 ~ 10800; 例外是被 LOBT 窗口合法投影到 >10800 的行
    nf = pred[non_fix]
    ok_lo = bool((nf >= CLIP_LO).all())
    over = non_fix & (pred > CLIP_HI)
    n_over = int(over.sum())
    # 这些行必须落在其 LOBT 窗口内 (证明是窗口投影而非 clip 缺失/越界)
    ctr = mvt_r - lobt_r
    lo_w, hi_w = ctr - 3606.0, ctr + 3606.0
    in_win = (pred[over] >= lo_w[over] - 0.5) & (pred[over] <= hi_w[over] + 0.5)
    ok_over = bool(lobt_ok_r[over].all() and in_win.all())
    checks["非修复行范围[30,10800]"] = ok_lo and ok_over
    checks["非修复行>10800均来自LOBT窗口"] = ok_over
    print(f"    {'✓' if ok_lo else '✗ FAIL'}  非修复行 ({len(nf):,}) "
          f"下界 ≥30: {nf.min():.0f}")
    print(f"    {'✓' if ok_over else '✗ FAIL'}  非修复行 >10800 共 {n_over} 行, "
          f"全部落在其 LOBT 窗口 [lo,hi] 内 (窗口投影, 非越界)")
    if n_over:
        det = pd.DataFrame({"MVT_ID": got[over], "pred": pred[over],
                            "lo_w": lo_w[over], "hi_w": hi_w[over]})
        print("      明细:")
        for _, r_ in det.iterrows():
            print(f"        MVT_ID={int(r_.MVT_ID)}  pred={r_.pred:.0f}  "
                  f"窗口=[{r_.lo_w:.0f}, {r_.hi_w:.0f}]  在窗口内="
                  f"{r_.lo_w - 0.5 <= r_.pred <= r_.hi_w + 0.5}")
    print(f"    非修复行 max={nf.max():.0f}  "
          f"({int((nf<=CLIP_HI).sum()):,} 行在 [30,10800], {n_over} 行为窗口投影)")

    for k, v in checks.items():
        if k not in ("列名正确", "行数=670790", "int64", "无NaN", "MVT_ID无重复",
                     "MVT_ID顺序一致", "MVT_ID集合一致", "与模板行数一致"):
            continue
    assert all(checks.values()), f"Gate 失败: {[k for k,v in checks.items() if not v]}"

    # ---- 报告 ----
    print("\n" + "=" * 78)
    print("[报告]")
    print("=" * 78)
    print(f"  正式文件: {SUB.name}  ({SUB.stat().st_size:,} bytes)")
    print(f"  备份文件: {BACKUP.name}  ({BACKUP.stat().st_size:,} bytes)  "
          f"存在={BACKUP.exists()}")

    print(f"\n  分月 (行数 / 预测中位 / 预测max):")
    mon = {}
    for mm in [1, 2, 6, 7]:
        m = mo == mm
        v = pred[m]
        mon[f"mo{mm}"] = {"n": int(m.sum()), "median": round(float(np.median(v)), 1),
                          "max": round(float(v.max()), 1),
                          "n_gt_10800": int((v > CLIP_HI).sum())}
        print(f"    mo={mm}: n={int(m.sum()):>7,}  median={np.median(v):>8.0f}  "
              f"max={v.max():>9.0f}  >10800={int((v>CLIP_HI).sum()):>3}")

    q = lambda p: float(np.percentile(pred, p))
    print(f"\n  全局: min={pred.min():.0f}  p25={q(25):.0f}  median={q(50):.0f}  "
          f"p75={q(75):.0f}  p95={q(95):.0f}  max={pred.max():.0f}")
    print(f"  >1800={int((pred>1800).sum()):,}  >3600={int((pred>3600).sum()):,}  "
          f">10800={int((pred>CLIP_HI).sum()):,}")

    # 与 clip 备份对比
    b = pd.read_parquet(BACKUP)
    bmap = pd.Series(b["TAXITIME_SEC_mvt"].to_numpy(),
                     index=b["MVT_ID_mvt"].astype(float))
    bpred = bmap.reindex(pd.Index(got)).to_numpy()
    diff = np.abs(pred - bpred) > 0.5
    print(f"\n  [与 clip 备份差异] 不同行数={int(diff.sum()):,} "
          f"(应 = 63 修复行)  非修复行差异={int((diff & non_fix).sum())}")
    print(f"    clip 备份 max={bpred.max():.0f}  |  noclip max={pred.max():.0f}")

    res = {"file": str(SUB), "size_bytes": SUB.stat().st_size,
           "backup": str(BACKUP), "backup_size_bytes": BACKUP.stat().st_size,
           "all_checks_passed": bool(all(checks.values())),
           "checks": {k: bool(v) for k, v in checks.items()},
           "n_fixed_rows": n_fix, "n_fixed_gt_10800": n_fix_gt,
           "n_fixed_eq_10800": n_fix_at,
           "n_nonfix_gt_10800": n_over,
           "nonfix_over_10800_detail": [
               {"MVT_ID": int(got[i]), "pred": float(pred[i]),
                "lobt_window": [float(lo_w[i]), float(hi_w[i])],
                "in_window": bool(lo_w[i] - 0.5 <= pred[i] <= hi_w[i] + 0.5)}
               for i in np.where(over)[0]],
           "fixed_pred": {"min": float(pred[is_fix].min()),
                          "p50": float(np.median(pred[is_fix])),
                          "max": float(pred[is_fix].max())},
           "per_month": mon,
           "global": {"min": float(pred.min()), "p25": q(25), "median": q(50),
                      "p75": q(75), "p95": q(95), "max": float(pred.max()),
                      "n_gt_1800": int((pred > 1800).sum()),
                      "n_gt_3600": int((pred > 3600).sum()),
                      "n_gt_10800": int((pred > CLIP_HI).sum())},
           "vs_clip_backup": {"n_diff_rows": int(diff.sum()),
                              "n_nonfix_diff": int((diff & non_fix).sum()),
                              "clip_max": float(bpred.max())},
           "upload_status": "NOT uploaded (用户手动上传)"}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")
    print(f"\n✅ 全部 Gate 通过。未上传。")


if __name__ == "__main__":
    main()
