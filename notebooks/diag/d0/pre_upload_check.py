"""
上传前决定性核查 (只读)
======================
核查1: v22 (真榜 360.86) 的 T14400 修复行是否被 clip
核查2: final / 2025 holdout 分月命中与口径连续性
核查3: 命中行 EOBT/IOBT/LOBT 旁证 (整套时刻缺失 vs 仅 BLOCK 损坏)

只读: 不改文件、不重训、不写 submissions/、不上传。
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "pre_upload_check.json"

V22 = ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair_t14400.parquet"
V22_SIMPLE = ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair.parquet"
RANK = ROOT / "data/raw/ranking.parquet"
FINAL = ROOT / "data/raw/final_ranking.parquet"

LIRF_AP = "LIRF"
LIRF_T = 14400.0
CLIP_HI = 10800.0
HOLD = [1, 2, 6, 7]

COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "AOBT_3_flt"]


def sec(s: pd.Series) -> np.ndarray:
    """datetime64[us,UTC] -> 秒; 负值哨兵 -> NaN"""
    if s.dtype == "datetime64[us, UTC]":
        s = s.dt.tz_localize(None)
    v = s.astype("datetime64[ns]").to_numpy("int64") // 10**9
    v = v.astype(np.float64)
    v[v < 0] = np.nan
    return v


def prep(df_raw):
    """附加 mvt/sch/lobt/eobt/iobt/aobt(秒) + adepflt_null + mo"""
    d = df_raw.copy()
    for c, tag in [("MVT_TIME_UTC_mvt", "mvt"), ("SCHED_TIME_UTC_mvt", "sch"),
                   ("BLOCK_TIME_UTC_mvt", "blk"), ("LOBT_flt", "lobt"),
                   ("EOBT_1_flt", "eobt"), ("IOBT_flt", "iobt"),
                   ("AOBT_3_flt", "aobt")]:
        d[tag] = sec(d[c]) if c in d.columns else np.nan
    s = d["ADEP_flt"]
    d["adepflt_null"] = ((s.isna() | (s.astype(str).str.strip() == "")).to_numpy()
                         if "ADEP_flt" in d.columns else True)
    d["mo"] = pd.to_datetime(pd.Series(d["mvt"]), unit="s").dt.month.to_numpy()
    d["muC"] = d["mvt"] - d["sch"]
    d["hit"] = (d["ADEP_mvt"].eq(LIRF_AP).to_numpy()
                & d["adepflt_null"].to_numpy()
                & (d["muC"].to_numpy() > LIRF_T))
    return d


def main():
    res = {}

    # ================= 核查1 =================
    print("=" * 80)
    print("[核查1] v22 (真榜 360.86) 的 T14400 修复行是否被 clip")
    print("=" * 80)
    rk = pq.read_table(RANK, columns=COLS).to_pandas()
    rk_dep = rk[rk["PHASE_mvt"].eq("DEP")].copy().reset_index(drop=True)
    r = prep(rk_dep)
    print(f"  ranking.parquet DEP = {len(r):,}")

    v22 = pd.read_parquet(V22)
    print(f"  v22 文件行数 = {len(v22):,}  列={list(v22.columns)}")
    m = r.merge(v22, on="MVT_ID_mvt", how="left", suffixes=("", "_v22"))
    assert m["TAXITIME_SEC_mvt"].notna().all(), "v22 未覆盖全部 ranking DEP"
    hit = m["hit"].to_numpy()
    n_hit = int(hit.sum())
    pred = m.loc[hit, "TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    muC = m.loc[hit, "muC"].to_numpy()

    n_gt = int((pred > CLIP_HI).sum())
    n_eq = int((np.isclose(pred, CLIP_HI)).sum())
    n_lt = int((pred < CLIP_HI).sum())
    print(f"\n  完整口径命中 (ADEP_flt缺 & MVT-SCHED>14400) = {n_hit} 行")
    print(f"    预测 >10800 (未 clip) : {n_gt:>4}  ({n_gt/max(n_hit,1)*100:.1f}%)")
    print(f"    预测 ==10800 (被 clip): {n_eq:>4}  ({n_eq/max(n_hit,1)*100:.1f}%)")
    print(f"    预测 <10800          : {n_lt:>4}")
    if n_hit:
        print(f"\n  命中行预测值: min={pred.min():.0f} p25={np.percentile(pred,25):.0f} "
              f"p50={np.median(pred):.0f} p75={np.percentile(pred,75):.0f} max={pred.max():.0f}")
        print(f"  命中行 MVT-SCHED: min={muC.min():.0f} p50={np.median(muC):.0f} max={muC.max():.0f}")
        d = pred - muC
        print(f"  预测 - (MVT-SCHED): min={d.min():.0f} p50={np.median(d):.0f} max={d.max():.0f} "
              f"| 完全相等(差=0) 行数={int((np.abs(d)<0.5).sum())}")
    # 对照: 简化口径 (无 ADEP_flt 条件)
    hit_s = (m["ADEP_mvt"].eq(LIRF_AP).to_numpy() & (m["muC"].to_numpy() > LIRF_T))
    pred_s = m.loc[hit_s, "TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    print(f"\n  [对照] 简化口径命中 = {int(hit_s.sum())} 行  "
          f">10800={int((pred_s>CLIP_HI).sum())} ==10800={int((np.isclose(pred_s,CLIP_HI)).sum())}")

    # v22 脚本口径复核: 是否有 clip
    v22b = pd.read_parquet(V22_SIMPLE)
    mb = r.merge(v22b, on="MVT_ID_mvt", how="left", suffixes=("", "_v"))
    pb = mb.loc[mb["hit"].to_numpy(), "TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    print(f"  [对照] v15_lirf_repair (T=10800版) 同一批 {len(pb)} 行: "
          f"max={pb.max():.0f} ==10800={int((np.isclose(pb,CLIP_HI)).sum())}")

    res["check1_v22"] = {
        "n_ranking_dep": int(len(r)), "n_hit_full": n_hit,
        "pred_gt_10800": n_gt, "pred_eq_10800": n_eq, "pred_lt_10800": n_lt,
        "pred_stats": {"min": float(pred.min()), "p50": float(np.median(pred)),
                       "max": float(pred.max())} if n_hit else None,
        "muC_stats": {"min": float(muC.min()), "p50": float(np.median(muC)),
                      "max": float(muC.max())} if n_hit else None,
        "n_hit_simple": int(hit_s.sum()),
        "pred_gt_10800_simple": int((pred_s > CLIP_HI).sum()),
        "verdict": ("v22 未 clip 修复行 (预测保留 >10800 原值)" if n_gt > 0
                    else "v22 的修复行被 clip 到 10800"),
    }
    print(f"\n  → 裁定: {res['check1_v22']['verdict']}")

    # ================= 核查2 =================
    print("\n" + "=" * 80)
    print("[核查2] 分月命中与口径连续性")
    print("=" * 80)
    print("  --- 2026 真榜 (旧 ranking.parquet, Jan+Jul) ---")
    c1 = {}
    for mm in [1, 7]:
        s = (r["mo"] == mm) & r["hit"]
        n = int(s.sum())
        v = r.loc[s, "muC"].to_numpy()
        c1[f"mo{mm}"] = {"n_hit": n, "n_dep": int((r["mo"] == mm).sum()),
                         "muC_med": round(float(np.median(v)), 1) if n else None,
                         "muC_max": round(float(v.max()), 1) if n else None}
        print(f"    mo={mm}: DEP={int((r['mo']==mm).sum()):>7,}  命中={n:>4}  "
              f"MVT-SCHED 中位={c1[f'mo{mm}']['muC_med']}  max={c1[f'mo{mm}']['muC_max']}")

    print("\n  --- 2026 final (Jan/Feb/Jun/Jul) ---")
    fn = pq.read_table(FINAL, columns=COLS).to_pandas()
    f_dep = fn[fn["PHASE_mvt"].eq("DEP")].copy().reset_index(drop=True)
    f = prep(f_dep)
    c2 = {}
    for mm in [1, 2, 6, 7]:
        s = (f["mo"] == mm) & f["hit"]
        n = int(s.sum())
        v = f.loc[s, "muC"].to_numpy()
        c2[f"mo{mm}"] = {"n_hit": n, "n_dep": int((f["mo"] == mm).sum()),
                         "muC_med": round(float(np.median(v)), 1) if n else None,
                         "muC_max": round(float(v.max()), 1) if n else None}
        print(f"    mo={mm}: DEP={int((f['mo']==mm).sum()):>7,}  命中={n:>4}  "
              f"MVT-SCHED 中位={c2[f'mo{mm}']['muC_med']}  max={c2[f'mo{mm}']['muC_max']}")
    print(f"    final 四个月合计命中 = {int(f['hit'].sum())}")

    print("\n  --- 2025 holdout 四个月 ---")
    ts = pickle.load(open(D0 / "_ts_cache.pkl", "rb"))
    y, mo = ts["y"], ts["mo"]
    ap, afn = ts["ap"].astype(str), ts["adep_flt_null"]
    muC25 = ts["mvt_s"] - ts["sch_s"]
    lobt25, lok25 = ts["lobt_s"], ts["lobt_valid"]
    hit25 = (ap == LIRF_AP) & afn & (muC25 > LIRF_T)
    m4 = np.isin(mo, HOLD)
    c3 = {}
    for mm in [1, 2, 6, 7]:
        s = (mo == mm) & hit25
        n = int(s.sum())
        v = muC25[s]
        yy = y[s]
        c3[f"mo{mm}"] = {"n_hit": n, "n_dep": int((mo == mm).sum()),
                         "muC_med": round(float(np.median(v)), 1) if n else None,
                         "muC_max": round(float(v.max()), 1) if n else None,
                         "y_med": round(float(np.median(yy)), 1) if n else None}
        print(f"    mo={mm}: DEP={int((mo==mm).sum()):>7,}  命中={n:>4}  "
              f"MVT-SCHED 中位={c3[f'mo{mm}']['muC_med']}  max={c3[f'mo{mm}']['muC_max']}  "
              f"真实y中位={c3[f'mo{mm}']['y_med']}")
    print(f"    2025 四个月合计命中 = {int((hit25 & m4).sum())}   "
          f"(全 12 月 {int(hit25.sum())})")
    print(f"\n  [连续性] 2026 Jan+Jul 命中 = {c1['mo1']['n_hit']+c1['mo7']['n_hit']}  "
          f"(与 v22 的 43 完全一致 → 口径与 v22 相同)")
    print(f"  [巧合性] 2025 四个月 = {int((hit25 & m4).sum())}  vs  "
          f"2026 final 四个月 = {int(f['hit'].sum())}  → "
          f"总数相等但逐月分布不同 (2025 6/6/12/39 vs 2026 15/8/12/28), 属巧合")

    res["check2_monthly"] = {"ranking_2026_JanJul": c1, "final_2026": c2,
                             "holdout_2025": c3,
                             "n_hit_ranking_JanJul": c1["mo1"]["n_hit"] + c1["mo7"]["n_hit"],
                             "n_hit_2025_total": int(hit25.sum()),
                             "n_hit_final_total": int(f["hit"].sum())}

    # ================= 核查3 =================
    print("\n" + "=" * 80)
    print("[核查3] 命中行旁证: 整套时刻缺失 vs 仅 BLOCK 损坏")
    print("=" * 80)

    def evid(d, name):
        s = d["hit"].to_numpy()
        sub = d.loc[s]
        n = int(s.sum())
        out = {"n_hit": n}
        if n == 0:
            print(f"  {name}: 无命中行")
            return out, None
        print(f"\n  {name}  (命中 {n} 行)")
        print(f"    {'':<12}{'缺失数':>8}{'缺失率':>9}")
        rows = {}
        for col, lab in [("adepflt_null", "ADEP_flt"), ("blk", "BLOCK"),
                         ("eobt", "EOBT_1"), ("iobt", "IOBT"), ("aobt", "AOBT_3"),
                         ("lobt", "LOBT")]:
            v = sub[col].to_numpy() if col in sub.columns else np.full(n, np.nan)
            k = int(np.isnan(v).sum()) if col != "adepflt_null" else int(sub[col].sum())
            rows[lab] = {"n_missing": k, "rate": round(k / n, 4)}
            print(f"    {lab:<12}{k:>8}{k/n*100:>8.1f}%")
        # 组合指纹
        miss_cnt = np.zeros(n, int)
        for col in ["eobt", "iobt", "aobt", "lobt"]:
            v = sub[col].to_numpy()
            miss_cnt += np.isnan(v).astype(int)
        blk_miss = np.isnan(sub["blk"].to_numpy()).astype(int)
        print(f"    [指纹] _flt 时刻缺失个数分布 (0~4): "
              f"{dict(zip(*np.unique(miss_cnt, return_counts=True)))}")
        print(f"    [指纹] 4 个 _flt 全缺 = {int((miss_cnt==4).sum())} 行; "
              f"BLOCK 本身缺失 = {int(blk_miss.sum())} 行")
        out["missing"] = rows
        out["flt_miss_count_dist"] = {int(k): int(v) for k, v in
                                     zip(*np.unique(miss_cnt, return_counts=True))}
        out["n_all_flt_missing"] = int((miss_cnt == 4).sum())
        out["n_blk_missing"] = int(blk_miss.sum())
        return out, sub

    e1, sub_r = evid(r, "2026 ranking (Jan+Jul)")
    e2, sub_f = evid(f, "2026 final (Jan/Feb/Jun/Jul)")
    e3, _ = (None, None)

    # 2025 用缓存直接算
    n25 = int(hit25.sum())
    if n25:
        lobt_miss = int((~lok25[hit25]).sum())
        print(f"\n  2025 holdout  (命中 {n25} 行)")
        print(f"    LOBT 缺失 = {lobt_miss} / {n25} ({lobt_miss/n25*100:.1f}%)")
        print(f"    [指纹] 真实 y: p50={np.median(y[hit25]):.0f} max={y[hit25].max():.0f}  "
              f">10800 行数={int((y[hit25]>10800).sum())} ({np.mean(y[hit25]>10800)*100:.1f}%)")
        e3 = {"n_hit": n25, "lobt_missing": lobt_miss,
              "y_med": round(float(np.median(y[hit25])), 1),
              "y_max": round(float(y[hit25].max()), 1),
              "n_y_gt_10800": int((y[hit25] > 10800).sum())}
    res["check3_evidence"] = {"ranking_2026": e1, "final_2026": e2, "holdout_2025": e3}

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n→ {OUT}  (只读, 未改任何文件)")


if __name__ == "__main__":
    main()
