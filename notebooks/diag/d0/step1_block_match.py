"""
step1_block_match.py — 训练集上 BLOCK 与四个计划时刻的精确匹配审计 (只读)

目的: 检验"BLOCK 缺失时组织方用 SCHED/LOBT/EOBT/IOBT 回填"这一假设。
      此前只查过 BLOCK==SCHED (LIRF 机制), 这里把另外三个计划时刻也查一遍。

口径:
  * 仅训练集 DEP (BLOCK 有真值)
  * 时间戳统一先转 datetime64[ns] 再整除 1e9 取秒
  * 哨兵值 (-9223372037 等) 先转 datetime 再识别, 避免秒级污染
  * 两档: |Δ| <= 2s 视为精确相等; <= 60s 视为近似相等
  * 一行可能同时匹配多个时刻 → 另给互斥类别

红线: 只读, 不改 label, 不写死机场。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "step1_block_match.json"
NS = 1_000_000_000
SENTINEL_SEC = -9223372037          # 训练/榜单共用的哨兵秒值
EXACT_S = 2
APPROX_S = 60
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt",
        "TAXITIME_SEC_mvt"]
TARGETS = ["SCHED", "LOBT", "EOBT", "IOBT"]
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def to_sec(s: pd.Series) -> np.ndarray:
    """datetime -> int64 纳秒 -> 秒。NaT / 哨兵 → NaN(用 float64 + nan)。"""
    v = pd.to_datetime(s, utc=True, errors="coerce")
    # 哨兵会变成 1970-01-01 附近的极小值或 NaT; 统一识别为无效
    v = v.dt.tz_convert("UTC").dt.tz_localize(None)
    ns = v.astype("datetime64[ns]").astype("int64")
    sec = ns // NS
    out = sec.astype(np.float64)
    # 哨兵: 转 datetime 后落在 1970-01-01 附近 (秒级约 0 或极小)
    out[(out >= -62135596800) & (out <= SENTINEL_SEC)] = np.nan   # 1970-01-01 之前
    out[out > 4_102_444_800] = np.nan                              # 2100 年之后
    return out


def month_file(m: int) -> str | None:
    nxt = "2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
    p = ROOT / f"data/raw/training_2025-{m:02d}-01_{nxt}.parquet"
    return str(p) if p.exists() else None


def main():
    parts = []
    for m in range(1, 13):
        f = month_file(m)
        if f is None:
            lap(f"  跳过 {m} 月")
            continue
        df = pd.read_parquet(f, columns=COLS)
        dep = df[df["PHASE_mvt"] == "DEP"].copy()
        dep["month"] = m
        parts.append(dep)
    raw = pd.concat(parts, ignore_index=True)
    lap(f"训练 DEP 全年: {len(raw):,}")

    blk = to_sec(raw["BLOCK_TIME_UTC_mvt"])
    mvt = to_sec(raw["MVT_TIME_UTC_mvt"])
    valid = ~np.isnan(blk) & ~np.isnan(mvt)
    lap(f"  BLOCK 有效 {int(valid.sum()):,} / MVT 有效 {int((~np.isnan(mvt)).sum()):,}")

    d = {"BLOCK": blk, "MVT": mvt}
    SRC_COL = {"SCHED": "SCHED_TIME_UTC_mvt", "LOBT": "LOBT_flt",
               "EOBT": "EOBT_1_flt", "IOBT": "IOBT_flt"}
    for t in TARGETS:
        d[t] = to_sec(raw[SRC_COL[t]])
    lap("  各计划时刻有效数: " + "  ".join(
        f"{t}={int((~np.isnan(d[t])).sum()):,}" for t in TARGETS))

    # |BLOCK - target| (仅在两者都有效时)
    diff = {}
    for t in TARGETS:
        m = valid & ~np.isnan(d[t])
        diff[t] = np.where(m, np.abs(blk - d[t]), np.nan)
        lap(f"  d_{t}: 有效配对 {int(m.sum()):,}")

    # ---------- 匹配掩码 ----------
    exact = {t: (diff[t] <= EXACT_S) for t in TARGETS}
    approx = {t: (diff[t] <= APPROX_S) for t in TARGETS}
    TAXI = np.where(valid, mvt - blk, np.nan)

    res = {"n_dep": int(len(raw)), "n_block_valid": int(valid.sum()),
           "exact_s": EXACT_S, "approx_s": APPROX_S,
           "taxi_p50": float(np.nanpercentile(TAXI, 50)),
           "counts": {}}

    lap("\n=== 匹配行数 (全训练集) ===")
    for tag, ms in (("exact_le2s", exact), ("approx_le60s", approx)):
        cnt = {t: int(ms[t].sum()) for t in TARGETS}
        any_t = np.zeros(len(raw), bool)
        for t in TARGETS:
            any_t |= ms[t]
        all_t = np.ones(len(raw), bool)
        for t in TARGETS:
            all_t &= ms[t]
        res["counts"][tag] = {**cnt, "any": int(any_t.sum()), "all": int(all_t.sum())}
        print(f"  {tag:<12} " + "  ".join(f"{t}={cnt[t]:>6,}" for t in TARGETS)
              + f"   any={int(any_t.sum()):>7,}  all4={int(all_t.sum()):>7,}")

    # ---------- 互斥类别 ----------
    lap("\n=== 互斥类别 (精确档) ===")
    keys = TARGETS
    excl = {}
    for t in keys:
        others = np.zeros(len(raw), bool)
        for u in keys:
            if u != t:
                others |= exact[u]
        only = exact[t] & ~others
        excl[f"only_{t}"] = int(only.sum())
    excl["none"] = int((~np.zeros(len(raw), bool) & np.ones(len(raw), bool)).sum())
    none_cnt = int((~np.any([exact[t] for t in TARGETS], axis=0)).sum())
    excl["none"] = none_cnt
    excl["all4"] = int(np.all([exact[t] for t in TARGETS], axis=0).sum())
    res["counts"]["exclusive_exact"] = excl
    for k, v in excl.items():
        print(f"  {k:<12} {v:>8,}")
    np.save(D0 / "_step1_exact_masks.npy",
            np.array([exact[t] for t in TARGETS]))
    np.save(D0 / "_step1_diff.npy",
            np.array([diff[t] for t in TARGETS]))
    np.save(D0 / "_step1_ids.npy", raw["MVT_ID_mvt"].to_numpy(float))
    np.save(D0 / "_step1_month.npy", raw["month"].to_numpy(np.int16))
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()
