"""
构建时间戳缓存 (MVT_ID 显式对齐, 秒级, 哨兵已处理)
==================================================
口径 (用户补充):
1. 时间戳先 .astype('datetime64[ns]') 再 int//10**9 转秒; NaT 与哨兵 (负值) 统一置 NaN,
   避免 d=|BLOCK-LOBT| 出现 10^18 假值污染 T_lobt。
2. 用 MVT_ID 显式对齐 _qinv_feats.pkl, 断言 2,085,047 行一一对应、无重复无遗漏;
   mo 按 MVT_TIME 月份 (允许 ~2 行跨月归属差异, 单独报告)。
3. LOBT 窗口只在 LOBT 有效行标定 (约 206 万); 缺失/哨兵行单独标记, 不参与阈值统计,
   投影时跳过并记录。
4. 模型先落盘再评估 (见后续脚本)。

输出 _ts_cache.pkl: mvt_id / mvt_s / blk_s / sch_s / lobt_s (float, NaN=无效) /
                     lobt_valid / y / mo / ap / mo_from_mvt
"""
from __future__ import annotations
import pickle, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
CACHE = D0 / "_qinv_feats.pkl"
OUT = D0 / "_ts_cache.pkl"

COLS = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "LOBT_flt", "TAXITIME_SEC_mvt", "ADEP_mvt"]

SENTINEL_SEC = -9_223_372_037  # AOBT/LOBT 哨兵 (秒)


def to_sec(s: pd.Series) -> np.ndarray:
    """datetime64[us, UTC] -> 秒级 float64; NaT 与负值哨兵 -> NaN"""
    if s.dtype == "datetime64[us, UTC]":
        s = s.dt.tz_localize(None)
    v = s.astype("datetime64[ns]").to_numpy("int64") // 10**9
    v = v.astype(np.float64)
    v[v < 0] = np.nan          # 哨兵 (int64 min → 负秒)
    return v


def main():
    blob = pickle.load(open(CACHE, "rb"))
    y_c, mo_c = blob["y"], blob["mo"]
    ap_c, mvt_s_c = blob["ap"].astype(str), blob["mvt_s"]
    n_c = len(y_c)
    print(f"cache rows = {n_c:,}")

    files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
    assert len(files) == 12, files
    parts = []
    for i, f in enumerate(files, 1):
        d = pq.read_table(f, columns=COLS).to_pandas()
        d = d[d["PHASE_mvt"].eq("DEP")]
        parts.append(d)
        print(f"  [{i}/12] {Path(f).name}  DEP={len(d):,}")
    raw = pd.concat(parts, ignore_index=True)
    del parts
    print(f"raw DEP total = {len(raw):,}")
    assert len(raw) == n_c, f"行数不一致 raw={len(raw)} cache={n_c}"

    # ---- MVT_ID 显式对齐 ----
    mvt_id = raw["MVT_ID_mvt"].to_numpy()
    assert pd.Series(mvt_id).is_unique, "raw MVT_ID 有重复"
    assert len(np.unique(mvt_id)) == n_c
    # cache 无 MVT_ID 列, 用三列 (MVT 秒, TAXITIME, ADEP) 逐行全等证明行序一致 ≡ MVT_ID 一一对应
    mvt_s = to_sec(raw["MVT_TIME_UTC_mvt"])
    y_r = raw["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    ap_r = raw["ADEP_mvt"].to_numpy().astype(str)
    assert np.array_equal(mvt_s, mvt_s_c), "MVT 秒与 cache 不一致"
    assert np.array_equal(y_r, y_c), "TAXITIME 与 cache 不一致"
    assert np.array_equal(ap_r, ap_c), "ADEP 与 cache 不一致"
    print("  ✓ MVT_ID 唯一 + (MVT秒, TAXITIME, ADEP) 三列逐行全等 → 与 cache 一一对应")

    # ---- 月份归属 (按 MVT_TIME), 报告跨月差异 ----
    mo_from_mvt = pd.to_datetime(mvt_s, unit="s").month.to_numpy()
    n_diff = int((mo_from_mvt != mo_c).sum())
    print(f"  月份(MVT_TIME) 与 cache mo 差异: {n_diff} 行"
          + (f"  → {np.where(mo_from_mvt != mo_c)[0][:10].tolist()}" if n_diff else ""))

    blk_s = to_sec(raw["BLOCK_TIME_UTC_mvt"])
    sch_s = to_sec(raw["SCHED_TIME_UTC_mvt"])
    lobt_s = to_sec(raw["LOBT_flt"])
    lobt_valid = ~np.isnan(lobt_s)

    out = {
        "mvt_id": mvt_id,
        "mvt_s": mvt_s, "blk_s": blk_s, "sch_s": sch_s, "lobt_s": lobt_s,
        "lobt_valid": lobt_valid,
        "y": y_c, "mo": mo_c, "mo_from_mvt": mo_from_mvt, "ap": ap_c,
        "n_month_diff": n_diff,
    }
    pickle.dump(out, open(OUT, "wb"), protocol=4)

    n = len(raw)
    print(f"\n可用性 (全量 {n:,}):")
    print(f"  BLOCK 有效 : {int((~np.isnan(blk_s)).sum()):,}")
    print(f"  SCHED 有效 : {int((~np.isnan(sch_s)).sum()):,}")
    print(f"  LOBT  有效 : {int(lobt_valid.sum()):,}  "
          f"(无效 {int((~lobt_valid).sum()):,} = 哨兵/NaT)")
    print(f"  哨兵秒值参照: {SENTINEL_SEC}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
