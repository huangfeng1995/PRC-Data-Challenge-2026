"""
给 _ts_cache.pkl 追加 ADEP_flt 有效性标记 (2025 四个月 holdout 验证用)
=======================================================================
final 上 T=14400 完整口径 = (ADEP==LIRF) AND (ADEP_flt 缺失) AND (MVT-SCHED>14400)。
阶段1 的 C 用了简化口径 (无 ADEP_flt 条件), 本脚本补上该条件以便对齐 final。

用 MVT_ID 显式对齐现有 _ts_cache 行序, 断言 2,085,047 一一对应。
"""
from __future__ import annotations
import pickle, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D0 = Path("/Users/openclaw/Documents/PRC_data_challenge/notebooks/diag/d0")
ROOT = D0.parents[2]
TS = D0 / "_ts_cache.pkl"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_flt"]


def main():
    ts = pickle.load(open(TS, "rb"))
    mvt_id_c = ts["mvt_id"]
    n = len(mvt_id_c)
    print(f"cache rows = {n:,}")

    parts = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))):
        d = pq.read_table(f, columns=COLS).to_pandas()
        parts.append(d[d["PHASE_mvt"].eq("DEP")])
    raw = pd.concat(parts, ignore_index=True)
    del parts
    print(f"raw DEP = {len(raw):,}")
    assert len(raw) == n, f"行数不一致 {len(raw)} vs {n}"

    mvt_id_r = raw["MVT_ID_mvt"].to_numpy()
    assert np.array_equal(mvt_id_r, mvt_id_c), "MVT_ID 顺序不一致 (行序已漂移)"
    print("  ✓ MVT_ID 逐行完全一致")

    s = raw["ADEP_flt"]
    # 空值判定: None / NaN / 空串 / 纯空白 都算缺失
    adepflt_null = (s.isna() | (s.astype(str).str.strip() == "")).to_numpy()
    print(f"  ADEP_flt 缺失: {int(adepflt_null.sum()):,} / {n:,} "
          f"({adepflt_null.mean()*100:.3f}%)")

    ts["adep_flt_null"] = adepflt_null
    pickle.dump(ts, open(TS, "wb"), protocol=4)
    print(f"→ 已追加 adep_flt_null 到 {TS.name}")

    # LIRF 子集统计 (与 final 对齐)
    ap = ts["ap"]
    mo = ts["mo"]
    is_lirf = ap == "LIRF"
    mu_C = ts["mvt_s"] - ts["sch_s"]
    for label, months in [("四个月(1/2/6/7)", [1, 2, 6, 7])]:
        m = np.isin(mo, months)
        lirf = is_lirf & m
        hit_simple = lirf & (mu_C > 14400)
        hit_full = lirf & adepflt_null & (mu_C > 14400)
        print(f"\n  [{label}] LIRF n={int(lirf.sum()):,}")
        print(f"    简化口径 (MVT-SCHED>14400)        命中 {int(hit_simple.sum()):,}")
        print(f"    完整口径 (ADEP_flt缺 & >14400)    命中 {int(hit_full.sum()):,}")
        print(f"    ADEP_flt 缺失但 <=14400           {int((lirf & adepflt_null & (mu_C<=14400)).sum()):,}")


if __name__ == "__main__":
    main()
