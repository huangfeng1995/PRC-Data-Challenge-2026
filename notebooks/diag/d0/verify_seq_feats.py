"""
verify_seq_feats.py — MVT-only 序列特征的独立交叉校验

用一份**完全独立的暴力实现**校验 mvtseq_feats.build_seq_features:
  * 暴力: 显式构造 (日,机场,跑道) 分组字典, 组内按 (MVT, 原行号) 排序,
          逐组切出前序间隔序列, 再按 MIN_PREV / 窗口长度规则取值。
  * 被校: 全局 lexsort + 纯 numpy 向量化。
两者代码路径完全不同, 若逐行一致即可排除索引错位/窗口偏移/跨组污染。

判定: max|Δ| < 1e-3 且 NaN 位置完全一致。

红线: 只读原始数据, 不写任何文件。
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(D0))
from mvtseq_feats import build_seq_features, SEQ_FEATS, MIN_PREV, AP_WINDOW_S  # noqa: E402

NS = 1_000_000_000
RAW_COLS = ["PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "RUNWAY_mvt"]


def _sec_and_keys(df: pd.DataFrame):
    mv = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    mv = mv.dt.tz_convert("UTC").dt.tz_localize(None)
    T = (mv.astype("datetime64[ns]").to_numpy(np.int64)) // NS
    day = T // 86400
    ap = df["ADEP_mvt"].astype(str).to_numpy()
    rw = df["RUNWAY_mvt"].astype(str).to_numpy()
    return T, day, ap, rw


def brute_seq(df: pd.DataFrame):
    """独立暴力实现, 返回 {特征名: float64 ndarray}。"""
    n = len(df)
    T, day, ap, rw = _sec_and_keys(df)
    valid = ~pd.isna(pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True,
                                   errors="coerce")).to_numpy()
    out = {c: np.full(n, np.nan) for c in SEQ_FEATS}

    g_rw = collections.defaultdict(list)
    g_ap = collections.defaultdict(list)
    for i in range(n):
        if not valid[i]:
            continue
        g_rw[(int(day[i]), ap[i], rw[i])].append(i)
        g_ap[(int(day[i]), ap[i])].append(i)
    for k in g_rw:
        g_rw[k].sort(key=lambda x: (T[x], x))
    for k in g_ap:
        g_ap[k].sort(key=lambda x: (T[x], x))

    for k, mem in g_rw.items():
        gaps = [T[mem[q]] - T[mem[q - 1]] for q in range(1, len(mem))]
        for r, i in enumerate(mem):                 # r = 前序架数
            for K, name in ((3, "seq_rwy_gap_mean_prev3"),
                            (10, "seq_rwy_gap_mean_prev10")):
                if r < MIN_PREV or r < K:
                    continue
                out[name][i] = float(np.mean(gaps[r - K:r]))
    for k, mem in g_ap.items():
        for r, i in enumerate(mem):
            out["seq_ap_dep_cnt_15min"][i] = float(
                sum(1 for x in mem[:r] if (T[i] - T[x]) < AP_WINDOW_S))
    return out


def verify(df: pd.DataFrame, tag: str) -> bool:
    A = build_seq_features(df)
    B = brute_seq(df)
    print(f"\n=== {tag} · n={len(df):,} ===")
    ok_all = True
    for c in SEQ_FEATS:
        a = A[c].to_numpy(float)
        b = B[c]
        na, nb = np.isnan(a), np.isnan(b)
        mism = int((na ^ nb).sum())
        fin = ~na & ~nb
        md = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0
        ok = (mism == 0 and md < 1e-3)
        ok_all &= ok
        v = a
        print(f"  {c:<28} NaN不一致={mism:<5} max|Δ|={md:<10.6g} "
              f"notna={np.isfinite(v).sum():>7,} ({100*np.isfinite(v).mean():5.2f}%)  "
              f"p50={np.nanpercentile(v,50):8.2f}  p95={np.nanpercentile(v,95):9.2f}  "
              f"max={np.nanmax(v):9.1f}  {'OK' if ok else 'FAIL'}")
    print(f"  → {'全部一致' if ok_all else '存在差异'}")
    return ok_all


def main():
    ok = True
    for tag, f in [("2025-01 (EDDF 为主)", "data/raw/training_2025-01-01_2025-02-01.parquet"),
                   ("2025-07", "data/raw/training_2025-07-01_2025-08-01.parquet")]:
        df = pd.read_parquet(ROOT / f, columns=RAW_COLS)
        dep = df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)
        ok &= verify(dep, tag)
    # ranking 侧同口径可算性
    rk = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=RAW_COLS)
    dep_rk = rk[rk["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    ok &= verify(dep_rk, "ranking 2026 (推理侧同口径)")
    print("\n最终:", "全部通过" if ok else "有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
