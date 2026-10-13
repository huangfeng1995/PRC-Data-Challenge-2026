"""
seq_feats_cache.py — 为 2025 全年 12 个月 + ranking 构造 MVT-only 序列特征并落盘

输出:
  data/interim/seq_feats_2025.parquet   月份 + MVT_ID_mvt + 3 个特征
  data/interim/seq_feats_ranking.parquet  MVT_ID_mvt + 3 个特征

口径与 mvtseq_feats.py 一致, 已在 verify_seq_feats.py 上三方交叉验证通过。
只读原始数据, 不改 label。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(D0))
from mvtseq_feats import build_seq_features, SEQ_FEATS  # noqa: E402

RAW_COLS = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "RUNWAY_mvt"]
OUT_DIR = ROOT / "data" / "interim"
def _train_file(m: int) -> str | None:
    """月度文件名: 起点 2025-{m}-01, 终点为下月 1 日 (12 月终点落到 2026-01-01)。"""
    nxt = f"2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
    p = ROOT / f"data/raw/training_2025-{m:02d}-01_{nxt}.parquet"
    return str(p) if p.exists() else None


TRAIN_FILES = {m: _train_file(m) for m in range(1, 13)}
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for m, f in TRAIN_FILES.items():
        if f is None:
            lap(f"  跳过 {m} 月 (文件不存在)")
            continue
        df = pd.read_parquet(f, columns=RAW_COLS)
        dep = df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)
        feats = build_seq_features(dep)
        out = feats.copy()
        out.insert(0, "MVT_ID_mvt", dep["MVT_ID_mvt"].to_numpy())
        out.insert(1, "month", m)
        parts.append(out)
        lap(f"  {m:>2} 月: DEP {len(dep):>7,} 行  "
            f"notna={int(np.isfinite(feats[SEQ_FEATS[0]].to_numpy(float)).sum()):>7,}")
    tr = pd.concat(parts, ignore_index=True)
    tp = OUT_DIR / "seq_feats_2025.parquet"
    tr.to_parquet(tp, index=False)
    lap(f"→ {tp.name}  {tr.shape}")

    rk = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=RAW_COLS)
    dep_rk = rk[rk["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    fr = build_seq_features(dep_rk)
    fr.insert(0, "MVT_ID_mvt", dep_rk["MVT_ID_mvt"].to_numpy())
    rp = OUT_DIR / "seq_feats_ranking.parquet"
    fr.to_parquet(rp, index=False)
    lap(f"→ {rp.name}  {fr.shape}")

    # 一致性检查: 两侧同名特征不应有结构性差异
    lap("\n两侧分布对比 (train 2025 全年 vs ranking 2026):")
    for c in SEQ_FEATS:
        a = tr[c].to_numpy(float)
        b = fr[c].to_numpy(float)
        print(f"  {c:<28} train p50={np.nanpercentile(a,50):8.2f} p95={np.nanpercentile(a,95):9.2f} | "
              f"rank p50={np.nanpercentile(b,50):8.2f} p95={np.nanpercentile(b,95):9.2f} | "
              f"NaN% {100*np.isnan(a).mean():5.2f} vs {100*np.isnan(b).mean():5.2f}")
    assert len(tr) > 2_000_000, f"训练侧行数异常 {len(tr)}"
    assert len(fr) == 344_841, f"ranking 侧行数异常 {len(fr)}"
    lap("OK")


if __name__ == "__main__":
    main()
