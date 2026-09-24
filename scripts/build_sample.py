"""
生成小样本数据 (data/sample/), 用于云端 clone 后快速验证特征管线
==================================================================
- 从 data/raw/training_2025_*.parquet × 12 + ranking.parquet 按机场抽少量行
- 每机场每月保留若干行 DEP + 对应 ARR (同 FLIGHT_ID 或同 MVT 时刻 ±5 分钟)
- 控制总量在几 MB ~ 几十 MB, 使仓库总增长 < 50 MB
- 不接触 .gitignore 列表里的敏感目录
- 输出:
    data/sample/
        training_2025-{MM}-{MM+1}.parquet  (12 个, 每个 < 1 MB)
        ranking.parquet
        submitting.parquet
        sample_meta.json  (本采样脚本参数 + 行数明细)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SAMPLE = ROOT / "data/sample"

# 机场 (10 个, 与比赛一致)
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
            "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]

# 训练: 每机场每月 N_DEP 行 (DEP) + 同月同机场 N_ARR 行 (ARR), 关联的 FLIGHT_ID 上下文
N_DEP_PER_AP_PER_MONTH = 40
N_ARR_PER_AP_PER_MONTH = 30  # ARR 抽样稍少

# ranking + submitting: 全量 (本来就小, ~45MB 总)
KEEP_RANKING_FULL = True


def sample_month(month: int) -> pd.DataFrame:
    """抽样某月: 每机场按 PHASE 分组各取若干行, 尽量保留时序"""
    if month < 12:
        fp = RAW / f"training_2025-{month:02d}-01_2025-{month+1:02d}-01.parquet"
    else:
        fp = RAW / "training_2025-12-01_2026-01-01.parquet"
    if not fp.exists():
        print(f"⚠️ 缺 {fp.name}, 跳过", flush=True)
        return pd.DataFrame()
    print(f"  [m={month:02d}] 读 {fp.name} ...", flush=True)
    # 全列 (列数稳定, 不大)
    df = pd.read_parquet(fp)
    print(f"    原始 {len(df):,} 行", flush=True)

    out_parts = []
    for ap in AIRPORTS:
        sub = df[df["ADEP_mvt"] == ap] if ap in df["ADEP_mvt"].unique() else df.iloc[:0]
        if len(sub) == 0:
            continue
        dep = sub[sub["PHASE_mvt"] == "DEP"].sample(
            n=min(N_DEP_PER_AP_PER_MONTH, len(sub[sub["PHASE_mvt"] == "DEP"])),
            random_state=42)
        arr = sub[sub["PHASE_mvt"] == "ARR"].sample(
            n=min(N_ARR_PER_AP_PER_MONTH, len(sub[sub["PHASE_mvt"] == "ARR"])),
            random_state=42)
        out_parts.append(pd.concat([dep, arr], ignore_index=True))
    if not out_parts:
        return pd.DataFrame()
    out = pd.concat(out_parts, ignore_index=True)
    print(f"    抽样后 {len(out):,} 行 ({len(out_parts)} 机场)", flush=True)
    return out


def main():
    if not RAW.exists():
        print(f"❌ 缺数据源 {RAW}/, 先跑 scripts/download_data.py", flush=True)
        return 1
    SAMPLE.mkdir(parents=True, exist_ok=True)

    meta = {"version": "v1", "n_dep_per_ap_per_month": N_DEP_PER_AP_PER_MONTH,
            "n_arr_per_ap_per_month": N_ARR_PER_AP_PER_MONTH, "airports": AIRPORTS,
            "files": {}}

    print("=" * 60)
    print("生成小样本 (data/sample/)")
    print("=" * 60)
    total_bytes = 0
    for m in range(1, 13):
        df = sample_month(m)
        if len(df) == 0:
            continue
        if m < 12:
            out_name = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        else:
            out_name = "training_2025-12-01_2026-01-01.parquet"
        out_path = SAMPLE / out_name
        df.to_parquet(out_path, index=False)
        sz = out_path.stat().st_size
        total_bytes += sz
        meta["files"][out_name] = {"rows": int(len(df)), "bytes": sz}
        print(f"  写 {out_name}: {len(df):,} 行, {sz/1024:.1f} KB", flush=True)

    # ranking + submitting 抽样 (云端不需全量, 5000 行够测试特征管线)
    N_RANK_SAMPLE = 5000
    for name, n_sample in [("ranking.parquet", N_RANK_SAMPLE), ("submitting.parquet", N_RANK_SAMPLE)]:
        src = RAW / name
        if not src.exists():
            print(f"⚠️ 缺 {name}, 跳过", flush=True)
            continue
        df = pd.read_parquet(src)
        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        out_path = SAMPLE / name
        df.to_parquet(out_path, index=False)
        sz2 = out_path.stat().st_size
        total_bytes += sz2
        meta["files"][name] = {"rows": int(len(df)), "bytes": sz2, "sampled": n_sample < len(pd.read_parquet(src)) if src.exists() else False}
        print(f"  写 {name}: {len(df):,} 行, {sz2/1024:.1f} KB", flush=True)

    meta["total_bytes"] = total_bytes
    (SAMPLE / "sample_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n✅ 总计 {total_bytes/1024:.1f} KB ({total_bytes/1e6:.2f} MB)")
    print(f"   落盘到 {SAMPLE}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())