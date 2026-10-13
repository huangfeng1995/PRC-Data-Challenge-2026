"""
C 异常指纹模块 — 7 个特征
============================
针对离港滑出秒数 TAXITIME_SEC_mvt, 长滑行(>30min) 约 4.5% 行, 承载约 63% 方差, 是最大未攻方向.

7 个特征:
  1. type_S: AOBT 实际推出 = SCHED 计划起飞 (信号: 推出立刻起飞, 不应有滑行时间)
  2. type_D: AOBT 实际推出 大幅晚于 EOBT (推迟推出)
  3. type_N: AOBT 缺失 (无实际推出时间, 训练数据缺位)
  4. type_H: 长滑行 (>=30min) 但 AOBT 早于 SCHED (推出早但滑行久, 异常信号)
  5. aobt_sched_clash: AOBT == SCHED (分钟内) 强烈异常信号
  6. crossday: AOBT 日期 != SCHED 日期 (跨日)
  7. LTFM后缀剥离: LTFM 的 STAND_mvt 后缀清理 (e.g. "01A" → "01")

红线:
  - BLOCK_TIME 在 ranking DEP 中被挖空, 绝不使用 BLOCK
  - MVT - AOBT 不能用于训练 (本模块遵守)
  - 字段必须有, 不能凭空造
"""
from __future__ import annotations

import numpy as np
import pandas as pd


C_FEATS = [
    "c_type_S",        # AOBT ≈ SCHED (推出立即起飞, 不应有长滑行)
    "c_type_D",        # AOBT 远晚于 EOBT (推迟推出)
    "c_type_N",        # AOBT 缺失
    "c_type_H",        # 长滑行但 AOBT 早于 SCHED
    "c_aobt_sched_clash",  # AOBT == SCHED (分钟内)
    "c_crossday",      # AOBT 日期 != SCHED 日期
    "c_ltfc_stand_strip",  # LTFM STAND 后缀剥离 (01A -> 01)
]


def _norm_strip_ltfc(stand: pd.Series, airport: pd.Series) -> pd.Series:
    """LTFM 的 STAND_mvt 后缀清理
    - 例: "01A" -> "01", "02R" -> "02"
    - 其它机场: 保持原样
    - NaN -> NaN
    """
    out = stand.astype(str).copy()
    is_ltfc = airport.astype(str).values == "LTFM"
    # LTFM 行: 取纯数字部分
    digits = out.str.extract(r"(\d+)", expand=False)
    out[is_ltfc] = digits[is_ltfc]
    return out


def compute_c_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """计算 7 个 C 异常指纹特征. 输入要求列:
      - AOBT_3_flt (实际推出, 可 NaN)
      - EOBT_1_flt (估计推出, 可 NaN)
      - SCHED_TIME_UTC_mvt (计划起飞, 必有)
      - ADEP_mvt (机场, 必有)
      - STAND_mvt (机位, 可 NaN)

    返回: 原 df + 7 个新列 (仅 DEP 行填, ARR 行 NaN)
    """
    df = df.copy()
    n = len(df)

    # 时间戳转 ns int64
    aobt = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
    eobt = pd.to_datetime(df["EOBT_1_flt"], utc=True, errors="coerce")
    sched = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
    mvt = pd.to_datetime(df.get("MVT_TIME_UTC_mvt"), utc=True, errors="coerce") if "MVT_TIME_UTC_mvt" in df.columns else None

    iNaT = np.iinfo(np.int64).min
    aobt_i = aobt.values.astype("datetime64[ns]").view("i8") if not aobt.empty else np.full(n, iNaT, dtype="int64")
    eobt_i = eobt.values.astype("datetime64[ns]").view("i8") if not eobt.empty else np.full(n, iNaT, dtype="int64")
    sched_i = sched.values.astype("datetime64[ns]").view("i8") if not sched.empty else np.full(n, iNaT, dtype="int64")
    aobt_v = aobt_i != iNaT
    eobt_v = eobt_i != iNaT
    sched_v = sched_i != iNaT

    # 1. type_S: AOBT ≈ SCHED (5 分钟内)
    type_S = np.zeros(n, dtype=np.int8)
    if mvt is not None:
        mvt_i = mvt.values.astype("datetime64[ns]").view("i8")
        # type_S: AOBT - SCHED 在 [-60s, 300s] 内 (推出后很快就起飞)
        diff_aobt_sched_s = (aobt_i - sched_i) / 1e9
        type_S = ((aobt_v & sched_v) & (diff_aobt_sched_s >= -60) & (diff_aobt_sched_s <= 300)).astype(np.int8)
    df["c_type_S"] = type_S

    # 2. type_D: AOBT 远晚于 EOBT (>30 分钟推迟推出)
    type_D = np.zeros(n, dtype=np.int8)
    diff_aobt_eobt_s = (aobt_i - eobt_i) / 1e9
    type_D = ((aobt_v & eobt_v) & (diff_aobt_eobt_s > 1800)).astype(np.int8)
    df["c_type_D"] = type_D

    # 3. type_N: AOBT 缺失 (无实际推出时间)
    type_N = (~aobt_v).astype(np.int8)
    df["c_type_N"] = type_N

    # 4. type_H: 长滑行 (>=30min) 但 AOBT 早于 SCHED (推出早但滑行久)
    #    滑行长用 TAXITIME_SEC_mvt 判 (>=1800)
    #    只有 DEP 行才填
    type_H = np.zeros(n, dtype=np.int8)
    if "TAXITIME_SEC_mvt" in df.columns and mvt is not None:
        is_dep = (df["PHASE_mvt"].astype(str).values == "DEP") if "PHASE_mvt" in df.columns else np.ones(n, dtype=bool)
        tt = pd.to_numeric(df["TAXITIME_SEC_mvt"], errors="coerce").values
        long_taxi = (tt >= 1800) & np.isfinite(tt)
        # AOBT 早于 SCHED (推出 < 计划起飞) 但滑行 >=30min
        aobt_before_sched = (aobt_i < sched_i) & aobt_v & sched_v
        type_H = (is_dep & long_taxi & aobt_before_sched).astype(np.int8)
    df["c_type_H"] = type_H

    # 5. aobt_sched_clash: AOBT 与 SCHED 一致 (分钟内)
    aobt_sched_clash = np.zeros(n, dtype=np.int8)
    diff_aobt_sched_s = (aobt_i - sched_i) / 1e9
    aobt_sched_clash = ((aobt_v & sched_v) & (np.abs(diff_aobt_sched_s) < 60)).astype(np.int8)
    df["c_aobt_sched_clash"] = aobt_sched_clash

    # 6. crossday: AOBT 日期 != SCHED 日期 (跨日)
    crossday = np.zeros(n, dtype=np.int8)
    if aobt_v.any() and sched_v.any():
        aobt_date = aobt.dt.date.values  # NaT -> NaT
        sched_date = sched.dt.date.values
        # NaT 的比较会抛错, 用 bitwise
        valid = aobt_v & sched_v
        crossday = np.where(valid, (aobt_date != sched_date).astype(np.int8), 0)
    df["c_crossday"] = crossday

    # 7. LTFM 后缀剥离 (LIRF 后缀剥离是针对 LTFM 而非 LIRF, 你的描述可能 typo)
    if "STAND_mvt" in df.columns and "ADEP_mvt" in df.columns:
        df["c_ltfc_stand_strip"] = _norm_strip_ltfc(df["STAND_mvt"], df["ADEP_mvt"]).values
    else:
        df["c_ltfc_stand_strip"] = np.nan

    return df


# 已知长滑行 (>= 30min) 占比 / 方差贡献验证函数
def long_taxi_stats(df_dep: pd.DataFrame) -> dict:
    """验证 C 模块定位: 长滑行 (>=30min) 行数占比 + 方差占比"""
    tt = pd.to_numeric(df_dep["TAXITIME_SEC_mvt"], errors="coerce").values
    valid = np.isfinite(tt) & (tt > 0)
    long_taxi = valid & (tt >= 1800)
    n_total = int(valid.sum())
    n_long = int(long_taxi.sum())

    # 全方差 (var of (y - mean))
    y_mean = tt[valid].mean()
    var_all = float(((tt[valid] - y_mean) ** 2).sum())

    # 长滑行贡献方差 (SSE_long)
    sse_long = float(((tt[long_taxi] - y_mean) ** 2).sum()) if n_long > 0 else 0.0
    sse_short = float(((tt[valid & ~long_taxi] - y_mean) ** 2).sum())

    return {
        "n_total": n_total,
        "n_long_taxi": n_long,
        "long_pct": n_long / max(n_total, 1),
        "sse_long": sse_long,
        "sse_short": sse_short,
        "sse_long_pct": sse_long / max(sse_long + sse_short, 1),
    }


if __name__ == "__main__":
    # 自测: 加载 02 月训练数据, 计算 C 特征 + 长滑行统计
    from pathlib import Path
    import sys
    ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
    sys.path.insert(0, str(ROOT))
    from src.data_loader import load_month, parse_time_columns

    df = load_month("data/raw/training_2025-02-01_2025-03-01.parquet")
    df = parse_time_columns(df, cols=["AOBT_3_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"])

    df_c = compute_c_fingerprints(df)
    print(f"\n=== C 异常指纹分布 (02月 DEP) ===")
    dep_c = df_c[df_c["PHASE_mvt"] == "DEP"]
    for f in C_FEATS:
        if f in ["c_ltfc_stand_strip"]:
            print(f"  {f}: n_valid={dep_c[f].notna().sum()}, top 5: "
                  f"{list(dep_c[f].value_counts().head(5).items())}")
        else:
            v = dep_c[f].value_counts()
            print(f"  {f}: {dict(v)}")

    print(f"\n=== 长滑行统计 (C 模块定位验证) ===")
    stats = long_taxi_stats(dep_c)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")