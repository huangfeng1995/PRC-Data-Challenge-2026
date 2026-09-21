"""
数据加载模块
============

负责从 data/raw/ 读取月度 parquet 文件,合并为全年数据,并做基础过滤。

设计要点:
- 只读 DEP 行(taxi-out 只对出发航班有意义)
- 保留必要的字段,减少内存
- 时间字段统一转成 pandas datetime + UTC
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 字段定义(来自 https://prc-data-challenge-2026.netlify.app/data.html)
# ---------------------------------------------------------------------------

# 训练数据中,我们关心的字段。后缀 _mvt 是机场侧,_flt 是 NM 航班侧。
TRAIN_COLS_MVT = [
    "MVT_ID_mvt",        # 唯一标识
    "FLIGHT_ID_mvt",
    "ADEP_mvt",          # 起飞机场 ICAO
    "ADES_mvt",          # 落地机场 ICAO
    "PHASE_mvt",         # DEP / ARR,我们要 DEP
    "MVT_TIME_UTC_mvt",  # 实际起飞时间(注意:这是标签终点,不能做特征)
    "BLOCK_TIME_UTC_mvt",  # 实际推出时间(注意:这是 taxi-out 起点,会泄漏,谨慎用)
    "SCHED_TIME_UTC_mvt",  # 计划起降时间(可以用,无泄漏)
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",  # 目标变量(秒)
]

TRAIN_COLS_FLT = [
    "LOBT_flt",
    "CALLSIGN_flt",
    "ADES_flt",
    "ADES_FILED_flt",    # 与 ADES_flt 不一致 = 航班改航
    "MARKET_SEGMENT_flt",
    "IOBT_flt",          # 初始计划推出时间(可用,无泄漏)
    "EOBT_1_flt",        # FPL 估计推出时间(可用,无泄漏)
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",    # wake turbulence category: L/M/H/J
    "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt",        # ⚠️ 实际推出时间,会泄漏目标起点,只能用于事后聚合
    "ARVT_3_flt",        # ⚠️ 实际到达时间,DEP 时也含泄漏信息
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def list_monthly_files(data_dir: Path | str, prefix: str = "training_") -> list[Path]:
    """列出 data_dir 下所有符合 prefix 的 parquet 文件,按文件名排序。"""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"{prefix}*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"在 {data_dir} 下没找到 {prefix}*.parquet 文件。"
            f"请确认数据已下载并放到此目录。"
        )
    return files


def load_month(file_path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """读取单个月份的 parquet 文件。

    用 pyarrow 显式指定列,可以大幅减少内存(原文件 ~25 MB,完整读会全量加载)。
    """
    file_path = Path(file_path)
    table = pq.read_table(file_path, columns=columns)
    return table.to_pandas()


def load_all_months(
    data_dir: Path | str,
    prefix: str = "training_",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """加载所有月度文件并拼接。

    内存占用预估:全量 417 万行,约 280 MB parquet 解压后 ~1.5 GB pandas DataFrame。
    如果机器内存 < 8 GB,建议改为按月训练或用 pyarrow 直接处理。
    """
    files = list_monthly_files(data_dir, prefix=prefix)
    print(f"找到 {len(files)} 个 parquet 文件,开始加载...")

    dfs = []
    for f in files:
        df = load_month(f, columns=columns)
        dfs.append(df)
        print(f"  已加载 {f.name}: {len(df):,} 行")

    df = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数: {len(df):,}")
    return df


def filter_departures(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 DEP 行,这是 taxi-out 预测的目标。"""
    n_before = len(df)
    df = df[df["PHASE_mvt"] == "DEP"].copy()
    print(f"过滤 DEP: {n_before:,} → {len(df):,} 行 ({(len(df) / n_before * 100):.1f}%)")
    return df


def parse_time_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """把 UTC 时间字符串列转成 pandas datetime(UTC)。"""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 主入口:加载 + 过滤 + 时间字段解析
# ---------------------------------------------------------------------------

def load_training(data_dir: Path | str = "data/raw") -> pd.DataFrame:
    """加载训练数据(2025 全年),只保留 DEP,并解析时间字段。

    返回的 DataFrame 已经过滤 PHASE=DEP,时间列都是 datetime64[ns, UTC]。
    """
    cols = TRAIN_COLS_MVT + TRAIN_COLS_FLT
    df = load_all_months(data_dir, prefix="training_", columns=cols)
    df = filter_departures(df)
    df = parse_time_columns(
        df,
        cols=[
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "IOBT_flt",
            "EOBT_1_flt",
            "AOBT_3_flt",
            "ARVT_3_flt",
            "LOBT_flt",
        ],
    )
    return df


def load_ranking(data_dir: Path | str = "data/raw") -> pd.DataFrame:
    """加载排名数据集(2026-01 + 2026-07,目标变量已被屏蔽)。

    BLOCK_TIME_UTC_mvt 和 TAXITIME_SEC_mvt 在 DEP 行上是空的,这正是我们要预测的部分。
    """
    cols = TRAIN_COLS_MVT + TRAIN_COLS_FLT
    df = load_month(Path(data_dir) / "ranking.parquet", columns=cols)
    df = filter_departures(df)
    df = parse_time_columns(
        df,
        cols=[
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "IOBT_flt",
            "EOBT_1_flt",
            "AOBT_3_flt",
            "ARVT_3_flt",
            "LOBT_flt",
        ],
    )
    return df


def load_submission_template(data_dir: Path | str = "data/raw") -> pd.DataFrame:
    """加载提交模板(只有 MVT_ID_mvt 和 TAXITIME_SEC_mvt 两列)。"""
    return pd.read_parquet(Path(data_dir) / "submitting.parquet")


if __name__ == "__main__":
    # 模块自测:python -m src.data_loader
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    print("=== 训练数据 ===")
    df = load_training(data_dir)
    print(f"\n形状: {df.shape}")
    print(f"\n字段类型:\n{df.dtypes}")
    print(f"\n目标变量描述:\n{df['TAXITIME_SEC_mvt'].describe()}")
