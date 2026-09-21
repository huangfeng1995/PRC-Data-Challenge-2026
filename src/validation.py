"""
时序验证模块
============

时序预测不能用随机切分(会泄漏未来信息),我们提供两种切分方式:

1. time_split(df, ...):按时间点硬切,主验证用这个
2. time_series_kfold(df, ...):按月滚动 K 折,辅验证/调优用

设计要点:
- 切分锚点是 SCHED_TIME_UTC_mvt(计划时间),不是实际时间,否则会泄漏
- 主验证月份的选择要匹配测试集:
  - 测试集 1 月 → 验证用 11/12 月(紧邻的前几个月)
  - 测试集 7 月 → 验证用 6 月(同季,去年同月)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE,这就是比赛的官方评分指标。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE,辅助看模型是否在长尾上栽跟头。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def p95_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """P95 绝对误差,反映长尾表现。"""
    return float(np.percentile(np.abs(y_true - y_pred), 95))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """一次性算 RMSE / MAE / P95。"""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "p95_error": p95_error(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 切分函数
# ---------------------------------------------------------------------------

@dataclass
class TimeSplit:
    """时序切分结果:训练 mask + 验证 mask + 描述信息。"""
    train_mask: np.ndarray
    val_mask: np.ndarray
    train_period: str
    val_period: str

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_val(self) -> int:
        return int(self.val_mask.sum())


def time_split_winter(df: pd.DataFrame) -> TimeSplit:
    """主验证:模拟"用 1~10 月训练,预测 11~12 月"。

    对应比赛测试集 = 2026 年 1 月(冬季)。11~12 月和 1 月的拥堵模式最接近。
    """
    t = df["SCHED_TIME_UTC_mvt"]
    train_mask = (t >= "2025-01-01") & (t < "2025-11-01")
    val_mask = (t >= "2025-11-01") & (t < "2026-01-01")
    return TimeSplit(
        train_mask=train_mask.values,
        val_mask=val_mask.values,
        train_period="2025-01 ~ 2025-10",
        val_period="2025-11 ~ 2025-12",
    )


def time_split_summer(df: pd.DataFrame) -> TimeSplit:
    """辅验证:模拟"用 1~5 月训练,预测 6 月"。

    对应比赛测试集 = 2026 年 7 月(夏季)。
    注意:不能用全量训练,因为验证月份必须在训练月份之后。
    """
    t = df["SCHED_TIME_UTC_mvt"]
    train_mask = (t >= "2025-01-01") & (t < "2025-06-01")
    val_mask = (t >= "2025-06-01") & (t < "2025-07-01")
    return TimeSplit(
        train_mask=train_mask.values,
        val_mask=val_mask.values,
        train_period="2025-01 ~ 2025-05",
        val_period="2025-06",
    )


def time_series_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    time_col: str = "SCHED_TIME_UTC_mvt",
) -> list[TimeSplit]:
    """K 折时序交叉验证。

    用 sklearn TimeSeriesSplit:每次增加一个时间窗口作为验证,前面的全是训练。
    """
    t = pd.to_datetime(df[time_col])
    order = t.argsort().values  # 按时序排的索引

    # sklearn 的 TimeSeriesSplit 期望连续序列,这里把 df 按时间排好再切
    df_sorted = df.iloc[order].reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df_sorted)):
        # 把 sorted 的 mask 转回原 df 的 mask
        sorted_indices = df_sorted.index.values  # 排序后的位置
        train_orig_idx = sorted_indices[train_idx]
        val_orig_idx = sorted_indices[val_idx]

        train_mask = np.zeros(len(df), dtype=bool)
        val_mask = np.zeros(len(df), dtype=bool)
        train_mask[train_orig_idx] = True
        val_mask[val_orig_idx] = True

        train_period = f"{df.iloc[train_orig_idx][time_col].min():%Y-%m} ~ {df.iloc[train_orig_idx][time_col].max():%Y-%m}"
        val_period = f"{df.iloc[val_orig_idx][time_col].min():%Y-%m} ~ {df.iloc[val_orig_idx][time_col].max():%Y-%m}"

        splits.append(TimeSplit(
            train_mask=train_mask,
            val_mask=val_mask,
            train_period=train_period,
            val_period=val_period,
        ))

    return splits


# ---------------------------------------------------------------------------
# 整合入口
# ---------------------------------------------------------------------------

def get_validation_splits(df: pd.DataFrame) -> dict:
    """返回所有可用的验证切分。

    返回:
        {
            "winter": TimeSplit,    # 主:1~10 → 11~12
            "summer": TimeSplit,    # 辅:1~5 → 6
            "kfold":  list[TimeSplit],  # 5 折时序 CV
        }
    """
    return {
        "winter": time_split_winter(df),
        "summer": time_split_summer(df),
        "kfold": time_series_kfold(df, n_splits=5),
    }


if __name__ == "__main__":
    # 模块自测
    print("=== 时序验证模块自测 ===\n")

    # 构造一些假数据
    times = pd.date_range("2025-01-01", "2025-12-31", freq="h", tz="UTC")
    df = pd.DataFrame({"SCHED_TIME_UTC_mvt": times})

    winter = time_split_winter(df)
    print(f"Winter split: 训练 {winter.n_train}, 验证 {winter.n_val} ({winter.val_period})")

    summer = time_split_summer(df)
    print(f"Summer split: 训练 {summer.n_train}, 验证 {summer.n_val} ({summer.val_period})")

    folds = time_series_kfold(df, n_splits=5)
    for i, fold in enumerate(folds):
        print(f"Fold {i+1}: 训练 {fold.n_train}, 验证 {fold.n_val} ({fold.val_period})")
