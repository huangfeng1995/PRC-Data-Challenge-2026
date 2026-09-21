"""
一键 EDA 脚本
============

拿到数据后,跑这个脚本会自动生成完整的探索性数据分析报告 + 图表。

用法:
    python -m scripts.eda --data-dir data/raw --output-dir notebooks/eda

输出:
    - notebooks/eda/taxi_out_distribution.png   : 总分布图
    - notebooks/eda/by_airport.png              : 各机场分布
    - notebooks/eda/by_hour.png                 : 按计划小时分布
    - notebooks/eda/by_month.png                : 按月分布(季节性)
    - notebooks/eda/by_airport_hour.png         : 机场 × 小时热力图
    - notebooks/eda/long_tail.png               : 长尾分析(P75/P90/P99)
    - notebooks/eda/missing_values.png          : 缺失值热力图
    - notebooks/eda/corr_basic.png              : 基础数值特征相关矩阵
    - notebooks/eda/summary_stats.csv           : 全量数值描述统计
    - notebooks/eda/airport_stats.csv           : 各机场 taxi-out 汇总
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互式后端,不依赖 display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import load_training


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def setup_style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["savefig.bbox"] = "tight"


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 各 EDA 步骤
# ---------------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame, out_dir: Path):
    """目标变量总分布 + 长尾分析。"""
    print_section("1. taxi-out 总分布")
    y = df["TAXITIME_SEC_mvt"].dropna()
    print(f"  样本数: {len(y):,}")
    print(f"  均值: {y.mean():.1f}s ({y.mean() / 60:.1f}min)")
    print(f"  中位数: {y.median():.1f}s")
    print(f"  P90: {y.quantile(0.9):.1f}s")
    print(f"  P99: {y.quantile(0.99):.1f}s")
    print(f"  最大: {y.max():.1f}s ({(y.max() / 3600):.1f}小时)")

    # v2:clip 到合理范围,排除极端异常值(36 小时的极端值会让 log 直方图崩)
    y_clean = y[(y > 30) & (y < 10800)]
    print(f"  清洗后样本(30 < y < 10800): {len(y_clean):,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始分布(clip 后的)
    axes[0].hist(y_clean, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(y_clean.mean(), color="red", linestyle="--", label=f"mean={y_clean.mean():.0f}s")
    axes[0].axvline(y_clean.median(), color="green", linestyle="--", label=f"median={y_clean.median():.0f}s")
    axes[0].set_xlabel("Taxi-out time (seconds)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Taxi-out Distribution (raw, clipped 30-10800s)")
    axes[0].legend()

    # log 变换后的分布
    axes[1].hist(np.log1p(y_clean), bins=80, color="seagreen", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("log(1 + Taxi-out)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Taxi-out Distribution (log scale, clipped)")

    fig.savefig(out_dir / "taxi_out_distribution.png", dpi=120)
    plt.close(fig)

    # 长尾分析
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [y.quantile(q) for q in quantiles]
    ax.bar([f"P{int(q * 1000) / 10}" for q in quantiles], vals, color="coral", alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}s\n({v / 60:.1f}min)", ha="center", va="bottom")
    ax.set_ylabel("Taxi-out (seconds)")
    ax.set_title("Taxi-out Long-tail Analysis")
    fig.savefig(out_dir / "long_tail.png", dpi=120)
    plt.close(fig)
    print(f"  → 已保存: taxi_out_distribution.png, long_tail.png")


def plot_by_airport(df: pd.DataFrame, out_dir: Path):
    """各机场 taxi-out 分布对比。"""
    print_section("2. 按机场分布")

    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median().sort_values().index
    sns.boxplot(
        data=df, x="ADEP_mvt", y="TAXITIME_SEC_mvt",
        order=order, ax=ax, showfliers=False, palette="Set2",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Taxi-out (seconds, log scale)")
    ax.set_xlabel("Departure airport")
    ax.set_title("Taxi-out by Departure Airport (DEP only)")
    plt.xticks(rotation=30)
    fig.savefig(out_dir / "by_airport.png", dpi=120)
    plt.close(fig)

    # 同时输出汇总表
    stats = df.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].agg(
        ["count", "mean", "median", "std",
         lambda x: x.quantile(0.9), lambda x: x.quantile(0.99)]
    ).rename(columns={"<lambda_0>": "p90", "<lambda_1>": "p99"}).round(1)
    stats.to_csv(out_dir / "airport_stats.csv")
    print(stats)
    print(f"  → 已保存: by_airport.png, airport_stats.csv")


def plot_by_hour(df: pd.DataFrame, out_dir: Path):
    """按计划小时分布(高峰小时模式)。"""
    print_section("3. 按计划小时分布")
    df = df.copy()
    df["sched_hour"] = df["SCHED_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.hour

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 中位数 taxi-out by hour
    hourly = df.groupby("sched_hour")["TAXITIME_SEC_mvt"].agg(["mean", "median"])
    axes[0].plot(hourly.index, hourly["mean"], "o-", color="steelblue", label="mean")
    axes[0].plot(hourly.index, hourly["median"], "s-", color="coral", label="median")
    axes[0].set_xlabel("Scheduled hour (UTC)")
    axes[0].set_ylabel("Taxi-out (seconds)")
    axes[0].set_title("Taxi-out by Scheduled Hour")
    axes[0].legend()
    axes[0].set_xticks(range(0, 24))

    # 起飞架次 by hour
    hourly_count = df.groupby("sched_hour").size()
    axes[1].bar(hourly_count.index, hourly_count.values, color="teal", alpha=0.7)
    axes[1].set_xlabel("Scheduled hour (UTC)")
    axes[1].set_ylabel("# Departures")
    axes[1].set_title("Departure Count by Hour")
    axes[1].set_xticks(range(0, 24))

    fig.savefig(out_dir / "by_hour.png", dpi=120)
    plt.close(fig)
    print(f"  → 已保存: by_hour.png")


def plot_by_month(df: pd.DataFrame, out_dir: Path):
    """按月分布(季节性)。"""
    print_section("4. 按月分布(季节性)")
    df = df.copy()
    df["month"] = df["SCHED_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.month

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    monthly = df.groupby("month")["TAXITIME_SEC_mvt"].agg(["mean", "median", "count"])
    axes[0].plot(monthly.index, monthly["mean"], "o-", color="steelblue", label="mean")
    axes[0].plot(monthly.index, monthly["median"], "s-", color="coral", label="median")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Taxi-out (seconds)")
    axes[0].set_title("Taxi-out by Month")
    axes[0].set_xticks(range(1, 13))
    axes[0].legend()

    axes[1].bar(monthly.index, monthly["count"], color="teal", alpha=0.7)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("# Departures")
    axes[1].set_title("Departure Count by Month")
    axes[1].set_xticks(range(1, 13))

    fig.savefig(out_dir / "by_month.png", dpi=120)
    plt.close(fig)
    print(f"  → 已保存: by_month.png")


def plot_heatmap_airport_hour(df: pd.DataFrame, out_dir: Path):
    """机场 × 计划小时 热力图(看哪些机场在哪些时段最拥堵)。"""
    print_section("5. 机场 × 小时 热力图")
    df = df.copy()
    df["sched_hour"] = df["SCHED_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.hour

    pivot = df.pivot_table(
        index="ADEP_mvt", columns="sched_hour",
        values="TAXITIME_SEC_mvt", aggfunc="median",
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False, fmt=".0f", ax=ax, cbar_kws={"label": "Median taxi-out (s)"})
    ax.set_xlabel("Scheduled hour (UTC)")
    ax.set_ylabel("Departure airport")
    ax.set_title("Median Taxi-out: Airport × Hour")
    fig.savefig(out_dir / "by_airport_hour.png", dpi=120)
    plt.close(fig)
    print(f"  → 已保存: by_airport_hour.png")


def plot_missing(df: pd.DataFrame, out_dir: Path):
    """缺失值分析。"""
    print_section("6. 缺失值分析")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    if len(miss) == 0:
        print("  ✅ 没有缺失值")
        return

    print(miss.to_string())

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.3)))
    miss.plot(kind="barh", ax=ax, color="salmon")
    ax.set_xlabel("Missing rate")
    ax.set_title("Missing Values by Column")
    fig.savefig(out_dir / "missing_values.png", dpi=120)
    plt.close(fig)
    print(f"  → 已保存: missing_values.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path):
    """基础数值特征相关矩阵。"""
    print_section("7. 相关矩阵(基础数值特征)")

    num_cols = [
        "TAXITIME_SEC_mvt",
        "sched_hour", "sched_dow", "sched_month",
        "iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation Matrix (basic numeric features)")
    fig.savefig(out_dir / "corr_basic.png", dpi=120)
    plt.close(fig)
    print(f"  → 已保存: corr_basic.png")


def summary_stats(df: pd.DataFrame, out_dir: Path):
    """汇总统计,落盘为 CSV。"""
    print_section("8. 汇总统计")
    desc = df.describe(include="all").T
    desc.to_csv(out_dir / "summary_stats.csv")
    print(desc)
    print(f"  → 已保存: summary_stats.csv")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_eda(data_dir: str, output_dir: str):
    """对训练数据(2025 全年)做完整 EDA。

    ⚠️ 注意:EDA 只看训练数据(ranking 数据的目标被屏蔽,看了也没用)。
    如果想对比训练和测试的"特征分布差异",可以单独写一个 `eda_compare.py`,
    但当前阶段不需要。
    """
    setup_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_section("加载数据")
    df = load_training(data_dir)
    print(f"  形状: {df.shape}")
    print(f"  字段:\n  {df.columns.tolist()}")

    # 跑所有 EDA
    plot_distribution(df, out_dir)
    plot_by_airport(df, out_dir)
    plot_by_hour(df, out_dir)
    plot_by_month(df, out_dir)
    plot_heatmap_airport_hour(df, out_dir)
    plot_missing(df, out_dir)
    plot_correlation(df, out_dir)
    summary_stats(df, out_dir)

    print("\n" + "=" * 70)
    print(f"  ✅ EDA 完成,所有图表已保存到 {out_dir}/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 一键 EDA")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="notebooks/eda")
    args = parser.parse_args()

    run_eda(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
