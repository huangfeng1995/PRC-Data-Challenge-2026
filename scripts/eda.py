"""
One-click EDA script
============

Once you have the data, running this script automatically generates a complete exploratory data analysis report + charts.

Usage:
    python -m scripts.eda --data-dir data/raw --output-dir notebooks/eda

Outputs:
    - notebooks/eda/taxi_out_distribution.png   : overall distribution plot
    - notebooks/eda/by_airport.png              : per-airport distribution
    - notebooks/eda/by_hour.png                 : distribution by scheduled hour
    - notebooks/eda/by_month.png                : distribution by month (seasonality)
    - notebooks/eda/by_airport_hour.png         : airport x hour heatmap
    - notebooks/eda/long_tail.png               : long-tail analysis (P75/P90/P99)
    - notebooks/eda/missing_values.png          : missing-value heatmap
    - notebooks/eda/corr_basic.png              : correlation matrix of basic numeric features
    - notebooks/eda/summary_stats.csv           : descriptive stats for all numeric values
    - notebooks/eda/airport_stats.csv           : per-airport taxi-out summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display dependency
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import load_training


# ---------------------------------------------------------------------------
# Utilities
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
# Individual EDA steps
# ---------------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame, out_dir: Path):
    """Overall target distribution + long-tail analysis."""
    print_section("1. taxi-out overall distribution")
    y = df["TAXITIME_SEC_mvt"].dropna()
    print(f"  sample size: {len(y):,}")
    print(f"  mean: {y.mean():.1f}s ({y.mean() / 60:.1f}min)")
    print(f"  median: {y.median():.1f}s")
    print(f"  P90: {y.quantile(0.9):.1f}s")
    print(f"  P99: {y.quantile(0.99):.1f}s")
    print(f"  max: {y.max():.1f}s ({(y.max() / 3600):.1f}hours)")

    # v2: clip to a reasonable range to exclude extreme outliers (the 36-hour extreme breaks the log histogram)
    y_clean = y[(y > 30) & (y < 10800)]
    print(f"  cleaned sample (30 < y < 10800): {len(y_clean):,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw distribution (after clipping)
    axes[0].hist(y_clean, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(y_clean.mean(), color="red", linestyle="--", label=f"mean={y_clean.mean():.0f}s")
    axes[0].axvline(y_clean.median(), color="green", linestyle="--", label=f"median={y_clean.median():.0f}s")
    axes[0].set_xlabel("Taxi-out time (seconds)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Taxi-out Distribution (raw, clipped 30-10800s)")
    axes[0].legend()

    # Distribution after the log transform
    axes[1].hist(np.log1p(y_clean), bins=80, color="seagreen", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("log(1 + Taxi-out)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Taxi-out Distribution (log scale, clipped)")

    fig.savefig(out_dir / "taxi_out_distribution.png", dpi=120)
    plt.close(fig)

    # Long-tail analysis
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
    print(f"  -> saved: taxi_out_distribution.png, long_tail.png")


def plot_by_airport(df: pd.DataFrame, out_dir: Path):
    """Comparison of taxi-out distribution across airports."""
    print_section("2. Distribution by airport")

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

    # Also emit the summary table
    stats = df.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].agg(
        ["count", "mean", "median", "std",
         lambda x: x.quantile(0.9), lambda x: x.quantile(0.99)]
    ).rename(columns={"<lambda_0>": "p90", "<lambda_1>": "p99"}).round(1)
    stats.to_csv(out_dir / "airport_stats.csv")
    print(stats)
    print(f"  -> saved: by_airport.png, airport_stats.csv")


def plot_by_hour(df: pd.DataFrame, out_dir: Path):
    """Distribution by scheduled hour (peak-hour pattern)."""
    print_section("3. Distribution by scheduled hour")
    df = df.copy()
    df["sched_hour"] = df["SCHED_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.hour

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Median taxi-out by hour
    hourly = df.groupby("sched_hour")["TAXITIME_SEC_mvt"].agg(["mean", "median"])
    axes[0].plot(hourly.index, hourly["mean"], "o-", color="steelblue", label="mean")
    axes[0].plot(hourly.index, hourly["median"], "s-", color="coral", label="median")
    axes[0].set_xlabel("Scheduled hour (UTC)")
    axes[0].set_ylabel("Taxi-out (seconds)")
    axes[0].set_title("Taxi-out by Scheduled Hour")
    axes[0].legend()
    axes[0].set_xticks(range(0, 24))

    # Departure count by hour
    hourly_count = df.groupby("sched_hour").size()
    axes[1].bar(hourly_count.index, hourly_count.values, color="teal", alpha=0.7)
    axes[1].set_xlabel("Scheduled hour (UTC)")
    axes[1].set_ylabel("# Departures")
    axes[1].set_title("Departure Count by Hour")
    axes[1].set_xticks(range(0, 24))

    fig.savefig(out_dir / "by_hour.png", dpi=120)
    plt.close(fig)
    print(f"  -> saved: by_hour.png")


def plot_by_month(df: pd.DataFrame, out_dir: Path):
    """Distribution by month (seasonality)."""
    print_section("4. Distribution by month (seasonality)")
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
    print(f"  -> saved: by_month.png")


def plot_heatmap_airport_hour(df: pd.DataFrame, out_dir: Path):
    """Airport x scheduled-hour heatmap (see which airports are most congested at which times)."""
    print_section("5. Airport x hour heatmap")
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
    print(f"  -> saved: by_airport_hour.png")


def plot_missing(df: pd.DataFrame, out_dir: Path):
    """Missing-value analysis."""
    print_section("6. Missing-value analysis")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    if len(miss) == 0:
        print("  ✅ no missing values")
        return

    print(miss.to_string())

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.3)))
    miss.plot(kind="barh", ax=ax, color="salmon")
    ax.set_xlabel("Missing rate")
    ax.set_title("Missing Values by Column")
    fig.savefig(out_dir / "missing_values.png", dpi=120)
    plt.close(fig)
    print(f"  -> saved: missing_values.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path):
    """Correlation matrix of the basic numeric features."""
    print_section("7. Correlation matrix (basic numeric features)")

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
    print(f"  -> saved: corr_basic.png")


def summary_stats(df: pd.DataFrame, out_dir: Path):
    """Summary statistics, written to CSV."""
    print_section("8. Summary statistics")
    desc = df.describe(include="all").T
    desc.to_csv(out_dir / "summary_stats.csv")
    print(desc)
    print(f"  -> saved: summary_stats.csv")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eda(data_dir: str, output_dir: str):
    """Run a full EDA on the training data (all of 2025).

    ⚠️ Note: EDA only looks at the training data (the ranking target is masked, so it would be useless anyway).
    If you want to compare "feature distribution shift" between train and test, you can write a separate `eda_compare.py`,
    but it is not needed at the current stage.
    """
    setup_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_section("Load data")
    df = load_training(data_dir)
    print(f"  shape: {df.shape}")
    print(f"  columns:\n  {df.columns.tolist()}")

    # Run all EDA steps
    plot_distribution(df, out_dir)
    plot_by_airport(df, out_dir)
    plot_by_hour(df, out_dir)
    plot_by_month(df, out_dir)
    plot_heatmap_airport_hour(df, out_dir)
    plot_missing(df, out_dir)
    plot_correlation(df, out_dir)
    summary_stats(df, out_dir)

    print("\n" + "=" * 70)
    print(f"  ✅ EDA complete, all charts saved to {out_dir}/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PRC 2026 one-click EDA")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="notebooks/eda")
    args = parser.parse_args()

    run_eda(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()