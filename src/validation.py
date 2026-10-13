"""
Time-series validation module
==============================

Time-series forecasting cannot use random splits (that would leak future information), so we provide two split strategies:

1. time_split(df, ...): hard split at a time point, used for the main validation
2. time_series_kfold(df, ...): rolling K-fold by month, used for the auxiliary validation / tuning

Design notes:
- The split anchor is SCHED_TIME_UTC_mvt (scheduled time), not the actual time, otherwise it would leak
- The month chosen for the main validation must match the test set:
  - test set January -> validate on Nov/Dec (the immediately preceding months)
  - test set July -> validate on June (same season, same month last year)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE, this is the official scoring metric of the competition."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE, an auxiliary metric to check whether the model falls apart on the long tail."""
    return float(np.mean(np.abs(y_true - y_pred)))


def p95_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """P95 absolute error, reflects the long-tail behaviour."""
    return float(np.percentile(np.abs(y_true - y_pred), 95))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE / MAE / P95 all at once."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "p95_error": p95_error(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Split functions
# ---------------------------------------------------------------------------

@dataclass
class TimeSplit:
    """Time-series split result: train mask + val mask + description."""
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
    """Main validation: simulates "train on Jan-Oct, predict Nov-Dec".

    Corresponding competition test set = January 2026 (winter). Nov-Dec and January have the closest congestion patterns.
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
    """Auxiliary validation: simulates "train on Jan-May, predict June".

    Corresponding competition test set = July 2026 (summer).
    Note: full training data cannot be used, because the validation month must come after the training months.
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
    """K-fold time-series cross validation.

    Uses sklearn TimeSeriesSplit: each time one more time window becomes the validation set, everything before it is training.
    """
    t = pd.to_datetime(df[time_col])
    order = t.argsort().values  # indices sorted by time

    # sklearn TimeSeriesSplit expects a contiguous sequence, so sort df by time before splitting
    df_sorted = df.iloc[order].reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df_sorted)):
        # Convert the mask of the sorted frame back to the mask of the original df
        sorted_indices = df_sorted.index.values  # positions after sorting
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
# Aggregation entry point
# ---------------------------------------------------------------------------

def get_validation_splits(df: pd.DataFrame) -> dict:
    """Return all available validation splits.

    Returns:
        {
            "winter": TimeSplit,    # main: Jan-Oct -> Nov-Dec
            "summer": TimeSplit,    # auxiliary: Jan-May -> Jun
            "kfold":  list[TimeSplit],  # 5-fold time-series CV
        }
    """
    return {
        "winter": time_split_winter(df),
        "summer": time_split_summer(df),
        "kfold": time_series_kfold(df, n_splits=5),
    }


if __name__ == "__main__":
    # Module self-test
    print("=== time-series validation module self-test ===\n")

    # Build some fake data
    times = pd.date_range("2025-01-01", "2025-12-31", freq="h", tz="UTC")
    df = pd.DataFrame({"SCHED_TIME_UTC_mvt": times})

    winter = time_split_winter(df)
    print(f"Winter split: train {winter.n_train}, val {winter.n_val} ({winter.val_period})")

    summer = time_split_summer(df)
    print(f"Summer split: train {summer.n_train}, val {summer.n_val} ({summer.val_period})")

    folds = time_series_kfold(df, n_splits=5)
    for i, fold in enumerate(folds):
        print(f"Fold {i+1}: train {fold.n_train}, val {fold.n_val} ({fold.val_period})")
