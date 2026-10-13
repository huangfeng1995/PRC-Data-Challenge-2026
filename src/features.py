"""
Feature Engineering Module (baseline version: basic features only)
==================================================================

This version only builds "the flight's own information" features, without congestion aggregation.
Once the baseline works, add congestion features in src/features_aggregate.py.

Leakage prevention principles (very important):
- OK: SCHED_TIME_UTC_mvt, IOBT_flt, EOBT_1_flt (scheduled times, available in real time)
- USE WITH CARE: AOBT_3_flt (actual pushback = taxi-out start), only for the training target / ex-post aggregation
- FORBIDDEN: MVT_TIME_UTC_mvt (actual takeoff = taxi-out end, which is exactly the label)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time features (based on scheduled time, no leakage)
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, time_col: str = "SCHED_TIME_UTC_mvt") -> pd.DataFrame:
    """Extract time-dimension features from the scheduled time.

    Note: uses SCHED_TIME_UTC_mvt instead of the actual AOBT/MVT time, otherwise it leaks.
    """
    if time_col not in df.columns:
        raise KeyError(f"missing time column {time_col}")

    # Convert to UTC (no timezone) to extract hour/dow etc.
    t = df[time_col].dt.tz_convert("UTC")

    df = df.copy()
    df["sched_hour"] = t.dt.hour.astype("int8")
    df["sched_dow"] = t.dt.dayofweek.astype("int8")  # Monday=0, Sunday=6
    df["sched_month"] = t.dt.month.astype("int8")
    df["sched_day"] = t.dt.day.astype("int8")
    df["sched_minute"] = t.dt.minute.astype("int8")

    # Is weekend / is typical peak hour
    df["is_weekend"] = (df["sched_dow"] >= 5).astype("int8")
    df["is_morning_peak"] = df["sched_hour"].isin([6, 7, 8, 9]).astype("int8")
    df["is_evening_peak"] = df["sched_hour"].isin([17, 18, 19, 20]).astype("int8")

    # Cyclical encoding (let the model understand that 23 o'clock and 0 o'clock are adjacent)
    df["hour_sin"] = np.sin(2 * np.pi * df["sched_hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["sched_hour"] / 24).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["sched_dow"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["sched_dow"] / 7).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * df["sched_month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["sched_month"] / 12).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------

def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract simple features related to the departure airport + arrival airport."""
    df = df.copy()

    # Domestic/international (rough judgement: same country counts as domestic, here roughly split by ICAO prefix)
    # The complete approach needs an airport-to-country mapping table, here use a simple one: within Europe = international
    df["is_domestic_estimate"] = (df["ADEP_mvt"].str[:2] == df["ADES_mvt"].str[:2]).astype("int8")

    # Departure airport x scheduled hour combination (implicitly carries "airport peak pattern" information)
    # Use string concatenation, LightGBM will treat it as a category
    df["airport_hour"] = df["ADEP_mvt"].astype(str) + "_" + df["sched_hour"].astype(str)

    return df


# ---------------------------------------------------------------------------
# Planning features (deviations based on IOBT/SCHED)
# ---------------------------------------------------------------------------

def add_planning_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the time differences related to "scheduled - estimated".

    These are structured features that reflect the "state" of the flight in the system:
    - difference from IOBT to SCHED = expected taxi from scheduled pushback to scheduled takeoff
    - difference from EOBT to SCHED = deviation from FPL estimate to scheduled takeoff
    """
    df = df.copy()

    # Scheduled pushback time (sched_hour has already been computed above)
    if "IOBT_flt" in df.columns:
        df["iobt_to_sched_sec"] = (
            (df["SCHED_TIME_UTC_mvt"] - df["IOBT_flt"]).dt.total_seconds().astype("float32")
        )

    if "EOBT_1_flt" in df.columns:
        df["eobt_to_sched_sec"] = (
            (df["SCHED_TIME_UTC_mvt"] - df["EOBT_1_flt"]).dt.total_seconds().astype("float32")
        )
        # Difference between estimated pushback and the initial plan, reflecting "how much the flight plan was updated"
        df["iobt_eobt_diff_sec"] = (
            (df["EOBT_1_flt"] - df["IOBT_flt"]).dt.total_seconds().astype("float32")
        )

    # Is diverted
    if "ADES_flt" in df.columns and "ADES_FILED_flt" in df.columns:
        df["is_diverted"] = (df["ADES_flt"] != df["ADES_FILED_flt"]).astype("int8")

    return df


# ---------------------------------------------------------------------------
# Categorical field normalization (LightGBM works best with category dtype)
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "ADEP_mvt",
    "ADES_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "airport_hour",  # combination feature
]


def to_categorical(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Convert the specified columns to pandas category type (LightGBM supports it natively and it also saves memory)."""
    cols = cols or CATEGORICAL_COLS
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------------------
# Main entry: complete feature pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Entry function that runs all feature engineering.

    Input: raw DataFrame loaded from data_loader (DEP only, time already parsed)
    Output: feature matrix that can be fed to LightGBM
    """
    df = add_time_features(df)
    df = add_route_features(df)
    df = add_planning_features(df)
    df = to_categorical(df)
    return df


# ---------------------------------------------------------------------------
# Target rescaling (learned from the 1st place team_likable_jelly in 2024)
# ---------------------------------------------------------------------------
#
# Their finding: the average taxi-out differences across airports are huge (FRA is ~50% longer
# than small/medium airports). If you scale by (target - group_min) / (group_max - group_min),
# the "target" of all airports is flattened to [0, 1], which is easier for the model to learn.
#
# Note: this can only be used for training (because it uses the target), and inference needs the saved mapping.
# We make it "optional" - off by default, use it if it works for you.
#

def fit_target_rescaler(
    df_train: pd.DataFrame,
    target_col: str = "TAXITIME_SEC_mvt",
    group_col: str = "ADEP_mvt",
) -> dict:
    """Learn the parameters of "target rescaling grouped by airport".

    For each airport:
        mean, std -> used for (target - mean) / std standardization

    Returns a serializable mapping.
    """
    grouped = df_train.groupby(group_col, observed=True)[target_col].agg(["mean", "std"])
    # Prevent std=0 (an airport with only one record)
    grouped["std"] = grouped["std"].fillna(1.0).clip(lower=1.0)

    mapping = {
        "group_col": group_col,
        "target_col": target_col,
        "per_group": {str(k): {"mean": float(v["mean"]), "std": float(v["std"])}
                       for k, v in grouped.to_dict(orient="index").items()},
        "global_mean": float(df_train[target_col].mean()),
        "global_std": float(df_train[target_col].std()),
    }
    return mapping


def apply_target_rescale(
    df: pd.DataFrame,
    mapping: dict,
    target_col: str = "TAXITIME_SEC_mvt",
    out_col: str = "target_rescaled",
) -> pd.DataFrame:
    """Apply the rescale mapping to the target variable, adding the out_col column."""
    df = df.copy()
    group_col = mapping["group_col"]
    per_group = mapping["per_group"]
    global_mean = mapping["global_mean"]
    global_std = mapping["global_std"]

    means = df[group_col].map({k: v["mean"] for k, v in per_group.items()}).fillna(global_mean)
    stds = df[group_col].map({k: v["std"] for k, v in per_group.items()}).fillna(global_std)

    df[out_col] = ((df[target_col] - means) / stds).astype("float32")
    return df


def inverse_target_rescale(
    predictions_rescaled: np.ndarray,
    group_values: np.ndarray,
    mapping: dict,
) -> np.ndarray:
    """Convert predictions in the "rescaled space" back to the original scale.

    Used at inference time: the model predicts rescaled values, we inverse-transform them back.
    """
    per_group = mapping["per_group"]
    global_mean = mapping["global_mean"]
    global_std = mapping["global_std"]

    out = np.empty_like(predictions_rescaled, dtype="float64")
    for i, g in enumerate(group_values):
        params = per_group.get(g, {"mean": global_mean, "std": global_std})
        out[i] = predictions_rescaled[i] * params["std"] + params["mean"]
    return out


# List of feature columns used by the baseline (no aggregation, no leakage)
BASELINE_FEATURE_COLS = [
    # Time
    "sched_hour", "sched_dow", "sched_month", "sched_day", "sched_minute",
    "is_weekend", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Spatial/route
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "airport_hour",
    "is_domestic_estimate",
    # Aircraft/flight
    "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt", "is_diverted",
    # Planning
    "iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
]


if __name__ == "__main__":
    # Module self-test: load data + run features
    import sys
    from src.data_loader import load_training

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = load_training(data_dir)
    print(f"raw: {df.shape}")

    df_feat = build_features(df)
    print(f"after features: {df_feat.shape}")

    # Check that all baseline features are present
    missing = [c for c in BASELINE_FEATURE_COLS if c not in df_feat.columns]
    if missing:
        print(f"missing features: {missing}")
    else:
        print(f"all {len(BASELINE_FEATURE_COLS)} baseline features are present")

    print(f"\ntarget variable distribution:")
    print(df_feat["TAXITIME_SEC_mvt"].describe())