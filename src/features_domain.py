"""
Domain feature module
=====================

Learned from the 1st place of 2025 ("domain knowledge" based):
- Heavy aircraft taxi slowly
- Taxiing rules differ across airlines
- Stand vs runway distance affects taxi-out
- Airline x airport interaction

What we add:
1. Stand zone (first letter extracted from STAND, e.g. zone A, zone B)
2. Runway family (runways of the same direction grouped together, e.g. "07L/07R" -> "07")
3. Target encoding of airline x airport (using historical taxi-out median)
4. Target encoding of aircraft type x airport

Note: target encoding must be fit on "past data" (to avoid leakage).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Stand zone / runway family / runway side
# ---------------------------------------------------------------------------

def add_stand_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the stand zone from STAND_mvt.

    For example "A12" -> "A", "B23" -> "B", "T2" -> "T", None -> "UNKNOWN"
    """
    df = df.copy()
    if "STAND_mvt" not in df.columns:
        df["stand_zone"] = "UNKNOWN"
        return df
    stand = df["STAND_mvt"].astype(str).fillna("UNKNOWN")
    # Take the first character (the letter part)
    df["stand_zone"] = stand.str.extract(r"^([A-Z])", expand=False).fillna("UNKNOWN")
    return df


def add_runway_family(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the runway family and side from RUNWAY_mvt.

    For example "07L" -> ("07", "L"), "25R" -> ("25", "R"), "18" -> ("18", "")
    - runway_family: all runways of the same direction grouped together (e.g. "07L"/"07R" -> "07")
    - runway_side: L/R/C (left/right/center)
    """
    df = df.copy()
    if "RUNWAY_mvt" not in df.columns:
        df["runway_family"] = "UNKNOWN"
        df["runway_side"] = "UNKNOWN"
        return df
    rw = df["RUNWAY_mvt"].astype(str).fillna("UNKNOWN")
    # Numeric part + letter part
    m = rw.str.extract(r"^(\d{1,2})([LRC]?)$", expand=True)
    df["runway_family"] = m[0].fillna("UNKNOWN")
    df["runway_side"] = m[1].fillna("")
    return df


# ---------------------------------------------------------------------------
# 2. Target encoding (target encoding)
# ---------------------------------------------------------------------------

def target_encode_smoothed(
    df_train: pd.DataFrame,
    df_target: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    smoothing: int = 20,
    new_col_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Smoothed target encoding.

    Idea: use the mean target of each (group_cols combination) as the feature value, but smooth small samples
        encoded = (n * group_mean + smoothing * global_mean) / (n + smoothing)

    Args:
        df_train: training data used to learn the encoding (with target)
        df_target: target data to apply the encoding to (can be df_train itself, to add the new column)
        group_cols: list of grouping columns, e.g. ["AIRCRAFT_OPERATOR_flt", "ADEP_mvt"]
        target_col: target variable name
        smoothing: smoothing coefficient, the larger the more conservative (closer to the global mean)
        new_col_name: output column name, default "_".join(group_cols) + "_te"

    Returns:
        (df_train, df_target, mapping_dict)
        both df_train and df_target get the new_col_name column added
        mapping_dict can be serialized and saved, used at inference time
    """
    global_mean = df_train[target_col].mean()

    # Make sure the target is numeric (categorical cannot compute mean)
    if hasattr(df_train[target_col], 'cat'):
        target_numeric = df_train[target_col].astype(float)
    else:
        target_numeric = df_train[target_col]

    # Learn the mean and count of each group
    grouped = df_train.assign(**{target_col: target_numeric}).groupby(
        group_cols, observed=True
    )[target_col].agg(["mean", "count"])
    grouped.columns = ["__te_mean__", "__te_count__"]

    # Smoothing
    grouped["__te_encoded__"] = (
        (grouped["__te_mean__"] * grouped["__te_count__"] + smoothing * global_mean)
        / (grouped["__te_count__"] + smoothing)
    )

    mapping = grouped["__te_encoded__"].to_dict()
    # If group_cols has multiple columns, the mapping key is a tuple
    if isinstance(next(iter(mapping.keys())), tuple):
        mapping = {k: v for k, v in mapping.items()}

    new_col = new_col_name or "_".join(group_cols) + "_te"

    # Apply to the training and target data
    if len(group_cols) == 1:
        df_train[new_col] = df_train[group_cols[0]].map(mapping).astype(float).fillna(global_mean)
        df_target[new_col] = df_target[group_cols[0]].map(mapping).astype(float).fillna(global_mean)
    else:
        # Multiple columns: join the group into a tuple as the key
        train_keys = list(zip(*(df_train[c] for c in group_cols)))
        target_keys = list(zip(*(df_target[c] for c in group_cols)))
        df_train[new_col] = pd.Series(train_keys).map(mapping).astype(float).fillna(global_mean).values
        df_target[new_col] = pd.Series(target_keys).map(mapping).astype(float).fillna(global_mean).values

    # Convert mapping into a serializable dict
    if isinstance(next(iter(mapping.keys())), tuple):
        mapping_save = {"__keys__": list(group_cols), "values": {str(k): v for k, v in mapping.items()}}
    else:
        mapping_save = {"__keys__": group_cols[0], "values": dict(mapping)}

    return df_train, df_target, {"mapping": mapping_save, "global_mean": global_mean, "new_col": new_col}


# ---------------------------------------------------------------------------
# 3. Main entry: full domain feature pipeline
# ---------------------------------------------------------------------------

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all structural domain features (the parts not depending on the historical target).

    Includes:
        - stand_zone: stand zone
        - runway_family: runway family
        - runway_side: runway side
        - aircraft_type_size: large/medium/small (inferred from wake turbulence category)

    Target encoding (train target encoding) has to be done separately in add_target_encodings(),
    because it needs historical taxi-out and cannot be used directly at inference time (unless the mapping is loaded).
    """
    df = add_stand_zone(df)
    df = add_runway_family(df)
    return df


# Column groups TE is fit on (single variable + two-way crosses)
TE_SPECS = [
    ("single", [("AIRCRAFT_OPERATOR_flt", "operator_te")]),
    ("single", [("AIRCRAFT_TYPE_mvt", "aircraft_type_te")]),
    ("single", [("RUNWAY_mvt", "runway_te")]),
    ("single", [("ADEP_mvt", "airport_te")]),
    ("cross", [(["AIRCRAFT_OPERATOR_flt", "ADEP_mvt"], "operator_airport_te")]),
    ("cross", [(["AIRCRAFT_TYPE_mvt", "ADEP_mvt"], "aircraft_airport_te")]),
    ("cross", [(["WK_TBL_CAT_flt", "ADEP_mvt"], "wake_airport_te")]),
    ("cross", [(["AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt"], "operator_wake_te")]),
]


def fit_te_mappings(
    df: pd.DataFrame,
    target_col: str = "TAXITIME_SEC_mvt",
    smoothing: int = 20,
) -> dict:
    """Learn all TE mappings from the training data.

    Return format:
        {
            "<te_col_name>": {
                "global_mean": float,
                "group_key_cols": list[str],  # 1 for single variable, 2 for two-way
                "mapping": dict,  # group_value -> encoded_value (after smoothing)
            },
            ...
        }
    """
    mappings = {}
    for kind, items in TE_SPECS:
        if kind == "single":
            col, name = items[0]
            if col not in df.columns:
                continue
            df_temp, _, info = target_encode_smoothed(
                df, df, [col], target_col, smoothing=smoothing, new_col_name=name,
            )
            mappings[name] = {
                "global_mean": info["global_mean"],
                "group_key_cols": [col],
                "mapping": info["mapping"]["values"],
            }
        else:
            cols, name = items[0]
            if not all(c in df.columns for c in cols):
                continue
            df_temp, _, info = target_encode_smoothed(
                df, df, cols, target_col, smoothing=smoothing, new_col_name=name,
            )
            mappings[name] = {
                "global_mean": info["global_mean"],
                "group_key_cols": cols,
                "mapping": info["mapping"]["values"],
            }
    return mappings


def add_target_encodings_with_mapping(
    df: pd.DataFrame,
    target_col: str = "TAXITIME_SEC_mvt",
    smoothing: int = 20,
) -> pd.DataFrame:
    """Called at training time: fit TE mapping + apply to df itself."""
    df = df.copy()
    mappings = fit_te_mappings(df, target_col, smoothing)
    df = apply_te_mappings(df, mappings)
    return df


def apply_te_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Apply the pretrained TE mapping to df (called during ranking inference)."""
    df = df.copy()
    for col_name, info in mappings.items():
        global_mean = info["global_mean"]
        key_cols = info["group_key_cols"]
        mapping = info["mapping"]

        if len(key_cols) == 1:
            df[col_name] = df[key_cols[0]].astype(str).map(mapping).astype(float).fillna(global_mean)
        else:
            # Two variables: join the multiple columns into a tuple as the key
            keys = list(zip(*(df[c].astype(str) for c in key_cols)))
            df[col_name] = pd.Series(keys).map(mapping).astype(float).fillna(global_mean).values

    return df


def add_target_encodings(
    df: pd.DataFrame,
    target_col: str = "TAXITIME_SEC_mvt",
    smoothing: int = 20,
) -> pd.DataFrame:
    """Wrapper compatible with the old API, equals add_target_encodings_with_mapping."""
    return add_target_encodings_with_mapping(df, target_col, smoothing)


# ---------------------------------------------------------------------------
# 4. Full pipeline (basic + aggregate + domain + target encoding)
# ---------------------------------------------------------------------------

def build_features_full(
    df: pd.DataFrame,
    include_aggregate: bool = True,
    include_target_encoding: bool = True,
) -> pd.DataFrame:
    """Full feature pipeline.

    Args:
        df: raw data
        include_aggregate: whether to add congestion aggregation features
        include_target_encoding: whether to add target encoding (requires TAXITIME_SEC_mvt to be non-null)
    """
    from src.features import build_features
    from src.features_aggregate import add_aggregate_features

    df = build_features(df)
    df = add_domain_features(df)
    if include_aggregate:
        df = add_aggregate_features(df)
    if include_target_encoding and "TAXITIME_SEC_mvt" in df.columns:
        non_null = df["TAXITIME_SEC_mvt"].notna().sum()
        if non_null > 100:  # only run TE with at least 100 valid samples
            df = add_target_encodings(df)
    return df


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== features_domain module self-test ===\n")

    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "STAND_mvt": np.random.choice(["A01", "A02", "B11", "B23", "T2", None], n),
        "RUNWAY_mvt": np.random.choice(["07L", "07R", "25L", "25R", "18"], n),
        "ADEP_mvt": np.random.choice(["EDDF", "EGLL", "LFPG"], n),
        "AIRCRAFT_OPERATOR_flt": np.random.choice(["DLH", "BAW", "RYR"], n),
        "AIRCRAFT_TYPE_mvt": np.random.choice(["A320", "B777"], n),
        "WK_TBL_CAT_flt": np.random.choice(["M", "H"], n),
        "TAXITIME_SEC_mvt": np.random.normal(600, 100, n),
    })

    print("1. Structural domain features:")
    df = add_domain_features(df)
    print(df[["STAND_mvt", "stand_zone", "RUNWAY_mvt", "runway_family", "runway_side"]].head(10))

    print("\n2. Target encoding:")
    df = add_target_encodings(df)
    new_cols = [c for c in df.columns if c.endswith("_te")]
    print(f"New TE columns: {new_cols}")
    print(df[new_cols].describe().round(1))
