"""
领域(domain)特征模块
=====================

从 2025 第 1 名的"基于领域知识"学到:
- 重型机滑行慢
- 不同航司的滑行规则不同
- 登机口 vs 跑道距离影响 taxi-out
- 航司 × 机场 交互

我们加入:
1. 登机口区(从 STAND 提取首字母,如 A 区、B 区)
2. 跑道家族(同一方向跑道归为一类,如 "07L/07R" → "07")
3. 航司 × 机场 的目标编码(用历史 taxi-out 中位数)
4. 飞机类型 × 机场 的目标编码

⚠️ 目标编码需要在"过去的数据"上做(避免 leakage)。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. 登机口区 / 跑道家族 / 跑道方向
# ---------------------------------------------------------------------------

def add_stand_zone(df: pd.DataFrame) -> pd.DataFrame:
    """从 STAND_mvt 提取登机口区。

    例如 "A12" → "A", "B23" → "B", "T2" → "T", None → "UNKNOWN"
    """
    df = df.copy()
    if "STAND_mvt" not in df.columns:
        df["stand_zone"] = "UNKNOWN"
        return df
    stand = df["STAND_mvt"].astype(str).fillna("UNKNOWN")
    # 取第一个字符(字母部分)
    df["stand_zone"] = stand.str.extract(r"^([A-Z])", expand=False).fillna("UNKNOWN")
    return df


def add_runway_family(df: pd.DataFrame) -> pd.DataFrame:
    """从 RUNWAY_mvt 提取跑道家族和方向。

    例如 "07L" → ("07", "L"), "25R" → ("25", "R"), "18" → ("18", "")
    - runway_family: 同一方向的所有跑道归一类(如 "07L"/"07R" → "07")
    - runway_side: L/R/C(左/右/中央)
    """
    df = df.copy()
    if "RUNWAY_mvt" not in df.columns:
        df["runway_family"] = "UNKNOWN"
        df["runway_side"] = "UNKNOWN"
        return df
    rw = df["RUNWAY_mvt"].astype(str).fillna("UNKNOWN")
    # 数字部分 + 字母部分
    m = rw.str.extract(r"^(\d{1,2})([LRC]?)$", expand=True)
    df["runway_family"] = m[0].fillna("UNKNOWN")
    df["runway_side"] = m[1].fillna("")
    return df


# ---------------------------------------------------------------------------
# 2. 目标编码(target encoding)
# ---------------------------------------------------------------------------

def target_encode_smoothed(
    df_train: pd.DataFrame,
    df_target: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    smoothing: int = 20,
    new_col_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """平滑目标编码。

    思路:把每个 (group_cols 组合) 的平均 target 作为特征值,但对小样本做平滑
        encoded = (n * group_mean + smoothing * global_mean) / (n + smoothing)

    参数:
        df_train: 用来学编码的训练数据(带 target)
        df_target: 要应用编码的目标数据(可以就是 df_train 自己,用来加新列)
        group_cols: 分组字段列表,如 ["AIRCRAFT_OPERATOR_flt", "ADEP_mvt"]
        target_col: 目标变量名
        smoothing: 平滑系数,越大越保守(向全局均值靠拢)
        new_col_name: 输出列名,默认 "_".join(group_cols) + "_te"

    返回:
        (df_train, df_target, mapping_dict)
        df_train 和 df_target 都加了 new_col_name 列
        mapping_dict 可以序列化保存,推理时用
    """
    global_mean = df_train[target_col].mean()

    # 确保 target 是数值类型(categorical 没法算 mean)
    if hasattr(df_train[target_col], 'cat'):
        target_numeric = df_train[target_col].astype(float)
    else:
        target_numeric = df_train[target_col]

    # 学每个组的 mean 和 count
    grouped = df_train.assign(**{target_col: target_numeric}).groupby(
        group_cols, observed=True
    )[target_col].agg(["mean", "count"])
    grouped.columns = ["__te_mean__", "__te_count__"]

    # 平滑
    grouped["__te_encoded__"] = (
        (grouped["__te_mean__"] * grouped["__te_count__"] + smoothing * global_mean)
        / (grouped["__te_count__"] + smoothing)
    )

    mapping = grouped["__te_encoded__"].to_dict()
    # 如果 group_cols 是多个,mapping 的 key 是 tuple
    if isinstance(next(iter(mapping.keys())), tuple):
        mapping = {k: v for k, v in mapping.items()}

    new_col = new_col_name or "_".join(group_cols) + "_te"

    # 应用到训练和目标数据
    if len(group_cols) == 1:
        df_train[new_col] = df_train[group_cols[0]].map(mapping).astype(float).fillna(global_mean)
        df_target[new_col] = df_target[group_cols[0]].map(mapping).astype(float).fillna(global_mean)
    else:
        # 多列:把 group 拼成 tuple 当作 key
        train_keys = list(zip(*(df_train[c] for c in group_cols)))
        target_keys = list(zip(*(df_target[c] for c in group_cols)))
        df_train[new_col] = pd.Series(train_keys).map(mapping).astype(float).fillna(global_mean).values
        df_target[new_col] = pd.Series(target_keys).map(mapping).astype(float).fillna(global_mean).values

    # 把 mapping 转成可序列化的 dict
    if isinstance(next(iter(mapping.keys())), tuple):
        mapping_save = {"__keys__": list(group_cols), "values": {str(k): v for k, v in mapping.items()}}
    else:
        mapping_save = {"__keys__": group_cols[0], "values": dict(mapping)}

    return df_train, df_target, {"mapping": mapping_save, "global_mean": global_mean, "new_col": new_col}


# ---------------------------------------------------------------------------
# 3. 主入口:完整 domain 特征流水线
# ---------------------------------------------------------------------------

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """加所有结构性 domain 特征(不依赖历史 target 的部分)。

    包括:
        - stand_zone: 登机口区
        - runway_family: 跑道家族
        - runway_side: 跑道侧
        - aircraft_type_size: 大/中/小(从 wake turbulence category 推断)

    目标编码(train target encoding)需要在 add_target_encodings() 里单独做,
    因为它需要历史 taxi-out,不能直接用于推理(除非加载 mapping)。
    """
    df = add_stand_zone(df)
    df = add_runway_family(df)
    return df


# TE 拟合的目标列组(单变量 + 双变量交叉)
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
    """从训练数据学所有 TE mapping。

    返回格式:
        {
            "<te_col_name>": {
                "global_mean": float,
                "group_key_cols": list[str],  # 单变量 1 个,双变量 2 个
                "mapping": dict,  # group_value → encoded_value(平滑后)
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
    """训练时调用:fit TE mapping + apply 到 df 自己。"""
    df = df.copy()
    mappings = fit_te_mappings(df, target_col, smoothing)
    df = apply_te_mappings(df, mappings)
    return df


def apply_te_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """应用预训练的 TE mapping 到 df(ranking 推理时调用)。"""
    df = df.copy()
    for col_name, info in mappings.items():
        global_mean = info["global_mean"]
        key_cols = info["group_key_cols"]
        mapping = info["mapping"]

        if len(key_cols) == 1:
            df[col_name] = df[key_cols[0]].astype(str).map(mapping).astype(float).fillna(global_mean)
        else:
            # 双变量:把多列拼成 tuple 作为 key
            keys = list(zip(*(df[c].astype(str) for c in key_cols)))
            df[col_name] = pd.Series(keys).map(mapping).astype(float).fillna(global_mean).values

    return df


def add_target_encodings(
    df: pd.DataFrame,
    target_col: str = "TAXITIME_SEC_mvt",
    smoothing: int = 20,
) -> pd.DataFrame:
    """兼容旧 API 的包装,等于 add_target_encodings_with_mapping。"""
    return add_target_encodings_with_mapping(df, target_col, smoothing)


# ---------------------------------------------------------------------------
# 4. 完整流水线(含基础 + 聚合 + domain + 目标编码)
# ---------------------------------------------------------------------------

def build_features_full(
    df: pd.DataFrame,
    include_aggregate: bool = True,
    include_target_encoding: bool = True,
) -> pd.DataFrame:
    """完整特征流水线。

    参数:
        df: 原始数据
        include_aggregate: 是否加拥堵聚合
        include_target_encoding: 是否加目标编码(需要 TAXITIME_SEC_mvt 非空)
    """
    from src.features import build_features
    from src.features_aggregate import add_aggregate_features

    df = build_features(df)
    df = add_domain_features(df)
    if include_aggregate:
        df = add_aggregate_features(df)
    if include_target_encoding and "TAXITIME_SEC_mvt" in df.columns:
        non_null = df["TAXITIME_SEC_mvt"].notna().sum()
        if non_null > 100:  # 至少 100 条有效样本才做 TE
            df = add_target_encodings(df)
    return df


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== features_domain 模块自测 ===\n")

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

    print("1. 结构性 domain 特征:")
    df = add_domain_features(df)
    print(df[["STAND_mvt", "stand_zone", "RUNWAY_mvt", "runway_family", "runway_side"]].head(10))

    print("\n2. 目标编码:")
    df = add_target_encodings(df)
    new_cols = [c for c in df.columns if c.endswith("_te")]
    print(f"新增 TE 列: {new_cols}")
    print(df[new_cols].describe().round(1))
