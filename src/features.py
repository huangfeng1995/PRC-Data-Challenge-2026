"""
特征工程模块(baseline 版:只基础特征)
====================================

这一版只做"航班自身信息"的特征,不含拥堵聚合。
等 baseline 跑通后,在 src/features_aggregate.py 里加拥堵特征。

防泄漏原则(非常重要):
- ✅ 可用:SCHED_TIME_UTC_mvt, IOBT_flt, EOBT_1_flt(计划时间,实时可拿)
- ⚠️ 慎用:AOBT_3_flt(实际推出 = taxi-out 起点),仅做训练目标/事后聚合
- ❌ 禁用:MVT_TIME_UTC_mvt(实际起飞 = taxi-out 终点,就是标签)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 时间特征(基于计划时间,无泄漏)
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, time_col: str = "SCHED_TIME_UTC_mvt") -> pd.DataFrame:
    """从计划时间提取时间维度的特征。

    注意:用 SCHED_TIME_UTC_mvt 而不是 AOBT/MVT 的实际时间,否则会泄漏。
    """
    if time_col not in df.columns:
        raise KeyError(f"缺少时间列 {time_col}")

    # 转 UTC(无时区)以便提取 hour/dow 等
    t = df[time_col].dt.tz_convert("UTC")

    df = df.copy()
    df["sched_hour"] = t.dt.hour.astype("int8")
    df["sched_dow"] = t.dt.dayofweek.astype("int8")  # Monday=0, Sunday=6
    df["sched_month"] = t.dt.month.astype("int8")
    df["sched_day"] = t.dt.day.astype("int8")
    df["sched_minute"] = t.dt.minute.astype("int8")

    # 是否周末 / 是否典型高峰小时
    df["is_weekend"] = (df["sched_dow"] >= 5).astype("int8")
    df["is_morning_peak"] = df["sched_hour"].isin([6, 7, 8, 9]).astype("int8")
    df["is_evening_peak"] = df["sched_hour"].isin([17, 18, 19, 20]).astype("int8")

    # 周期性编码(让模型理解 23 点和 0 点是相邻的)
    df["hour_sin"] = np.sin(2 * np.pi * df["sched_hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["sched_hour"] / 24).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["sched_dow"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["sched_dow"] / 7).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * df["sched_month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["sched_month"] / 12).astype("float32")

    return df


# ---------------------------------------------------------------------------
# 空间特征
# ---------------------------------------------------------------------------

def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取起飞机场 + 落地机场相关的简单特征。"""
    df = df.copy()

    # 国内/国际(粗略判断:同国家算国内,这里用 ICAO 前缀粗分)
    # 完整做法需要机场-国家映射表,这里先用简单的:欧洲内 = 国际
    df["is_domestic_estimate"] = (df["ADEP_mvt"].str[:2] == df["ADES_mvt"].str[:2]).astype("int8")

    # 起飞机场 x 计划小时 组合(隐含"机场高峰模式"信息)
    # 用字符串拼接,LightGBM 会作为类别处理
    df["airport_hour"] = df["ADEP_mvt"].astype(str) + "_" + df["sched_hour"].astype(str)

    return df


# ---------------------------------------------------------------------------
# 计划性特征(基于 IOBT/SCHED 的偏差)
# ---------------------------------------------------------------------------

def add_planning_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取与"计划-估计"相关的时间差。

    这些是结构化特征,反映航班在系统里的"状态":
    - IOBT 到 SCHED 的差 = 计划推出到计划起飞的滑行预期
    - EOBT 到 SCHED 的差 = FPL 估计到计划起飞的偏差
    """
    df = df.copy()

    # 计划推出时间(sched_hour 已经在前面算过)
    if "IOBT_flt" in df.columns:
        df["iobt_to_sched_sec"] = (
            (df["SCHED_TIME_UTC_mvt"] - df["IOBT_flt"]).dt.total_seconds().astype("float32")
        )

    if "EOBT_1_flt" in df.columns:
        df["eobt_to_sched_sec"] = (
            (df["SCHED_TIME_UTC_mvt"] - df["EOBT_1_flt"]).dt.total_seconds().astype("float32")
        )
        # 估计推出 vs 初始计划的差,反映"航班计划被更新了多少"
        df["iobt_eobt_diff_sec"] = (
            (df["EOBT_1_flt"] - df["IOBT_flt"]).dt.total_seconds().astype("float32")
        )

    # 是否改航
    if "ADES_flt" in df.columns and "ADES_FILED_flt" in df.columns:
        df["is_diverted"] = (df["ADES_flt"] != df["ADES_FILED_flt"]).astype("int8")

    return df


# ---------------------------------------------------------------------------
# 类别字段规范化(LightGBM 用 category dtype 效果最好)
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
    "airport_hour",  # 组合特征
]


def to_categorical(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """把指定列转成 pandas category 类型(LightGBM 原生支持,内存也省)。"""
    cols = cols or CATEGORICAL_COLS
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------------------
# 主入口:完整特征流水线
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """跑完所有特征工程的入口函数。

    输入:从 data_loader 加载的原始 DataFrame(DEP only, 时间已 parse)
    输出:可喂给 LightGBM 的特征矩阵
    """
    df = add_time_features(df)
    df = add_route_features(df)
    df = add_planning_features(df)
    df = to_categorical(df)
    return df


# ---------------------------------------------------------------------------
# 目标重缩放(从 2024 第 1 名 team_likable_jelly 学到)
# ---------------------------------------------------------------------------
#
# 他们的发现:不同机场的 taxi-out 平均值差异巨大(FRA 比中小机场长 ~50%)。
# 如果按 (target - group_min) / (group_max - group_min) 缩放,
# 所有机场的"目标"被拉平到 [0, 1],模型更容易学。
#
# 注意:这只能用于训练(因为要用 target),推理时需要保存的 mapping。
# 我们做成"可选"——默认关闭,你能用就用。
#

def fit_target_rescaler(
    df_train: pd.DataFrame,
    target_col: str = "TAXITIME_SEC_mvt",
    group_col: str = "ADEP_mvt",
) -> dict:
    """学"按机场分组的 target 重缩放"参数。

    对每个机场:
        mean, std → 用于 (target - mean) / std 标准化

    返回可序列化的 mapping。
    """
    grouped = df_train.groupby(group_col, observed=True)[target_col].agg(["mean", "std"])
    # 防止 std=0(某机场只有一条记录)
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
    """应用重缩放 mapping 到目标变量,新增 out_col 列。"""
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
    """把"重缩放空间"的预测还原回原始尺度。

    用在推理时:模型预测出 rescaled 值,我们反变换回去。
    """
    per_group = mapping["per_group"]
    global_mean = mapping["global_mean"]
    global_std = mapping["global_std"]

    out = np.empty_like(predictions_rescaled, dtype="float64")
    for i, g in enumerate(group_values):
        params = per_group.get(g, {"mean": global_mean, "std": global_std})
        out[i] = predictions_rescaled[i] * params["std"] + params["mean"]
    return out


# baseline 用到的特征列清单(不含聚合,不含 leakage)
BASELINE_FEATURE_COLS = [
    # 时间
    "sched_hour", "sched_dow", "sched_month", "sched_day", "sched_minute",
    "is_weekend", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # 空间/路线
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "airport_hour",
    "is_domestic_estimate",
    # 飞机/航班
    "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt", "is_diverted",
    # 计划性
    "iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
]


if __name__ == "__main__":
    # 模块自测:加载数据 + 跑特征
    import sys
    from src.data_loader import load_training

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = load_training(data_dir)
    print(f"原始: {df.shape}")

    df_feat = build_features(df)
    print(f"特征后: {df_feat.shape}")

    # 检查所有 baseline 特征都在
    missing = [c for c in BASELINE_FEATURE_COLS if c not in df_feat.columns]
    if missing:
        print(f"⚠️ 缺失特征: {missing}")
    else:
        print(f"✅ 所有 {len(BASELINE_FEATURE_COLS)} 个 baseline 特征都齐全")

    print(f"\n目标变量分布:")
    print(df_feat["TAXITIME_SEC_mvt"].describe())
