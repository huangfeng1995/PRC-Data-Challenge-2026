"""
将逐小时天气数据锚定到航班 (按 MVT_TIME_UTC)
============================================

锚定规则:
  每航班 = (ADEP_mvt, MVT_TIME_UTC)  取所在 UTC 小时的天气 (round down 到整点)
  不锚 BLOCK_TIME (业务上 BLOCK_TIME 早于 MVT_TIME, 代表起飞前,
                   我们需要的是"当前航班"的天气 —— 起飞时刻更贴近 taxi-out)
  不插值到亚小时精度 (样本上逐小时密度够, 整点值直接 merge)

输出特征 (逐航班 DataFrame 新增这些列):
  weather_temperature      (float)  temperature_2m
  weather_dewpoint         (float)  dewpoint_2m
  weather_dewpoint_deficit (float)  temperature - dewpoint (除冰关键)
  weather_snowfall         (float)  snowfall (mm/h)
  weather_snow_depth       (float)  snow_depth (cm)
  weather_precipitation    (float)  precipitation (mm/h)
  weather_wind_speed       (float)  wind_speed_10m (km/h)
  weather_wind_gust        (float)  wind_gusts_10m (km/h)
  weather_cloud_cover      (int)    cloud_cover (0~100)
  weather_pressure         (float)  surface_pressure (hPa)
  weather_code             (int)    weather_code (WMO 0~99 编码)

Open-Meteo archive hourly 不支持的已跳过:
  visibility, cape, cloud_base, freezing_level
  → 报告里写清, 不尝试补齐, 不伪造

数据来源: data/external/weather/weather_hourly.parquet
锚定字段: ADEP_mvt + MVT_TIME_UTC (两键)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WEATHER_HOURLY = Path(__file__).resolve().parents[1] / "data" / "external" / "weather" / "weather_hourly.parquet"

HOURLY_TO_FEATURE = [
    # (src col -> out col, dtype)
    ("temperature_2m",       "weather_temperature",      "float32"),
    ("dewpoint_2m",          "weather_dewpoint",         "float32"),
    ("precipitation",        "weather_precipitation",    "float32"),
    ("snowfall",             "weather_snowfall",         "float32"),
    ("snow_depth",           "weather_snow_depth",       "float32"),
    ("wind_speed_10m",       "weather_wind_speed",       "float32"),
    ("wind_gusts_10m",       "weather_wind_gust",        "float32"),
    ("cloud_cover",          "weather_cloud_cover",      "Int64"),
    ("surface_pressure",     "weather_pressure",         "float32"),
    ("weather_code",         "weather_code",             "Int64"),
]


def add_weather_features(df_flight: pd.DataFrame, weather_path: Path | str = WEATHER_HOURLY,
                         icao_col: str = "ADEP_mvt", time_col: str = "MVT_TIME_UTC_mvt") -> pd.DataFrame:
    """给航班 DataFrame 追加天气特征。

    Args:
        df_flight: 需包含 `ADEP_mvt` / `MVT_TIME_UTC_mvt` 两列 (或用 icao_col/time_col 指定)
        weather_path: 逐小时天气 parquet, schema (icao, time, *_hourly)

    Returns:
        原 df_flight 的副本 + 新增天气列
        未匹配到天气的行: 新增列为 NaN (正常是因为数据范围没覆盖)
    """
    wdf = pd.read_parquet(weather_path)
    wdf["icao"] = wdf["icao"].astype(str)
    wdf["time"] = pd.to_datetime(wdf["time"], utc=True)

    out = df_flight.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)

    # 锚定: 航班时间 round down 到 UTC 整点, 与逐小时天气 time 对齐
    out["_weather_key_time"] = out[time_col].dt.floor("h")
    out[icao_col] = out[icao_col].astype(str)

    # 列重命名
    keep_w_cols = ["icao", "time"] + [src for src, _, _ in HOURLY_TO_FEATURE]
    wdf_r = wdf[keep_w_cols].rename(columns={"time": "_weather_key_time", "icao": icao_col})

    out = out.merge(wdf_r, on=[icao_col, "_weather_key_time"], how="left")

    # 重命名 hourly -> weather_*
    for src, out_col, dtype in HOURLY_TO_FEATURE:
        out[out_col] = out[src].astype(dtype)
        if src != out_col:
            out.drop(columns=[src], inplace=True)

    # 派生: 温露差 (除冰关键: 负差 → 过饱和 → 可能结冰)
    out["weather_dewpoint_deficit"] = (
        out["weather_temperature"] - out["weather_dewpoint"]
    ).astype("float32")

    out.drop(columns=["_weather_key_time"], inplace=True)

    return out


def report_merge(df_before: pd.DataFrame, df_after: pd.DataFrame,
                 icao_col: str = "ADEP_mvt") -> dict:
    """报告 merge 覆盖率, 不打印, 返回 dict."""
    result = {
        "total_rows": len(df_after),
    }
    feature_cols = [o for _, o, _ in HOURLY_TO_FEATURE] + ["weather_dewpoint_deficit"]

    # 总体覆盖
    match_mask = df_after["weather_temperature"].notna()  # 用第一个 feature 作 proxy
    result["match_overall_n"] = int(match_mask.sum())
    result["match_overall_rate"] = round(float(match_mask.mean()), 4)

    # 每机场覆盖
    per_icao = {}
    for icao, grp in df_after.groupby(icao_col):
        m = grp["weather_temperature"].notna().sum()
        per_icao[str(icao)] = {"total": len(grp), "matched": int(m), "rate": round(float(m / len(grp)), 4)}
    result["per_icao"] = per_icao

    # 字段非空率 + 量级
    stats = {}
    for c in feature_cols:
        s = df_after[c]
        stats[c] = {
            "nonnull_rate": round(float(s.notna().mean()), 4),
            "mean": round(float(s.mean(skipna=True)), 3),
            "std": round(float(s.std(skipna=True)), 3),
            "min": round(float(s.min(skipna=True)), 3),
            "max": round(float(s.max(skipna=True)), 3),
        }
    result["field_stats"] = stats

    # 没匹配上的原因分布 (哪个机场有航班, 但该机场全部 miss)
    all_weather_icaos = set(pd.read_parquet(WEATHER_HOURLY, columns=["icao"])["icao"].unique())
    miss_icaos = set(df_after[icao_col].unique()) - all_weather_icaos
    if miss_icaos:
        result["miss_reason_no_weather_data_icaos"] = sorted(miss_icaos)
    return result


if __name__ == "__main__":
    # 最小自测: 读 sample, 拉 weather, 打印报告
    from src.data_loader import load_month, filter_departures, parse_time_columns

    months = [
        "data/sample/training_2025-01-01_2025-02-01.parquet",
        "data/sample/training_2025-02-01_2025-03-01.parquet",
        "data/sample/training_2025-03-01_2025-04-01.parquet",
        "data/sample/training_2025-07-01_2025-08-01.parquet",  # 2025-07 (与 ranking 同时段)
    ]
    dfs = []
    for m in months:
        df = load_month(m)
        df = filter_departures(parse_time_columns(
            df, cols=["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                      "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]
        ))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"sample DEP rows: {len(df)}", file=sys.stderr)

    wdf = add_weather_features(df)
    rep = report_merge(df, wdf)

    print(f"\n=== Merge 覆盖: {rep['match_overall_n']}/{rep['total_rows']} ({rep['match_overall_rate']:.1%}) ===")
    print("Per icao:")
    for icao, v in rep["per_icao"].items():
        print(f"  {icao}: {v['matched']}/{v['total']} ({v['rate']:.1%})")

    print(f"\n=== 字段统计 ===")
    for c, s in rep["field_stats"].items():
        print(f"  {c:26} nn={s['nonnull_rate']:.1%}  μ={s['mean']:>8.2f}  σ={s['std']:>7.2f}  min={s['min']:>7.2f}  max={s['max']:>7.2f}")

    # 单独看冬季(2025-01) 覆盖 + 数值
    df_w = filter_departures(parse_time_columns(
        load_month("data/sample/training_2025-01-01_2025-02-01.parquet"),
        cols=["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
              "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]
    ))
    wdf_w = add_weather_features(df_w)
    print(f"\n=== 2025-01 (冬季除冰/风暴) 覆盖 + 数值 ===")
    rw = report_merge(df_w, wdf_w)
    for c, s in rw["field_stats"].items():
        print(f"  {c:26} nn={s['nonnull_rate']:.1%}  μ={s['mean']:>8.2f}  min={s['min']:>7.2f}  max={s['max']:>7.2f}")

    print("\n冬季异常提示: snowfall/snow_depth 在 01 月应为正值 (尤其 EGLL/LSZH)")
    print("              LEMD/LIRF 冬季应该温和, 极端值 < 30 mm snowfall 合理")
