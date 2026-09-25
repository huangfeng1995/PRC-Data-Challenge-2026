"""
拉取 Open-Meteo archive hourly 天气数据
====================================

用途: 为 10 个比赛机场拉取逐小时天气特征(2025 全年 + 2026-01 + 2026-07),
存到 data/external/weather/weather_hourly.parquet。

数据来源: Open-Meteo Archive API (CC BY 4.0)
  https://open-meteo.com/en/docs/historical-weather-api

坐标读取: data/external/xplane/runways.csv (跑道端点均值)

可用 hourly 字段 (已实测 2025-01 EDDF,48/48 有效值):
  temperature_2m, relative_humidity_2m, dewpoint_2m, apparent_temperature,
  precipitation, rain, snowfall, snow_depth, weather_code, cloud_cover,
  pressure_msl, surface_pressure,
  wind_speed_10m, wind_direction_10m, wind_gusts_10m

archive hourly 不可用 (HTTP 200 但全 None 或 400):
  visibility, cape, cloud_base (HTTP 200 但 None)
  freezing_level (HTTP 400 invalid)
  已跳过,不在请求里出现。

用法:
    python3 scripts/fetch_weather.py
输出:
    data/external/weather/weather_hourly.parquet
    data/external/weather/_coverage_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XP_RUNWAYS = REPO / "data" / "external" / "xplane" / "runways.csv"
WEATHER_DIR = REPO / "data" / "external" / "weather"
OUT_PARQUET = WEATHER_DIR / "weather_hourly.parquet"
OUT_COVERAGE = WEATHER_DIR / "_coverage_report.json"

BASE = "https://archive-api.open-meteo.com/v1/archive"

# 已实测可用的 15 个 hourly 字段
HOURLY_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "snow_depth", "weather_code", "cloud_cover",
    "pressure_msl", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]

# 已实测不可用 (HTTP 200 但全 None / 或 400):
SKIPPED_FIELDS = ["visibility", "cape", "cloud_base", "freezing_level"]


def load_airport_coords() -> dict[str, tuple[float, float]]:
    """从 runways.csv 读 10 机场 (lat,lon),跑道端点均值。"""
    pts = defaultdict(list)
    with open(XP_RUNWAYS) as f:
        for r in csv.DictReader(f):
            pts[r["icao"]].append((float(r["lat1"]), float(r["lon1"])))
            pts[r["icao"]].append((float(r["lat2"]), float(r["lon2"])))
    coords = {}
    for icao, ls in pts.items():
        lat = sum(p[0] for p in ls) / len(ls)
        lon = sum(p[1] for p in ls) / len(ls)
        coords[icao] = (round(lat, 5), round(lon, 5))
    return coords


def fetch_one_month(icao: str, lat: float, lon: float,
                    start: str, end: str, ctx: ssl.SSLContext,
                    retries: int = 3, delay: float = 0.3) -> list[dict] | None:
    params = "&".join(f"hourly={f}" for f in HOURLY_FIELDS)
    url = (f"{BASE}?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={end}&{params}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=45, context=ctx) as r:
                j = json.loads(r.read())
            hourly = j.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                print(f"  ⚠ 空响应 {icao} {start}~{end}", file=sys.stderr)
                return []
            rows = []
            for i, t in enumerate(times):
                row = {"icao": icao, "time": t}
                for f in HOURLY_FIELDS:
                    row[f] = hourly.get(f, [None] * len(times))[i]
                rows.append(row)
            return rows
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"  重试 {attempt + 1}/{retries} {icao} {start}~{end}: {e}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"  异常 {icao} {start}~{end}: {e}", file=sys.stderr)
            return None
        time.sleep(delay)
    print(f"  ❌ 放弃 {icao} {start}~{end}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-delay", type=float, default=0.3)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    coords = load_airport_coords()
    print(f"坐标 ({len(coords)} 机场,来自 {XP_RUNWAYS}):")
    for icao, (lat, lon) in sorted(coords.items()):
        print(f"  {icao}: {lat}, {lon}")

    # 月份列表:2025-01 ~ 2025-12 + 2026-01 + 2026-07
    # 每月起止用最后一天 31 / 30 / 29 / 28 都不影响 archive (多拉 1 小时不会报错)
    months = []  # (start, end) 闭区间 (YYYY-MM-DD)
    month_end_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                      7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    for y, ms in [(2025, range(1, 13)), (2026, [1, 7])]:
        for m in ms:
            months.append((f"{y}-{m:02d}-01", f"{y}-{m:02d}-{month_end_days[m]}"))
    print(f"\n月份范围 ({len(months)} 段): {months[0][0]}~{months[-1][1]}")

    ctx = ssl.create_default_context()
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    report = {"hourly_fields": HOURLY_FIELDS,
              "skipped_fields": SKIPPED_FIELDS,
              "n_requests_expected": len(coords) * len(months),
              "requests_succeeded": 0,
              "requests_failed": 0,
              "per_icao": {},
              "coverage_notes": "数据来源 Open-Meteo Archive (CC BY 4.0), archive hourly 端点不支持 visibility/cape/cloud_base/freezing_level, 已跳过"}

    for icao, (lat, lon) in sorted(coords.items()):
        icao_rows = []
        icao_rep = {"months_ok": [], "months_fail": [], "n_rows_per_month": {}}
        for s, e in months:
            rows = fetch_one_month(icao, lat, lon, s, e, ctx, retries=args.retries, delay=args.rate_delay)
            if rows is None:
                icao_rep["months_fail"].append(f"{s}~{e}")
                report["requests_failed"] += 1
            else:
                icao_rep["months_ok"].append(f"{s}~{e}")
                icao_rep["n_rows_per_month"][f"{s}~{e}"] = len(rows)
                report["requests_succeeded"] += 1
                icao_rows.extend(rows)
            time.sleep(args.rate_delay)
        all_rows.extend(icao_rows)

        # 覆盖率:每机场总体 + 按字段
        if icao_rows:
            df_icao = pd.DataFrame(icao_rows)
            icao_rep["total_rows"] = len(df_icao)
            icao_rep["first_time"] = df_icao["time"].min()
            icao_rep["last_time"] = df_icao["time"].max()
            icao_rep["field_nonnull_ratio"] = {
                f: round(df_icao[f].notna().mean(), 4) for f in HOURLY_FIELDS
            }
            icao_rep["month_ok_rate"] = round(len(icao_rep["months_ok"]) / len(months), 4)
        report["per_icao"][icao] = icao_rep
        print(f"  ✓ {icao}: {len(icao_rows)} rows, {len(icao_rep['months_ok'])}/{len(months)} months OK",
              file=sys.stderr)

    if not all_rows:
        print("❌ 没有任何数据,放弃写 parquet", file=sys.stderr)
        report["error"] = "no rows fetched"
        OUT_COVERAGE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["icao", "time"]).reset_index(drop=True)

    # 字段类型优化: weather_code / cloud_cover 整数,其他 float
    for f in HOURLY_FIELDS:
        if f in ("weather_code", "cloud_cover"):
            df[f] = pd.to_numeric(df[f], errors="coerce").astype("Int64")
        else:
            df[f] = pd.to_numeric(df[f], errors="coerce").astype("float32")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n✅ {OUT_PARQUET}: {len(df):,} rows × {len(df.columns)} cols", file=sys.stderr)
    print(f"   每机场平均 {len(df)/len(coords):.0f} rows", file=sys.stderr)
    print(f"   每机场×月份平均 {len(df)/(len(coords)*len(months)):.0f} rows", file=sys.stderr)

    report["total_rows"] = len(df)
    report["n_months_per_icao"] = len(months)
    report["per_field_global_nonnull"] = {
        f: round(df[f].notna().mean(), 4) for f in HOURLY_FIELDS
    }
    OUT_COVERAGE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"📄 {OUT_COVERAGE}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
