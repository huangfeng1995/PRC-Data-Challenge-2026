"""
阶段1 队列特征单元测试(修正版)
=============================

ref_i = AOBT_3_flt 优先,缺失回退 EOBT_1_flt。

构造 6 行 DEP + 2 行 ARR,EDDF 同一机场。
已知每行的 AOBT/MVT,手算每行的 queue_n / heavy_n / dep_mvt_cnt_* / arr_mvt_cnt_*。

期望:
- a (ref=base-100): queue=0, heavy=0
- b (ref=base-50):  queue=1 (a), heavy=0
- c (ref=base-30):  queue=2 (a,b), heavy=0 (a 是 M)
- d (ref=base+20):  queue=3 (a,b,c), heavy=2 (b=H, c=J)
- e (ref=base-200): queue=0 (所有 AOBT > -200)
- f (AOBT=NaN, ref=EOBT=base, estimated=1): queue=3 (a,b,c), heavy=2

dep_mvt_cnt_airport_*: 所有 DEP 的 MVT 都在 ref 之后,无 DEP MVT 落入 ref 之前的窗口 → 全 0

ARR 计数(只对 DEP 行算,ARR 行的 arr_mvt_cnt_* 为 NaN):
- 对 f (ref=base): ARR x MVT=-10 在 (-10, 0] → 1; ARR y MVT=-60 < -15 ✗ → 0/15m=1
- 对其他 DEP 行 ref 远离 base, ARR x/y 都不在窗口 → 0
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

from src.features_queue import add_queue_features


def make_test_df():
    base = pd.Timestamp("2025-06-15 08:00:00", tz="UTC")
    rows = [
        {"PHASE_mvt": "DEP", "MVT_ID_mvt": "a", "ADEP_mvt": "EDDF", "RUNWAY_mvt": "07L",
         "AOBT_3_flt": base + pd.Timedelta(minutes=-100), "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=50),
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "M"},
        {"PHASE_mvt": "DEP", "MVT_ID_mvt": "b", "ADEP_mvt": "EDDF", "RUNWAY_mvt": "07L",
         "AOBT_3_flt": base + pd.Timedelta(minutes=-50), "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=200),
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "H"},
        {"PHASE_mvt": "DEP", "MVT_ID_mvt": "c", "ADEP_mvt": "EDDF", "RUNWAY_mvt": "07L",
         "AOBT_3_flt": base + pd.Timedelta(minutes=-30), "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=150),
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "J"},
        {"PHASE_mvt": "DEP", "MVT_ID_mvt": "d", "ADEP_mvt": "EDDF", "RUNWAY_mvt": "07L",
         "AOBT_3_flt": base + pd.Timedelta(minutes=20), "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=80),
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "M"},
        {"PHASE_mvt": "DEP", "MVT_ID_mvt": "e", "ADEP_mvt": "EDDF", "RUNWAY_mvt": "07L",
         "AOBT_3_flt": base + pd.Timedelta(minutes=-200), "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=-150),
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "M"},
        {"PHASE_mvt": "DEP", "MVT_ID_mvt": "f", "ADEP_mvt": "EDDF", "RUNWAY_mvt": "07L",
         "AOBT_3_flt": pd.NaT, "MVT_TIME_UTC_mvt": pd.NaT,
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "M"},
        {"PHASE_mvt": "ARR", "MVT_ID_mvt": "x", "ADEP_mvt": "EDDM", "ADES_mvt": "EDDF",
         "RUNWAY_mvt": "07L",
         "AOBT_3_flt": pd.NaT, "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=-10),
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "M"},
        {"PHASE_mvt": "ARR", "MVT_ID_mvt": "y", "ADEP_mvt": "EDDM", "ADES_mvt": "EDDF",
         "RUNWAY_mvt": "07L",
         "AOBT_3_flt": pd.NaT, "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=-60),
         "EOBT_1_flt": base, "WK_TBL_CAT_flt": "M"},
    ]
    return pd.DataFrame(rows)


def test_queue_features():
    df = make_test_df()
    df_out = add_queue_features(df, include_a=True, include_b=True, include_c=True,
                                include_d=False, include_e=True)

    print("输出列:", list(df_out.columns))
    print(df_out[["MVT_ID_mvt", "PHASE_mvt", "ref_was_estimated",
                  "dep_mvt_cnt_airport_5m", "dep_queue_n_airport",
                  "dep_queue_heavy_n_airport",
                  "arr_mvt_cnt_airport_10m", "arr_mvt_cnt_airport_15m"]].to_string())

    expected_queue_n = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 0, "f": 3}
    # heavy_n: H(行 b), J(行 c)
    #   a: queue=0 → heavy=0
    #   b: queue=[a(M)] → heavy=0
    #   c: queue=[a(M), b(H)] → heavy=1
    #   d: queue=[a(M), b(H), c(J)] → heavy=2
    #   e: queue=0 → heavy=0
    #   f: queue=[a(M), b(H), c(J)] → heavy=2
    expected_heavy_n = {"a": 0, "b": 0, "c": 1, "d": 2, "e": 0, "f": 2}

    # ===== 断言 (1) 自身不计入 =====
    print("\n===== 断言 (1) 自身不计入 =====")
    for idx, row in df_out[df_out["PHASE_mvt"] == "DEP"].iterrows():
        mvt_id = row["MVT_ID_mvt"]
        qn = row["dep_queue_n_airport"]
        exp = expected_queue_n[mvt_id]
        assert qn == exp, f"{mvt_id} queue_n 应是 {exp},实际 {qn}"
        print(f"  {mvt_id}: queue_n={qn:.0f} (期望 {exp}) ✓")

    # ===== 断言 (2)+(3) ref 之后 MVT 不计入 / 窗口边界严格 < ref =====
    print("\n===== 断言 (2)+(3) ref 之后 MVT 不计入 / 窗口边界 =====")
    # 所有 DEP 的 MVT 都在 ref 之后 → dep_mvt_cnt = 0
    for idx, row in df_out[df_out["PHASE_mvt"] == "DEP"].iterrows():
        mvt_id = row["MVT_ID_mvt"]
        for w in [5, 10, 15]:
            v = row[f"dep_mvt_cnt_airport_{w}m"]
            assert v == 0, f"{mvt_id} dep_mvt_cnt_{w}m 应是 0,实际 {v}"
    print("  dep_mvt_cnt_airport_{5,10,15}m: 所有 DEP MVT 都在各自 ref 之后 → 全 0 ✓")

    # ===== 断言 (4) queue_N 手算 = 代码值 =====
    print("\n===== 断言 (4) queue_N 手算 = 代码值 =====")
    for idx, row in df_out[df_out["PHASE_mvt"] == "DEP"].iterrows():
        mvt_id = row["MVT_ID_mvt"]
        hn = row["dep_queue_heavy_n_airport"]
        exp = expected_heavy_n[mvt_id]
        assert hn == exp, f"{mvt_id} heavy_n 应是 {exp},实际 {hn}"
        print(f"  {mvt_id}: queue_heavy_n={hn:.0f} (期望 {exp}) ✓")

    # ===== 断言 (5) NaN 冷启动只在预期边界 =====
    print("\n===== 断言 (5) NaN 冷启动只在预期边界 =====")
    f_row = df_out[df_out["MVT_ID_mvt"] == "f"].iloc[0]
    print(f"  f (AOBT=NaN): ref_was_estimated={f_row['ref_was_estimated']} (期望 1)")
    assert f_row["ref_was_estimated"] == 1

    # ARR 计数(只有 DEP 行有值,ARR 行为 NaN)
    print("\n===== ARR 计数 =====")
    f_row = df_out[df_out["MVT_ID_mvt"] == "f"].iloc[0]
    # f ref=base=0:
    #   区间 (ref-10m, ref] = (-10, 0] 半开半闭
    #   ARR x MVT=-10: -10 == ref-10 → 不严格 > ref-10 → 不在 (-10, 0] 内 → arr_10m=0
    #   区间 (ref-15m, ref] = (-15, 0]
    #   ARR x MVT=-10: -10 严格 > -15 ✓ 且 ≤ 0 ✓ → 计入 → arr_15m=1
    #   ARR y MVT=-60: -60 严格 > -15 ✗ → 不在 (-15, 0] 内
    assert f_row["arr_mvt_cnt_airport_10m"] == 0, f"f arr_10m 应是 0 (-10 == t_lo 不算),实际 {f_row['arr_mvt_cnt_airport_10m']}"
    assert f_row["arr_mvt_cnt_airport_15m"] == 1, f"f arr_15m 应是 1,实际 {f_row['arr_mvt_cnt_airport_15m']}"
    print(f"  f: arr_10m={f_row['arr_mvt_cnt_airport_10m']}, arr_15m={f_row['arr_mvt_cnt_airport_15m']} ✓")

    # ARR x 行的 arr_* 应是 NaN(我们只对 DEP 算)
    x_row = df_out[df_out["MVT_ID_mvt"] == "x"].iloc[0]
    assert pd.isna(x_row["arr_mvt_cnt_airport_10m"])
    print(f"  ARR x: arr_10m=NaN ✓ (ARR 行不算)")

    # 其余 DEP 行的 arr_mvt_cnt:
    # a (ref=-100): (-110,-100] y=-60 > -110 ✓ ≤ -100 ✗, x=-10 > -100 ✗ → 0 / 0
    # b (ref=-50):  (-60,-50] y=-60 == -60 ✗ → 0; (-65,-50] y=-60 ∈ → 1
    # c (ref=-30):  (-40,-30] → 0; (-45,-30] → 0
    # d (ref=+20):  (10,20] → 0; (5,20] → 0
    # e (ref=-200): (-210,-200] → 0; (-215,-200] → 0
    expected_arr = {
        "a": (0, 0),
        "b": (0, 1),  # y=-60 落入 (-65,-50]
        "c": (0, 0),
        "d": (0, 0),
        "e": (0, 0),
    }
    for mid, (e10, e15) in expected_arr.items():
        r = df_out[df_out["MVT_ID_mvt"] == mid].iloc[0]
        assert r["arr_mvt_cnt_airport_10m"] == e10, f"{mid} arr_10m 应是 {e10},实际 {r['arr_mvt_cnt_airport_10m']}"
        assert r["arr_mvt_cnt_airport_15m"] == e15, f"{mid} arr_15m 应是 {e15},实际 {r['arr_mvt_cnt_airport_15m']}"
        print(f"  {mid}: arr_10m={r['arr_mvt_cnt_airport_10m']}, arr_15m={r['arr_mvt_cnt_airport_15m']} ✓")

    print("\n✅ 所有 5 项断言通过")


if __name__ == "__main__":
    test_queue_features()
