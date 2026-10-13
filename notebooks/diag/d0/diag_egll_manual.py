"""
排查 6: EGLL 2025-03 抽样手算 dep_queue_n_runway

挑 5-10 航班, 列出 ref(AOBT) 时刻同跑道已 AOBT 未起飞 DEP 的时间线,
逐架数, 对比管线重建值

判据:
  - 手算 ≠ 重建 → 计数 bug, 修
  - 手算 = 重建 → 计数逻辑正确, 停止逆向
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns

# ============================================================
# 1. 加载 2025-03 EGLL DEP+ARR 全部数据
# ============================================================
print("=" * 70)
print("[1] 加载 2025-03 EGLL DEP+ARR")
print("=" * 70)
TRAIN_COLS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
    "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt",
    "TAXITIME_SEC_mvt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]

df_mar = load_month("data/raw/training_2025-03-01_2025-04-01.parquet", columns=TRAIN_COLS)
df_mar = parse_time_columns(df_mar, cols=TIME_COLS)

# EGLL 全部 (DEP+ARR 都需要, 因为 dep_queue_n 的 enter=ref_dt 默认 AOBT)
df_egll = df_mar[df_mar["ADEP_mvt"] == "EGLL"].copy().reset_index(drop=True)
print(f"  EGLL 总 (DEP+ARR): {len(df_egll):,} (DEP: {(df_egll['PHASE_mvt']=='DEP').sum()}, ARR: {(df_egll['PHASE_mvt']=='ARR').sum()})")

# EGLL DEP
df_egll_dep = df_egll[df_egll["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
print(f"  EGLL DEP: {len(df_egll_dep):,}")

# ============================================================
# 2. 重建管线的 dep_queue_n_runway (来自 add_queue_features)
# ============================================================
print()
print("=" * 70)
print("[2] 重建管线 dep_queue_n_runway")
print("=" * 70)

from src.features import build_features as build_basic
from src.features_queue import add_queue_features

df_mar_basic = build_basic(df_mar.copy())
df_mar_full = add_queue_features(
    df_mar_basic.copy(),
    include_a=True, include_b=True, include_c=True,
    include_d=True, include_e=True,
)
print(f"  add_queue_features 后 shape: {df_mar_full.shape}")
# 看 EGLL 的 dep_queue_n_runway
egll_qr_pipe = df_mar_full[df_mar_full["ADEP_mvt"] == "EGLL"][["MVT_ID_mvt", "PHASE_mvt", "RUNWAY_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "dep_queue_n_runway"]].copy()

# 管线产出值分布
v = egll_qr_pipe["dep_queue_n_runway"]
print(f"\n  管线 EGLL dep_queue_n_runway 分布 (DEP+ARR 全部):")
print(f"    有效: {v.notna().sum()}/{len(v)}")
print(f"    min={v.min()}, p50={v.median()}, p90={v.quantile(0.9)}, max={v.max()}")

# 只看 DEP
egll_dep_qr_pipe = egll_qr_pipe[egll_qr_pipe["PHASE_mvt"] == "DEP"].copy()
print(f"\n  EGLL DEP (管线):")
print(f"    n={len(egll_dep_qr_pipe)}")
print(f"    min={egll_dep_qr_pipe['dep_queue_n_runway'].min()}, p50={egll_dep_qr_pipe['dep_queue_n_runway'].median()}, "
      f"p90={egll_dep_qr_pipe['dep_queue_n_runway'].quantile(0.9)}, max={egll_dep_qr_pipe['dep_queue_n_runway'].max()}")

# ============================================================
# 3. 抽样手算 10 个 DEP 航班
# ============================================================
print()
print("=" * 70)
print("[3] 抽样 10 个 EGLL 03 月 DEP 手算")
print("=" * 70)

# 按 AOBT 排序, 均匀抽样
egll_dep_sorted = egll_dep_qr_pipe.sort_values("AOBT_3_flt").reset_index(drop=True)
# 取 dep_queue_n_runway 跨度大的 (0, 中, 高)
sample_idx = np.linspace(0, len(egll_dep_sorted) - 1, 10, dtype=int)

samples = egll_dep_sorted.iloc[sample_idx].copy()
print(f"\n  抽样 (n=10, 按 AOBT 排序均匀取):")
print(f"  {'idx':>4s} {'MVT_ID':>10s} {'RW':>4s} {'AOBT':>20s} {'MVT':>20s} {'管线值':>8s} {'手算值':>8s} {'匹配':>6s}")
print("-" * 95)

# 手算: 对每个样本, 找同 (EGLL, RW) 中:
#   AOBT_j <= AOBT_i AND (MVT_j NaT OR MVT_j > AOBT_i) AND MVT_ID_j != i
all_egll_dep = egll_dep_qr_pipe.copy()
all_egll_dep["AOBT_i"] = pd.to_datetime(all_egll_dep["AOBT_3_flt"], utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
all_egll_dep["MVT_i"] = pd.to_datetime(all_egll_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
all_egll_dep["AOBT_v"] = all_egll_dep["AOBT_i"] != iNaT
all_egll_dep["MVT_v"] = all_egll_dep["MVT_i"] != iNaT

matches = []
for idx, row in samples.iterrows():
    mvt_id_i = row["MVT_ID_mvt"]
    rw_i = row["RUNWAY_mvt"]
    aobt_i = row["AOBT_3_flt"]
    pipe_val = row["dep_queue_n_runway"]

    # 手算: 同 RW 的其他 DEP, AOBT <= aobt_i, 未起飞 (MVT NaT or MVT > aobt_i)
    aobt_i_ns = pd.to_datetime(aobt_i, utc=True).value if pd.notna(aobt_i) else None
    if aobt_i_ns is None:
        print(f"  {idx:4d} {mvt_id_i:>10.0f} {str(rw_i):>4s} {str(aobt_i):>20s} - AOBT 缺失, 跳过")
        continue

    same_rw = all_egll_dep[
        (all_egll_dep["RUNWAY_mvt"] == rw_i) &
        (all_egll_dep["MVT_ID_mvt"] != mvt_id_i)
    ]
    cond_aobt = (same_rw["AOBT_i"] <= aobt_i_ns) & same_rw["AOBT_v"]
    cond_not_takeoff = (~same_rw["MVT_v"]) | (same_rw["MVT_i"] > aobt_i_ns)
    hand_count = int((cond_aobt & cond_not_takeoff).sum())

    match = "✅" if hand_count == pipe_val else "❌"
    matches.append(hand_count == pipe_val)

    print(f"  {idx:4d} {mvt_id_i:>10.0f} {str(rw_i):>4s} {str(aobt_i):>20s} {str(row['MVT_TIME_UTC_mvt']):>20s} "
          f"{pipe_val:>8.0f} {hand_count:>8d} {match:>6s}")

# 看每个样本的同跑道前后架次 (时间线)
print()
print("=" * 70)
print("[4] 时间线示例 (样本 1)")
print("=" * 70)
if len(samples) > 0:
    ex_row = samples.iloc[0]
    ex_rw = ex_row["RUNWAY_mvt"]
    ex_aobt = ex_row["AOBT_3_flt"]
    print(f"  样本: MVT_ID={ex_row['MVT_ID_mvt']:.0f}, RW={ex_rw}, AOBT={ex_aobt}")
    print(f"  ref 时刻: {ex_aobt}")
    print(f"  管线 dep_queue_n_runway: {ex_row['dep_queue_n_runway']}")
    print()
    print(f"  整个 EGLL 03 月 {ex_rw} 跑道的 DEP 时间线:")
    tl = all_egll_dep[all_egll_dep["RUNWAY_mvt"] == ex_rw].sort_values("AOBT_i")
    ex_aobt_ns = pd.to_datetime(ex_aobt, utc=True).value
    for _, r in tl.iterrows():
        mvt_id = r["MVT_ID_mvt"]
        aobt = r["AOBT_3_flt"]
        mvt = r["MVT_TIME_UTC_mvt"]
        # 标记是否在样本的 ref 时刻"已 AOBT 但未起飞"
        flag = ""
        if mvt_id != ex_row["MVT_ID_mvt"]:
            aobt_ns = r["AOBT_i"] if r["AOBT_v"] else None
            mvt_ns = r["MVT_i"] if r["MVT_v"] else None
            if aobt_ns is not None and aobt_ns <= ex_aobt_ns:
                if mvt_ns is None or mvt_ns > ex_aobt_ns:
                    flag = "  ← 在样本 ref 时刻 (已 AOBT 未起飞)"
        print(f"    MVT_ID={mvt_id:.0f}, AOBT={aobt}, MVT={mvt}{flag}")

print()
print("=" * 70)
print("[5] 匹配汇总")
print("=" * 70)
print(f"  抽样 {len(matches)} 个, 匹配: {sum(matches)}/{len(matches)}")
if all(matches):
    print(f"  ✅ 手算全部等于管线重建值 → 计数逻辑正确")
    print(f"     停止逆向 → EGLL 问题是机制性缺信号 (AOBT 锚定看不到跑道口持点/穿越)")
else:
    print(f"  ❌ 手算与管线不一致 → 计数 bug, 修")
    # 显示不一致的
    for i, m in enumerate(matches):
        if not m:
            print(f"    idx={sample_idx[i]}: 管线={samples.iloc[i]['dep_queue_n_runway']}, 手算=?")