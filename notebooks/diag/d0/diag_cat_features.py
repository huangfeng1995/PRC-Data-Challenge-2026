"""
诊断 11 个 LGB cat 列的基数和 dtype
- 列名 + 当前 dtype + unique 基数
- 高基数列(>50) → 不走 LGB 原生 cat_features, 改用工程 *_cat_mappings
- 低基数列(≤50) → 保留 LGB 原生 cat
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

# 加载 holdout (1+7) 拿 cat 列现状
TRAIN_COLS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
    "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
    "MARKET_SEGMENT_flt", "AIRCRAFT_TYPE_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt",
    "ADES_flt", "ADES_FILED_flt",
    "LOBT_flt", "TAXITIME_SEC_mvt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt", "LOBT_flt"]

print("=" * 70)
print("[加载训练月 (2025-03) 看 cat 列]")
print("=" * 70)
df = load_month("data/raw/training_2025-03-01_2025-04-01.parquet", columns=TRAIN_COLS)
df = parse_time_columns(df, cols=TIME_COLS)
df_dep = df[df["PHASE_mvt"] == "DEP"].copy()
print(f"  03月 DEP: {len(df_dep):,}")

# 也加载 holdout 看基数稳定性
print("\n[加载 holdout (2025-01+07)]")
ho_dfs = []
for m in [1, 7]:
    fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_ho = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho, cols=TIME_COLS)
df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy()
print(f"  holdout DEP: {len(df_ho_dep):,}")

# 加载 v15 的 cat mapping (LGB 训练用)
cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
cat_map = json.loads(cat_map_path.read_text())
print(f"\n[v15 cat mapping: {len(cat_map)} 列]")
for k in cat_map:
    print(f"  {k}: {len(cat_map[k])} 个映射 (含 __nan__)")

# 加载 holdout_v1 的 add_queue_features 看 cat dtype 来源
# 看 LGB 调用时传的 categorical_feature
print("\n" + "=" * 70)
print("[FEATS69 中所有 category 列的基数]")
print("=" * 70)

# 训练 FEATS69 中的所有 category 列 (按 holdout_v1_queue_ab 推断)
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

# 重建一份完整特征, 看 71 列中所有 category 列
lut = L.build_t_unimp_lut(df_ho_dep)  # 用 holdout DEP 作 LUT (够用)
df_ho_basic = build_basic(df_ho.copy())
df_ho_full = add_queue_features(df_ho_basic, include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
df_ho_full["ref_was_estimated"] = df_ho_full["AOBT_3_flt"].isna().astype("int8")
df_ho_full["is_diverted"] = (df_ho_full["ADES_flt"] != df_ho_full["ADES_FILED_flt"]).astype("int8") if "ADES_flt" in df_ho_full.columns else 0
d_dict = D.compute_drift_features(df_ho_full, ref_dep=df_ho, label="diag")
for k, v in d_dict.items():
    df_ho_full[k] = v
t_vals, _ = L.lookup_t_unimp(df_ho_full, lut)
df_ho_full["t_unimp_r7"] = t_vals
dep_mask = (df_ho_full["PHASE_mvt"] == "DEP").values
extra_dict = L.compute_extra_features(df_ho_full[dep_mask], label="diag")
for k, v in extra_dict.items():
    df_ho_full[k] = np.nan
    df_ho_full.loc[dep_mask, k] = v
mvt_dict = L.compute_mvt_anchor_features(df_ho_full[dep_mask], t_unimp_vals=t_vals[dep_mask], label="diag")
for k, v in mvt_dict.items():
    df_ho_full[k] = np.nan
    df_ho_full.loc[dep_mask, k] = v

cat_in_feats = [f for f in df_ho_full.columns if df_ho_full[f].dtype.name == "category"]
print(f"\n  重构 holdout 后 category 列: {len(cat_in_feats)}")
print(f"\n{'#':>3s} {'列名':30s} {'holdout unique':>15s} {'03月 unique':>15s} {'dtype':>12s} {'cat_mapping 字典大小':>20s}")
print("-" * 100)

for f in sorted(cat_in_feats):
    h_unique = df_ho_full[f].nunique()
    m_unique = df_dep[f].nunique() if f in df_dep.columns else "N/A"
    dtype_str = str(df_ho_full[f].dtype)
    map_size = len(cat_map.get(f, {}))
    flag = " ← 高基数 (>50)" if h_unique > 50 else " ← 低基数"
    print(f"{cat_in_feats.index(f)+1:>3d} {f:30s} {h_unique:>15d} {str(m_unique):>15s} {dtype_str:>12s} {map_size:>20d}{flag}")

print("\n" + "=" * 70)
print("[判断]")
print("=" * 70)
high = [f for f in cat_in_feats if df_ho_full[f].nunique() > 50]
low = [f for f in cat_in_feats if df_ho_full[f].nunique() <= 50]
print(f"  高基数 (>50, 不用 LGB 原生 cat_features, 改用工程 *_cat_mappings): {len(high)}")
for f in high:
    print(f"    {f}: {df_ho_full[f].nunique()}")
print(f"  低基数 (≤50, 保留 LGB 原生 cat_features): {len(low)}")
for f in low:
    print(f"    {f}: {df_ho_full[f].nunique()}")