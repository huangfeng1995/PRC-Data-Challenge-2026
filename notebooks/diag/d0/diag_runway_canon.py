"""
排查 3: 跑道号口径, 重点

- 列 holdout RUNWAY 实际格式 (数字 vs 带字母、两端号)
- 查跑道级分组匹配率, 确认跑道号是否归一
- 漏匹配会让所有跑道级特征 (queue/mu/q) 系统性偏小 → 低估拥堵 → RMSE升高
- 按两端号+L/R/C 统一后重算跑道级特征, 重跑 holdout
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns

TRAIN_COLS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
    "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "AOBT_3_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "AOBT_3_flt"]

# ============================================================
# 1. 加载 holdout (1+7)
# ============================================================
ho_dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_ho_dep = df_ho_dep[df_ho_dep["TAXITIME_SEC_mvt"] > 0].reset_index(drop=True)
print(f"Holdout DEP: {len(df_ho_dep):,}")

# ============================================================
# 2. 加载训练 (2-12)
# ============================================================
months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months:
    fname = (f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
             if m < 12 else "training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[(df_tr["PHASE_mvt"] == "DEP") & (df_tr["TAXITIME_SEC_mvt"] > 0)].reset_index(drop=True)
print(f"训练 DEP: {len(df_tr_dep):,}")

# ============================================================
# 3. RUNWAY 格式分析
# ============================================================
print()
print("=" * 70)
print("[RUNWAY 格式分析]")
print("=" * 70)
print(f"\n训练 RUNWAY 原始 unique 值:")
rw_tr = df_tr_dep["RUNWAY_mvt"].astype(str)
print(f"  数量: {rw_tr.nunique()}, dtype: {df_tr_dep['RUNWAY_mvt'].dtype}")
print(f"  前 30 个: {sorted(rw_tr.unique())[:30]}")
print(f"  后 30 个: {sorted(rw_tr.unique())[-30:]}")

print(f"\nholdout RUNWAY 原始 unique 值:")
rw_ho = df_ho_dep["RUNWAY_mvt"].astype(str)
print(f"  数量: {rw_ho.nunique()}")
print(f"  全部: {sorted(rw_ho.unique())}")

# 字符类型分布
def classify_rw(s):
    """分类跑道号格式: 纯数字 / 带L/R/C / 其他"""
    if s.isdigit():
        return "纯数字"
    elif s.endswith(("L", "R", "C")):
        return f"末尾{s[-1]}"
    else:
        return "其他"

print(f"\n跑道号格式分类 (训练):")
cnt = rw_tr.apply(classify_rw).value_counts()
print(cnt.to_string())

# 检查两端口号 (e.g. "09L/27R" 风格)
print(f"\n是否有两端口号格式 (e.g. '09L/27R')?")
slash_rw = [r for r in rw_tr.unique() if "/" in r]
print(f"  含 '/' 的跑道号: {len(slash_rw)} 个, 例子: {slash_rw[:5]}")

# ============================================================
# 4. 匹配率诊断
# ============================================================
print()
print("=" * 70)
print("[匹配率诊断] holdout 跑道号在训练中能否匹配")
print("=" * 70)

tr_rw_set = set(rw_tr.unique())
ho_rw_set = set(rw_ho.unique())

print(f"  训练 unique 跑道: {len(tr_rw_set)}")
print(f"  holdout unique 跑道: {len(ho_rw_set)}")
print(f"  重叠: {len(tr_rw_set & ho_rw_set)}")
print(f"  holdout 中仅在训练中存在的: {len(ho_rw_set - tr_rw_set)} 个")

unmatched = ho_rw_set - tr_rw_set
if unmatched:
    print(f"\n  仅 holdout 有的跑道号:")
    for r in sorted(unmatched):
        n = int((rw_ho == r).sum())
        print(f"    '{r}': {n} 航班")

# 同样看 (ADEP, RUNWAY) 组合的匹配率
print()
print("=" * 70)
print("[(ADEP, RUNWAY) 组合匹配率]")
print("=" * 70)

tr_pairs = set(zip(df_tr_dep["ADEP_mvt"].astype(str), df_tr_dep["RUNWAY_mvt"].astype(str)))
ho_pairs = df_ho_dep[["ADEP_mvt", "RUNWAY_mvt"]].astype(str).apply(tuple, axis=1)

ho_pairs_unique = set(ho_pairs)
matched_pairs = ho_pairs_unique & tr_pairs
print(f"  训练 unique (ADEP, RW): {len(tr_pairs):,}")
print(f"  holdout unique (ADEP, RW): {len(ho_pairs_unique):,}")
print(f"  重叠 (匹配): {len(matched_pairs):,} ({100*len(matched_pairs)/len(ho_pairs_unique):.1f}%)")
print(f"  不匹配: {len(ho_pairs_unique - tr_pairs):,} ({100*len(ho_pairs_unique-tr_pairs)/len(ho_pairs_unique):.1f}%)")

# 不匹配的 pair 例子
unmatched_pairs = ho_pairs_unique - tr_pairs
if unmatched_pairs:
    print(f"\n  不匹配 pair 例子 (前 10):")
    for ap, rw in sorted(unmatched_pairs)[:10]:
        n = int(((df_ho_dep["ADEP_mvt"].astype(str) == ap) & (df_ho_dep["RUNWAY_mvt"].astype(str) == rw)).sum())
        print(f"    ({ap}, {rw}): {n} 航班")

# 归一化: 把 09L/27R, 09R/27L 拆开
# 例如 EDDF/07C/25C 是同一跑道, 09L 和 27R 互为反向
# ============================================================
# 5. 归一化方案: 跑道号 -> canonical 形式
# ============================================================
print()
print("=" * 70)
print("[归一化方案]")
print("=" * 70)

def canonical_runway(rw):
    """归一化跑道号: 处理 09L/27R 这种两端口号
    规则:
      - 09L <-> 27R 是同一物理跑道 (L 端 09, R 端 27)
      - 取较小的编号 (主方向) + 反向端字母
      - 例如 27R -> 09R, 09L -> 09L (保留)
    """
    if "/" in rw:
        # 两端口号
        parts = rw.split("/")
        return parts[0]  # 简单取第一个
    # 09L -> 09L (不变), 27R -> ?
    return rw

# 但更安全的归一化是: 09L <-> 27R 视为同一
# 这里采用简化: 两端取小编号
def canonical_runway_v2(rw):
    """09L/27R -> 09R (同跑道, 取最小编号+反向端字母)"""
    if "/" in rw:
        parts = rw.split("/")
        nums = []
        for p in parts:
            num = "".join(c for c in p if c.isdigit())
            nums.append(int(num) if num else 0)
        # 取最小编号, 对应反向端字母
        # 09L/27R: 09 -> 09, 字母 = parts[0][-1] (L)
        min_idx = nums.index(min(nums))
        return parts[min_idx]
    return rw

df_tr_dep["rw_orig"] = df_tr_dep["RUNWAY_mvt"].astype(str)
df_tr_dep["rw_canon"] = df_tr_dep["rw_orig"].apply(canonical_runway_v2)
df_ho_dep["rw_orig"] = df_ho_dep["RUNWAY_mvt"].astype(str)
df_ho_dep["rw_canon"] = df_ho_dep["rw_orig"].apply(canonical_runway_v2)

print(f"\n  归一化后训练 unique 跑道: {df_tr_dep['rw_canon'].nunique()} (原 {df_tr_dep['rw_orig'].nunique()})")
print(f"  归一化后 holdout unique 跑道: {df_ho_dep['rw_canon'].nunique()} (原 {df_ho_dep['rw_orig'].nunique()})")

# 匹配率 (归一化后)
tr_pairs2 = set(zip(df_tr_dep["ADEP_mvt"].astype(str), df_tr_dep["rw_canon"]))
ho_pairs2 = set(zip(df_ho_dep["ADEP_mvt"].astype(str), df_ho_dep["rw_canon"]))
matched2 = ho_pairs2 & tr_pairs2
print(f"\n  归一化后 (ADEP, RW) 匹配率: "
      f"{len(matched2)}/{len(ho_pairs2)} ({100*len(matched2)/len(ho_pairs2):.1f}%)")
unmatched2 = ho_pairs2 - tr_pairs2
print(f"  归一化后不匹配 pair: {len(unmatched2)}")

# ============================================================
# 6. 重算 dep_queue_n_runway 用归一化跑道号
# ============================================================
print()
print("=" * 70)
print("[重算 dep_queue_n_runway 用归一化跑道号]")
print("=" * 70)

# 先看旧的 dep_queue_n_runway 分布
import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

old_qr_full = pd.to_numeric(hr7.df_ho_full["dep_queue_n_runway"], errors="coerce").values
print(f"\n  旧 dep_queue_n_runway (用原始 RW, 在 full 上):")
print(f"    有效: {np.sum(~np.isnan(old_qr_full))}/{len(old_qr_full)}")
print(f"    p50={np.nanmedian(old_qr_full):.1f}, p90={np.percentile(old_qr_full, 90):.1f}, max={np.nanmax(old_qr_full):.0f}")

# 重算 (用归一化 RW), 在 holdout 上
def compute_dep_queue_n_runway(df_dep):
    """对每个 DEP, 统计同 (ADEP, RUNWAY) 中 ref 时刻 'AOBT<=ref AND (MVT NaT OR MVT>ref)' 的航班数
    用 AOBT 作 enter, MVT 作 leave. 排除自身.
    """
    aobt = pd.to_datetime(df_dep["AOBT_3_flt"], utc=True, errors="coerce")
    mvt = pd.to_datetime(df_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    aep = df_dep["ADEP_mvt"].astype(str).values
    rw = df_dep["rw_canon"].values

    aobt_i = aobt.values.astype("datetime64[ns]").view("i8")
    mvt_i = mvt.values.astype("datetime64[ns]").view("i8")
    iNaT = np.iinfo(np.int64).min
    aobt_v = aobt_i != iNaT
    mvt_v = mvt_i != iNaT

    n = len(df_dep)
    result = np.full(n, np.nan, dtype=np.float32)
    # 用 AOBT 作 ref (旧版 dep_queue_n 的口径)
    ref_i = aobt_i.copy()

    for ap_rwy in np.unique(np.char.add(np.char.add(aep, "_"), rw)):
        ap, rwy = ap_rwy.split("_", 1)
        mask = (aep == ap) & (rw == rwy)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        ref_g = ref_i[mask]
        aobt_v_g = aobt_v[mask]
        mvt_v_g = mvt_v[mask]
        mvt_g = mvt_i[mask]
        valid_g = aobt_v_g
        if valid_g.sum() < 2:
            continue
        # 对每个 ref 时刻, 统计: #{j: aobt_j<=ref AND mvt_j NaT or >ref} - self
        # 简单 O(n^2) 因为组内一般不大
        for ii in range(len(idx)):
            if not aobt_v_g[ii]:
                continue
            t_ref = ref_g[ii]
            # 条件: aobt_j <= t_ref AND (mvt_j NaT OR mvt_j > t_ref) AND j != i
            other = np.arange(len(idx)) != ii
            cond_aobt = (aobt_v_g) & (aobt_i[mask] <= t_ref) & other
            cond_mvt_ok = (~mvt_v_g) | (mvt_g > t_ref)
            cnt = (cond_aobt & cond_mvt_ok).sum()
            result[idx[ii]] = float(cnt)
    return result

# 取 df_ho_dep (DEP 子集, 全 holdout)
new_qr = compute_dep_queue_n_runway(df_ho_dep)
print(f"\n  新 dep_queue_n_runway (用归一化 RW):")
print(f"    p50={np.nanmedian(new_qr):.1f}, p90={np.percentile(new_qr, 90):.1f}, max={np.nanmax(new_qr):.0f}")

# 推理 (替换 dep_queue_n_runway)
print()
print("=" * 70)
print("[推理 RMSE 对比]")
print("=" * 70)

import json
bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names
from src import train_xgb as tx
from src.train_xgb import _encode_categoricals
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())

# 直接用 df_ho_dep 的特征 + 替换 dep_queue_n_runway
dep_mask_full = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = hr7.df_ho_full[dep_mask_full].copy().reset_index(drop=True)

# 用 MVT_ID 对齐: 新计算的 new_qr 对应 df_ho_dep (按位置)
# df_ho_dep 已经按 (PHASE=DEP) 过滤, 但 hr7.df_ho_full 经过更多过滤
# 让我用 MVT_ID 对齐
new_qr_map = dict(zip(df_ho_dep["MVT_ID_mvt"].values, new_qr))
mapped_qr = df_ho_dep_full["MVT_ID_mvt"].map(new_qr_map).astype(np.float32).values

X = df_ho_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)
y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)

# 基准
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
rmse_base = float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))
print(f"  基准 RMSE: {rmse_base:.2f}")

# 替换 dep_queue_n_runway
X_mod = X_enc.copy()
qr_idx = FEATS69.index("dep_queue_n_runway")
X_mod[:, qr_idx] = mapped_qr
dmat = xgb.DMatrix(X_mod, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
rmse_new = float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))
print(f"  新 dep_queue_n_runway (归一化) RMSE: {rmse_new:.2f} (Δ={rmse_new-rmse_base:+.2f})")