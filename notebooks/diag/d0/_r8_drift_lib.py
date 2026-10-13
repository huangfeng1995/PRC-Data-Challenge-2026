"""
_r8_drift_lib.py — 漂移特征库（重建版，v15 用了 2 个）
只保留 v15 实际使用的 2 个特征: rwy_month_share, ap_month_hourly_density
"""
import time
import numpy as np
import pandas as pd

D_FEATS = ["rwy_month_share", "ap_month_hourly_density"]
KEY_TIME = "MVT_TIME_UTC_mvt"


def compute_drift_features(df_tgt, ref_dep=None, ref_all=None, label=""):
    """计算 D 特征。

    df_tgt: 目标帧 (含 DEP+ARR, 但 D 特征只对 DEP 有意义)
    ref_dep: 更早的 DEP 池 (past-only 参考), None 则用 df_tgt 自身
    ref_all: 更早的全量池, None 则用 df_tgt 自身
    label: 调试用标签

    返回: dict {feat_name: np.ndarray(len(df_tgt))}
    """
    t0 = time.time()
    n = len(df_tgt)
    out = {c: np.full(n, np.nan) for c in D_FEATS}

    dep_mask = (df_tgt["PHASE_mvt"] == "DEP").values
    dep_idx = np.where(dep_mask)[0]
    if len(dep_idx) == 0:
        return out

    # 参考池: 用 ref_dep 的 DEP 行做统计, 没有 ref 就用自身
    if ref_dep is not None:
        ref = ref_dep[ref_dep["PHASE_mvt"] == "DEP"].copy()
    else:
        ref = df_tgt.iloc[dep_idx].copy()

    ref["__month__"] = ref[KEY_TIME].dt.month
    tgt_dep = df_tgt.iloc[dep_idx].copy()
    tgt_dep["__month__"] = tgt_dep[KEY_TIME].dt.month

    # --- rwy_month_share: 同 (ADEP, RUNWAY, MONTH) 内, 该 RUNWAY 占机场当月起飞的比例 ---
    ap_month_total = ref.groupby(["ADEP_mvt", "__month__"], observed=True).size().rename("total")
    rwy_month_cnt = ref.groupby(["ADEP_mvt", "RUNWAY_mvt", "__month__"], observed=True).size().rename("cnt")
    df_aps = ap_month_total.reset_index(name="ap_total")
    df_rwy = rwy_month_cnt.reset_index(name="rwy_cnt")
    df_merged = df_rwy.merge(df_aps, on=["ADEP_mvt", "__month__"], how="left")
    df_merged["share"] = df_merged["rwy_cnt"] / df_merged["ap_total"]
    share_map = df_merged.set_index(["ADEP_mvt", "RUNWAY_mvt", "__month__"])["share"]

    tgt_keys = pd.MultiIndex.from_arrays([
        tgt_dep["ADEP_mvt"].values,
        tgt_dep["RUNWAY_mvt"].values,
        tgt_dep["__month__"].values,
    ])
    out["rwy_month_share"][dep_idx] = share_map.reindex(tgt_keys).values

    # --- ap_month_hourly_density: 同 (ADEP, MONTH) 每小时平均起飞数 ---
    ref["__hour__"] = ref[KEY_TIME].dt.hour
    hourly_cnt = ref.groupby(["ADEP_mvt", "__month__", "__hour__"]).size()
    ap_month_hourly = hourly_cnt.groupby(["ADEP_mvt", "__month__"]).mean()
    ap_density_map = ap_month_hourly

    tgt_ap_keys = pd.MultiIndex.from_arrays([
        tgt_dep["ADEP_mvt"].values,
        tgt_dep["__month__"].values,
    ])
    out["ap_month_hourly_density"][dep_idx] = ap_density_map.reindex(tgt_ap_keys).values

    print(f"  D 特征 ({label}): {time.time()-t0:.1f}s, DEP {len(dep_idx):,}",
          flush=True)
    return out
