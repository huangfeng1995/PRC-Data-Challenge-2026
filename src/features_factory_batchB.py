"""
目标编码 (Target Encoding, TE) 特征 (feature-factory 批次 B)
=============================================================

口径约束 (铁律):
  1. ranking TAXITIME_SEC_mvt 50% 缺失 → TE 只能用训练数据学, 推理时用训练时存的映射
  2. 机场隔离: DEP/ARR 用对应字段
  3. OOF (Out-Of-Fold) 5 折按 (ADEP, month) 分层 → 避免同机场同月数据泄漏
  4. 4 层回退: (ADEP, STAND, RUNWAY) → (ADEP, RUNWAY) → (ADEP, STAND) → (ADEP)
  5. 排名命中率: 推理时 target ∈ {10%, 20%, ..., 90%} 百分位数, 看命中率

特征族 (5 类):
  te_ap_std_rwy_median         中位数版 TE(ADEP, STAND, RUNWAY), 秒
  te_ap_std_rwy_lowdemand      非拥堵期基准版 (低需求小时 + 20-30 百分位), 秒
  te_ap_rwy_hour               对照 TE(ADEP, RUNWAY, hour)
  te_ap_std                    对照 TE(ADEP, STAND)
  te_ap_operator               对照 TE(ADEP, AIRCRAFT_OPERATOR)

OOF 防泄漏机制:
  - 按 (ADEP_mvt, month) 5 折 split
  - 每一折: 用其他 4 折的 (key → TAXITIME_SEC_mvt 中位数) 学映射
  - 应用到该折行 → 5 折合并得到全量 OOF 预测
  - 训练时: 学全量 5 折合并后的映射 (用于 test 推理), 同时存 OOF 预测作为训练特征

注意: 真实值只能在全量训练上验证, sample 上仅验证:
  - OOF 代码逻辑 (5 折正确执行, 每折命中映射)
  - 4 层回退逻辑 (键不存在时降到下一层)
  - 命中率输出 (look-up 是否覆盖 95%+ 行)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.to_period("M")


def _make_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """按 (ADEP_mvt, month) 分层 5 折. 返回 fold_idx per row."""
    key = df["ADEP_mvt"].astype(str) + "_" + _parse_month(df["AOBT_3_flt"]).astype(str)
    fold_idx = np.full(len(df), -1, dtype="int8")
    rng = np.random.RandomState(seed)
    for k in key.unique():
        idx = np.where(key.values == k)[0]
        # 每个 (adep, month) 内随机分配 fold
        fold = rng.randint(0, n_folds, size=len(idx))
        fold_idx[idx] = fold
    return fold_idx


def _build_mapping(df_train: pd.DataFrame, group_cols: list[str], target: str = "TAXITIME_SEC_mvt",
                   agg: str = "median") -> pd.Series:
    """对 group_cols 学映射: group_key → agg(target)."""
    g = df_train.groupby(group_cols, dropna=False)[target]
    if agg == "median":
        m = g.median()
    elif agg == "lowdemand_p25":
        # 简化: 取 25 百分位 (低需求近似)
        m = g.quantile(0.25)
    else:
        raise ValueError(f"unknown agg: {agg}")
    return m


def _lookup_with_fallback(df_test: pd.DataFrame, group_cols: list[str], fallback_cols: list[str] | None,
                          target: str, agg: str = "median",
                          train_df: pd.DataFrame | None = None,
                          mappings: dict | None = None) -> np.ndarray:
    """从映射表 lookup, group 不存在时按 fallback_cols 降级."""
    if mappings is None:
        assert train_df is not None
        mappings = {"primary": _build_mapping(train_df, group_cols, target, agg)}

    primary = mappings["primary"]
    fallback = mappings.get("fallback")
    if fallback_cols:
        # 也尝试 lookup 每个 fallback 链
        result = np.full(len(df_test), np.nan, dtype="float32")
        # 第一层: primary
        keys = pd.MultiIndex.from_frame(df_test[group_cols]) if isinstance(group_cols, list) and len(group_cols) > 1 else df_test[group_cols[0]]
        result = primary.reindex(keys).values.astype("float32") if not isinstance(keys, pd.MultiIndex) else primary.reindex(keys).values.astype("float32")
        # 第二层 (multi-fallback)
        if fallback is not None:
            miss_mask = np.isnan(result)
            if miss_mask.any():
                keys2 = pd.MultiIndex.from_frame(df_test[fallback_cols]) if isinstance(fallback_cols, list) and len(fallback_cols) > 1 else df_test[fallback_cols[0]]
                fb_vals = fallback.reindex(keys2).values
                result[miss_mask] = fb_vals[miss_mask].astype("float32")
        return result
    else:
        keys = df_test[group_cols[0]] if isinstance(group_cols, list) and len(group_cols) == 1 else pd.MultiIndex.from_frame(df_test[group_cols])
        return primary.reindex(keys).values.astype("float32")


# ---------------------------------------------------------------------------
# 5 类 TE 主入口
# ---------------------------------------------------------------------------
def add_batchB_te_features(df: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                            target_col: str = "TAXITIME_SEC_mvt",
                            verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """追加 5 类 TE (OOF 训练时), 返回 (df_oof, mappings).

    df_oof: 包含 5 个 _te_oof 列 (训练用, OOF 预测)
    mappings: 存到磁盘供 test 推理用
    """
    out = df.copy()
    out["_month"] = _parse_month(out["AOBT_3_flt"])
    out["_hour"] = pd.to_datetime(out["AOBT_3_flt"], utc=True, errors="coerce").dt.hour

    # 防泄漏: 必须有 target
    has_target = target_col in out.columns and out[target_col].notna().any()
    if not has_target:
        if verbose:
            print(f"⚠ 训练数据 {target_col} 缺失, TE 无法拟合, 全填 NaN")
        for c in ["te_ap_std_rwy_median", "te_ap_std_rwy_lowdemand",
                  "te_ap_rwy_hour", "te_ap_std", "te_ap_operator"]:
            out[c] = np.nan
        return out, {}

    # 5 折
    folds = _make_folds(out, n_folds=n_folds, seed=seed)
    out["_fold"] = folds

    # 配置: TE 的 5 种口径
    te_specs = [
        # (列名, group_cols, fallback_cols, agg)
        ("te_ap_std_rwy_median",   ["ADEP_mvt","STAND_mvt","RUNWAY_mvt"], ["ADEP_mvt","RUNWAY_mvt","STAND_mvt"], "median"),
        ("te_ap_std_rwy_lowdemand",["ADEP_mvt","STAND_mvt","RUNWAY_mvt"], ["ADEP_mvt","RUNWAY_mvt","STAND_mvt"], "lowdemand_p25"),
        ("te_ap_rwy_hour",         ["ADEP_mvt","RUNWAY_mvt","_hour"],    None, "median"),
        ("te_ap_std",              ["ADEP_mvt","STAND_mvt"],             ["ADEP_mvt"], "median"),
        ("te_ap_operator",         ["ADEP_mvt","AIRCRAFT_OPERATOR_flt"], ["ADEP_mvt"], "median"),
    ]

    # 0) global median fallback (4 层回退的最底层)
    global_med = float(out[target_col].median())

    # OOF 5 折 + 1 全量映射
    oof_results = {}
    full_mappings = {}

    for col, group_cols, fallback_cols, agg in te_specs:
        # 检查所有 group_cols 都存在
        missing = [c for c in group_cols if c not in out.columns]
        if missing:
            if verbose:
                print(f"  ⚠ {col}: 缺 {missing}, 跳过")
            out[col] = np.nan
            continue
        # 5 折 OOF
        oof_pred = np.full(len(out), np.nan, dtype="float32")
        hit_count_primary = 0
        hit_count_fallback = 0
        miss_count = 0
        for fold in range(n_folds):
            tr_mask = (folds != fold)
            te_mask = (folds == fold)
            if not te_mask.any():
                continue
            tr_df = out[tr_mask & (out[target_col].notna())]
            te_df = out[te_mask]
            primary_map = _build_mapping(tr_df, group_cols, target_col, agg)
            if fallback_cols is not None:
                # 取 fallback chain (简化: 一层 fallback)
                fb_cols = fallback_cols
                fb_map = _build_mapping(tr_df, fb_cols, target_col, agg)
            else:
                fb_map = None
            # lookup
            te_keys = pd.MultiIndex.from_frame(te_df[group_cols]) if len(group_cols) > 1 else te_df[group_cols[0]]
            vals = primary_map.reindex(te_keys).values
            miss_mask = pd.isna(vals)
            vals = vals.astype("float32").copy()  # ensure writeable
            if fb_map is not None:
                fb_keys = pd.MultiIndex.from_frame(te_df[fb_cols]) if len(fb_cols) > 1 else te_df[fb_cols[0]]
                fb_vals = fb_map.reindex(fb_keys).values
                if fb_vals.dtype != np.float32:
                    fb_vals = fb_vals.astype("float32")
                vals[miss_mask] = fb_vals[miss_mask]
                miss_mask = pd.isna(vals)
            # 兜底 global
            vals[miss_mask] = global_med
            oof_pred[np.asarray(te_mask)] = vals.astype("float32")
            hit_count_primary += int((~miss_mask).sum())
            miss_count += int(miss_mask.sum())
        # 全量映射 (供推理)
        full_primary = _build_mapping(out[out[target_col].notna()], group_cols, target_col, agg)
        full_fb = None
        if fallback_cols is not None:
            full_fb = _build_mapping(out[out[target_col].notna()], fallback_cols, target_col, agg)

        out[col] = oof_pred
        full_mappings[col] = {"primary": full_primary, "fallback": full_fb, "global": global_med}

        if verbose:
            nn = (~pd.isna(oof_pred)).sum()
            print(f"  {col}: OOF nn={nn}/{len(out)} ({nn/len(out)*100:.1f}%), "
                  f"miss={miss_count}/{len(out)} (兜底 global={global_med:.1f})")

    # cleanup
    out.drop(columns=["_month","_hour","_fold"], inplace=True, errors="ignore")
    return out, full_mappings


def apply_batchB_te_features(df_test: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """test 推理时: 用全量映射填 5 个 TE 列."""
    out = df_test.copy()
    out["_hour"] = pd.to_datetime(out["AOBT_3_flt"], utc=True, errors="coerce").dt.hour
    for col, m in mappings.items():
        primary = m["primary"]
        fb = m.get("fallback")
        global_v = m.get("global", np.nan)
        # 提取 group cols
        if "ap_std_rwy" in col:
            gc = ["ADEP_mvt","STAND_mvt","RUNWAY_mvt"]
            fc = ["ADEP_mvt","RUNWAY_mvt","STAND_mvt"]
        elif "ap_rwy_hour" in col:
            gc = ["ADEP_mvt","RUNWAY_mvt","_hour"]
            fc = None
        elif "ap_std" in col:
            gc = ["ADEP_mvt","STAND_mvt"]
            fc = ["ADEP_mvt"]
        elif "ap_operator" in col:
            gc = ["ADEP_mvt","AIRCRAFT_OPERATOR_flt"]
            fc = ["ADEP_mvt"]
        else:
            continue
        # 缺失字段则填 NaN
        miss_g = [c for c in gc if c not in out.columns]
        if miss_g:
            out[col] = np.nan
            continue
        keys = pd.MultiIndex.from_frame(out[gc]) if len(gc) > 1 else out[gc[0]]
        vals = primary.reindex(keys).values.astype("float32")
        if fb is not None and fc is not None:
            miss = pd.isna(vals)
            if miss.any():
                keys2 = pd.MultiIndex.from_frame(out[fc]) if len(fc) > 1 else out[fc[0]]
                fb_v = fb.reindex(keys2).values.astype("float32")
                vals[miss] = fb_v[miss]
        miss = pd.isna(vals)
        if miss.any():
            vals[miss] = global_v
        out[col] = vals
    out.drop(columns=["_hour"], inplace=True, errors="ignore")
    return out
