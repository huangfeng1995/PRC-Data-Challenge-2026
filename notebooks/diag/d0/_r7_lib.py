"""
_r7_lib.py — R7 slim 特征库（重建版）
=====================================
v15 用的 R7slim = Gd_ABD_runway(39) + MVT锚定(16) + 计数(9) + 计划延误(4) + t_unimp(1) = 69

当前进度:
  ✅ dly_* (4)
  ✅ dep_cnt_m_* (3), aobt_cnt_m_* (3), eobt_ahead_cnt_* (3) = 9
  ✅ t_unimp_r7 (1) — LUT 用训练数据构建（不是 ranking，否则全 NaN）
  ⏳ compute_mvt_anchor_features (16个, bit_sweep + 尾流) — 待重建

接口对齐旧版:
  - compute_extra_features(df_dep, df_arr, label) -> dict of arrays (9计数+4延误)
  - build_t_unimp_lut(df_train) -> dict
  - lookup_t_unimp(df, lut) -> (values, used_level)
  - compute_mvt_anchor_features(df_dep, label) -> dict of arrays (16个mvt_*)
  - join_feats_pos_train(df_full, df_dep_raw, feat_dict, feat_names, label)
  - MVT_FEATS_R5 / G2_FEATS / G3_FEATS / G5_FEATS / G7_FEATS (兼容名)

================================================================
【已验证工程修复】（TE add-one 实验沉淀，2026-09-25）
  - LUT 必用训练数据构建，不能用 ranking（后者无 TAXITIME 标签 → t_unimp 全 NaN → 
    跑道级队列特征（mvt_n_ahead/pos_rw/w_actual_*）全 0/1 → 模型退化到 G0 弱基线）
  - 类别编码查表：用 pd.Series.reindex(.) 替代 .map(dict)，前者对未登录类别返回 NaN，
    可统一 fillna(-1)；map 默认填旧值会污染新样本
  - TE (target encoding) 三件套（见 te_v3 pipeline，可复用）：
      1) OOF（K-Fold）防泄漏
      2) 全局均值回退（防冷启动）
      3) 命中率监控（暴露缺失键占比）
  - 验证方法：在标准 holdout（train 2025-02~12 → val 2025-01+07）上整体跑 + 集成 RMSE 
    必须复现 ≈ 281.65，才算口径一致；本题不可用 ranking-vs-submission 自洽差异当基准
================================================================
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

# ============================================================
# 兼容用的特征名列表 (v15 实际只用 MVT_FEATS_R5 + G2+G3+G5+G7)
# R7slim 69 = Gd(39) + MVT锚定(16) + 计数(9) + dly(4) + t_unimp(1)
# ============================================================
MVT_FEATS_R5 = [
    "mvt_n_ahead", "mvt_mu_rwy_5", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
    "mvt_w_actual_5", "mvt_w_actual_15", "mvt_w_actual_30",
    "mvt_q_peak", "mvt_q_mean",
    "mvt_na_J", "mvt_na_H", "mvt_na_M", "mvt_na_L", "mvt_na_heavy_frac",
    "mvt_pos_rw", "mvt_na_heavy_x_pos",
]  # 16 个
G2_FEATS = [
    "dep_cnt_m_5", "dep_cnt_m_15", "dep_cnt_m_30",
]  # 服务率计数 G2
G3_FEATS = [
    "aobt_cnt_m_5", "aobt_cnt_m_15", "aobt_cnt_m_30",
]  # 时间窗 AOBT 计数 G3
G5_FEATS = [
    "eobt_ahead_cnt_5", "eobt_ahead_cnt_15", "eobt_ahead_cnt_30",
]  # 未来计划窗 G5
G7_FEATS = [
    "dly_aobt_eobt", "dly_aobt_iobt", "dly_aobt_sched", "dly_lobt_aobt",
    "t_unimp_r7",
]  # 延误 + t_unimp = G7


# ============================================================
# 工具: 时间窗计数 (向量化)
# ============================================================
def _window_count_1d(target_times_ns, event_times_ns, window_sec,
                     direction="past"):
    """对每个 target, 计算 window 内 event 数.
    target_times_ns: (n,) int64 ns
    event_times_ns:  (m,) int64 ns (已排序)
    window_sec: 窗口宽度(秒)
    direction: 'past' = [t-w, t], 'future' = [t, t+w]
    返回: (n,) int32
    """
    w_ns = int(window_sec * 1e9)
    if direction == "past":
        lo = target_times_ns - w_ns
        hi = target_times_ns
    else:
        lo = target_times_ns
        hi = target_times_ns + w_ns
    # searchsorted: 右边界 idx - 左边界 idx = 窗口内数量
    left = np.searchsorted(event_times_ns, lo, side="left")
    right = np.searchsorted(event_times_ns, hi, side="right")
    return (right - left).astype(np.int32)


# ============================================================
# 1. 计划延误 4 个 + 计数 9 个 (共 13)
# ============================================================
def compute_extra_features(df_dep, df_arr=None, label=""):
    """对 DEP 帧计算额外特征 (dly_4 + dep_cnt_3 + aobt_cnt_3 + eobt_ahead_cnt_3).

    df_dep: DEP 帧 (含 AOBT_3_flt, EOBT_1_flt, IOBT_flt, LOBT_flt,
                   MVT_TIME_UTC_mvt, ADEP_mvt, SCHED_TIME_UTC_mvt)
    df_arr: 未用 (保留接口)
    返回: dict {feat_name: np.ndarray(len(df_dep))}
    """
    t0 = time.time()
    n = len(df_dep)
    out = {}

    # --- 4 个计划延误 (秒) ---
    aobt = pd.to_datetime(df_dep["AOBT_3_flt"], utc=True, errors="coerce").values
    eobt = pd.to_datetime(df_dep["EOBT_1_flt"], utc=True, errors="coerce").values
    iobt = pd.to_datetime(df_dep["IOBT_flt"], utc=True, errors="coerce").values
    lobt = pd.to_datetime(df_dep["LOBT_flt"], utc=True, errors="coerce").values
    sched = pd.to_datetime(df_dep["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce").values

    def _diff_sec(a, b):
        """a - b (秒), 任一 NaT -> NaN"""
        with np.errstate(invalid="ignore"):
            diff = (a.astype("datetime64[ns]").view("i8")
                    - b.astype("datetime64[ns]").view("i8"))
        nan_mask = (a.view("i8") == 0) | (b.view("i8") == 0)
        # NaT 是 iinfo(int64).min? 不, pandas 的 NaT 在 int64 是 iNaT
        iNaT = np.iinfo(np.int64).min
        nan_mask = ((a.view("i8") == iNaT) | (b.view("i8") == iNaT))
        result = diff / 1e9
        result[nan_mask] = np.nan
        return result.astype(np.float32)

    out["dly_aobt_eobt"] = _diff_sec(aobt, eobt)
    out["dly_aobt_iobt"] = _diff_sec(aobt, iobt)
    out["dly_aobt_sched"] = _diff_sec(aobt, sched)
    out["dly_lobt_aobt"] = _diff_sec(lobt, aobt)

    # --- 计数特征 (按机场分组) ---
    mvt = pd.to_datetime(df_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").values
    aep = df_dep["ADEP_mvt"].astype(str).values
    unique_aep = np.unique(aep)

    for name in ["dep_cnt_m", "aobt_cnt_m", "eobt_ahead_cnt"]:
        for w in [5, 15, 30]:
            out[f"{name}_{w}"] = np.full(n, np.nan, dtype=np.float32)

    mvt_i = mvt.astype("datetime64[ns]").view("i8")
    aobt_i = aobt.astype("datetime64[ns]").view("i8")
    eobt_i = eobt.astype("datetime64[ns]").view("i8")
    iNaT = np.iinfo(np.int64).min

    for ap in unique_aep:
        mask = aep == ap
        if mask.sum() == 0:
            continue
        mvt_ap = mvt_i[mask]
        mvt_sorted = np.sort(mvt_ap[mvt_ap != iNaT])
        # dep_cnt_m_*: MVT 为参考点, 过去 w 分钟内的 MVT 数 (起飞流量)
        for w in [5, 15, 30]:
            cnt = _window_count_1d(mvt_ap, mvt_sorted, w * 60, "past")
            out[f"dep_cnt_m_{w}"][mask] = cnt.astype(np.float32)
        # aobt_cnt_m_*: MVT 为参考点, 过去 w 分钟内的 AOBT 数 (推出流量)
        aobt_ap = aobt_i[mask]
        aobt_valid = np.sort(aobt_ap[aobt_ap != iNaT])
        for w in [5, 15, 30]:
            cnt = _window_count_1d(mvt_ap, aobt_valid, w * 60, "past")
            out[f"aobt_cnt_m_{w}"][mask] = cnt.astype(np.float32)
        # eobt_ahead_cnt_*: MVT 为参考点, 未来 w 分钟内的 EOBT 数 (预计后续流量)
        eobt_ap = eobt_i[mask]
        eobt_valid = np.sort(eobt_ap[eobt_ap != iNaT])
        for w in [5, 15, 30]:
            cnt = _window_count_1d(mvt_ap, eobt_valid, w * 60, "future")
            out[f"eobt_ahead_cnt_{w}"][mask] = cnt.astype(np.float32)

    print(f"  extra ({label}): {time.time()-t0:.1f}s, DEP {n:,}", flush=True)
    return out


# ============================================================
# 2. t_unimp_r7 (未受阻碍滑行时间, LUT 查表)
# ============================================================
def build_t_unimp_lut(df_train, quantile=0.3, min_sec=120, max_sec=3600,
                       min_obs=10, clamp_lo=80, clamp_hi=4000):
    """从训练帧 DEP 构建 t_unimp LUT (干净版).
    键层级: (ADEP, STAND, RUNWAY) → (ADEP, STAND) → (ADEP,)
    值: TAXITIME_SEC_mvt 的 quantile 分位数 (默认 0.3, 代表未受阻碍滑行时间)

    干净化处理 (2026-09-25 落地):
      - 训练过滤: TAXITIME in [min_sec, max_sec] (剔除 < 120s 的脏数据和 > 3600s 的极端长尾)
      - 样本阈值: L1/L2/L3 group 内 obs < min_obs 时 q30 标 NaN (回退)
      - 截断到物理范围: q30 ∉ [clamp_lo, clamp_hi] 时标 NaN (避免脏数据污染)
      - 全局兜底: dep_clean 的 q30 中位数
    """
    dep = df_train[df_train["PHASE_mvt"] == "DEP"]
    dep_clean = dep[(dep["TAXITIME_SEC_mvt"] >= min_sec) &
                    (dep["TAXITIME_SEC_mvt"] <= max_sec)]

    g1 = dep_clean.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")
    g2 = dep_clean.groupby(["ADEP_mvt", "STAND_mvt"])["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")
    g3 = dep_clean.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")

    for g in [g1, g2, g3]:
        g.loc[g["n"] < min_obs, "q"] = np.nan
        g.loc[(g["q"] < clamp_lo) | (g["q"] > clamp_hi), "q"] = np.nan

    lut = {
        "quantile": quantile, "min_sec": min_sec, "max_sec": max_sec,
        "min_obs": min_obs, "clamp_lo": clamp_lo, "clamp_hi": clamp_hi,
        "L1_q": g1["q"], "L1_n": g1["n"],
        "L2_q": g2["q"], "L2_n": g2["n"],
        "L3_q": g3["q"], "L3_n": g3["n"],
        "global_q30": float(dep_clean["TAXITIME_SEC_mvt"].quantile(quantile)),
    }
    # 兼容旧接口 (L1/L2/L3 直接 Series)
    lut["L1"] = g1["q"]
    lut["L2"] = g2["q"]
    lut["L3"] = g3["q"]
    return lut


def build_t_unimp_lut_legacy(df_train, quantile=0.3):
    """旧版 LUT (无干净化), 仅供审计/对比使用. 新代码请用 build_t_unimp_lut."""
    dep = df_train[df_train["PHASE_mvt"] == "DEP"]
    lut = {"quantile": quantile}
    lut["L1"] = dep.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])["TAXITIME_SEC_mvt"].quantile(quantile)
    lut["L2"] = dep.groupby(["ADEP_mvt", "STAND_mvt"])["TAXITIME_SEC_mvt"].quantile(quantile)
    lut["L3"] = dep.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].quantile(quantile)
    return lut


def lookup_t_unimp(df, lut, min_obs=5):
    """查表 t_unimp. 返回 (values, used_level_array)

    兼容新旧 LUT 格式:
      - 新版 (build_t_unimp_lut): 用 L1_q/L2_q/L3_q 字段 + global_q30 兜底
      - 旧版 (build_t_unimp_lut_legacy): 用 L1/L2/L3 字段
    """
    dep_mask = (df["PHASE_mvt"] == "DEP").values
    n = len(df)
    vals = np.full(n, np.nan, dtype=np.float32)
    levels = np.full(n, "", dtype=object)

    dep_idx = np.where(dep_mask)[0]
    sub = df.iloc[dep_idx]
    result = np.full(len(dep_idx), np.nan, dtype=np.float32)
    used = np.full(len(dep_idx), "", dtype=object)

    ap = sub["ADEP_mvt"].astype(str).values
    st = sub["STAND_mvt"].astype(str).values
    rw = sub["RUNWAY_mvt"].astype(str).values

    # L1
    L1_key = "L1_q" if "L1_q" in lut else "L1"
    keys1 = pd.MultiIndex.from_arrays([ap, st, rw])
    v1 = lut[L1_key].reindex(keys1).values
    filled = ~np.isnan(v1)
    result[filled] = v1[filled]
    used[filled] = "L1"

    # L2 回退
    need = np.isnan(result)
    if need.sum() > 0:
        L2_key = "L2_q" if "L2_q" in lut else "L2"
        keys2 = pd.MultiIndex.from_arrays([ap[need], st[need]])
        v2 = lut[L2_key].reindex(keys2).values
        filled2 = ~np.isnan(v2)
        idx_need = np.where(need)[0]
        result[idx_need[filled2]] = v2[filled2]
        used[idx_need[filled2]] = "L2"

    # L3 回退
    need = np.isnan(result)
    if need.sum() > 0:
        L3_key = "L3_q" if "L3_q" in lut else "L3"
        idx_need = np.where(need)[0]
        v3 = lut[L3_key].reindex(ap[need]).values
        filled3 = ~np.isnan(v3)
        result[idx_need[filled3]] = v3[filled3]
        used[idx_need[filled3]] = "L3"

    # 全局兜底 (新版有, 旧版没有)
    need = np.isnan(result)
    if need.sum() > 0 and "global_q30" in lut:
        idx_need = np.where(need)[0]
        result[idx_need] = lut["global_q30"]
        used[idx_need] = "GLOBAL"

    vals[dep_idx] = result
    levels[dep_idx] = used
    return vals, levels


# ============================================================
# 3. MVT 锚定特征 (bit_sweep + 尾流) — 占位, 待重建
# ============================================================
def compute_mvt_anchor_features(df_dep, t_unimp_vals=None, label=""):
    """MVT 锚定特征 (bit_sweep 同跑道队列 + 尾流 7 量 + q_peak/q_mean + mu_rwy).

    队列定义 (跑道级):
      enter 时间 = AOBT + t_unimp (估计到达跑道入口时间)
      leave 时间 = MVT (起飞时间)
      n_ahead = 在 MVT_i 时刻, 同跑道上已到达跑道入口但未起飞的前机数
    
    若 t_unimp_vals 为 None, 则退化为 AOBT enter (旧行为, 不推荐)。
    
    mu_rwy 单位: 架次/分钟 (过去 w 分钟起飞架次 / w)
    w_actual 单位: 分钟 (n_ahead / mu_rwy)

    输入: DEP 帧, 含 AOBT_3_flt, MVT_TIME_UTC_mvt, RUNWAY_mvt, WK_TBL_CAT_flt, ADEP_mvt
          t_unimp_vals: 每个航班的未受阻碍滑行时间 (秒), 用于估计到达跑道入口时间
    返回: dict {feat_name: np.ndarray(len(df_dep))}
    """
    t0 = time.time()
    n = len(df_dep)
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in MVT_FEATS_R5}

    aobt = pd.to_datetime(df_dep["AOBT_3_flt"], utc=True, errors="coerce").values
    mvt = pd.to_datetime(df_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").values
    rw = df_dep["RUNWAY_mvt"].astype(str).values
    aep = df_dep["ADEP_mvt"].astype(str).values
    wake = df_dep["WK_TBL_CAT_flt"].astype(str).values

    aobt_i = aobt.astype("datetime64[ns]").view("i8")
    mvt_i = mvt.astype("datetime64[ns]").view("i8")
    iNaT = np.iinfo(np.int64).min
    aobt_v = aobt_i != iNaT
    mvt_v = mvt_i != iNaT
    
    # === enter 时间: AOBT + t_unimp (到达跑道入口) ===
    if t_unimp_vals is not None:
        t_unimp_ns = (t_unimp_vals * 1e9).astype("int64")  # 秒 → 纳秒
        enter_i = aobt_i + t_unimp_ns
        enter_i = np.minimum(enter_i, mvt_i)  # 不晚于 MVT
        enter_v = aobt_v & ~np.isnan(t_unimp_vals) & mvt_v
    else:
        # 旧行为: enter = AOBT
        enter_i = aobt_i.copy()
        enter_v = aobt_v.copy()

    # ===== 按跑道分组 bit_sweep =====
    unique_rw = np.unique(rw)
    mvt_arr = mvt_i.copy()
    mvt_arr[~mvt_v] = np.iinfo(np.int64).max  # NaT → 末尾排序最后

    for rwy in unique_rw:
        mask = rw == rwy
        idx = np.where(mask)[0]
        m_g = len(idx)
        if m_g < 2:
            continue

        enter_g = enter_i[idx]
        mvt_g = mvt_arr[idx]
        enter_v_g = enter_v[idx]
        mvt_v_g = mvt_v[idx]
        wake_g = wake[idx]

        # === 构造事件数组 ===
        #   enter 事件: 到达跑道入口, kind=+1
        #   leave 事件: 起飞, kind=-1
        #   query 事件: 起飞时刻查询队列长度, kind=0

        n_enter = int(enter_v_g.sum())
        # leave 事件只有 enter 有效的航班才产生
        both_v = enter_v_g & mvt_v_g
        n_leave = int(both_v.sum())

        ev_times = np.concatenate([
            enter_g[enter_v_g],  # enter
            mvt_g[both_v],        # leave
            mvt_g,               # query
        ])
        ev_kinds = np.concatenate([
            np.ones(n_enter, dtype="int8"),
            -np.ones(n_leave, dtype="int8"),
            np.zeros(m_g, dtype="int8"),
        ])
        # 同时刻优先级: enter(+1) > query(0) > leave(-1)
        ev_wake = np.concatenate([
            wake_g[enter_v_g],
            wake_g[both_v],
            np.full(m_g, "", dtype=wake_g.dtype),
        ])
        ev_query_idx = np.concatenate([
            np.full(n_enter, -1, dtype=np.int32),
            np.full(n_leave, -1, dtype=np.int32),
            np.arange(m_g, dtype=np.int32),
        ])

        # 排序: 时间 asc, 优先级 desc
        order = np.lexsort((-ev_kinds.astype(int), ev_times))

        # sweep line
        sweep_n = 0
        sweep_J = 0
        sweep_H = 0
        sweep_M = 0
        sweep_L = 0

        q_n = np.full(m_g, np.nan, dtype="float32")
        q_J = np.full(m_g, np.nan, dtype="float32")
        q_H = np.full(m_g, np.nan, dtype="float32")
        q_M = np.full(m_g, np.nan, dtype="float32")
        q_L = np.full(m_g, np.nan, dtype="float32")

        for p in range(len(order)):
            k = int(ev_kinds[order[p]])
            w = ev_wake[order[p]]
            qi = ev_query_idx[order[p]]
            if k == 1:  # enter
                sweep_n += 1
                if w == "J": sweep_J += 1
                elif w == "H": sweep_H += 1
                elif w == "M": sweep_M += 1
                elif w == "L": sweep_L += 1
            elif k == -1:  # leave
                sweep_n -= 1
                if w == "J": sweep_J -= 1
                elif w == "H": sweep_H -= 1
                elif w == "M": sweep_M -= 1
                elif w == "L": sweep_L -= 1
            else:  # query
                self_in = enter_v_g[qi]
                if self_in:
                    q_n[qi] = float(max(sweep_n - 1, 0))
                    self_wake = wake_g[qi]
                    self_J = 1 if self_wake == "J" else 0
                    self_H = 1 if self_wake == "H" else 0
                    self_M = 1 if self_wake == "M" else 0
                    self_L = 1 if self_wake == "L" else 0
                    q_J[qi] = float(max(sweep_J - self_J, 0))
                    q_H[qi] = float(max(sweep_H - self_H, 0))
                    q_M[qi] = float(max(sweep_M - self_M, 0))
                    q_L[qi] = float(max(sweep_L - self_L, 0))
                else:
                    q_n[qi] = float(sweep_n)
                    q_J[qi] = float(sweep_J)
                    q_H[qi] = float(sweep_H)
                    q_M[qi] = float(sweep_M)
                    q_L[qi] = float(sweep_L)

        out["mvt_n_ahead"][idx] = q_n
        out["mvt_na_J"][idx] = q_J
        out["mvt_na_H"][idx] = q_H
        out["mvt_na_M"][idx] = q_M
        out["mvt_na_L"][idx] = q_L
        out["mvt_na_heavy_frac"][idx] = np.where(
            q_n > 0, (q_J + q_H) / q_n, 0.0).astype(np.float32)

        # pos_rw: 队列中的位置 (1-based) = n_ahead + 1
        pos_rw = (q_n + 1).astype(np.float32)
        out["mvt_pos_rw"][idx] = pos_rw
        out["mvt_na_heavy_x_pos"][idx] = (
            (q_J + q_H) * pos_rw).astype(np.float32)

        # === mu_rwy: 过去 w 分钟内同跑道 MVT 数 / w (分钟) → 架次/分钟 ===
        mvt_sorted = np.sort(mvt_g[mvt_v_g])
        for w, name in [(5, "mvt_mu_rwy_5"), (15, "mvt_mu_rwy_15"),
                         (30, "mvt_mu_rwy_30")]:
            w_ns = w * 60 * int(1e9)
            left = np.searchsorted(mvt_sorted, mvt_g - w_ns, side="left")
            right = np.searchsorted(mvt_sorted, mvt_g, side="right")
            cnt = (right - left).astype(np.float32)
            mu = cnt / w  # 架次/分钟
            out[name][idx] = mu

        # === w_actual = n_ahead / mu (分钟) ===
        for w in [5, 15, 30]:
            mu = out[f"mvt_mu_rwy_{w}"][idx]
            n_a = out["mvt_n_ahead"][idx]
            result = np.where(mu > 0, n_a / mu, np.nan)
            out[f"mvt_w_actual_{w}"][idx] = result.astype(np.float32)

    # ===== q_peak / q_mean (机场级, [AOBT, MVT] 窗口内) =====
    # 对每个机场, 计算航班 i 在 [AOBT_i, MVT_i] 时段内机场级队列的峰值和均值
    # 队列定义: 已 AOBT (推出) 未 MVT (起飞) 的航班数 (机场级, 全机场总拥堵)
    # 注意: 这里用 AOBT 做 enter, 与跑道级队列 (enter = AOBT + t_unimp) 不同
    for ap in np.unique(aep):
        mask = aep == ap
        idx_ap = np.where(mask)[0]
        m_ap = len(idx_ap)
        if m_ap < 10:
            continue

        aobt_ap = aobt_i[mask]
        mvt_ap = mvt_arr[mask]  # NaT → inf
        aobt_v_ap = aobt_v[mask]
        mvt_v_ap = mvt_v[mask]

        # --- 构造事件 (仅 aobt 有效的 enter, 且 aobt+mvt 都有效的 leave) ---
        both_v_ap = aobt_v_ap & mvt_v_ap
        n_ent = int(aobt_v_ap.sum())
        n_lev = int(both_v_ap.sum())

        ev_t = np.concatenate([aobt_ap[aobt_v_ap], mvt_ap[both_v_ap]])
        ev_k = np.concatenate([
            np.ones(n_ent, dtype="int8"),
            -np.ones(n_lev, dtype="int8"),
        ])
        # 排序: 时间 asc, 同时间 enter 先于 leave
        order_ap = np.lexsort((-ev_k.astype(int), ev_t))

        times_sorted = ev_t[order_ap]
        cum = np.cumsum(ev_k[order_ap].astype("int32"))

        # 前缀时间差 (用于积分)
        # q 在 (times_sorted[k], times_sorted[k+1]) 区间内的值 = cum[k]
        dt = np.diff(times_sorted.astype("int64"))  # 纳秒
        # 积分前缀和: prefix_int[k] = Σ_{j<k} dt[j] * cum[j]
        if len(dt) > 0:
            q_vals = cum[:-1].astype(np.float64)
            int_vals = dt.astype(np.float64) * q_vals
            prefix_int = np.concatenate([[0.0], np.cumsum(int_vals)])
        else:
            prefix_int = np.array([0.0])

        q_peak_arr = np.full(m_ap, np.nan, dtype="float32")
        q_mean_arr = np.full(m_ap, np.nan, dtype="float32")

        for i in range(m_ap):
            t_start = aobt_ap[i] if aobt_v_ap[i] else mvt_ap[i]
            t_end = mvt_ap[i]
            if t_start >= t_end or not mvt_v_ap[i]:
                continue

            # 窗口内事件: times_sorted 在 [t_start, t_end] 内
            left = np.searchsorted(times_sorted, t_start, side="left")
            right = np.searchsorted(times_sorted, t_end, side="right") - 1

            if left > right:
                # 窗口内没有事件, 取窗口起点的队列长度
                q_start = cum[left - 1] if left > 0 else 0
                # 减去自己 (如果自己 aobt 有效且在窗口内)
                if aobt_v_ap[i]:
                    q_start = max(q_start - 1, 0)
                q_peak_arr[i] = float(q_start)
                q_mean_arr[i] = float(q_start)
                continue

            # 窗口起点处的队列长度
            q_start = cum[left - 1] if left > 0 else 0
            # 窗口终点处的队列长度 (cum[right] 是处理完 right 事件后的队列长度)
            q_end = cum[right]

            # 峰值: max(窗口起点 q, 窗口内所有 cum)
            peak = max(int(q_start), int(cum[left: right + 1].max()))

            # 均值: 积分 / 时长
            int_total = 0.0
            if right > left:
                int_total += (prefix_int[right] - prefix_int[left])
            # 起始段: (times_sorted[left] - t_start) * q_start
            int_total += (times_sorted[left] - t_start) * q_start
            # 结束段: (t_end - times_sorted[right]) * q_end
            int_total += (t_end - times_sorted[right]) * q_end

            duration_ns = t_end - t_start
            mean_q = int_total / duration_ns if duration_ns > 0 else float(q_start)

            # 排除自己 (自己的 enter 在窗口起点, leave 在窗口终点外)
            peak_self = max(peak - 1, 0)
            mean_self = max(mean_q - 1, 0.0)

            q_peak_arr[i] = float(peak_self)
            q_mean_arr[i] = float(mean_self)

        out["mvt_q_peak"][idx_ap] = q_peak_arr
        out["mvt_q_mean"][idx_ap] = q_mean_arr

    print(f"  mvt_anchor ({label}): {time.time()-t0:.1f}s, DEP {n:,}", flush=True)
    return out


# ============================================================
# 4. join 工具
# ============================================================
def join_feats_pos_train(df_full, df_dep_raw, feat_dict, feat_names, label="join"):
    """把 DEP 级特征 join 回 full 帧 (按行号对齐, DEP 才有效, ARR 为 NaN).
    df_full: 含 DEP+ARR 的 full 特征帧
    df_dep_raw: 原始 DEP 帧 (用于定位行号)
    feat_dict: {feat_name: array(n_dep)}
    feat_names: 要 join 的特征名列表
    """
    # 假设 df_full 与 df_dep_raw 顺序一致且 DEP 行同序 (build_full_features 不重排)
    # 实际更安全: 用 MVT_ID_mvt 对齐
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    for name in feat_names:
        if name not in feat_dict:
            continue
        arr = feat_dict[name]
        df_full[name] = np.nan
        df_full.loc[dep_mask, name] = arr
    return None
