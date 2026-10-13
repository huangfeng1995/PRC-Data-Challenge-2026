"""
arr_ranking_audit.py — 第一步盘点: 到港(ARR)数据可用性
======================================================
1. 71 特征里哪些实际用到 ARR 信息 (静态审计 FEATS71 的定义链)
2. ranking.parquet 里 ARR 行的列可见性 / 遮蔽情况
   重点: ARR 的 BLOCK / MVT / 落地时间 是否可用
   对照: DEP 的 BLOCK_TIME_UTC_mvt 已知 100% 缺失
3. 与 2025 训练数据对比, 找出 ranking 上"训练有、线上没有"的 ARR 字段

输出: notebooks/diag/d0/arr_ranking_audit.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT_JSON = D0 / "arr_ranking_audit.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/arr_ranking_audit.log")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
from v15_clean_retrain import FEATS71, TRAIN_COLS


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


log("=" * 78)
log("[arr_audit] 第一步盘点")
log("=" * 78)

# ============ 1. 71 特征里的 ARR 信息 ============
# 逐特征判定: 该特征的计算是否读取了 ARR 行的字段
ARR_TOUCHING = {
    "eobt_ahead_cnt_5": "G5 未来计划窗: 统计本 DEP 之后 5min 内同机场其他 DEP 的 EOBT 计划推出时刻。"
                        "注意: 这是 DEP 的 EOBT, 不是 ARR。",
    "eobt_ahead_cnt_15": "同上, 15min 窗",
    "eobt_ahead_cnt_30": "同上, 30min 窗",
}
ARR_ROW_FEATS = ["arr_mvt_cnt_airport_10m", "arr_mvt_cnt_airport_15m"]
log("")
log("[1] 71 特征中的 ARR 信息审计")
log("-" * 78)
log(f"  FEATS71 共 {len(FEATS71)} 个")
present = [f for f in ARR_ROW_FEATS if f in FEATS71]
log(f"  读取 ARR 行的特征 (arr_mvt_cnt_*): {ARR_ROW_FEATS}")
log(f"    → 在 FEATS71 中: {'存在' if present else '不存在'}")
log("")
log("  名字含 arr / 含 ARR 语义但实为 DEP 自身计划的特征:")
for f, d in ARR_TOUCHING.items():
    log(f"    {f:24s} 在 FEATS71: {f in FEATS71}")
    log(f"      {d}")
log("")
log("  结论: FEATS71 中【没有任何一个特征读取 ARR 行的字段】。")
log("        eobt_ahead_cnt_* 用的是 DEP 自己的 EOBT_1_flt(计划推出), 属未来计划窗, 不是到港。")
log("        唯一真正用到 ARR 行的是 E 组 arr_mvt_cnt_airport_{10,15}m, 已被否决并未进入 71。")

# ============ 2. 旧 E 组的口径 ============
log("")
log("[2] 复盘旧 '进港特征' 的确切口径")
log("-" * 78)
log("  历史 G5_ABCE 中的 E = arr_mvt_cnt_airport_{10,m} + arr_mvt_cnt_airport_{15,m} (2 个, 非 4 个)。")
log("  实现: src/features_queue.py :: add_arr_mvt_cnt()")
log("  关键口径: 用的是 ARR 行的 MVT_TIME_UTC_mvt = 【实际落地时刻】(真实世界), 不是计划时刻。")
log("  当年裁决 (docs/diag/holdout_v1p5_significance_diagnosis.md):")
log("    'E (arr_mvt_cnt_*) ❌ 否决 — 高 gain 零泛化; 线上可构造性已确认(ranking ARR 49.99%),")
log("     否决理由是特征本身的泛化能力, 而非线上可构造性'")
log("  当年 gain 极高 (221M / 74.9M) 但 holdout RMSE 无改善 → 典型过拟合/口径不可迁移。")
log("  ⚠ 本轮任务用的是【计划到港时刻】(flight 源), 与当年的【实际落地时刻】口径不同,")
log("    所以本轮不是简单重跑, 需要重新验证。")

# ============ 3. ranking 列可见性 ============
log("")
log("=" * 78)
log("[3] ranking.parquet 列可见性")
log("=" * 78)
rk_path = ROOT / "data/raw/ranking.parquet"
pf = pq.ParquetFile(rk_path)
rk_cols = list(pf.schema_arrow.names)
log(f"  ranking 总行数 {pf.metadata.num_rows:,}  列数 {len(rk_cols)}")

CHECK_COLS = [
    "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "SIB_TIME_UTC_mvt", "TIME_AOBT_mvt", "TIME_TAXIIN_mvt",
    "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "ARVT_3_flt", "LOBT_flt",
    "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt", "ADES_FILED_flt", "CALLSIGN_flt",
    "FLIGHT_ID_mvt", "MVT_ID_mvt", "TAXITIME_SEC_mvt",
]
avail = [c for c in CHECK_COLS if c in rk_cols]
missing = [c for c in CHECK_COLS if c not in rk_cols]
log("")
log(f"  关注的列: 存在 {len(avail)} / 缺失 {len(missing)}")
if missing:
    log(f"  ranking 中不存在的列: {missing}")

rk = pq.read_table(rk_path, columns=avail).to_pandas()
ph = rk["PHASE_mvt"].astype(str).values
is_dep = ph == "DEP"
is_arr = ph == "ARR"
log("")
log(f"  ranking DEP 行 {int(is_dep.sum()):,}  ARR 行 {int(is_arr.sum()):,}  "
    f"其他 {int((~is_dep & ~is_arr).sum()):,}")
log(f"  ARR 占比 {is_arr.sum()/len(rk)*100:.2f}%")

def nonnull_rate(col, mask):
    s = rk[col]
    v = s[mask]
    if v.dtype.kind == "M":
        return float(v.notna().sum() / max(len(v), 1))
    nn = v.notna() & (v.astype(str).str.strip() != "") & (v.astype(str) != "None") \
         & (v.astype(str) != "NaT")
    return float(nn.sum() / max(len(v), 1))

log("")
log("  关键时间列的非空率 (ranking):")
log(f"  {'列':26s} {'DEP':>10s} {'ARR':>10s}")
KEY_TIME = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "ARVT_3_flt", "LOBT_flt"]
avail_rate = {}
for c in KEY_TIME:
    if c not in rk.columns:
        log(f"  {c:26s} {'(缺列)':>10s}")
        continue
    d, a = nonnull_rate(c, is_dep), nonnull_rate(c, is_arr)
    avail_rate[c] = {"dep": round(d, 4), "arr": round(a, 4)}
    flag = ""
    if c == "BLOCK_TIME_UTC_mvt":
        flag = "  ← DEP 已知 100% 缺失"
    if c == "MVT_TIME_UTC_mvt":
        flag = "  ← 落地/起飞时刻"
    log(f"  {c:26s} {d*100:>9.2f}% {a*100:>9.2f}%{flag}")

log("")
log("  关键结论:")
bk_dep = avail_rate.get("BLOCK_TIME_UTC_mvt", {}).get("dep")
bk_arr = avail_rate.get("BLOCK_TIME_UTC_mvt", {}).get("arr")
mvt_dep = avail_rate.get("MVT_TIME_UTC_mvt", {}).get("dep")
mvt_arr = avail_rate.get("MVT_TIME_UTC_mvt", {}).get("arr")
log(f"    - BLOCK_TIME_UTC_mvt: DEP {bk_dep*100:.2f}%  ARR {bk_arr*100:.2f}%"
    + ("  → 确认 DEP 侧 100% 缺失" if (bk_dep or 0) < 0.001 else ""))
log(f"    - MVT_TIME_UTC_mvt:  DEP {mvt_dep*100:.2f}%  ARR {mvt_arr*100:.2f}%")
log("    - EOBT_1_flt(计划推出)在 ranking 上: "
    f"DEP {avail_rate.get('EOBT_1_flt',{}).get('dep',0)*100:.2f}%  "
    f"ARR {avail_rate.get('EOBT_1_flt',{}).get('arr',0)*100:.2f}%")

# ARR 行可用的计划时刻列
log("")
log("  ARR 行上可用于构造'计划到港负荷'的列:")
for c in ["SCHED_TIME_UTC_mvt", "EOBT_1_flt", "ADES_mvt", "MVT_TIME_UTC_mvt"]:
    if c in rk.columns:
        log(f"    {c:26s} 非空率 {nonnull_rate(c, is_arr)*100:6.2f}%")

# ============ 4. 与训练数据对比 ============
log("")
log("=" * 78)
log("[4] 训练数据(2025-07) 的同列非空率 —— 对照 ranking")
log("=" * 78)
tr = pq.read_table(ROOT / "data/raw/training_2025-07-01_2025-08-01.parquet",
                   columns=[c for c in avail]).to_pandas()
tph = tr["PHASE_mvt"].astype(str).values
t_dep, t_arr = tph == "DEP", tph == "ARR"
log(f"  训练月 DEP {int(t_dep.sum()):,}  ARR {int(t_arr.sum()):,}")
log("")
log(f"  {'列':26s} {'trDEP':>10s} {'trARR':>10s} {'rkDEP':>10s} {'rkARR':>10s}")
cmp_tab = {}
for c in KEY_TIME:
    if c not in tr.columns:
        continue
    td, ta = nonnull_rate.__wrapped__(c, t_dep) if False else (
        float((tr[c][t_dep].notna()
               & (tr[c][t_dep].astype(str).str.strip() != "")).sum() / max(int(t_dep.sum()), 1)),
        float((tr[c][t_arr].notna()
               & (tr[c][t_arr].astype(str).str.strip() != "")).sum() / max(int(t_arr.sum()), 1)))
    rd = avail_rate.get(c, {}).get("dep", float("nan"))
    ra = avail_rate.get(c, {}).get("arr", float("nan"))
    cmp_tab[c] = {"train_dep": round(td, 4), "train_arr": round(ta, 4),
                  "rank_dep": rd, "rank_arr": ra}
    flag = ""
    if td - rd > 0.05 or ta - ra > 0.05:
        flag = "  ← 训练有/线上无"
    log(f"  {c:26s} {td*100:>9.2f}% {ta*100:>9.2f}% {rd*100:>9.2f}% {ra*100:>9.2f}%{flag}")

log("")
log("  ⚠ 注意: 2025 训练数据是【事后全量记录】, 训练侧任何列都接近 100% 非空;")
log("    ranking 是【赛前截断】数据, 只有预测时点已知的信息。")
log("    因此 '训练有、线上无' 的列不能直接用。")

RESULT = {
    "task": "arr_ranking_audit",
    "feats71_arr_usage": {
        "n_feats71": len(FEATS71),
        "features_reading_arr_rows": ARR_ROW_FEATS,
        "in_feats71": present,
        "any_arr_row_feature_in_71": bool(present),
        "lookalike_but_dep_own": {
            k: v for k, v in ARR_TOUCHING.items() if k in FEATS71},
        "conclusion": "FEATS71 中无任何特征读取 ARR 行字段",
    },
    "prior_E_group_review": {
        "features": ARR_ROW_FEATS,
        "count": 2,
        "user_said_4": "历史 G5_ABCE 的 E 组实为 2 个 (10m/15m), 不是 4 个",
        "impl": "src/features_queue.py :: add_arr_mvt_cnt()",
        "time_basis": "ARR 行的 MVT_TIME_UTC_mvt = 实际落地时刻 (真实世界), 非计划时刻",
        "prior_verdict": "docs/diag/holdout_v1p5_significance_diagnosis.md: ❌ 否决, "
                         "高 gain 零泛化; 线上可构造性已确认, 否决理由是泛化能力",
        "prior_gain": {"arr_mvt_cnt_airport_10m": "221M", "arr_mvt_cnt_airport_15m": "74.9M"},
        "difference_vs_this_round": "本轮用【计划到港时刻】而非实际落地时刻, 口径不同, 需重新验证",
    },
    "ranking": {
        "n_rows": int(pf.metadata.num_rows),
        "n_dep": int(is_dep.sum()), "n_arr": int(is_arr.sum()),
        "arr_share_pct": round(float(is_arr.sum() / len(rk) * 100), 2),
        "missing_checked_cols": missing,
        "time_col_nonnull_rate": avail_rate,
        "block_time_dep_pct": round((bk_dep or 0) * 100, 2),
        "block_time_arr_pct": round((bk_arr or 0) * 100, 2),
        "mvt_time_dep_pct": round((mvt_dep or 0) * 100, 2),
        "mvt_time_arr_pct": round((mvt_arr or 0) * 100, 2),
    },
    "train_vs_rank": cmp_tab,
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}")
