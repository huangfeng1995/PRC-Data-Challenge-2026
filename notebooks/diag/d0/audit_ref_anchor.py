"""
audit_ref_anchor.py — FEATS71 参考时刻锚点审计 (只读, 不训练)
================================================================
目的: 判定每个 mvt/计数类特征构造时用的"参考时刻"到底锚在哪一列:
    - AOBT_3_flt / EOBT_1_flt  -> 合法 (推出时刻, 预测时点已知)
    - MVT_TIME_UTC_mvt (自己)   -> 泄漏 (真实未来起飞时刻)

两路证据:
  (1) 源码静态审计: 逐特征记录计算时用的 query/参考时刻变量
  (2) 数值实证: 构造"反事实 MVT" (把 MVT 随机打乱 / 加巨大噪声) 重算特征,
      合法特征应完全不变, 泄漏特征会随之变化。
      实证是最硬的证据, 因为它不依赖对源码的理解。

输出: notebooks/diag/d0/audit_ref_anchor.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
WORK = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b")
OUT_JSON = D0 / "audit_ref_anchor.json"
OUT_LOG = WORK / "audit_ref_anchor.log"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from v15_clean_retrain import FEATS71
import _r7_lib as R7


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


# ============ (1) 源码静态审计表 ============
# src: 源码位置; query: 计算时作为"参考时刻/查询点"的列
STATIC = {
    # ---- compute_mvt_anchor_features (_r7_lib.py:291) ----
    # 全部 16 个都是"query 事件 = mvt_g" (行366: mvt_g 拼进 ev_times 的 query 部分)
    "mvt_n_ahead":        ("_r7_lib.py:291 compute_mvt_anchor_features", "MVT_TIME_UTC_mvt (自身, query 事件)", "泄漏"),
    "mvt_pos_rw":         ("_r7_lib.py:445-447 (pos_rw = q_n+1)", "MVT_TIME_UTC_mvt (自身, query)", "泄漏"),
    "mvt_na_J":           ("_r7_lib.py:438", "MVT_TIME_UTC_mvt (自身, query)", "泄漏"),
    "mvt_na_H":           ("_r7_lib.py:439", "MVT_TIME_UTC_mvt (自身, query)", "泄漏"),
    "mvt_na_M":           ("_r7_lib.py:440", "MVT_TIME_UTC_mvt (自身, query)", "泄漏"),
    "mvt_na_L":           ("_r7_lib.py:441", "MVT_TIME_UTC_mvt (自身, query)", "泄漏"),
    "mvt_na_heavy_frac":  ("_r7_lib.py:442-443", "MVT_TIME_UTC_mvt (自身, query)", "泄漏"),
    "mvt_na_heavy_x_pos": ("_r7_lib.py:448-449", "MVT_TIME_UTC_mvt (自身, query)", "泄漏"),
    "mvt_mu_rwy_5":       ("_r7_lib.py:452-460 (mvt_g - w_ns 窗口)", "MVT_TIME_UTC_mvt (自身中心)", "泄漏"),
    "mvt_mu_rwy_15":      ("_r7_lib.py:452-460", "MVT_TIME_UTC_mvt (自身中心)", "泄漏"),
    "mvt_mu_rwy_30":      ("_r7_lib.py:452-460", "MVT_TIME_UTC_mvt (自身中心)", "泄漏"),
    "mvt_w_actual_5":     ("_r7_lib.py:463-467 (= n_ahead/mu)", "间接经 n_ahead/mu_rwy → MVT", "泄漏"),
    "mvt_w_actual_15":    ("_r7_lib.py:463-467", "间接经 n_ahead/mu_rwy → MVT", "泄漏"),
    "mvt_w_actual_30":    ("_r7_lib.py:463-467", "间接经 n_ahead/mu_rwy → MVT", "泄漏"),
    "mvt_q_peak":         ("_r7_lib.py:473-562 (窗口 [AOBT_i, MVT_i])", "窗口终点 = MVT_TIME_UTC_mvt (自身)", "泄漏"),
    "mvt_q_mean":         ("_r7_lib.py:473-562 (窗口 [AOBT_i, MVT_i])", "窗口终点 = MVT_TIME_UTC_mvt (自身)", "泄漏"),
    # ---- compute_extra_features (_r7_lib.py:93) ----
    "dep_cnt_m_5":        ("_r7_lib.py:150-153 (_window_count_1d(mvt_ap, ...))", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "dep_cnt_m_15":       ("_r7_lib.py:150-153", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "dep_cnt_m_30":       ("_r7_lib.py:150-153", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "aobt_cnt_m_5":       ("_r7_lib.py:154-159 (查询点 mvt_ap, 事件 AOBT)", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "aobt_cnt_m_15":      ("_r7_lib.py:154-159", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "aobt_cnt_m_30":      ("_r7_lib.py:154-159", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "eobt_ahead_cnt_5":   ("_r7_lib.py:160-165 (查询点 mvt_ap, 事件 EOBT)", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "eobt_ahead_cnt_15":  ("_r7_lib.py:160-165", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    "eobt_ahead_cnt_30":  ("_r7_lib.py:160-165", "MVT_TIME_UTC_mvt (自身, 查询点)", "泄漏"),
    # ---- features_queue.py: ref_dt = AOBT (fallback EOBT) ----
    "dep_mvt_cnt_airport_5m":  ("features_queue.py:38-47 add_ref_time + :173", "AOBT_3_flt (回退 EOBT_1_flt) → 合法", "合法"),
    "dep_mvt_cnt_airport_10m": ("features_queue.py:38-47 + :173", "AOBT_3_flt (回退 EOBT_1_flt) → 合法", "合法"),
    "dep_mvt_cnt_airport_15m": ("features_queue.py:38-47 + :173", "AOBT_3_flt (回退 EOBT_1_flt) → 合法", "合法"),
    "dep_mvt_cnt_runway_5m":   ("features_queue.py:38-47 + :195-201", "AOBT_3_flt (回退 EOBT_1_flt) → 合法", "合法"),
    "dep_mvt_cnt_runway_10m":  ("features_queue.py:38-47 + :195-201", "AOBT_3_flt (回退 EOBT_1_flt) → 合法", "合法"),
    "dep_mvt_cnt_runway_15m":  ("features_queue.py:38-47 + :195-201", "AOBT_3_flt (回退 EOBT_1_flt) → 合法", "合法"),
    # ---- _r7_lib.py:125-128 计划延误两列之差, 不含 MVT ----
    "dly_aobt_eobt":          ("_r7_lib.py:125 (AOBT − EOBT)", "AOBT_3_flt / EOBT_1_flt 两列之差, 不含 MVT", "合法"),
    "dly_aobt_iobt":          ("_r7_lib.py:126 (AOBT − IOBT)", "AOBT_3_flt / IOBT_flt 两列之差, 不含 MVT", "合法"),
    "dly_aobt_sched":         ("_r7_lib.py:127 (AOBT − SCHED)", "AOBT_3_flt / SCHED_TIME 两列之差, 不含 MVT", "合法"),
    "dly_lobt_aobt":          ("_r7_lib.py:128 (LOBT − AOBT)", "LOBT_flt / AOBT_3_flt 两列之差, 不含 MVT", "合法"),
    # ---- _r8_drift_lib.py: 只按 (机场/跑道, 月, 小时) 聚合, 无自身时间窗 ----
    "rwy_month_share":         ("_r8_drift_lib.py:42-56", "无自身时刻窗口(仅 (ADEP,RUNWAY,MONTH) 频率)", "合法"),
    "ap_month_hourly_density": ("_r8_drift_lib.py:58-68", "无自身时刻窗口(仅 (ADEP,MONTH,HOUR) 频率)", "合法"),
}

log("=" * 92)
log("[audit] FEATS71 参考时刻锚点审计")
log("=" * 92)
log("")
log("[1] 静态审计 (源码逐特征)")
log("-" * 92)
leak_feats = [f for f in STATIC if STATIC[f][2] == "泄漏"]
ok_feats = [f for f in STATIC if STATIC[f][2] == "合法"]
log(f"  泄漏 (锚自身 MVT): {len(leak_feats)}")
log(f"  合法             : {len(ok_feats)}")
log(f"  审计覆盖         : {len(STATIC)} / 71")
covered = set(STATIC) & set(FEATS71)
log(f"  与 FEATS71 交集   : {len(covered)}")
not_in_71 = [f for f in STATIC if f not in FEATS71]
if not_in_71:
    log(f"  ⚠ 静态表里不在 FEATS71 的: {not_in_71}")

# ============ (2) 数值实证: 反事实 MVT ============
log("")
log("=" * 92)
log("[2] 数值实证: 把 MVT 替换成反事实值, 重算特征")
log("-" * 92)
log("  思路: 合法特征不应依赖自身 MVT, 换成任意反事实值后必须【逐行完全不变】。")
log("        泄漏特征会明显变化。")

# 取一个月的小样本做实证(只读, 不训练)
from src.data_loader import load_month, parse_time_columns
import pandas as pd
MM = ROOT / "data/raw/training_2025-07-01_2025-08-01.parquet"
df = pq_read = pd.read_parquet(MM)
df = parse_time_columns(df, cols=["SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
                                  "BLOCK_TIME_UTC_mvt"])
dep = df[df["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
log(f"  样本: 2025-07 DEP {len(dep):,} 行")

t_unimp = R7.lookup_t_unimp(dep, R7.build_t_unimp_lut(dep))[0]

# 基线
base = {}
base.update(R7.compute_mvt_anchor_features(dep, t_unimp, label="base"))
base.update(R7.compute_extra_features(dep, label="base"))

# 反事实: 每行 MVT 平移一个航班自身独有的巨大随机量 (打乱时间轴)
rng = np.random.RandomState(7)
dep_cf = dep.copy()
shift = rng.randint(-6 * 3600, 6 * 3600, size=len(dep_cf))  # ±6h
dep_cf["MVT_TIME_UTC_mvt"] = (
    pd.to_datetime(dep_cf["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    + pd.to_timedelta(shift, unit="s"))
t_unimp_cf = R7.lookup_t_unimp(dep_cf, R7.build_t_unimp_lut(dep_cf))[0]
cf = {}
cf.update(R7.compute_mvt_anchor_features(dep_cf, t_unimp_cf, label="cf"))
cf.update(R7.compute_extra_features(dep_cf, label="cf"))

log("")
log(f"  {'特征':24s} {'反事实前后变化行数':>20s} {'占比':>9s} {'判定':>8s}")
emp = {}
for f in sorted(base):
    a = np.asarray(base[f], float)
    b = np.asarray(cf[f], float)
    ok = np.isfinite(a) & np.isfinite(b)
    same = (np.isclose(a[ok], b[ok], rtol=1e-9, atol=1e-9)
            if ok.sum() else np.array([], bool))
    changed = int((~same).sum()) if ok.sum() else 0
    frac = changed / max(int(ok.sum()), 1)
    verdict = "泄漏" if frac > 0.01 else "合法"
    emp[f] = {"n_valid": int(ok.sum()), "n_changed": changed,
              "changed_frac": round(frac, 6), "verdict": verdict}
    log(f"  {f:24s} {changed:>20,d} {frac*100:>8.2f}% {verdict:>8s}")

n_leak_emp = sum(1 for f in emp if emp[f]["verdict"] == "泄漏")
n_ok_emp = sum(1 for f in emp if emp[f]["verdict"] == "合法")
log("")
log(f"  实证结论: 泄漏 {n_leak_emp} 个 / 合法 {n_ok_emp} 个")

# 静态 vs 实证 一致性
log("")
log("[3] 静态 vs 实证 一致性检查")
log("-" * 92)
mismatch = [f for f in emp if emp[f]["verdict"] != STATIC.get(f, ("", "", "?"))[2]]
if mismatch:
    for f in mismatch:
        log(f"  ⚠ 不一致: {f}  静态={STATIC.get(f,('?','?','?'))[2]}  实证={emp[f]['verdict']}")
else:
    log("  ✓ 静态审计与数值实证完全一致")

# ============ 总结 ============
FINAL_LEAK = [f for f in FEATS71 if emp.get(f, {}).get("verdict") == "泄漏"]
log("")
log("=" * 92)
log("[结论]")
log("-" * 92)
log(f"  FEATS71 总数            : {len(FEATS71)}")
log(f"  锚定自身 MVT (泄漏) 数量: {len(FINAL_LEAK)}")
log(f"  剔除后特征数            : {len(FEATS71) - len(FINAL_LEAK)}")
log(f"  泄漏特征名: {FINAL_LEAK}")

RESULT = {
    "task": "audit_ref_anchor_feats71",
    "method": {
        "static": "逐特征读源码, 记录计算时的参考时刻/查询点列",
        "empirical": "把 MVT_TIME_UTC_mvt 换成反事实值(每行±6h随机平移)重算, "
                     "合法特征必须逐行不变; 不依赖对源码的理解",
    },
    "static_table": {f: {"source": s, "ref_time_col": q, "verdict": v}
                     for f, (s, q, v) in STATIC.items()},
    "empirical": emp,
    "consistency": {"mismatches": mismatch, "consistent": not mismatch},
    "n_feats71": len(FEATS71),
    "n_leak": len(FINAL_LEAK),
    "leak_features": FINAL_LEAK,
    "n_after_removal": len(FEATS71) - len(FINAL_LEAK),
    "legal_features": sorted([f for f in FEATS71 if f not in FINAL_LEAK]),
}
OUT_JSON.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"→ {OUT_JSON}")
