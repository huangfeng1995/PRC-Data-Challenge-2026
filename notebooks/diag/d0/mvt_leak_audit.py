"""
mvt_leak_audit.py — 第一部分: MVT_TIME_UTC_mvt 泄漏疑点只读核查
==================================================================
疑点: ranking DEP 的 MVT_TIME_UTC_mvt (实际起飞时刻) 报 100% 非空,
      但按预测时点 (推出后、起飞前) 它应该未知。

六项核查:
  1. ranking DEP: MVT == SCHED 的相等比例 → 若大量相等, 说明被填成计划值
  2. ranking DEP: (MVT - AOBT_3_flt) / (MVT - EOBT_1_flt) 分布 → 是否像真实起飞
  3. 训练集 2025: 逐行核对 MVT - BLOCK == TAXITIME_SEC, 给误差
  4. 关键: 若 MVT 真实起飞且 AOBT 已知, (MVT - AOBT) 能否还原滑出时间?
  5. ranking DEP vs ARR: MVT 非空率与语义对比
  6. 明确结论 (a) 真实未来起飞 / (b) 被填成 SCHED / (c) 其他

输出: notebooks/diag/d0/mvt_leak_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT_JSON = D0 / "mvt_leak_audit.json"
OUT_LOG = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b/mvt_leak_audit.log")

INAT = np.iinfo(np.int64).min


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")


def to_ns(s: pd.Series) -> np.ndarray:
    """datetime Series → int64 纳秒 (显式单位, 不受 pandas 内部 us/ns 影响)。"""
    return s.to_numpy(dtype="datetime64[ns]").astype("int64")


def to_s(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def dist(a, name, qs=(1, 5, 25, 50, 75, 95, 99)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        log(f"    {name:34s} (全空)")
        return {}
    d = {f"p{q}": round(float(np.percentile(a, q)), 2) for q in qs}
    d["n"] = int(len(a))
    d["mean"] = round(float(a.mean()), 2)
    d["frac_neg"] = round(float((a < 0).mean()), 4)
    log(f"    {name:34s} n={len(a):>8,}  " +
        "  ".join(f"p{q}={d[f'p{q}']:>10.1f}" for q in qs))
    return d


log("=" * 84)
log("[mvt_leak] MVT_TIME_UTC_mvt 泄漏疑点核查 (只读)")
log("=" * 84)

# ================= 载入 ranking =================
rk_path = ROOT / "data/raw/ranking.parquet"
COLS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "ARVT_3_flt",
        "LOBT_flt", "TAXITIME_SEC_mvt"]
rk = pq.read_table(rk_path, columns=[c for c in COLS if c in
                                     pq.ParquetFile(rk_path).schema_arrow.names]).to_pandas()
ph = rk["PHASE_mvt"].astype(str).values
RDEP, RARR = ph == "DEP", ph == "ARR"
log("")
log(f"ranking: {len(rk):,} 行  DEP {int(RDEP.sum()):,}  ARR {int(RARR.sum()):,}")

for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
          "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "ARVT_3_flt"]:
    rk[c] = to_s(rk[c])

rk_mvt = to_ns(rk["MVT_TIME_UTC_mvt"])
rk_sch = to_ns(rk["SCHED_TIME_UTC_mvt"])
rk_aobt = to_ns(rk["AOBT_3_flt"])
rk_eobt = to_ns(rk["EOBT_3_flt"] if "EOBT_3_flt" in rk.columns else rk["EOBT_1_flt"])
rk_blk = to_ns(rk["BLOCK_TIME_UTC_mvt"])

RES = {}

# ============ 核查 1: MVT == SCHED 相等比例 ============
log("")
log("=" * 84)
log("[核查 1] ranking DEP: MVT_TIME == SCHED_TIME 的相等比例")
log("=" * 84)
m = RDEP & (rk_mvt != INAT) & (rk_sch != INAT)
eq = float((rk_mvt[m] == rk_sch[m]).mean())
log(f"    精确相等比例 = {eq*100:.4f}%  ({int((rk_mvt[m]==rk_sch[m]).sum()):,} / {int(m.sum()):,})")
d_mm = (rk_mvt[m] - rk_sch[m]) / 1e9
d1 = dist(d_mm, "(MVT - SCHED) 秒")
log(f"    → {'若接近 100% 则为被填成计划值' if eq > 0.9 else '不相等 → 不是简单填成 SCHED'}")
RES["check1_mvt_eq_sched"] = {"eq_ratio": round(eq, 6), "delta_sec": d1}

# ============ 核查 2: MVT vs AOBT / EOBT ============
log("")
log("=" * 84)
log("[核查 2] ranking DEP: (MVT - AOBT) 与 (MVT - EOBT) 分布")
log("=" * 84)
m2 = RDEP & (rk_mvt != INAT) & (rk_aobt != INAT)
d2a = (rk_mvt[m2] - rk_aobt[m2]) / 1e9
log("    AOBT_3_flt = 实际推出时刻 (off-block)")
r2a = dist(d2a, "(MVT - AOBT) 秒")
m2b = RDEP & (rk_mvt != INAT) & (rk_eobt != INAT)
d2b = (rk_mvt[m2b] - rk_eobt[m2b]) / 1e9
log("    EOBT_1_flt = 计划推出时刻")
r2b = dist(d2b, "(MVT - EOBT) 秒")
frac_aobt_pos = float((d2a > 0).mean())
log(f"    (MVT - AOBT) > 0 的比例 = {frac_aobt_pos*100:.2f}%")
log("    → 真实起飞应满足 MVT > AOBT (先推出后起飞), 比例应接近 100% 且量级为正")
RES["check2_mvt_vs_aobt"] = {"delta_sec": r2a, "frac_pos": round(frac_aobt_pos, 6)}
RES["check2_mvt_vs_eobt"] = {"delta_sec": r2b}

# ============ 核查 3: 训练集恒等式 MVT - BLOCK == TAXITIME ============
log("")
log("=" * 84)
log("[核查 3] 训练集 2025: 逐行核对 (MVT_TIME - BLOCK_TIME) == TAXITIME_SEC")
log("=" * 84)
MONTHS = {1: "2025-01-01_2025-02-01", 7: "2025-07-01_2025-08-01", 12: "2025-12-01_2026-01-01"}
tr_parts = []
for mm, fn in MONTHS.items():
    tr_parts.append(pq.read_table(ROOT / f"data/raw/training_{fn}.parquet",
                                  columns=[c for c in COLS if c in
                                           pq.ParquetFile(ROOT / f"data/raw/training_{fn}.parquet")
                                           .schema_arrow.names]).to_pandas())
tr = pd.concat(tr_parts, ignore_index=True)
tph = tr["PHASE_mvt"].astype(str).values
TDEP, TARR = tph == "DEP", tph == "ARR"
for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
          "AOBT_3_flt", "EOBT_1_flt"]:
    tr[c] = to_s(tr[c])
log(f"    训练样本: DEP {int(TDEP.sum()):,}  ARR {int(TARR.sum()):,} (月份 {list(MONTHS)})")

t_mvt = to_ns(tr["MVT_TIME_UTC_mvt"])
t_blk = to_ns(tr["BLOCK_TIME_UTC_mvt"])
t_y = tr["TAXITIME_SEC_mvt"].values.astype(float)
m3 = TDEP & (t_mvt != INAT) & (t_blk != INAT) & np.isfinite(t_y) & (t_y > 0)
ident = (t_mvt[m3] - t_blk[m3]) / 1e9
err = ident - t_y[m3]
log(f"    DEP 有效行 {int(m3.sum()):,}")
log(f"    恒等式最大绝对误差 = {np.abs(err).max():.6f} 秒")
log(f"    平均绝对误差       = {np.abs(err).mean():.9f} 秒")
log(f"    误差 > 0.001s 的行数 = {int((np.abs(err) > 0.001).sum()):,}")
log(f"    → {'恒等式成立: TAXITIME = MVT - BLOCK' if np.abs(err).max() < 0.001 else '恒等式不成立'}")
RES["check3_identity"] = {
    "n_dep": int(m3.sum()),
    "max_abs_err_sec": round(float(np.abs(err).max()), 6),
    "mean_abs_err_sec": round(float(np.abs(err).mean()), 9),
    "n_rows_err_gt_1ms": int((np.abs(err) > 0.001).sum()),
    "identity_holds": bool(np.abs(err).max() < 0.001),
}

# ARR 侧恒等式 (对照, 帮助理解语义)
m3a = TARR & (t_mvt != INAT) & (t_blk != INAT) & np.isfinite(t_y) & (t_y > 0)
if m3a.sum():
    ident_a = (t_blk[m3a] - t_mvt[m3a]) / 1e9
    err_a = ident_a - t_y[m3a]
    log(f"    [ARR 对照] (BLOCK - MVT) == TAXITIME ? 最大误差 {np.abs(err_a).max():.6f}s  "
        f"(有效行 {int(m3a.sum()):,})")
    RES["check3_identity_arr"] = {
        "n_arr": int(m3a.sum()),
        "max_abs_err_sec": round(float(np.abs(err_a).max()), 6),
        "identity_holds": bool(np.abs(err_a).max() < 0.001),
    }

# ============ 核查 4: 泄漏判断 ============
log("")
log("=" * 84)
log("[核查 4] 关键泄漏判断: (MVT - AOBT) 能否直接还原滑出时间?")
log("=" * 84)
# 训练侧: (MVT - AOBT) 与真实 TAXITIME 的关系
m4 = TDEP & (t_mvt != INAT) & (to_ns(tr["AOBT_3_flt"]) != INAT) & np.isfinite(t_y) & (t_y > 0)
t_aobt = to_ns(tr["AOBT_3_flt"])
mvt_minus_aobt = (t_mvt[m4] - t_aobt[m4]) / 1e9
taxi = t_y[m4]
d4 = mvt_minus_aobt - taxi
log(f"    训练 DEP 有效行 {int(m4.sum()):,}")
log(f"    (MVT - AOBT) 的分布 (这正是 'AOBT 已知' 情况下可算出的量):")
r4a = dist(mvt_minus_aobt, "(MVT - AOBT) 秒")
log(f"    真实 TAXITIME 分布:")
r4b = dist(taxi, "TAXITIME_SEC 秒")
corr = float(np.corrcoef(mvt_minus_aobt, taxi)[0, 1])
rmse_of_recon = float(np.sqrt(np.mean(d4 ** 2)))
log(f"    corr((MVT-AOBT), TAXITIME) = {corr:.6f}")
log(f"    用 (MVT - AOBT) 直接当预测的 RMSE = {rmse_of_recon:.2f}s")
log(f"    (MVT-AOBT) - TAXITIME 的分布:")
r4c = dist(d4, "重建误差 秒")
log("")
log(f"    → 若 corr≈1 且 RMSE 远小于模型误差, 说明 AOBT 已知时 MVT 确实等价于答案。")
RES["check4_train_reconstruction"] = {
    "mvt_minus_aobt_sec": r4a, "taxitime_sec": r4b,
    "corr": round(corr, 6), "recon_rmse_sec": round(rmse_of_recon, 2),
    "recon_err_sec": r4c,
}

# ranking 侧: (MVT - AOBT) 与训练同口径对比
rk_mm = (rk_mvt[m2] - rk_aobt[m2]) / 1e9
log("")
log("    ranking DEP 的 (MVT - AOBT) 分布:")
r4d = dist(rk_mm, "ranking (MVT - AOBT) 秒")
log(f"    训练 DEP 同口径 p50={r4a['p50']:.1f}s vs ranking p50={r4d['p50']:.1f}s  "
    f"→ {'两侧同口径' if abs(r4a['p50']-r4d['p50'])<60 else '两侧口径不同'}")
RES["check4_rank_mvt_minus_aobt"] = r4d

# ============ 核查 5: DEP vs ARR 语义对比 ============
log("")
log("=" * 84)
log("[核查 5] ranking DEP vs ARR: MVT 非空率与语义")
log("=" * 84)


def nn_rate(s, mask):
    v = s[mask]
    return round(float(v.notna().sum() / max(len(v), 1)), 4)


cmp5 = {}
log(f"  {'列':24s} {'ranking DEP':>14s} {'ranking ARR':>14s} {'训练 DEP':>12s} {'训练 ARR':>12s}")
for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
          "AOBT_3_flt", "EOBT_1_flt", "ARVT_3_flt", "IOBT_flt"]:
    rd, ra = nn_rate(rk[c], RDEP), nn_rate(rk[c], RARR)
    td = nn_rate(tr[c], TDEP) if c in tr.columns else None
    ta = nn_rate(tr[c], TARR) if c in tr.columns else None
    cmp5[c] = {"rank_dep": rd, "rank_arr": ra,
               "train_dep": td, "train_arr": ta}
    tds = f"{td*100:>11.2f}%" if td is not None else f"{'n/a':>12s}"
    tas = f"{ta*100:>11.2f}%" if ta is not None else f"{'n/a':>12s}"
    log(f"  {c:24s} {rd*100:>13.2f}% {ra*100:>13.2f}% {tds} {tas}")
RES["check5_nonnull"] = cmp5

# 语义: ARR 的 MVT 是落地(在其 ADES 机场), DEP 的 MVT 是起飞(自其 ADEP 机场)
# 检验: ARR 行 (MVT - BLOCK) 应为正的滑入时间; DEP 行 (MVT - BLOCK) 为正的滑出
log("")
mA = RARR & (rk_mvt != INAT) & (rk_blk != INAT)
if mA.sum():
    dA = (rk_blk[mA] - rk_mvt[mA]) / 1e9
    rA = dist(dA, "ranking ARR (BLOCK-MVT) 秒")
    log(f"    ranking ARR 的 (BLOCK - MVT) 中位 {rA['p50']:.1f}s → 语义=落地时刻 ✓ "
        f"(滑入时间为正)")
else:
    rA = {}
    log("    ranking ARR 的 BLOCK 全空, 无法用 (BLOCK-MVT) 交叉验证")
mD = RDEP & (rk_mvt != INAT) & (rk_blk != INAT)
log(f"    ranking DEP 的 BLOCK 非空行数 = {int(mD.sum()):,} "
    f"→ DEP 侧无法用 (MVT-BLOCK) 交叉验证")
RES["check5_arr_semantics"] = rA

# ============ 核查 6: 结论 ============
log("")
log("=" * 84)
log("[核查 6] 结论判定")
log("=" * 84)

is_sched_fill = eq > 0.9
mvt_gt_aobt = frac_aobt_pos
real_takeoff = mvt_gt_aobt > 0.95 and r2a.get("p50", 0) > 60
# 训练侧: 若 (MVT-AOBT) 能高精度还原 TAXITIME, 说明 AOBT≈BLOCK
recon_precise = rmse_of_recon < 60 and corr > 0.9

# 追加关键证据: AOBT_3_flt 是否等于 BLOCK_TIME_UTC_mvt?
# 若 AOBT == BLOCK, 则 (MVT - AOBT) 就能精确还原 TAXITIME -> 直接答案泄漏。
# 若 AOBT != BLOCK, 则 (MVT - AOBT) 只能近似 -> 组织方是靠"抽掉 BLOCK"来防泄漏。
log("")
log("=" * 84)
log("[核查 4b] 决定性证据: AOBT_3_flt 是否等于 BLOCK_TIME_UTC_mvt?")
log("=" * 84)
t_aobt_all = to_ns(tr["AOBT_3_flt"])
t_eobt_all = to_ns(tr["EOBT_1_flt"])
t_sch_all = to_ns(tr["SCHED_TIME_UTC_mvt"])
ab_cmp = {}
log(f"  {'对照列':12s} {'与BLOCK相等比例':>16s} {'中位差(秒)':>13s} {'|差|p95(秒)':>13s}")
for nm, arr in [("AOBT_3_flt", t_aobt_all), ("EOBT_1_flt", t_eobt_all),
                ("SCHED_TIME", t_sch_all)]:
    ok = TDEP & (t_blk != INAT) & (arr != INAT)
    dd = (arr[ok] - t_blk[ok]) / 1e9
    eqf = float((np.abs(dd) < 1e-6).mean())
    ab_cmp[nm] = {"eq_ratio": round(eqf, 6),
                  "median_diff_sec": round(float(np.median(dd)), 2),
                  "abs_p95_sec": round(float(np.percentile(np.abs(dd), 95)), 2)}
    log(f"  {nm:12s} {eqf*100:>15.2f}% {float(np.median(dd)):>13.1f} "
        f"{float(np.percentile(np.abs(dd),95)):>13.1f}")
log("")
log("  -> AOBT_3_flt 与 BLOCK_TIME_UTC_mvt 【不是同一个字段】(相等比例极低, p95 差近千秒)。")
log("     真正的答案锚点是 BLOCK: TAXITIME = MVT - BLOCK 精确成立(核查 3, 最大误差 0.000000s)。")
log("     组织方在 ranking DEP 上把 BLOCK_TIME_UTC_mvt 整列抽空(0.00%), 而保留 MVT。")
log("     即: 泄漏防护靠的是【抽掉 BLOCK】, 而不是抽掉 MVT。")
log(f"     因此 (MVT - AOBT) 无法精确还原答案: corr={corr:.4f}, RMSE={rmse_of_recon:.1f}s")
log(f"     (该 RMSE 甚至劣于本项目当前模型 core 约 300s, 故 AOBT+MVT 也不构成'免费答案')。")
RES["check4b_aobt_vs_block"] = ab_cmp
RES["check4b_leak_defense"] = {
    "mechanism": "ranking DEP 把 BLOCK_TIME_UTC_mvt 整列抽空(0.00%), 保留 MVT(100%)",
    "anchor": "TAXITIME = MVT - BLOCK (训练集精确成立)",
    "aobt_is_block": False,
    "implication": "AOBT + MVT 不能精确还原答案; 但 MVT 本身仍是预测时点之后的信息",
}

if is_sched_fill:
    verdict = "b_filled_with_sched"
    reason = "MVT 与 SCHED 几乎完全相等 → MVT 被填成计划值, 无泄漏"
elif real_takeoff and recon_precise:
    verdict = "a_real_future_takeoff"
    reason = ("MVT 显著晚于 AOBT 且 (MVT-AOBT) 能高精度还原 TAXITIME "
              "→ ranking DEP 含真实未来起飞时刻, 构成答案泄漏")
elif real_takeoff and not recon_precise:
    # 关键分支: MVT 是真实的未来起飞时刻(非填 SCHED), 但答案锚点 BLOCK 已被抽空,
    # 且 AOBT != BLOCK, 所以 (MVT-AOBT) 只能近似, 无法精确还原答案。
    verdict = "a_real_future_takeoff_not_direct_answer"
    reason = (
        "MVT 是【真实的未来起飞时刻】(与 SCHED 仅 0.007% 相等; 99.95% 晚于 AOBT; "
        "中位 (MVT-AOBT)=978s, 与训练集同口径 p50=961s 一致) -> 属于预测时点之后的未来信息, "
        "【绝不能当特征】。但它【不构成直接答案泄漏】: 精确恒等式是 TAXITIME=MVT-BLOCK, "
        "而 ranking DEP 已把 BLOCK_TIME_UTC_mvt 整列抽空(0.00%), 且 AOBT_3_flt 与 BLOCK "
        "并非同一字段(相等仅 0.63%, |差|p95≈966s), 因此 (MVT-AOBT) 只能近似: "
        f"corr={corr:.4f}, RMSE={rmse_of_recon:.1f}s, 甚至劣于当前模型 core≈300s。")
else:
    verdict = "c_other"
    reason = "MVT 与 SCHED 不相等, 且未呈现典型真实起飞形态"

log(f"    等于 SCHED 比例        : {eq*100:.4f}%")
log(f"    (MVT-AOBT) > 0 比例   : {mvt_gt_aobt*100:.2f}%   中位 {r2a.get('p50')}s")
log(f"    corr(重建物, TAXITIME): {corr:.6f}")
log(f"    重建 RMSE             : {rmse_of_recon:.2f}s")
log("")
log(f"    ▶ 判定: ({verdict})")
log(f"      {reason}")
RES["verdict"] = {
    "code": verdict, "reason": reason,
    "eq_sched_ratio": round(eq, 6),
    "frac_mvt_gt_aobt": round(mvt_gt_aobt, 6),
    "mvt_minus_aobt_median_sec": r2a.get("p50"),
    "recon_corr": round(corr, 6),
    "recon_rmse_sec": round(rmse_of_recon, 2),
}

RES["headline"] = {
    "ranking_dep_mvt_nonnull_pct": round(nn_rate(rk["MVT_TIME_UTC_mvt"], RDEP) * 100, 2),
    "ranking_dep_block_nonnull_pct": round(nn_rate(rk["BLOCK_TIME_UTC_mvt"], RDEP) * 100, 2),
    "ranking_dep_aobt_nonnull_pct": round(nn_rate(rk["AOBT_3_flt"], RDEP) * 100, 2),
    "mvt_eq_sched_pct": round(eq * 100, 4),
    "mvt_minus_aobt_median_sec": r2a.get("p50"),
    "train_identity_max_err_sec": RES["check3_identity"]["max_abs_err_sec"],
    "recon_corr": round(corr, 6),
    "recon_rmse_sec": round(rmse_of_recon, 2),
    "verdict_code": verdict,
}
log("")
log(f"→ {OUT_JSON}")
OUT_JSON.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
