"""
读取 notebooks/diag/d0/tunimp_trust_twostage_result.json
打印 Step0-Step4 对照表 + LIRF/EGLL/LTFM/LFPG 重点机场分解
"""
import json, sys
from pathlib import Path

D0 = Path("/Users/openclaw/Documents/PRC_data_challenge/notebooks/diag/d0")
fn = D0 / "tunimp_trust_twostage_result.json"
if not fn.exists():
    print(f"NOT FOUND: {fn}"); sys.exit(1)
data = json.loads(fn.read_text())
splits = ["FoldA_Jan2Jul", "FoldB_Jul2Jan", "MAIN"]

GROUPS = [("g1_base71", "①71特征"),
          ("g2_offset", "②+offset"),
          ("g3_trust", "③+trust"),
          ("g4_twostage", "④两阶段")]
AP = ["LIRF", "EGLL", "LTFM", "LFPG"]


def fmt(v, w=8):
    if v is None: return f"{'-':>{w}}"
    return f"{v:>{w}.2f}"


print("=" * 90)
print("Step0-Step4 汇总  (3 seed 平均, ROUNDS_REG=3000, ROUNDS_CLS=600)")
print("=" * 90)
for sp in splits:
    if sp not in data:
        continue
    r = data[sp]
    print(f"\n### {sp}   train={r['train_months']} val={r['val_months']}   "
          f"offset.global={r['offset_stats']['global_offset']:+.1f}s  "
          f"LIRF_offset={r['offset_stats'].get('LIRF_offset'):+.1f}  "
          f"(groups={r['offset_stats']['n_groups_used']}/{r['offset_stats']['n_groups_total']})")
    print("-" * 90)
    print(f"{'组':18s} {'短段':>10s} {'长段':>10s} {'核心':>10s} {'全样本':>10s} "
          f"{'shortσ':>8s} {'longσ':>8s} {'AUC':>8s}")
    for k, lbl in GROUPS:
        g = r["results"][k]
        print(f"{lbl:18s} {fmt(g['rmse_short'])} {fmt(g['rmse_long'])} {fmt(g['rmse_core'])} "
              f"{fmt(g['rmse_full'])} {fmt(g.get('seed_std', 0), 7)} "
              f"{fmt(g.get('seed_std', 0), 7)} {fmt(g.get('cls_auc'), 8)}")

    print(f"\n  重点机场 核心RMSE  (列顺序 ①②③④):")
    print("  " + " ".join(f"{a:>10s}" for a in AP))
    for k, lbl in GROUPS:
        per = r["results"][k]["per_ap"]
        cells = []
        for ap in AP:
            v = per.get(ap, {}).get("rmse_core")
            cells.append(f"{v:>10.1f}" if v else f"{'-':>10s}")
        print(f"  {lbl:>12s} " + " ".join(cells))

    # Step1 offset gain
    if "g2_offset" in r["results"]:
        o = r["results"]["g2_offset"]
        print(f"\n  [Step1] t_unimp_bias_offset gain_pct={o['gain_pct_t_unimp_bias_offset']:.3f}%  "
              f"(t_unimp_r7 gain_pct={o['gain_t_unimp_r7']/sum(x['gain'] for x in o['top10_gain'])*100:.3f}%)" if False else "")
        # 用全量 gain 表 (从 model 未存), 只显示 offset pct
        print(f"  [Step1] offset gain_pct={o['gain_pct_t_unimp_bias_offset']:.3f}%")
    # Step2 trust
    if "g3_trust" in r["results"]:
        t = r["results"]["g3_trust"]
        print(f"  [Step2] trust_level gain_pct={t['gain_pct_trust_level']:.3f}%  "
              f"iobt_echoes={t['gain_pct_iobt_echoes_sched']:.3f}%  "
              f"aobt_iobt_absdiff={t['gain_pct_aobt_iobt_absdiff_sec']:.3f}%")
        for lv, d in t["by_trust_level"].items():
            print(f"          trust={lv}: n={d['n']:,} G1={d['g1']:.1f} G3={d['g3']:.1f} Δ={d['delta']:+.1f}s")
    # Step3
    if "g4_twostage" in r["results"]:
        c = r["results"]["g4_twostage"]
        print(f"  [Step3] cls AUC={c['cls_auc']:.4f}  PR-AUC={c['cls_pr_auc']:.4f}  "
              f"n_pred_long@0.5={c['n_pred_long_thr05']:,}  "
              f"n_train_short={c['n_train_short']:,}  n_train_long={c['n_train_long']:,}")

print()
print("DONE")