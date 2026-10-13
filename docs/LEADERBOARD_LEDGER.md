# Leaderboard Ledger · PRC 2026 Taxi-Time

Last updated: 2026-10-13

## Final submission (uploaded)

| Item | Value |
|---|---|
| **File** | `submissions/jubilant-puzzle_final.parquet` (670,790 rows) |
| **Uploaded** | 2026-10-13 |
| **Official receipt** | `Succeeded`, used_pairs=670,790, inputs=`final_submitting.parquet` |
| **Scoring** | The `jubilant-puzzle_final.parquet` slot does **not** score immediately; it is evaluated after the deadline |
| **Four-month holdout RMSE** | **324.76** (mo1 405.49 / mo2 223.58 / mo6 300.62 / mo7 338.02) |
| **vs LGB baseline 493.05** | **−34.1%**, 10/10 airports no worse, LIRF 1483.94 → 654.77 |
| **Recipe** | noclip E2: LIRF T=14400 full-rule hard replacement (63 rows with pred == MVT−SCHED, **not clipped**) + global LOBT window `[MVT − LOBT ± 3606]` projection + LGB 3-seed baseline elsewhere + clip[30, 10800] |

**Best leaderboard score (immediate-scoring path)**: v22 = 360.8565 (`truthing.parquet`, 344,841 rows). The final slot uses `final_submitting.parquet` (670,790 rows) instead, which is not directly comparable to the truthing scope; extrapolating from the v22 offset (holdout 342.40 → leaderboard 360.86) puts the final official score in the **335–370** range.

### The noclip decision chain (why 63 rows are allowed to exceed 10,800 s)

- v22's leaderboard run (360.86) was **unclipped**: all 43 repaired rows predicted exactly `MVT − SCHED` (difference = 0), max = 111,654 s, accepted by the official scorer
- True labels on hit rows: 2025 median 20,576 s, 88.2% > 10,800 s, max 131,167 s — genuine extreme operations, not outliers
- Clipping inflates RMSE on those rows from 7,918 → 33,313 (4.2×); four-month holdout noclip 324.76 vs clip 460.87
- Hit rows have 100% of the schedule missing (`ADEP_flt`/`BLOCK`/`EOBT_1`/`IOBT`/`AOBT_3`/`LOBT`), so the LOBT window is structurally useless for them
- All 13 Gate checks pass (commit 138a122, `notebooks/diag/d0/gate_noclip.py`)

## All submission records

| Version | Leaderboard RMSE | Δ vs previous | Note |
|---|---:|---:|---|
| v15 | 532.2765 | — | Full-year trained LGB baseline |
| v17 | 578.60 | +46.3 | Older XGB/LGB blend |
| v18 | 538.22 | −40.4 | LGB 3-seed full |
| v21 | 363.5146 | **−168.77** | LIRF corrupted-row replication, T=21600, 26 rows changed |
| **v22** | **360.8565** | **−2.65** | Same, T=14400, 43 rows changed (17 new) |
| v23 | 392.5762 | **+31.72** | P50 (quantile α=0.5) full-year retrain + LIRF T=14400 — **objective change made it worse** |
| final | post-deadline | — | **noclip E2** (holdout 324.76), 670,790 rows, uploaded 2026-10-13, receipt `Succeeded` |

## v23 failure attribution (verified on the leaderboard)

v23 = P50 objective + LIRF T=14400, i.e. the objective-function replacement of v15 (L2). Leaderboard result **392.58**, **31.72 s worse** than v22.

Difference structure (v23 − v22, 344,798 rows, excluding the 43 LIRF repairs):

| Statistic | v22 | v23 | Ratio |
|---|---:|---:|---:|
| mean | 1026.8 | 1046.0 | 1.019 |
| p50 | 972.5 | 994.2 | 1.022 |
| p90 | 1451.4 | 1485.1 | 1.023 |
| p99 | 2155.4 | 2088.2 | 0.969 |
| p99.9 | 3706.8 | 3022.3 | **0.815** |

- **The body of the distribution shifts up ~2.2%** (consistent +2.2% from p50 to p90) — not random noise
- **The tail is compressed**: p99.9 ratio 0.815; bucketed by v22, the mean of the `[3600, 10800)` segment drops from 4,794.7 to 3,376.9 (**−1,417.8 s**)
- diff-SSE attribution: `[600, 1800)` accounts for 52.1%, `[1800, 3600)` 26.0%, `[3600, 10800)` 20.9%
- Bucketed by |Δ|: the 23,154 rows with `|Δ| ∈ [200, 1000)` contribute 48.1% of diff-SSE; the 558 rows with `|Δ| > 1000` contribute 28.3%

**Conclusion: P50 is unusable on this leaderboard.** On internal holdout, P50 beats L2 (core 306 vs 457); on the leaderboard the opposite holds — because the leaderboard RMSE is dominated by the **extreme tail**, and P50 as a median estimator compresses the upper tail by construction (p99.9 only 81.5% of L2), which is heavily penalised under squared loss.

Root cause of the holdout/leaderboard disagreement: the holdout `core` scope (`y <= 10800`) filters out the tail-dominated terms, while leaderboard full RMSE applies no filtering. **The holdout core scope systematically over-estimates P50.**

### Ablation: separating three confounded factors

`notebooks/diag/d0/v23_ablation.py` — same ES split (last 10% of the year by MVT order, 208,505 rows), seed=42, isolating objective / rounds / label cleaning one group at a time.

| Combination | Training labels | Rounds | Objective | ES full | ES core |
|---|---|---:|---|---:|---:|
| L2 r2000 dirty | uncleaned | 1160 | L2 | **345.00** | 268.19 |
| **L2 r3000 clean (= v15)** | cleaned | 3000 | L2 | **354.70** | **254.70** |
| L2 r2000 clean | cleaned | 1996 | L2 | 356.29 | 256.78 |
| P50 r2000 dirty (= v23) | uncleaned | 2000 | P50 | 371.21 | 272.87 |
| P50 r3000 clean | cleaned | 3000 | P50 | 378.67 | 274.60 |
| P50 r2000 clean | cleaned | 2000 | P50 | 382.22 | 277.62 |

**Three conclusions:**

1. **The objective function is the only main cause.** In every like-for-like pair, P50 is **23–26 s** worse than L2: dirty/r2000 by 26.21 s, clean/r2000 by 25.93 s, clean/r3000 by 23.97 s — consistent in magnitude with the 31.72 s degradation measured on the leaderboard.
2. **Rounds are not the cause.** L2 from 1,996 → 3,000 rounds is marginally better on full (356.29 → 354.70); P50 from 2,000 → 3,000 rounds improves only 3.55 s (382.22 → 378.67), far short of closing the gap.
3. **Label cleaning is not the cause.** Uncleaned labels affect only 590 rows (0.028%); the dirty↔clean difference is directionally consistent across both objectives and of comparable magnitude (+9.70 s / −3.55 s) — i.e. noise.

**Why P50 must be worse:** the optimal estimator for RMSE is the conditional **mean**, which L2 optimises directly. P50 optimises the conditional **median**. When `y|x` is right-skewed the median sits below the mean, so P50 systematically underestimates, and the more the tail is compressed (p99.9 only 81.5% of L2) the heavier the squared-loss penalty. This is an objective/metric mismatch, not insufficient training. **Quantile objectives are permanently closed for this task.**

## Best recipe

```
Base: v15 full-year trained ranking prediction
Rule: PHASE_mvt=='DEP' AND ADEP_mvt=='LIRF'
      AND ADEP_flt missing AND (MVT-SCHED) > 14400
      → prediction becomes (MVT - SCHED)
```

Generator: `notebooks/diag/d0/ranking_lirf_repair_t14400.py`

## Mechanism (verified on the leaderboard)

LIRF has **label corruption**: `BLOCK_TIME_UTC_mvt` is overwritten by `SCHED_TIME_UTC_mvt`, which makes `TAXITIME = MVT - SCHED`. On corrupted rows this identity holds with max residual 60.0 s, without exception.

- Corrupted rows are **8.835%** of the full holdout DEP set (30,428/344,419)
- Baseline rate of `BLOCK==SCHED` at LIRF is **20.725%**, and **90.7%** within extreme rows
- On the ranking side BLOCK is blanked (100% of DEP rows carry the sentinel `-9223372037`), so repair must go through `MVT-SCHED`
- The key: `ADEP_flt` missing ≡ `AOBT` sentinel (both exactly 5,290 rows on the ranking side — perfectly collinear)
- **LIRF is the only repairable airport**: the 3 LFPG rows are genuine long ground stays (BLOCK earlier than SCHED by 23 h / 15.6 h), and the extreme rows at EHAM/EGLL are all uncorrupted. Extending to all airports degrades by +54 to +112 s

## Closed directions (do not retry)

| Direction | Conclusion | Evidence |
|---|---|---|
| Isotonic-regression calibration | Ceiling ≈ 0 (cross-validation −0.04 s) | calib_ceiling |
| Stacking on `oof_pred` only | No gain; the optimal conditional-mean solution is the identity map | corrector_ab variant B |
| Residual corrector (same family, same features) | Equivalent to just boosting further; no new information | corrector_ab variant D, +50–87 s |
| Direct correction (with features) | Degrades likewise | corrector_ab variant C |
| ARR arrival-load features | Independent of load at both quantile levels; net AUC 0.5146 < existing density 0.5400 | arr_load_diag |
| log / Huber objectives | log makes the tail more conservative; Huber needs alpha chosen by residual scale | ab_tail_objective |
| Fluid queue inversion | Physical RMSE 90%–968% worse on long-taxi segments | fluid_inversion |
| Lowering the threshold further | T=14400 achieves only 28% (predicted 9.22 s → actual 2.65 s) | v22 leaderboard |
| **P50 / quantile objectives** | **Leaderboard 392.58, 31.72 s worse than L2; ablation shows the objective itself is responsible, unrelated to rounds or label cleaning** | **v23 leaderboard + v23_ablation** |
| **Back-filling `BLOCK` into LOBT/EOBT/IOBT** | **Does not hold**: `only_LOBT=0`, `only_IOBT=46`, and the 12,674 `only_EOBT` rows have a 0% ADEP miss rate with precision 0.0000 | step1/2/3 forensics |
| P80 tail amplification / P50+P80 gating | Local long-taxi improvement, both full and core get worse | ab_ratio_quantile |
| Learned gating fusion (physical vs data) | AUC 0.939 but core barely moves (308.20 → 308.19) | gate_ab |
| Eight further non-LIRF tail families | EVT/GPD, quantile targets, log target, Huber, winsorisation, weighted loss, quantile compression, extreme-row rescue — all falsified. The tail is real physics (congestion/de-icing/weather) and cannot be compressed without extra information | method.md §4 |

## Pitfall notes

1. **The P0 evaluation scope leaks**: the `p0_nine_model` training set includes months 1 and 7, so core 304.94 is inflated. Under the clean scope (train 2-6, 8-12) the same model scores core 431.27 — a 164.84 s gap. **Not usable as a baseline anchor.**
2. **`BLOCK_TIME_UTC_mvt` is a sentinel on ranking DEP, not a real time.** Misreading it as "usable" leads to entirely wrong conclusions. The sentinel is the negative value `-9223372037` (search in the negative direction).
3. **`ADEP_mvt` is the airport key.** `ADEP_flt` is itself missing on corrupted rows, so filtering by it yields only 2 rows.
4. **Hit sets differ between holdout and ranking**: under the v21 rule, LIRF matches 21 holdout rows vs 26 ranking rows.
5. **pandas datetime → int64 yields microseconds**: cast with `.astype("datetime64[ns]")` before dividing by 1e9.
6. **Feature cache filenames carry no smoke/full marker** — this once caused a full run to silently reuse a smoke-sample cache.
7. **The holdout `core` scope (`y <= 10800`) systematically over-estimates P50 / quantile-type objectives.** v23 looked better on internal core yet degraded 31.72 s on the leaderboard. Quantile objectives must be judged with **full RMSE**.
8. **P50 lifts the body of the distribution by ~2.2%** (not just tail compression), meaning it changes the conditional median itself — it is not a pure trimming effect.

## Post-deadline status (wrap-up 2026-10-13)

- The final submission was uploaded and accepted by the official scorer (`jubilant-puzzle_final.parquet`, `Succeeded`, 670,790 rows)
- Main line after 10-03 (see `docs/method.md`): noclip E2 locked → physics-constrained second-moment mixture σ → damage-aware stratified conformal (h2 coverage deviation ≤ 0.0023) → `supplementary/final_uncertainty.parquet`
- New falsified results: the μ_total probabilistic mixture as a point predictor scores 788.01 (0/10 airports improve) — so UQ does not produce point predictions
- The external-data route was assessed as infeasible (< 24 h remaining, adsb.lol archive 420 GiB / 30 h, OpenSky quota 4.5 days, METAR worth only 2–8 s). The final decision was to upload at 324.76
- Early-mining notes (10-02, kept for post-mortem): 340 rows remain unrepaired by the LIRF rule (`ADEP_flt` missing + `MVT − SCHED ≤ 14400`, phys median 6,450 s); the 4,907 non-LIRF rows with missing `ADEP_flt` are the largest untapped pool, but naive airport-wide extension measured +54 to +112 s worse, so a finer discriminating key is required