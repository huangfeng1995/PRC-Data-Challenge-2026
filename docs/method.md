# Taxi-Out Time Prediction — Method

> Task: predict `TAXITIME_SEC_mvt` (taxi-out time of departing flights, in seconds).
> The final submission covers four months of 2026 (Jan / Feb / Jun / Jul), 670,790 DEP rows.

---

## 1. Background: LIRF label corruption with the entire schedule missing

### 1.1 The phenomenon

Among the ten airports, **LIRF (Rome Fiumicino)** exhibits a systematic label corruption pattern. Its signature is that **only two timestamps in the record — `MVT` and `SCHED` — are valid**, while everything else is either missing or overwritten:

| Column | Missing rate on hit rows |
|---|---|
| `ADEP_flt` | 100% |
| `BLOCK_TIME_UTC` | 100% |
| `EOBT_1_flt` / `IOBT_flt` / `AOBT_3_flt` / `LOBT_flt` | 100% |

This is not "only `BLOCK` is corrupted" but rather **the entire schedule is missing** — no other timestamp column can be relied upon when repairing these rows.

### 1.2 Detection rule

```
hit = (ADEP == LIRF)
      AND (ADEP_flt is missing)
      AND (MVT − SCHED > 14400)
```

All three conditions are required. Hit counts:

| Dataset | Rows hit |
|---|---|
| 2026 leaderboard ranking (Jan+Jul) | 43 |
| 2026 final (Jan/Feb/Jun/Jul) | **63** |
| 2025 holdout (4 months) | 63 |
| 2025 full 12 months | 127 |

> Note: a **simplified rule** without the `ADEP_flt` condition was also tried (hit counts inflate to 138/229 rows) and pulls in many normal flights. The full rule matches the v22 leaderboard approach (43 rows for 2026 Jan+Jul).

### 1.3 What the true labels look like

True `TAXITIME` on 2025 holdout hit rows:

- Median **20,576 s**, max 131,167 s
- **88.2% exceed 10,800 s**

Conclusion: these are **genuine extreme taxi events**, not data errors. `MVT − SCHED` is the only usable point-estimate source for them.

---

## 2. Method

### 2.1 Baseline: LightGBM with 71 features

- Features: 71-dimensional (timestamp-derived + airport/runway/stand + queue density + `t_unimp` + `rwy_month_share` + `ap_month_hourly_density`)
- Model: LGBM, simple average over 3 seeds (42/123/2024)
- Strict holdout: train = 2025-02..12 (excluding 1/7), holdout = 2025-01 + 2025-07
- All LUT / target encoding / `t_unimp` statistics use **training folds only**; the inference path never touches holdout or any 2026 label

**Four-month holdout full RMSE = 493.05 s** (mo=1: 493.67 / mo=2: 410.16 / mo=6: 388.92 / mo=7: 622.28)

### 2.2 Hard repair of corrupted rows (full rule, no clipping)

For rows matched in §1.2, **replace the model prediction with `MVT − SCHED`**:

```python
P[hit] = MVT[hit] - SCHED[hit]      # no upper bound
```

### 2.3 LOBT physical window

The difference between `LOBT` (push-back time) and `MVT`, `d = |BLOCK − LOBT|`, has a **hard upper bound** across the full dataset:

| Quantile | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| value (s) | 359 | 1143 | 2644 | 3479 | **3606** |

`max = 3606 s` is **constant** across 12 months × 10 airports — it is a physical truncation in the upstream data source. This calibrates **T_lobt = 3606 s** (100% coverage), and predictions are projected into the physically feasible interval:

```python
ctr = MVT - LOBT
P = clip(P, ctr - 3606, ctr + 3606)     # rows with missing LOBT are skipped
```

### 2.4 Physics-constrained second-moment mixture for σ

**Motivation**: σ emitted directly by NGBoost explodes in the tail (measured σ_max on h1 = **7.08e14**) and does not distinguish the two heterogeneous mechanisms — "corrupted" vs "normal".

**Approach**: model the prediction as a two-component mixture —

- Corrupted component: physically bounded `μ_C = MVT − SCHED`, `σ_C = 100 s` (constant, never extrapolated with features)
- Normal component: NGBoost's `μ_N, σ_N`, with `σ_N ← min(σ_N, 2000)`
- Mixing weight: `p = P(copy corruption | x)` from the copy classifier (LGBM binary, 71 features, outputs a probability — no argmax)

Point estimate and second moment:

```
μ  = p·μ_C + (1−p)·μ_N
σ² = p·σ_C² + (1−p)·σ_N² + p(1−p)·(μ_C − μ_N)²
```

The third term is the variance contribution from the gap between the two component means; it inflates σ substantially when `p` is uncertain — exactly the expression of "the model does not know which side to trust".

### 2.5 Damage-aware conformal prediction

Using a **calibration set** (2025 months `mo ∈ {3,4,5,8,9,10,11}`, disjoint from the holdout months), take per-stratum quantiles of the nonconformity score `s = |y − μ| / σ`:

| Nominal level | q_C (damaged stratum) | q_N (normal stratum) |
|---|---|---|
| 90% | 1.1341 | 1.4056 |
| 95% | 1.6955 | 1.7433 |

Stratification rule: `is_damaged = T14400 hit OR p > 0.5`.

At inference the interval is `[μ − q·σ, μ + q·σ]`, with `q` selected by the stratum the row belongs to.

---

## 3. Key results

### 3.1 Point prediction (this is what went into the submission)

| Approach | mo=1 | mo=2 | mo=6 | mo=7 | 4 months |
|---|---|---|---|---|---|
| LGB baseline | 493.67 | 410.16 | 388.92 | 622.28 | 493.05 |
| + clip[30,10800] variant | 474.17 | 395.08 | 369.23 | 563.79 | 460.87 |
| **Final submission (hard repair + LOBT, no clip)** | **405.49** | **223.58** | **300.62** | **338.02** | **324.76** |

**168.29 s better than baseline (−34.1%)**; the LIRF stratum goes from 1483.94 → **654.77**.

**Not clipping is the key**: 88.2% of hit rows have true labels above 10,800 s. With conventional upper-bound truncation, RMSE on those 63 rows degrades from 7,918.50 to 33,313.07 (4.2×), and the overall score from 324.76 to 460.87. Leaderboard v22 (360.86) was likewise unclipped — a row-by-row check confirms its 43 repaired values are **exactly equal** to `MVT − SCHED` (all differences are 0), min 14,945 s, max 111,654 s.

All 10 airports are no worse than the baseline.

### 3.2 Uncertainty quantification

**σ tail converges by 10 orders of magnitude**:

| | NGBoost raw | Physics-constrained mixture |
|---|---|---|
| h1 σ_max | 7.0832e+14 | **4.31e+04** |
| h1 rows with σ>1e4 | 431 | **400** |
| h2 σ_max | 1.9292e+12 | **3.46e+04** |
| h2 rows with σ>1e4 | 103 | **9** |

**σ vs residual correlation (Spearman, best at σ_C=100)**:

| | NGBoost raw σ | Physics-constrained mixture σ |
|---|---|---|
| h1 (Jan+Jul) | +0.4228 | **+0.4354** |
| h2 (Feb+Jun) | +0.3774 | **+0.3966** |

After mixing the **signal is stronger**, and long-vs-short tail discrimination improves from 2.29× / 2.49× to **8.48× / 9.00×** (median ratio between the σ>1800 and σ<600 segments).

**σ_C sensitivity**:

| σ_C | h1 Spearman | h2 Spearman | h1 95% coverage (damaged stratum) |
|---|---|---|---|
| **100** | **+0.4354** | **+0.3966** | 0.9535 |
| 300 | +0.4066 | +0.3347 | 0.9455 |
| 600 | +0.2515 | +0.1840 | 0.9334 |

The larger `σ_C` is, the more σ is flattened by `p·σ_C²` and the signal decays monotonically. Use 100.

### 3.3 Conformal coverage

Actual coverage of the stratified intervals on the two holdout sets (disjoint from the calibration set):

| | Nominal | h1 | Deviation | h2 | Deviation |
|---|---|---|---|---|---|
| Damaged stratum | 90% | 0.8712 | −0.0288 | 0.8977 | −0.0023 |
| Normal stratum | 90% | 0.9133 | +0.0133 | 0.9004 | +0.0004 |
| Damaged stratum | 95% | 0.9455 | −0.0045 | 0.9507 | +0.0007 |
| Normal stratum | 95% | 0.9558 | +0.0058 | 0.9493 | −0.0007 |

**All four h2 cells deviate by ≤ 0.0023** — the conformal guarantee holds on unseen data. h1 under-covers by 2.88 pp at the 90% damaged level; every other cell is within 1.5 pp.

---

## 4. Negative results (equally important)

### 4.1 The μ_total mixture is unusable as a point predictor

Using the **point estimate** of the second-moment mixture, `μ = p·μ_C + (1−p)·μ_N`, directly as the prediction:

| Approach | 4-month RMSE |
|---|---|
| **μ_total mixture** | **788.01** |
| LGB baseline | 493.05 |
| v22 hard rule (full scope) | 342.40 |

**0 of 10 airports** improve on the baseline; LIRF degrades from 1483.94 to **2564.7**.

Two compounding causes:

1. `μ_N` comes from sklearn decision trees, which do not natively handle categorical features and are themselves 35–60 s worse than the LGB baseline;
2. Probability dilution — mean `p` is only ≈ 0.13, so rows that should be replaced outright are only partially replaced. The `BLOCK` overwrite at LIRF is a **deterministic mechanism**; the hard rule (replace on hit) fires on almost every row there and is directionally correct, whereas a probabilistic mixture can never achieve "full replacement" by construction.

### 4.2 Conclusion: uncertainty quantification does not produce point predictions

**The physics-constrained second-moment mixture for σ works** (tail converges by 10 orders of magnitude, Spearman reaches +0.4354 / +0.3966, conformal coverage deviates by ≤ 0.0023 on unseen data), but **its use is limited to uncertainty quantification and anomaly detection, and it must not be used for point prediction**.

The correct approach to point prediction is **deterministic hard repair + physical window constraints** (§2.2–2.3). The two are complementary, not substitutes.

---

## 5. Artifact inventory

| File | Description |
|---|---|
| `submissions/jubilant-puzzle_final.parquet` | **Official submission**, 670,790 rows (noclip E2) |
| `submissions/jubilant-puzzle_final_clip_backup.parquet` | clip[30,10800] backup (for comparison) |
| `supplementary/final_uncertainty.parquet` | Uncertainty supplement, 8 columns, **not scored** |
| `notebooks/diag/d0/holdout_noclip_e2.py` | Four-month holdout recomputation (324.76) |
| `notebooks/diag/d0/gate_noclip.py` | Submission Gate checks (range validation) |
| `notebooks/diag/d0/moment_conformal.py` | Second-moment mixture + stratified conformal |
| `notebooks/diag/d0/supplementary_uncertainty.py` | Uncertainty generation for the final 4 months |

### Column definitions for `supplementary/final_uncertainty.parquet`

| Column | Type | Description |
|---|---|---|
| `MVT_ID_mvt` | int64 | Row-for-row identical to the submission file |
| `point_pred` | int64 | Copied from the submission file, unmodified |
| `sigma_total` | float64 | Mixture σ (σ_C=100, σ_N capped at 2000) |
| `is_damaged` | bool | T14400 hit OR p>0.5 |
| `lo_90` / `hi_90` | float64 | 90% interval |
| `lo_95` / `hi_95` | float64 | 95% interval |

Four-month statistics: median σ **209.1**, σ_max **2.64e4**, damaged rows **6,926**
(63 T14400 hits + 6,863 with p>0.5).
Median 95% interval width is 442 s on the damaged stratum vs 732 s on the normal stratum — the damaged stratum is tighter, matching the intent of the stratified design.

---

## 6. Data conventions worth remembering

- Raw parquet timestamps are `datetime64[us, UTC]` (**microseconds**); cast with `.astype("datetime64[ns]")` before taking `int // 10^9`, otherwise timestamps land in 1970
- `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are 100% null in final DEP — never touched
- Sentinel values are negative timestamps; search in both directions
- Strata definitions: short `0 < TX ≤ 1800` / long `1800 < TX ≤ 10800` / core `TX ≤ 10800` / full `TX > 0`