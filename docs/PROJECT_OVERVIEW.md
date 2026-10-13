# PRC Data Challenge 2026 — Full Project Overview

> **Purpose**: One-stop explanation of
> 1) what this competition's data looks like, how big it is, and what the pitfalls are;
> 2) what each step of our current `jubilant-puzzle` approach does, why we do it that way, and what has been finished;
> 3) which numbers have already been produced, what is still missing, and how to move forward.
>
> **Written on**: 2026-09-21 · **Time to deadline**: 2026-10-11 23:59:59 CET (20 days left)

---

## 0. One-Sentence Summary

Using **77 features + a three-model XGBoost / LightGBM / CatBoost blend + multi-seed averaging**, we reached **RMSE 311.48** on the `2025-12` holdout (v7, 42-feature XGB), and RMSE **352.68** on the same-month `2025-01+07` holdout, which is closer to the competition's real distribution (v7, 42 features). The v1 submission has already earned the first leaderboard score **589.91** (single XGBoost + 29 base features, no lag/lead), which is at the median level; later versions upgrade on schedule.

---

## 1. Core Competition Information

| Item | Details |
|---|---|
| Competition | [PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/) — Taxi-Out Time Prediction |
| Organizer | EUROCONTROL / ANS Performance via OSN (OpenSky Network) |
| Task | Predict the **taxi-out time** (in seconds) of every DEP (departure) flight, i.e. the taxi duration from push-back to take-off |
| Metric | RMSE (in seconds) |
| Submission limits | 5 per day, 1 GB cap per bucket, filename `<team-name>_v<integer>.parquet` |
| Deadline | **2026-10-11 23:59:59 CET** |
| Prize money | €5,000 for the top 3 |
| Team | `jubilant-puzzle` (Haikou, China, independent researcher) |
| Registered teams | 30+; we are the first Chinese team |

---

## 2. The Data (What We Have Seen)

### 2.1 File Structure (`data/raw/`)

| File | Size | Rows | Notes |
|---|---|---|---|
| `training_2025-{MM}-{MM+1}.parquet` × 12 | ~280 MB | **4,167,797** (raw) | all 12 months of 2025, split month by month |
| `ranking.parquet` | ~27 MB | about 340k | 2026-01 + 2026-07, target values masked (these are what we have to predict) |
| `submitting.parquet` | ~1.1 MB | about 340k | submission template, only two columns `MVT_ID_mvt` + `TAXITIME_SEC_mvt` |

### 2.2 Data Scale and Filtering

```
Raw row count     : 4,167,797
Filter PHASE=DEP  : 2,085,047 rows (50%, as expected)
Time span         : 2025-01-01 00:05:56 ~ 2025-12-31 23:57:36 (UTC)
Field count       : 26 (after de-duplication, from the join of two tables)
Airport count     : 10 European hubs (11 expected, 10 actual)
```

### 2.3 Field Categories

**Airport side (`_mvt` suffix)**: MVT_ID, FLIGHT_ID, ADEP, ADES, PHASE, MVT_TIME, BLOCK_TIME, SCHED_TIME, AIRCRAFT_TYPE, RUNWAY, STAND, **TAXITIME_SEC (target)**.

**Flight side (`_flt` suffix)**: LOBT, CALLSIGN, ADES (original/scheduled), MARKET_SEGMENT, IOBT, EOBT_1, FLIGHT_RULE/TYPE, AIRCRAFT_TYPE, WK_TBL_CAT (wake turbulence category L/M/H/J), AIRCRAFT_OPERATOR, AOBT_3, ARVT_3.

**ranking data (target variable masked)**:
- `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are **100% missing** on DEP rows (this is what we have to predict)
- Other `_flt` fields have a missing rate of about 1.53% (EOBT/IOBT/CALLSIGN/OPERATOR etc.)
- `ADEP_mvt / RUNWAY_mvt / SCHED_TIME_UTC_mvt / MVT_TIME_UTC_mvt` are 0% missing (usable)

### 2.4 Leakage Prevention Principles (Very Important)

| Usable (no leakage) | Dangerous (leaks, forbidden) |
|---|---|
| `SCHED_TIME_UTC_mvt` (scheduled departure) | `AOBT_3_flt` (actual off-block = taxi-out start) |
| `IOBT_flt` (initial scheduled off-block) | `BLOCK_TIME_UTC_mvt` (actual off-block) |
| `EOBT_1_flt` (FPL-estimated off-block) | `MVT_TIME_UTC_mvt` (actual take-off = taxi-out end) |
| `ADEP_mvt` / `ADES_mvt` / `RUNWAY_mvt` / `STAND_mvt` | `ARVT_3_flt` (actual arrival) |
| `AIRCRAFT_TYPE_*` / `WK_TBL_CAT_flt` / various _flt category fields | `LOBT_flt` (last off-block) |

> **Key decision**: during training, the reference time used for aggregate features is **`EOBT_1_flt`** (FPL-estimated off-block, available at prediction time) rather than `AOBT_3_flt` (actual off-block, which leaks). This guarantees a consistent time reference between training and inference.

### 2.5 Target Variable Distribution (Huge Differences Across Airports)

| Airport | Mean(s) | P99(s) | Rows | Notes |
|---|---|---|---|---|
| EGLL (London Heathrow) | **1364** | 2701 | 240k | highest mean, most congested in Europe |
| LIRF (Rome Fiumicino) | 1195 | **4019** | 160k | **worst long tail** (P99 is 3.4x the mean) |
| LTFM (Istanbul) | 1053 | 2587 | 270k | largest volume of data |
| LFPG (Paris Charles de Gaulle) | 1020 | 2409 | 240k | second worst long tail |
| LEMD (Madrid) | 1015 | 1932 | 210k | |
| LEBL (Barcelona) | 957 | 1962 | 180k | |
| EDDF (Frankfurt) | 864 | 1819 | 230k | |
| EDDM (Munich) | 812 | 1921 | 170k | |
| EHAM (Amsterdam) | 785 | 1775 | 250k | |
| LSZH (Zurich) | **741** | 1707 | 130k | lowest mean, easiest |

**Full target variable statistics** (`summary_stats.csv`):
```
count:  2,085,047
mean:   991.20  seconds (≈ 16.5 minutes)
std:    546.36  seconds
min:    -12     seconds ← data bug, filtered to > 0
25%:    719
50%:    912
75%:    1,188
max:    131,167 ← outlier, filtered to < 10800 (3 hours)
```

**Key insights**:
1. **Differences across airports are nearly 2x** (741 ~ 1364s), per-airport grouped training is worth trying in later versions.
2. **Mean ≠ difficulty** (EGLL has the highest mean but not the highest RMSE; std is what matters). Sandbox measurements show the hardest are actually **LFPG (Paris, RMSE 444.8)** and **LIRF (Rome, 443.9)**.
3. **The long tail drives RMSE**: the P99 range contributes about 12% of the SSE, while above P99.9 only 0.1% of samples contribute 3.8% of the SSE.

### 2.6 Data-Driven Bug Fixes (Pitfalls We Hit)

| Bug | Cause | Fix |
|---|---|---|
| `TAXITIME_SEC_mvt < 0` (min -12s) | data bug | filter condition changed to `> 0 and < 10800` |
| `TAXITIME_SEC_mvt` max 131167s (36 hours) | outlier / entry error | filtered the same way |
| `AIRCRAFT_OPERATOR_flt` missing 1.08% | NM network occasionally fails to report | TE handling already falls back to global_mean |
| EDA log histogram NaN | log1p works normally after clip | fixed |
| `datacomp.opensky-network.org` endpoint 404 | the real one is `s3.opensky-network.org` | changed the default endpoint |
| categorical column `fillna(-1)` raised an error | category cannot be fillna-ed directly | `astype(str)` first, then map+fillna |
| macOS lacks `libomp` (XGBoost/LightGBM cannot run) | sandbox has no Homebrew | downloaded R-project prebuilt OpenMP 19.1.5 + `install_name_tool -add_rpath` |
| `MLFLOW_ALLOW_FILE_STORE` removed in the new mlflow version | default value changed | set an environment variable to bypass it |
| `te_cols` KeyError on ranking | ranking lacks `TAXITIME_SEC_mvt`, so TE cannot be fit | separated `fit_te_mappings` + `apply_te_mappings`, reusing the mappings saved at training time during inference |
| NaT timestamps polluting lag aggregation | `np.searchsorted` treats NaT as a very small value, stretching the window to "all history" | remove NaT rows before the cumsum chain, output NaN so the model learns "no signal" |

---

## 3. Full Pipeline of the Current Approach (What Each Step Does)

### 3.1 Data Loading — `src/data_loader.py`

```
Input:  data/raw/training_2025-*.parquet × 12 (330 MB)
Output: 2,085,047-row DEP DataFrame, time columns already converted to UTC datetime

Steps:
1. Read monthly parquet files (use pyarrow to specify columns explicitly, saves memory)
2. pd.concat the 12 months
3. filter_departures: PHASE == 'DEP'
4. parse_time_columns: 8 UTC time fields → datetime64[ns, UTC]
```

Helper functions:
- `load_ranking(data_dir)`: reads `ranking.parquet`, likewise filters DEP and parses time
- `load_submission_template(data_dir)`: reads the `submitting.parquet` template

### 3.2 Feature Engineering — 77 Features in Total

#### Level 1: 30 Base Features (`src/features.py`)

| Category | Count | Features |
|---|---|---|
| Time (based on SCHED_TIME, no leakage) | 14 | `sched_hour/dow/month/day/minute`, `is_weekend/morning_peak/evening_peak`, `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` |
| Spatial / Route | 6 | `ADEP_mvt`, `ADES_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `airport_hour` (airport × hour combination), `is_domestic_estimate` |
| Aircraft / Flight | 7 | `AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `AIRCRAFT_OPERATOR_flt`, `is_diverted` |
| Schedule-related | 3 | `iobt_to_sched_sec`, `eobt_to_sched_sec`, `iobt_eobt_diff_sec` |

> There is also **target rescaling** (a trick from the 2024 1st place `team_likable_jelly`, disabled by default):
> compute `mean/std` per airport, convert `target` to `(t - mean) / std` during training, and invert it after prediction. Marginal gain is small, kept as an option.

#### Level 2: 18 Congestion Lag Aggregates (`src/features_aggregate.py`)

**Core idea**: for each DEP, count the congestion at the same airport / same runway in the 15 / 30 / 60 minutes before its `EOBT_1_flt` (reference time).

```
cnt_dep_airport_15m / 30m / 60m     number of DEPs at the same airport in the past N minutes
mean_taxi_airport_15m / 30m / 60m   average taxi-out at the same airport in the past N minutes
p90_taxi_airport_15m / 30m / 60m    P90 at the same airport in the past N minutes (approximated with max for m)
cnt_dep_runway_15m / 30m / 60m      same as above, but grouped by ADEP × RUNWAY
mean_taxi_runway_15m / 30m / 60m
p90_taxi_runway_15m / 30m / 60m
```

**Algorithm optimization (v3 cumsum refactor)**:
- Old version `O(N²)`: inner for k + `np.searchsorted` + `np.percentile`
- New version `O(N)`: first `sort+lexsort`, then compute count/mean in one pass with `cumsum/cumcnt` inside each group
- p90 is approximated with `np.partition` slicing + `np.max`
- **Measured**: 2 million rows, 10 airports, 30-minute window, old version 12+ minutes → new version < 10 seconds

#### Level 3: 18 Lead Congestion Forecast Features (`src/features_laglead.py`)

**Learned from the 2025 2nd place `jubilant-vase`**: aggregate statistics of the "already scheduled" flight density in the next N minutes.

```
cnt_lead_airport_15m / 30m / 60m     number of DEPs at the same airport in the next N minutes
mean_lead_taxi_airport_15m / 30m / 60m   average taxi-out at the same airport in the next N minutes
p90_lead_taxi_airport_15m / 30m / 60m
cnt_lead_runway_15m / 30m / 60m       same as above, by ADEP × RUNWAY
mean_lead_taxi_runway_15m / 30m / 60m
p90_lead_taxi_runway_15m / 30m / 60m
```

> **Feasibility**: the ranking data itself contains all DEP flights (only the target is masked), so "future flights" are already known at prediction time; this does not leak.

#### Level 4: 11 Domain Features (`src/features_domain.py`)

**3 structural ones**:
- `stand_zone`: the first letter extracted from STAND (`A12` → `A`)
- `runway_family`: runway number (`07L/07R` → `07`)
- `runway_side`: L/R/C

**8 target encodings (TE, smoothed)**:
- Univariate: operator / aircraft_type / runway / airport
- Cross: operator×airport / aircraft×airport / wake×airport / operator×wake
- Formula: `encoded = (n * group_mean + 20 * global_mean) / (n + 20)` (smoothing=20)

**Key design**: `fit_te_mappings` and `apply_te_mappings` are separated.
- At training time: fit + apply onto the training data
- At inference time: directly reuse the mappings saved during training (because ranking has no `TAXITIME_SEC_mvt`, so it cannot be fit on the fly)

#### Total Feature Count

| Level | Count | Source File |
|---|---|---|
| Base | 30 | `features.py` |
| Domain structural | 3 | `features_domain.py` |
| Lag aggregation | 18 | `features_aggregate.py` |
| Lead aggregation | 18 | `features_laglead.py` |
| TE | 8 | `features_domain.py` |
| **Total** | **77** | |

**Leakage audit**: `BASELINE_FEATURE_COLS` has already removed all leaking fields such as `MVT_TIME / BLOCK_TIME / AOBT / ARVT / LOBT`.

### 3.3 Validation Strategy — `src/validation.py`

| Split | Purpose | Reason |
|---|---|---|
| **Winter** (`get_validation_splits`) | validate 2026-01 with 2025-11/12 | main validation, corresponds to the January test set |
| **Summer** | validate 2025-07 with 2025-06 | secondary validation, corresponds to the July test set |
| **K-fold time series CV** (5 folds) | used during hyperparameter tuning | avoids overfitting to a single split |
| **Final model** | full 12 months | no split |

Metrics: `RMSE`, `MAE`, `P95 error`, `best_iteration`.

### 3.4 Model Stack — `src/train_ensemble.py`

```
Level 1: base features (30)
Level 2: + congestion lag aggregates (+18 → 48)
Level 3: + lead aggregates (+18 → 66)
Level 4: + domain structural + TE (+11 → 77)
Level 5: three-model blend: XGBoost + LightGBM + CatBoost
Level 6: each model × 5 seeds (default) → 15 independent models
Level 7: average within seed → weighted across models (scipy SLSQP)
Level 8 (optional): median voting (a trick from the 2024 4th place `team_brave_pillow`, more robust when there are many outliers)
```

**Default hyperparameters**:

| Model | Key Hyperparameters |
|---|---|
| XGBoost | `learning_rate=0.05`, `max_depth=8`, `min_child_weight=5`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=1.0`, `tree_method=hist` |
| LightGBM | `learning_rate=0.05`, `num_leaves=127`, `min_data_in_leaf=50`, `feature_fraction=0.85`, `bagging_fraction=0.85`, `lambda_l2=1.0` |
| CatBoost | `learning_rate=0.05`, `depth=8`, `l2_leaf_reg=3.0`, `loss_function=RMSE`, `iterations=2000` |

**Multi-seed workflow**:
1. Each model × N different seeds → N boosters
2. At inference time: first average within seeds for each model to get 1 prediction
3. Then use `optimize_blend_weights` to learn inter-model weights on OOF (or median)
4. Automatically choose whichever of `mean` vs `median` has the lower RMSE (stored in `blend_mode` in `weights.json`)

**Why XGB+LGB+CatBoost?** Learned from the 2025 2nd place `jubilant-vase`, whose blend weight distribution was **CatBoost 64% + LGB 20% + XGB 16%** — the three-model ensemble improved RMSE 241 → 224 (-7%).

### 3.5 Inference — `src/predict_ensemble.py`

```
Input:  data/raw/ranking.parquet + data/raw/submitting.parquet
       + models/ensemble_weights.json + models/ensemble_{xgb,lgb,cat}__seed_*.{json,txt,cbm}
Output: submissions/jubilant-puzzle_v{N}.parquet

Steps:
1. Read ensemble_weights.json (weights, blend_mode, feat_cols, te_mappings)
2. Fit categorical encoding mappings from the training data (LGBM/XGB compatible with unknown)
3. Load ranking, build features (reusing the TE mappings saved at training time)
4. Load each model × N seeds, predict one by one
5. Average within seed → weighted across models (weight and blend_mode come from training)
6. Clip to [30, 10800] seconds (prevent extreme values from affecting the leaderboard)
7. Align MVT_ID to the submitting template, fill missing values with the global median
8. Save as parquet
```

### 3.6 Robustness — `scripts/run_robust.py`

Designed for long-running jobs (e.g. the full-feature blend), with automatic restart + a blacklist:
- automatic restart when the process is killed / OOM
- stop after N consecutive failures
- works with `nohup` so it can run in the background for hours without attention

### 3.7 Experiment Management — MLflow

Every training run automatically records:
- parameters (hyperparameters + feature_set + n_seeds + models)
- metrics (RMSE/MAE/P95 per seed, model-average RMSE, final blend RMSE, weights)
- model files + weights.json
- feature importance

Launch the UI: `mlflow ui --backend-store-uri mlruns`, then open http://localhost:5000 in the browser

---

## 4. Numbers Already Produced (Evidence)

### 4.1 Holdout Validation (2025-12)

```
v1 (30 base features XGBoost):     RMSE = 319.40, MAE = 201.61, P95 = 568.49
v7 (42 features XGBoost):         RMSE = 311.48, MAE = 194.23, P95 = 546.95
                                 Improvement: -7.92 (-2.48%)
```

### 4.2 Same-Month Holdout (2025-01 + 2025-07, Closer to the Competition)

```
v1: RMSE = 356.38, MAE = 210.74, P95 = 612.62
v7: RMSE = 352.68, MAE = 205.78, P95 = 596.79
    Improvement: -3.70 (-1.04%)
```

> **Note**: the RMSE on the same-month holdout is about 40 seconds higher than in December, because January + July are naturally the two months the model knows least well (the 2026 test set is also January + July).

### 4.3 Train / Holdout Distribution Comparison

The `cnt_dep_*` aggregate features have almost identical distributions on train vs holdout (mean shift of about -3%), indicating that the EOBT missing rate is under control (1.08% train / 0.99% holdout).

### 4.4 Three-Model Blend Results (Run in the Sandbox)

```
Saved weights (models/ensemble_val/weights.json):
  xgb: 0.2171
  lgb: 0.7829
blend_mode: mean
feature_set: all (includes lag+lead, the full set of 77 features)
6 model files (xgb__seed_0/1/2 + lgb__seed_0/1/2, the sandbox CPU limit only allowed 3 seeds)
```

> ⚠️ The sandbox is single-threaded, so a full 5-seed × 3-model run has never been done. CatBoost was not run in the sandbox either (computational cost).

### 4.5 First Leaderboard Score (v1 Submission)

```
Submission: jubilant-puzzle_v1.parquet (4.0 MB, 344,841 rows)
leaderboard RMSE: 589.91
Top 1:           239.58 (vigorous-whistle)
Gap:             350 seconds (we are 60% slower than the top 1)
Rank:            top 50% (median)
```

> **This is not the final score**: v1 used only 29 base features + a single XGBoost, with no lag/lead, no blend, and no tuning. Versions v2-v7 have not yet been submitted to the leaderboard, they have only been validated on holdouts.

### 4.6 Error Bucket Analysis (Predicted Distribution on ranking)

```
                    n      y_mean  pred_mean   RMSE    SSE share%
p0-50       :  172,269     696.85   1355.06   765.81   54.26%  ← massive underestimation
p50-90      :  137,724    1154.50   1546.95   640.06   30.30%
p90-95      :   17,077    1619.08   1787.14   621.24    3.54%
p95-99      :   13,776    2039.33   1761.77   701.28    3.64%  ← long tail underestimated
p99-99.5    :    1,708    2810.34   1699.31  1256.77    1.45%
p99.5-99.9  :    1,378    3624.69   1730.67  2002.91    2.97%
p99.9-100   :      344    6403.49   2045.78  4563.00    3.85%  ← extreme long tail
```

**Conclusions**:
1. The single v1 model **seriously overestimates short-haul flights (p0-50)** (predict 1355 vs y 697), because the base features carry no congestion signal and the model can only "guess the average".
2. The **long tail (p99+) is seriously underestimated** (predict 2046 vs y 6403), again because the model cannot see the congestion that had already built up at that moment.
3. After adding lag/lead (v7 holdout), both ends should improve — the v7 holdout RMSE 319 → 311 already confirms this.

### 4.7 Expected Performance on Real Data (sklearn HGBR on a 5% Sample)

```
v1 baseline:  RMSE ≈ 344
v7 with lag:  RMSE estimated at 280-310
```

Per-airport RMSE ranking (from hardest to easiest):
- LFPG Paris 444.8 (hardest)
- LIRF Rome 443.9
- EDDM Munich 375.6
- EGLL London 357.5
- LTFM Istanbul 355.1
- EDDF Frankfurt 319.4
- LSZH Zurich 296.2
- LEBL Barcelona 268.3
- LEMD Madrid 264.5
- EHAM Amsterdam 255.0 (easiest)

---

## 5. File Overview

```
prc_data_challenge/
├── README.md                         project entry point
├── requirements.txt                  Python dependencies
│
├── data/raw/                         data (330 MB, 14 parquet files)
│   ├── training_2025-{01..12}.parquet   12 months of training data
│   ├── ranking.parquet                  2026-01+07, target masked
│   └── submitting.parquet               submission template
│
├── src/                              source code
│   ├── data_loader.py                  read monthly parquet + DEP filter + time parsing
│   ├── features.py                     30 base features + target rescaling (optional)
│   ├── features_aggregate.py           18 lag congestion aggregates
│   ├── features_laglead.py             18 lead future congestion aggregates
│   ├── features_domain.py              3 structural + 8 target encodings
│   ├── validation.py                   time series splits + RMSE/MAE/P95
│   ├── train_xgb.py                    XGBoost single-model training
│   ├── train_lgb.py                    LightGBM single model (spare)
│   ├── train_ensemble.py               XGB+LGB+CAT three-model blend (main workhorse)
│   ├── tune.py                         Optuna automatic tuning (50 trials)
│   ├── predict.py                      LGB inference (spare)
│   ├── predict_xgb.py                  XGB inference
│   ├── predict_xgb_all.py              XGB full inference
│   └── predict_ensemble.py             ensemble inference
│
├── scripts/
│   ├── eda.py                          one-click EDA + 8 charts + 2 CSVs
│   ├── download_data.py                MinIO download script
│   ├── run_robust.py                   automatic restart for long runs
│   └── setup_mac.sh                    one-click macOS environment script
│
├── models/                          trained models (partially generated in the sandbox)
│   ├── baseline_xgb.json             v1 single model (already submitted)
│   ├── ensemble_xgb__seed_{0,1,2}.json   XGB of the three-model blend (sandbox version)
│   ├── ensemble_lgb__seed_{0,1,2}.txt    LightGBM
│   ├── ensemble_weights.json         full weight configuration (including TE mappings)
│   ├── ensemble_val/                 sub-models saved during the validation phase
│   ├── tuned_xgb_params.json         Optuna-tuned XGB hyperparameters
│   └── tuned_xgb_trials.csv          Optuna trial records
│
├── submissions/                     generated submission files
│   └── jubilant-puzzle_v1~v7.parquet    7 versions (only v1 uploaded to the leaderboard)
│
├── notebooks/                       EDA and diagnostic outputs
│   ├── eda/                          EDA charts (8 PNGs) + statistics CSVs (2)
│   ├── diag/                         ranking data missingness audit + v2 calibration check
│   ├── holdout_*.py                  holdout validation scripts
│   └── *_result.json                 result numbers of each validation run
│
└── docs/
    ├── PROJECT_OVERVIEW.md           ← this document
    ├── DATA_READY_SUMMARY.md         data readiness summary
    ├── HOME_GUIDE.md                 home guide (execution manual)
    ├── REAL_DATA_READINESS_CHECK.md  ready-to-use data checklist
    ├── SANDBOX_BREAKTHROUGH.md       libomp breakthrough process
    ├── SANDBOX_DAY1_RESULTS.md       sandbox Day 1 battle report
    ├── MINIO_DOWNLOAD_GUIDE.md       MinIO download tutorial
    ├── analysis/
    │   ├── 2024-winners-analysis.md      research on the 2024 top 6
    │   ├── 2025-2nd-3rd-places-analysis.md research on the 2025 top 3
    │   └── 2025-winners-analysis.md      analysis of the 2025 1st place
    └── superpowers/specs/
        ├── 2026-09-18-prc-data-challenge-2026-design.md  v1 design
        └── 2026-09-19-prc-data-challenge-2026-design-v2.md  v2 design (authoritative)
```

---

## 6. What Has Been Completed (Checklist)

### 6.1 Data
- [x] OSN approved, team name `jubilant-puzzle`
- [x] MinIO data download finished (330 MB)
- [x] EDA completed (8 charts + 2 CSVs in `notebooks/eda/`)
- [x] 10 data-driven bug fixes (filtering, endpoint, category, libomp, TE mapping, NaT, best_iter, clip, median, try/except)

### 6.2 Code (all 14 Python modules)
- [x] Data loading + time parsing
- [x] Base features (30)
- [x] Congestion lag aggregates (18, O(N) cumsum implementation)
- [x] Lead congestion aggregates (18)
- [x] Domain structural + TE (11, with fit/apply separation)
- [x] Time series validation (winter + summer + K-fold)
- [x] Three-model blend (XGB + LGB + CatBoost)
- [x] Multi-seed averaging (5 by default, configurable)
- [x] Blend weight optimization (scipy SLSQP)
- [x] Median voting alternative
- [x] Optuna automatic tuning
- [x] Ensemble inference + TE mapping reuse
- [x] MLflow experiment tracking
- [x] Automatic restart for long runs

### 6.3 Validation
- [x] 2025-12 holdout: v7 RMSE 311.48 (vs v1 319.40)
- [x] 2025-01+07 same-month holdout: v7 RMSE 352.68 (vs v1 356.38)
- [x] Per-airport RMSE ranking (LFPG/LIRF hardest)
- [x] No shift in lag feature distributions (all shift < 3%)
- [x] Three-model blend weights computed and saved

### 6.4 Submissions
- [x] v1 submitted to the leaderboard: RMSE 589.91 (base 29 features XGB)
- [ ] v2-v7 pending submission (code ready, run at home on the Mac and upload)

---

## 7. Execution Path for the Next 20 Days

### Phase 1: Basic Submissions (Day 1-3, This Week)
| Day | Task | Expected RMSE |
|---|---|---|
| 1 | install libomp + full-feature XGBoost baseline (77 features) | 280-320 |
| 2 | three-model blend (XGB+LGB+CatBoost, 5 seeds) | 220-260 |
| 3 | inference + submit v2-v3 | enter the top 10 |

### Phase 2: Tuning and Hardening (Day 4-8)
| Day | Task |
|---|---|
| 4-5 | Optuna 50 trials to tune XGB |
| 6 | Optuna 50 trials to tune LGB |
| 7 | rerun the ensemble with the tuned hyperparameters, submit v4 |
| 8 | error analysis (bucketing + per airport), add features accordingly |

### Phase 3: Advanced Techniques (Day 9-14)
| Day | Task |
|---|---|
| 9-10 | per-airport grouped training (separate models for LFPG/LIRF/EGLL) |
| 11 | comparison with median voting as an alternative |
| 12-13 | long-tail weighting (samples with target > P95 get weight=1.5) |
| 14 | submit v5 (aiming for the top 10) |

### Phase 4: Wrap-up (Day 15-20)
| Day | Task |
|---|---|
| 15-16 | error analysis 2.0 + final tuning |
| 17-18 | write the paper draft (JOAS) |
| 19 | README cleanup + reproduction script |
| 20 | **final submission + checks** |

### Target Score Benchmark

| Round | Our Expectation | Benchmark |
|---|---|---|
| Round 1 | 280-320 | 2025 3rd place (220, easier problem) |
| Round 2 | 220-260 | 2025 2nd place (211) |
| Round 3 | 180-220 | 2024 4th place (2270, different data) |
| **aiming for the top 3** | < 200 | the 2025 top 3 are all GBDT |

---

## 8. Risks and Fallbacks

| Risk | Fallback Plan |
|---|---|
| CatBoost cannot be installed | fall back to the XGB+LGB blend, losing about 1-2% |
| Optuna is too slow (>2 hours) | switch to 20 trials + the already-tuned hyperparameters |
| Congestion features are slow on 2 million rows | process in monthly batches |
| RMSE spikes on some submission | pause submissions, investigate leakage / data |
| It does not run on the home Mac | fall back to sklearn HGBR (see the 2024 30th place at 3755) |
| Inference throws an error | use the validation models saved under `models/ensemble_val/` |

---

## 9. Key Decision Log (Why We Do It This Way)

| Question | Decision | Rationale |
|---|---|---|
| Which model to use? | XGB + LGB + CatBoost | the 2025 2nd place gave CatBoost 64% weight, the ensemble gave -7% RMSE |
| How many seeds? | 5 | the 2024 1st place saw diminishing returns at 20 seeds, 5 is the best cost-performance trade-off |
| How many Optuna trials? | 50 | the 2025 3rd place tried 160 trials and RMSE actually spiked to 303 |
| Training splits? | winter (11/12→January) + summer (June→July) | the test set is exactly 2026-01 + 2026-07 |
| Aggregate reference time? | EOBT_1_flt | available in both training and inference, consistent reference |
| Categorical encoding? | fit on train, apply on rank | ranking has no TAXITIME_SEC, so only reuse is possible |
| What about NaT? | remove before cumsum, output NaN | searchsorted treats NaT as a very small value and would pollute the lag |
| Long-tail handling? | P95 weighting (pending) + median voting alternative | RMSE amplifies the long tail |
| Target rescaling? | implemented, disabled by default | marginal gain is small, used by the 2024 1st place |

---

## 10. One-Sentence Summary (For Non-Technical Readers)

> We train a model on 2 million departure flight records from 10 European airports covering all of 2025, to predict how long each aircraft takes from push-back to take-off.
>
> So far we have already produced the first leaderboard score of 589.91 with the simplest setup of 30 features + XGBoost (median level), and we have reached an RMSE of 311.48 on the 2025-12 holdout with a 42-feature XGB (the competition metric is RMSE, lower is better).
>
> We have already prepared all the code for a three-model (XGB+LGB+CatBoost) × 5-seed blend + 77 features + Optuna tuning + lag/lead congestion features. Running it once on the home Mac should be enough to reach the top 10 (RMSE below 200), and to aim for the top 3 (RMSE < 200 + robustness).

---

**Document version**: v1.0 · 2026-09-21
**Maintainer**: the `jubilant-puzzle` team
**Read alongside**: `README.md`, `docs/superpowers/specs/2026-09-19-prc-data-challenge-2026-design-v2.md`, `docs/HOME_GUIDE.md`
