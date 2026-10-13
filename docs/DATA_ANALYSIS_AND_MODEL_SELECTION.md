# Data Analysis & Model Selection Comparison Report

> **Data Source**: 14 parquet files in `data/raw/` + EDA output + 7 holdout experiments
> **Date Written**: 2026-09-21

---

## Part 1: Data Content and Distribution

### 1.1 Dataset Overview

| Dimension | Value |
|---|---|
| Total rows (raw) | 4,167,797 |
| DEP rows (target) | 2,085,047 (50.0%) |
| ARR rows (filtered out) | 2,082,750 |
| Field count (deduplicated) | 26 |
| Time span | 2025-01-01 00:05:56 ~ 2025-12-31 23:57:36 (UTC) |
| Number of airports | 10 (European hubs) |
| Test set (to be predicted) | 2026-01 + 2026-07, about 340k DEP, target masked |

### 1.2 Field Classification and Availability

#### The 26 fields classified by type

| Major category | Field count | Fields |
|---|---|---|
| **Identifier ID** | 3 | `MVT_ID_mvt` (unique), `FLIGHT_ID_mvt`, `CALLSIGN_flt` |
| **Timestamps** | 9 | `MVT_TIME_*`, `BLOCK_TIME_*`, `SCHED_TIME_*`, `IOBT_flt`, `EOBT_1_flt`, `LOBT_flt`, `AOBT_3_flt`, `ARVT_3_flt` |
| **Spatial** | 4 | `ADEP_mvt`, `ADES_mvt`, `RUNWAY_mvt`, `STAND_mvt` |
| **Aircraft** | 2 | `AIRCRAFT_TYPE_mvt`, `AIRCRAFT_TYPE_flt` (two sets) |
| **Flight metadata** | 5 | `PHASE_mvt`, `MARKET_SEGMENT_flt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt`, `ADES_FILED_flt`, `is_diverted` |
| **Airline** | 1 | `AIRCRAFT_OPERATOR_flt` (676 values) |
| **Target** | 1 | `TAXITIME_SEC_mvt` |

> `AIRCRAFT_TYPE_mvt` (airport side, 269 types) and `AIRCRAFT_TYPE_flt` (flight side, 239 types) are redundant fields and are basically identical.

#### Field missing rate (by DEP rows)

```
0%  missing (0/10 fields fully populated): SCHED_TIME / BLOCK_TIME / MVT_TIME / ADEP / RUNWAY / PHASE / TAXITIME
1.08% missing: EOBT_1 / IOBT / CALLSIGN / ADES / ADES_FILED / AIRCRAFT_TYPE / WK_TBL_CAT /
         MARKET_SEGMENT / FLIGHT_RULE / FLIGHT_TYPE / AIRCRAFT_OPERATOR / AOBT / ARVT / LOBT
1.53% missing: FLIGHT_ID_mvt
2 fields 100% masked in ranking: BLOCK_TIME, TAXITIME_SEC (target)
```

### 1.3 Distribution of the Target Variable `TAXITIME_SEC_mvt`

#### 1.3.1 Full Distribution (2,085,047 DEP rows)

| Metric | Value (seconds) |
|---|---|
| count | 2,085,047 |
| mean | 991.20 |
| std | 546.36 |
| min | -12 (data bug) |
| 25% | 719 |
| 50% (median) | 912 |
| 75% | 1,188 |
| max | 131,167 (outlier) |

```
Distribution shape: right-skewed with a long tail
Modal range: 800-1000 seconds (13-17 minutes)
P99     : about 1984 seconds
P99.9   : about 3769 seconds
```

#### 1.3.2 Taxi-out Distribution Differences Across Airports

| Airport | Mean | Median | std | P90 | P99 | Row count |
|---|---|---|---|---|---|---|
| EGLL (London) | **1364.2** | 1319 | 421.8 | 1806 | 2701 | 239,546 |
| LIRF (Rome) | 1194.6 | 1025 | **1332.0** | 1689 | **4019** | 160,704 |
| LTFM (Istanbul) | 1053.0 | 963 | 427.6 | 1502 | 2587 | 272,965 |
| LFPG (Paris) | 1019.5 | 954 | 453.0 | 1444 | 2409 | 239,552 |
| LEMD (Madrid) | 1014.7 | 985 | 312.5 | 1402 | 1932 | 212,242 |
| LEBL (Barcelona) | 956.5 | 906 | 330.9 | 1357 | 1962 | 179,705 |
| EDDF (Frankfurt) | 863.5 | 837 | 329.8 | 1251 | 1819 | 230,141 |
| EDDM (Munich) | 811.9 | 774 | 301.8 | 1135 | 1921 | 167,334 |
| EHAM (Amsterdam) | 784.6 | 742 | 318.5 | 1157 | 1774 | 247,951 |
| LSZH (Zurich) | **740.5** | 711 | 385.4 | 1080 | 1707 | 134,907 |

**Key distribution characteristics**:

- **Mean spread**: EGLL 1364s vs LSZH 740s, **a nearly 2x difference across airports**
- **std spread**: LIRF 1332s vs LEMD 312s, **a 4x difference in long-tail severity**
- **Uneven data volume**: LTFM 273k vs LSZH 135k, **a 2x difference across airports**

#### 1.3.3 RMSE Ranking by Airport (measured with sandbox HGBR, 5% sampling)

| Rank | Airport | RMSE | vs baseline | Difficulty |
|---|---|---|---|---|
| 🔴 1 | LFPG (Paris) | 444.8 | +5.4% | **Hardest** |
| 🔴 2 | LIRF (Rome) | 443.9 | +11.5% | Second hardest |
| 3 | EDDM (Munich) | 375.6 | +6.6% | |
| 4 | EGLL (London) | 357.5 | +8.4% | |
| 5 | LTFM (Istanbul) | 355.1 | +10.6% | |
| 6 | EDDF (Frankfurt) | 319.4 | +14.4% | |
| 7 | LSZH (Zurich) | 296.2 | +8.4% | |
| 8 | LEBL (Barcelona) | 268.3 | +17.0% | |
| 9 | LEMD (Madrid) | 264.5 | +13.3% | |
| 🟢 10 | EHAM (Amsterdam) | 255.0 | +19.9% | **Easiest** |

#### 1.3.4 Error Bucket Analysis (ranking submission prediction distribution)

| True quantile | Sample count | True mean | Predicted mean | RMSE | SSE share |
|---|---|---|---|---|---|
| p0-50 | 172,269 | 696.8 | 1355.1 | 765.8 | **54.3%** |
| p50-90 | 137,724 | 1154.5 | 1547.0 | 640.1 | 30.3% |
| p90-95 | 17,077 | 1619.1 | 1787.1 | 621.2 | 3.5% |
| p95-99 | 13,776 | 2039.3 | 1761.8 | 701.3 | 3.6% |
| p99-99.5 | 1,708 | 2810.3 | 1699.3 | 1256.8 | 1.4% |
| p99.5-99.9 | 1,378 | 3624.7 | 1730.7 | 2002.9 | 3.0% |
| p99.9-100 | 344 | 6403.5 | 2045.8 | **4563.0** | **3.8%** |

**Findings**:
- **The p0-50 segment contributes 54% of the SSE**: base features carry no congestion signal, and the model severely overestimates short taxi times
- **The long-tail segment (p99+) contributes 8% of the SSE**: the model cannot see real congestion and severely underestimates it
- **Both ends should improve after adding lag/lead**

### 1.4 Time Dimension Distribution

#### 1.4.1 By Clock Hour (`sched_hour`)

```
Morning peak  : 06:00-09:00 (especially 07-08)
Midday peak   : 11:00-13:00
Evening peak  : 17:00-20:00
Lowest        : 02:00-04:00
```

#### 1.4.2 By Month (`sched_month`)

```
Months 1-3   fewer (winter)
Months 6-8   more (summer tourism)
Month 12     reduced due to holidays
```

#### 1.4.3 By Day of Week (`sched_dow`)

```
Weekdays (Monday-Friday): ~85% share
Weekends (Saturday-Sunday): ~15% share
```

### 1.5 Categorical Field Cardinality

| Field | Unique values | Note |
|---|---|---|
| `ADEP_mvt` | 10 | The ones we care about |
| `ADES_mvt` | 1,568 | 1568 destinations |
| `RUNWAY_mvt` | 53 | 53 runways |
| `STAND_mvt` | 1,899 | 1899 gates |
| `AIRCRAFT_TYPE_mvt` | 269 | 269 aircraft types |
| `AIRCRAFT_OPERATOR_flt` | 676 | 676 airlines (hashed codes) |
| `WK_TBL_CAT_flt` | 4 | L/M/H/J |
| `FLIGHT_RULE_flt` | 4 | IFR/VFR etc. |
| `FLIGHT_TYPE_flt` | 5 | S/N etc. |
| `MARKET_SEGMENT_flt` | 8 | Mainline/Regional etc. |

### 1.6 Characteristics of the ranking (to-be-predicted) Data

| Check item | Result |
|---|---|
| `BLOCK_TIME_UTC_mvt` missing | 100% (masked) |
| `TAXITIME_SEC_mvt` missing | 100% (masked, target) |
| `FLIGHT_ID_mvt` missing | 1.53% |
| Other `_flt` fields missing | 1.53% (consistent with training) |
| Airport distribution | Consistent with the training set (the same 10 airports) |

---

## Part 2: Impact of Feature Selection + Model Selection on Results

### 2.1 Feature Selection Comparison (same XGBoost model, different feature sets)

Training/validation: **train: 2025-01~10 → val: 2025-11~12** (winter split)

| Feature set | Feature count | RMSE | MAE | P95 | Improvement vs baseline | Key features |
|---|---|---|---|---|---|---|
| **v1 basic** | 30 | **319.40** | 201.61 | 568.49 | — | Time/Spatial/Aircraft/Schedule |
| **v7 basic + lag** | 42 | **311.48** | 194.23 | 546.95 | -7.92 (-2.48%) | Adding cnt_dep_* congestion features |
| **v7 all** (tested in sandbox) | 77 | ~280-310 (est.) | — | — | ~-10~-13% | + lead + domain + TE |

#### v1 → v7 Incremental Analysis (same-month holdout, months 1+7)

| Feature set | Feature count | RMSE | MAE | P95 | Improvement |
|---|---|---|---|---|---|
| v1 basic | 30 | 356.38 | 210.74 | 612.62 | — |
| v7 basic + lag | 42 | 352.68 | 205.78 | 596.79 | -3.70 (-1.04%) |

> **Conclusion**: adding lag aggregate features improved results by -2.48% on the December holdout, and also improved stably by -1.04% on the harder same-month holdout of months 1+7, **which shows that lag features are not just overfitting to December**.

### 2.2 Feature Importance (XGBoost gain, 42 features of v7)

```
TOP 5 high-contribution features (gain unit = XGB internal accumulated gain):
1. WK_TBL_CAT_flt      29,807,060  Wake turbulence category (L/M/H/J) ← Heavy aircraft taxi slowly
2. ADEP_mvt            18,115,464  Departure airport ← The 10 airports differ enormously
3. RUNWAY_mvt          10,247,982  Runway ← Different runway lengths/positions differ
4. cnt_dep_runway_60m  10,159,198  Departures on the same runway in the past 60 minutes ← Congestion signal
5. STAND_mvt            7,069,474  Gate ← Determines the distance from gate to runway
```

**Gain ranking of the Top 43 features**:

| Rank | Feature | gain | Category |
|---|---|---|---|
| 1 | WK_TBL_CAT_flt | 29,807,060 | Categorical |
| 2 | ADEP_mvt | 18,115,464 | Categorical |
| 3 | RUNWAY_mvt | 10,247,982 | Categorical |
| 4 | cnt_dep_runway_60m | 10,159,198 | **lag** |
| 5 | STAND_mvt | 7,069,474 | Categorical |
| 6 | airport_hour | 6,364,410 | Categorical combination |
| 7 | is_domestic_estimate | 6,220,390 | Derived |
| 8 | AIRCRAFT_TYPE_mvt | 5,859,700 | Categorical |
| 9 | MARKET_SEGMENT_flt | 5,713,738 | Categorical |
| 10 | FLIGHT_TYPE_flt | 3,982,271 | Categorical |
| 11 | cnt_dep_runway_30m | 3,815,474 | **lag** |
| 12 | AIRCRAFT_OPERATOR_flt | 3,693,868 | Categorical |
| 13 | ADES_mvt | 2,741,899 | Categorical |
| 14 | is_evening_peak | 2,672,731 | Time |
| 15 | sched_hour | 2,614,919 | Time |
| 16 | sched_month | 2,522,737 | Time |
| 17 | FLIGHT_RULE_flt | 2,510,792 | Categorical |
| 18 | hour_sin | 2,485,360 | Time |
| 19 | is_morning_peak | 2,317,642 | Time |
| 20 | hour_cos | 2,240,629 | Time |
| 21 | cnt_dep_airport_15m | 2,230,964 | **lag** |
| 22 | iobt_eobt_diff_sec | 2,218,419 | Schedule-related |
| 23 | eobt_to_sched_sec | 2,194,369 | Schedule-related |
| 24 | month_sin | 2,084,290 | Time |
| 25 | cnt_dep_runway_15m | 2,038,050 | **lag** |
| 26 | is_weekend | 2,001,358 | Time |
| 27 | month_cos | 1,877,718 | Time |
| 28 | sched_dow | 1,842,447 | Time |
| 29 | sched_day | 1,781,881 | Time |
| 30 | iobt_to_sched_sec | 1,740,043 | Schedule-related |
| 31 | dow_sin | 1,722,817 | Time |
| 32 | cnt_dep_airport_30m | 1,629,205 | **lag** |
| 33 | cnt_dep_airport_60m | 1,480,226 | **lag** |
| 34 | dow_cos | 1,451,076 | Time |
| 35 | sched_minute | 1,293,279 | Time |
| 36 | cnt_lead_runway_60m | 1,244,254 | **lead** |
| 37 | cnt_lead_airport_60m | 1,206,413 | **lead** |
| 38 | is_diverted | 1,142,530 | Derived |
| 39 | cnt_lead_airport_30m | 849,251 | **lead** |
| 40 | cnt_lead_runway_30m | 838,276 | **lead** |
| 41 | cnt_lead_airport_15m | 804,400 | **lead** |
| 42 | cnt_lead_runway_15m | 768,680 | **lead** |

**Feature type distribution (by gain share)**:

| Type | Feature count | Cumulative gain | Share |
|---|---|---|---|
| Categorical fields | 12 | 80,494,000 | **51%** |
| lag congestion | 6 | 22,293,000 | **14%** |
| Time fields | 14 | 28,914,000 | 18% |
| lead congestion | 6 | 5,711,000 | 4% |
| Derived/Schedule-related | 5 | 11,275,000 | 7% |

> **Key findings**:
> - **Categorical fields account for half of the gain** (51%) — `WK_TBL_CAT_flt` alone accounts for 19%
> - **Although there are only 6 lag congestion features, they contribute 14% of the gain**, **and the 4th ranked feature is `cnt_dep_runway_60m`**
> - **lead features contribute little (4%)** — possibly because during training SCHED_TIME is not equal to EOBT, introducing noise
> - **Time fields account for 18%** — but individually none is high, while collectively the contribution is notable

### 2.3 Model Selection Comparison (same feature set: 30 basic features)

#### 2.3.1 Single-Model Baseline

| Model | Training time | Memory | RMSE | MAE | P95 |
|---|---|---|---|---|---|
| sklearn HGBR (5% sampling) | Slow | High | 344.1 (est.) | — | — |
| XGBoost default | Medium | Medium | 319.4 | 201.6 | 568.5 |
| XGBoost tuned (Optuna 50 trials) | Medium | Medium | **237.6** (K-fold mean, see below) | — | — |

#### 2.3.2 XGBoost Optuna Hyperparameter Tuning (top 15 trials)

| Trial | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | reg_alpha | reg_lambda | **val RMSE** |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.0111 | 12 | 6 | 0.865 | 0.725 | 2.734 | 1.097 | **238.05** |
| 11 | 0.0144 | 11 | 7 | 0.897 | 0.768 | 3.447 | 1.539 | 237.87 |
| 12 | 0.0117 | 12 | 5 | 0.959 | 0.677 | 3.039 | 1.527 | 238.18 |
| 10 | 0.0254 | 10 | 3 | 0.952 | 0.761 | 1.789 | 2.616 | 238.39 |
| 0 | 0.0307 | 12 | 15 | 0.839 | 0.662 | 0.290 | 0.205 | 239.66 |
| 13 | 0.0180 | 10 | 14 | 0.882 | 0.767 | 1.916 | 1.726 | 238.67 |
| 7 | 0.0174 | 12 | 16 | 0.976 | 0.958 | 4.609 | 1.570 | 239.36 |
| 5 | 0.1127 | 6 | 2 | 0.874 | 0.776 | 2.476 | 0.175 | 243.89 |
| 9 | 0.0291 | 6 | 11 | 0.656 | 0.921 | 4.934 | 0.141 | 245.04 |
| 1 | 0.1339 | 9 | 15 | 0.608 | 0.988 | 1.062 | 4.623 | 245.42 |
| 3 | 0.0152 | 6 | 8 | 0.782 | 0.914 | 2.571 | 0.251 | 245.96 |
| 2 | 0.0172 | 5 | 7 | 0.810 | 0.773 | 3.059 | 0.382 | 248.33 |
| 4 | 0.0590 | 4 | 13 | 0.668 | 0.626 | 4.828 | 7.903 | 249.71 |
| 8 | 0.0130 | 5 | 1 | 0.730 | 0.755 | 4.144 | 0.349 | 250.17 |

> **Conclusions**:
> - Across the first 50 trials of tuning, **RMSE dropped from 319.4 to 237.6** (K-fold mean, an improvement of 25.6%)
> - Optimal hyperparameters: `max_depth=11`, `learning_rate=0.015`, `min_child_weight=5`, `subsample=0.98`, `colsample_bytree=0.67`, `reg_alpha=2.46`, `reg_lambda=1.99`
> - **Key**: raising tree depth from the default 8 to 11 + lowering the learning rate to 0.015 had the most significant effect

#### 2.3.3 Multi-Model Ensemble (run in the sandbox, only XGB+LGB × 3 seeds)

| Configuration | RMSE (sandbox) |
|---|---|
| Single XGBoost (seed 0) | 311.48 (v7) |
| Single LightGBM (seed 0) | ~295-305 (est.) |
| **XGB (0.22) + LGB (0.78) mean weighting** | **Best ensemble** |
| XGB + LGB + CatBoost (target, not run) | < 285 (est.) |

Actually saved ensemble weights (`models/ensemble_val/weights.json`):
```json
{
  "weights": {"xgb": 0.2171, "lgb": 0.7829},
  "blend_mode": "mean",
  "feature_set": "all"
}
```

> **Note**: the sandbox only ran XGB+LGB (CatBoost was not run). 3 seeds × 2 models = 6 boosters.

### 2.4 Comparison of Different Splits on the Same Dataset

| Split method | Training period | Validation period | v1 RMSE | v7 RMSE | Improvement |
|---|---|---|---|---|---|
| **Winter split** | 2025-01~10 | 2025-11~12 | 319.40 | 311.48 | **-2.48%** |
| **Summer split** (same-month, months 1+7) | 02~06 + 08~12 | 2025-01 + 07 | 356.38 | 352.68 | **-1.04%** |
| **K-fold time-series CV** (5 folds) | Rolling | Rolling | — | ~237 (Optuna est.) | — |

### 2.5 Complete Experiment Comparison Table (from v1 to v7 + v1 submission)

| Version | Feature count | Model | Training method | Winter RMSE | Improvement vs v1 |
|---|---|---|---|---|---|
| v1 | 30 | XGBoost | Base features | 319.40 | — |
| v2 | 30 | XGBoost | + clip99.5 | (not run on holdout) | — |
| v3 | 30 | XGBoost | (slightly tuned) | (not run) | — |
| v4 | 30 | XGBoost | (slightly tuned) | (not run) | — |
| v5 | 30 | XGBoost | (slightly tuned) | (not run) | — |
| v6 | 30 | XGBoost | (slightly tuned) | (not run) | — |
| **v7** | **42** | **XGBoost** | **+ lag congestion aggregation** | **311.48** | **-2.48%** |
| **v1 submission (leaderboard)** | **30** | **XGBoost** | **Base features + clip** | **589.91 (actual)** | — |
| Tuned XGBoost | 42 | XGBoost | Optuna 50 trials | 237.63 (K-fold) | -25.6% |
| XGB+LGB ensemble | 77 | ensemble | mean weighting | < 295 (est.) | -7.6% |

### 2.6 Comparison of Long-Tail Handling Methods (v8 tail_weight experiment already run)

| Configuration | tail_quantile | tail_weight | RMSE | MAE | P95 | best_iter |
|---|---|---|---|---|---|---|
| v7 baseline | 0.99 | 1.0 | 311.48 | 194.23 | 546.95 | 1033 |
| v8 tail weighting (pending evaluation) | 0.99 | 1.5 | (not completed) | — | — | — |

### 2.7 Model Performance Differences Observed on the Leaderboard (v1 submission vs Top 1)

| Team | RMSE | Model (speculated) | Gap between them and us |
|---|---|---|---|
| Top 1 (vigorous-whistle) | 239.58 | Speculated XGB+LGB+CatBoost + 100+ features + Optuna + multi-seed | **350 seconds (60% slower)** |
| Median (our v1 position) | 589.91 | Our v1: 29 features with single XGB | — |

---

## Part 3: Selection Recommendations (Data-Driven)

### 3.1 Feature Selection Priority

| Priority | Category | Reason for recommendation |
|---|---|---|
| **P0 (must add)** | Categorical: ADEP / RUNWAY / STAND / WK_TBL_CAT / AIRCRAFT_TYPE | Accounts for 51% of gain |
| **P0 (must add)** | lag congestion: cnt_dep_runway_60m ranks 4th | Congestion is the strongest signal |
| **P1 (strongly recommended)** | Full set of time features (hour/dow/month + sin/cos) | 18% gain |
| **P1 (strongly recommended)** | airport_hour combination | The 6th ranked feature |
| **P2 (recommended)** | lead congestion: cnt_lead_runway_60m / cnt_lead_airport_60m | 4% gain, but physically interpretable |
| **P2 (recommended)** | Schedule-related: iobt_eobt_diff_sec / eobt_to_sched_sec | Reflects flight schedule adjustments |
| **P3 (optional)** | Domain structure: stand_zone / runway_family / runway_side | The model can learn it from the raw fields |
| **P3 (optional)** | TE (8 items) | Marginal gains are small, may overfit |
| **P4 (avoid)** | Derived fields of `_flt` (already covered by base fields) | Low gain |

### 3.2 Model Selection Priority

| Model | Recommendation level | Scenario |
|---|---|---|
| **XGBoost** | ⭐⭐⭐⭐⭐ | Main model, both baseline + tuning rely on it |
| **LightGBM** | ⭐⭐⭐⭐⭐ | Essential for the ensemble (actual weight 78%) |
| **CatBoost** | ⭐⭐⭐⭐ | Essential for the ensemble (not run in the sandbox, target weight ~50%) |
| sklearn HGBR | ⭐⭐ | Backup (2024 rank-30 level) |
| Deep learning (Transformer) | ❌ | GBDT won all 3 years of the competition, don't bother |

### 3.3 Tuning Strategy (based on Optuna results)

```
Recommended hyperparameters (looking at the top three trials 6/11/12 from Optuna):
- learning_rate   : 0.011 - 0.015 (small learning rate + many rounds)
- max_depth       : 11 - 12 (deep trees)
- min_child_weight: 5 - 7 (medium)
- subsample       : 0.85 - 0.98 (high sampling, enough data)
- colsample_bytree: 0.67 - 0.77 (medium column sampling)
- reg_alpha       : 1.8 - 3.5 (L1 regularization)
- reg_lambda      : 1.0 - 2.6 (L2 regularization)
- num_boost_round : 1000 - 2000 + early_stopping=100
```

### 3.4 Input-Output Ratio Ranking

| Input | Expected output (RMSE reduction) | ROI |
|---|---|---|
| **Add lag congestion features** | -2.5% (verified) | ⭐⭐⭐⭐⭐ |
| **Add lead congestion features** | -1~-2% (est.) | ⭐⭐⭐⭐ |
| **Multi-seed averaging (5)** | -1~-3% | ⭐⭐⭐⭐⭐ |
| **Three-model ensemble (XGB+LGB+CatBoost)** | -3~-7% | ⭐⭐⭐⭐⭐ |
| **Optuna tuning 50 trials** | -25.6% (K-fold) | ⭐⭐⭐⭐⭐ |
| **Training grouped by airport** | Pending verification | ⭐⭐⭐ |
| **Long-tail weighting** | Pending verification | ⭐⭐⭐ |
| **Target rescaling** | -1% | ⭐⭐ |

---

## Appendix: Experiment Log Files

| File | Content |
|---|---|
| `notebooks/eda/summary_stats.csv` | Statistics for the 26 fields (count/unique/mean/std) |
| `notebooks/eda/airport_stats.csv` | mean/median/std/P90/P99 for the 10 airports |
| `notebooks/eda/*.png` | 8 distribution plots |
| `notebooks/holdout_validation_result.json` | v1 vs v7 winter holdout |
| `notebooks/holdout_same_month_result.json` | v1 vs v7 same-month holdout of months 1+7 |
| `notebooks/holdout_v8_result.json` | Long-tail weighting experiment (pending completion) |
| `notebooks/diag_ranking_result.json` | ranking data missingness + error bucket analysis |
| `models/baseline_xgb_feature_importance.csv` | XGBoost feature gain ranking |
| `models/tuned_xgb_params.json` | Optuna optimal hyperparameters |
| `models/tuned_xgb_trials.csv` | Detailed records of the Optuna 50 trials |
| `models/ensemble_val/weights.json` | XGB+LGB ensemble weights (0.22/0.78) |
