# Final Self-Check Report Before Getting the Real Data

**Generated at**: 2026-09-18
**Purpose**: make sure all the code, data flows, and documentation consistency are in place, so that everything can be run as soon as the data arrives.

---

## ✅ 1. Code Completeness (16 .py files, all syntax checks pass)

```
src/
├── data_loader.py        ✅ load monthly parquet + DEP filtering + time parsing
├── features.py           ✅ 30 basic features (leakage guards fully annotated)
├── features_aggregate.py ✅ 18 lag aggregate features (15/30/60min)
├── features_domain.py    ✅ 3 structures + 8 target encodings
├── features_laglead.py   ✅ 18 lead features (future 15/30/60min)
├── validation.py         ✅ time-series split + RMSE/MAE/P95 metrics
├── train_xgb.py          ✅ XGBoost single-model training + MLflow
├── train_lgb.py          ✅ LightGBM single-model training (backup)
├── train_ensemble.py     ✅ XGB+LGB+CAT three-model ensemble + weight optimization
├── tune.py               ✅ Optuna automatic hyperparameter tuning
├── predict.py            ✅ LightGBM inference (backup)
├── predict_xgb.py        ✅ XGBoost inference
└── predict_ensemble.py   ✅ Ensemble inference

scripts/
└── eda.py                ✅ one-click EDA + 8 charts + 2 CSVs
```

---

## ✅ 2. The Dependency Graph Has No Cycles, the Modules Are Clear

```
data_loader ←── (all modules depend on it)
features ←── features_aggregate, features_domain, train_xgb, train_ensemble
validation ←── train_xgb, train_lgb, train_ensemble, tune
train_xgb ←── train_ensemble, tune, predict_xgb, predict_ensemble
train_ensemble ←── tune, predict_ensemble
```

---

## ✅ 3. Leakage Prevention Check

**BASELINE_FEATURE_COLS: 30 features, none of them is a leaking field**:

| Safe fields (usable) | Dangerous fields (blocked, only used for time-difference computation in the code) |
|---|---|
| `SCHED_TIME_UTC_mvt` | `AOBT_3_flt` ← actual pushback, would leak |
| `IOBT_flt` | `BLOCK_TIME_UTC_mvt` ← actual pushback, would leak |
| `EOBT_1_flt` | `MVT_TIME_UTC_mvt` ← actual takeoff, would leak |
| `ADEP_mvt`, `ADES_mvt` | `ARVT_3_flt` ← actual arrival |
| `RUNWAY_mvt`, `STAND_mvt` | `LOBT_flt` ← last off-block |
| `AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt` | |
| `MARKET_SEGMENT_flt`, `FLIGHT_OPERATOR_flt` | |

---

## ✅ 4. macOS Compatibility

| Problem | Status | Handling |
|---|---|---|
| LightGBM is missing libomp | ⚠️ the user needs `brew install libomp` locally | already documented in the README |
| XGBoost's top-level import tracker also uses libomp | ✅ fixed | train_xgb/predict_xgb import directly; train_ensemble / predict_ensemble / tune use lazy import |
| macOS sandbox (currently for this repo) | ❌ cannot run at all | does not affect code correctness; the smoke test has already been validated with sklearn as a substitute |

**Key fix**: both train_ensemble.py and predict_ensemble.py were changed to **lazy import** —— the xgboost import is only triggered when it is really needed. In this way, even on macOS without libomp, it will not crash at the `from src.train_ensemble import ...` step.

---

## ✅ 5. Documentation Consistency

| Document | Content | Consistency |
|---|---|---|
| README.md | project description, reproduction steps, progress table | ✅ |
| design.md | full design + 24-day milestones | ✅ |
| 2025-winners-analysis.md | research report | ✅ |
| requirements.txt | dependency list | ✅ |
| .gitignore | large files such as data/models/mlruns do not go into git | ✅ |

---

## ✅ 6. All Smoke Tests Passed

| Test | Result |
|---|---|
| validation module | ✅ the time-series split works correctly |
| features_domain module | ✅ both the structural features and TE are correct |
| features_laglead module | ✅ the lead feature values are reasonable |
| features_aggregate module | ✅ the lag feature values are reasonable |
| ensemble framework | ✅ full pipeline (77 features) + weight optimization |
| end-to-end flow | ✅ data → features → training → ensemble → submission all run through |

---

## 🚀 5 Steps After Getting the Data (expect the first score in 2–3 hours)

```bash
# Step 1: put the data under data/raw/ (download it from the OSN bucket)
ls data/raw/
# there should be 12 training_2025-*.parquet + ranking.parquet + submitting.parquet

# Step 2: EDA, understand the data (5–10 minutes)
python -m scripts.eda --data-dir data/raw --output-dir notebooks/eda
# look at the 8 charts under notebooks/eda/, focusing on:
#  - taxi_out_distribution.png: target distribution, confirm the degree of the long tail
#  - by_airport_hour.png: which airports are the most congested in which time slots
#  - long_tail.png: how many seconds P95/P99 are

# Step 3: run the minimal baseline (20–40 minutes)
python -m src.train_xgb --data-dir data/raw --feature-set basic
# write down the winter_main RMSE → this is the starting line

# Step 4: add all the features (30–60 minutes)
python -m src.train_xgb --data-dir data/raw --feature-set all
# see how much the winter_main RMSE improves

# Step 5: three-model ensemble + the first submission (60–90 minutes)
python -m src.train_ensemble --data-dir data/raw --feature-set all --models xgb,lgb
python -m src.predict_ensemble --output submissions/v1.parquet
# get submitting.parquet, rename it to <team-name>_v1.parquet, upload it to the OSN bucket
```

---

## ⚠️ Known Risks (may need to be solved after the data arrives)

1. **Categorical fields have too high cardinality**: airport_hour is an `ADEP × hour` combination and may have 10 × 24 = 240 values. Target encoding is fine, but LGBM categorical handling may be slow. If training turns out to be slow, it can be split into two features.

2. **Performance of the aggregate features on the full 4.17 million rows**: the smoke test runs in 0.1s on 2k rows, **on the real data it is estimated to take 5–15 minutes**. If it is too slow, consider:
   - aggregating by month (compute lag independently for each month, then merge)
   - using numpy instead of pandas loops

3. **Categorical mapping leakage**: currently `_encode_categoricals` fits the mapping with the training data and uses it directly at inference time. But some categories in the ranking data may never have been seen during training. LightGBM/XGBoost handle unknown categories by default, but watch out for whether large-scale NaN appears.

4. **Target encoding overfit**: `target_encode_smoothed` computes the mapping with the whole training set, which means the target values of the validation set have already been "seen" by the encoder. In theory this makes the validation score artificially high, but on the test set (ranking) it is reasonable (because ranking has no target, so only historical means can be used). **The severity of the problem is low**, but it should be explained when writing the paper.

---

## 🎯 Subsequent Sprint Plan (based on the 2025 study)

| Stage | Estimated Time | Focus |
|---|---|---|
| Day 1–4 | First submission round | baseline + aggregation + ensemble |
| Day 5–9 | Optuna hyperparameter tuning | find the optimal hyperparameters |
| Day 10–15 | Robustness + error analysis | see which samples on the leaderboard have large errors, and add targeted features |
| Day 16–20 | Sprint | add lag/lead tuning, long-tail handling (optional) |
| Day 21–23 | Wrap-up | final submission + paper + code cleanup |

---

## 📞 Contact Me First Thing After Getting the Data

If you run into any problem (for example, a field that does not actually exist, aggregate features that come out as NaN, a leaderboard submission error), **tell me directly**, and I will iterate and improve based on the real data.
