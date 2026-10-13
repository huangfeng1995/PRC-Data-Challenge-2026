# Data Ready Summary (2026-09-20)

**What happened today:**
1. OSN application approved, team name confirmed: `jubilant-puzzle`
2. Data download complete (330 MB, 14 parquet files)
3. Finished EDA, discovered the real characteristics of the data
4. Fixed 2 data-driven bugs (filter conditions, endpoint)

---

## 📊 Data Characteristics at a Glance

| Item | Actual Value |
|---|---|
| Total rows | 4,167,797 |
| **DEP rows** | **2,085,047** (50%, as expected) |
| Number of airports | **10** (not 11) |
| Time span | 2025-01-01 ~ 2025-12-31 |
| Number of fields | 26 (after deduplication) |

### Taxi-out distribution by airport

| Airport | Mean | P99 | Data Volume |
|---|---|---|---|
| EGLL (London) | 1364s | 2701s | 240K |
| LIRF (Rome) | 1195s | 4019s (worst long tail) | 160K |
| LTFM (Istanbul) | 1053s | 2587s | 270K (most) |
| LFPG (Paris) | 1020s | 2409s | 240K |
| LEMD (Madrid) | 1015s | 1932s | 210K |
| LEBL (Barcelona) | 957s | 1962s | 180K |
| EDDF (Frankfurt) | 864s | 1819s | 230K |
| EDDM (Munich) | 812s | 1921s | 170K |
| EHAM (Amsterdam) | 785s | 1775s | 250K |
| LSZH (Zurich) | **741s (easiest)** | 1707s | 130K |

**Key insights**:
- **EGLL has the highest mean (22.7 minutes)** —— Heathrow is one of the most congested airports in Europe, and the hardest to predict
- **LIRF has the worst long tail** (P99 is 3.4 times the mean) —— Rome exhibits a "long queue buildup" phenomenon
- The differences between airports are huge (mean 741 ~ 1364s, nearly 2 times apart) → **the "train grouped by airport" approach mentioned in v2 should add further gains**

---

## 🐛 Data-Driven Bug Fixes

| Bug | Cause | Fix |
|---|---|---|
| `TAXITIME_SEC_mvt < 0` (minimum -12s) | data bug | filter changed to `> 0 and < 10800` |
| `TAXITIME_SEC_mvt` maximum 131167s (36 hours) | data bug / long tail | filtered out as well |
| NaN in the EDA log histogram | works fine with log1p after clip | fixed |
| `datacomp.opensky-network.org` endpoint 404 | it is actually `s3.opensky-network.org` | changed the default endpoint |

---

## 🆕 New Findings on the Real Data

| Finding | Impact |
|---|---|
| There are actually **10 airports** (not 11) | the progress table / docs may need to be updated; it affects the robustness of the rank/ranking loading logic |
| `AIRCRAFT_OPERATOR_flt` missing rate 1.08% | our TE handling already falls back to global_mean, no impact |
| The same flight ID sometimes has multiple movements | chronological order must be considered during training; data loading is already sorted by month |
| TAXITIME_SEC_mvt has negative values and extreme values | training filtering has been strengthened |

---

## 🚀 Commands You Can Run Immediately Once Home

Environment assumption: once you are back home on your own Mac, the data has already been downloaded (330 MB in `~/Documents/PRC_data_challenge/data/raw/`).

```bash
cd ~/Documents/PRC_data_challenge

# 1. Minimal baseline (basic features, XGBoost) —— see the starting RMSE
python -m src.train_xgb --data-dir data/raw --feature-set basic

# 2. Full-feature baseline (basic + congestion + lead + domain + TE, XGBoost)
python -m src.train_xgb --data-dir data/raw --feature-set all

# 3. Three-model ensemble (XGB + LGB + CatBoost, 5 seeds)
python -m src.train_ensemble --data-dir data/raw --feature-set all \
    --models xgb,lgb,cat --n-seeds 5

# 4. Hyperparameter tuning (25% of the data, about 30 minutes)
python -m src.tune --data-dir data/raw --feature-set all --n-trials 50

# 5. Inference + generate the submission
python -m src.predict_ensemble --output submissions/jubilant-puzzle_v1.parquet

# 6. Upload to OSN
mc cp submissions/jubilant-puzzle_v1.parquet \
    osn/prc-2026-jubilant-puzzle/
```

---

## 📋 Mac Environment Requirements

Our code depends on: `xgboost`, `lightgbm`, `catboost`, `scipy`, `mlflow`, `optuna`, `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `minio`

**libomp is needed on macOS** (otherwise XGBoost/LightGBM will crash):
```bash
brew install libomp
```

After it is installed, run the baseline.

---

## 📦 Bucket Information (for mc)

| Item | Value |
|---|---|
| Endpoint | `https://s3.opensky-network.org` |
| **Data bucket** | `prc-2026-datasets` (shared by all teams) |
| **Your submission bucket** | `prc-2026-jubilant-puzzle` |

**Configure mc**:
```bash
mc alias set osn https://s3.opensky-network.org <ACCESS_KEY> <SECRET_KEY>
```

---

## 🎯 Execution Path for the Next 21 Days

| Day | Task | Expected RMSE |
|---|---|---|
| 4 | Run the baseline (basic → all) | 250-300 |
| 5 | Run the three-model 5-seed ensemble | 200-250 |
| 6 | **First submission jubilant-puzzle_v1** | enter the leaderboard |
| 7-8 | Optuna hyperparameter tuning | 180-220 |
| 11-12 | Review error analysis, add lag/lead optimizations | 150-180 |
| 13-15 | **Second/third submission** | push for top 10 |
| 16-22 | Robustness + paper | continuous optimization |

---

## ✅ Project Status

- ✅ Data download complete (330 MB, 14 files)
- ✅ EDA complete (8 charts + 2 CSVs in `notebooks/eda/`)
- ✅ Fixed 2 data-driven bugs (filter conditions, endpoint)
- ✅ 6 high/medium-priority bug fixes (TE mapping, final best_iter, Optuna subsample, clip_max, median, try/except, blend_mode)
- ✅ All 14 Python files pass the syntax check
- ✅ The team name is unified as `jubilant-puzzle` across all docs and code
- ⏳ Waiting for you to go home and run the baseline on your Mac

**The first thing to do once home**: run the baseline and look at the RMSE number.
