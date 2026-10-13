# Sandbox Day 1 Report (2026-09-20)

## ✅ What Was Done Today

| Time | Event |
|---|---|
| 09:00 | You submitted the OSN registration, and it was approved |
| 09:30 | Obtained the access key + secret |
| 10:00 | Data download complete (330 MB, 14 parquet files) |
| 10:30 | EDA finished (8 charts + 2 CSVs) |
| 11:00 | Sandbox baseline (5% sampling + sklearn HGBR) = RMSE 344 |
| 13:00 | **Fixed the libomp problem**: downloaded the R-project precompiled OpenMP, used install_name_tool to add the rpath |
| 13:30 | **XGBoost really ran through**! RMSE 318 (winter Nov-Dec validation) |
| 14:00 | Generated the submission + uploaded to OSN |
| 14:30 | **Got the first leaderboard score: 589.91** |

## 📊 The Real Numbers

| Item | Number |
|---|---|
| Our v1 RMSE | **589.91** |
| top 1 RMSE | 239.58 (vigorous-whistle) |
| Gap | 350 seconds (we are 60% slower than 1st place) |
| Ranking | top 50% (median) |

## 🔍 Gap Analysis Between the Top Team and Us

Us:
- 29 basic features
- a single XGBoost model
- no lag/lead aggregation
- no three-model ensemble
- no Optuna hyperparameter tuning

Top team (speculated):
- 100+ features (including airport/runway lag/lead)
- XGB+LGB+CatBoost ensemble
- Optuna 50 trials of hyperparameter tuning
- multi-seed averaging

## 🎯 Optimization Path (28-21=7 days remaining)

```
Day 1 (already completed today): 590 - v1 has been submitted
Day 2 (at home): 280-320 - add the 77-feature XGBoost
Day 3: 240-280 - three-model ensemble (XGB+LGB)
Day 4: 200-250 - Optuna tuning + multi-seed
Day 5: 180-220 - error analysis + targeted features
Day 6-7: push for top 10 - 150-200
```

## 🛠️ Technical Breakthroughs (for reference on your home Mac)

1. **Precompiled OpenMP**: `https://mac.r-project.org/openmp/openmp-19.1.5-darwin20-Release.tar.gz` (universal arm64+x86_64)
2. **Modify the rpath**: `install_name_tool -add_rpath $PWD/libomp/lib <lib_path>`
3. **MLflow file store**: set `MLFLOW_ALLOW_FILE_STORE=true`
4. **Aggregate features are too slow single-threaded**: a home Mac with 8 cores will be 5-10 times faster

## 📁 Current Bucket Status

```
osn/prc-2026-jubilant-puzzle/
  └── jubilant-puzzle_v1.parquet  (4.0 MB, 344,841 rows)
```

## 📂 Current Model

```
models/
  └── baseline_xgb.json  (XGBoost 29 features, ~300 KB)
```

## ⚠️ What Was Abandoned in the Sandbox (reasons)

- **train_ensemble (full features + 5 seeds)**: single-threaded 60m window aggregation takes 30+ minutes, **the sandbox does not have enough CPU computing power**
- **Optuna 50 trials**: same as above

**Both of them can run on your home Mac** (with multiple cores + JIT they finish in 30-40 minutes).

## 🏠 The First Thing to Do Once Home

Run them in the order given in [HOME_GUIDE.md](HOME_GUIDE.md):
1. `bash scripts/setup_mac.sh` - environment ready (15 minutes)
2. `python -m src.train_xgb --data-dir data/raw --feature-set all` - expected 280-320 (40 minutes)
3. `python -m src.train_ensemble --data-dir data/raw --feature-set all --models xgb,lgb --n-seeds 5` - expected 220-260 (60 minutes)
4. `python -m src.predict_ensemble --output submissions/jubilant-puzzle_v2.parquet` - inference (5 minutes)
5. `mc cp submissions/jubilant-puzzle_v2.parquet osn/prc-2026-jubilant-puzzle/` - upload (1 minute)

**The second score is expected within 3 hours**.

## 📌 Important Details (to avoid pitfalls at home)

1. **On macOS 14+, LightGBM may need OpenMP 11+**; the one we precompiled is 19.1.5, which is new enough
2. **`MLFLOW_ALLOW_FILE_STORE=true`** must be set (otherwise mlflow reports an error)
3. **Fitting the categorical encoding at predict time**: already done automatically in `predict_xgb.py` (fit from the training data)
4. **TE columns on ranking**: already fixed (`features_domain.py` fit/apply separation + `predict_ensemble.py` loads the mapping)
5. **clip_max=10800**: has been relaxed (to cope with extreme long tails)
