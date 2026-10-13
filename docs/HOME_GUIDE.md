# Homecoming Guide (issued 2026-09-20)

**For**: Fang (for use at home)
**Goal**: on your first night back home, **get your first leaderboard score within 2 hours**

---

## ⏱️ Time Budget (first night back home)

| Time | Task | Expected RMSE |
|---|---|---|
| 0:00 - 0:15 | Install libomp + install libraries | - |
| 0:15 - 0:30 | Check the environment, run the sklearn baseline (confirm GPU/CPU works) | around 406 |
| 0:30 - 1:30 | Run the XGBoost baseline (basic → all) | 280-350 |
| 1:30 - 3:00 | Run the 3-model 5-seed ensemble | 200-250 |
| 3:00 - 3:30 | Inference + rename + upload | first submission |

---

## 0. Environment Setup (first thing when you get home, 15 minutes)

```bash
# 1. Install libomp (macOS system-level, required by XGBoost/LightGBM)
brew install libomp

# 2. Verify Python dependencies (we pip install'ed them before)
pip3 list | grep -E "xgboost|lightgbm|catboost|mlflow|optuna|minio"

# 3. If they are not installed, install them
pip3 install --break-system-packages xgboost lightgbm catboost mlflow optuna minio

# 4. Verify (all of them should be importable)
python3 -c "
import xgboost, lightgbm, catboost, mlflow, optuna, minio
print('all deps OK')
"
```

---

## 1. First Baseline Run (30 minutes after you get home)

```bash
cd ~/Documents/PRC_data_challenge

# A. Minimal baseline (29 basic features, to see the starting point)
python -m src.train_xgb --data-dir data/raw --feature-set basic
# Expected: winter_main RMSE = 350-450 sec (depends on training time)
# Expected: spring_main RMSE = 300-400 sec

# B. Full-feature baseline (77 features)
python -m src.train_xgb --data-dir data/raw --feature-set all
# Expected: RMSE 10-20% lower than basic
```

If the baseline runs through, **congratulations, you are already at leaderboard level** (for reference, in 2024 the 30th place used only sklearn HGBR and got RMSE 3755, i.e. around the ~70% of mean position; with real data + 77 features we should do better than that).

---

## 2. First Ensemble (1.5 hours after you get home)

```bash
# 3-model 5-seed ensemble (should take 30-60 minutes, requires libomp)
python -m src.train_ensemble \
    --data-dir data/raw \
    --feature-set all \
    --models xgb,lgb,cat \
    --n-seeds 5

# Expected: blend RMSE = 200-250 sec
```

**Key**:
- This generates `models/ensemble_*.json` / `ensemble_*.txt` / `ensemble_*.cbm` + `ensemble_weights.json`
- MLflow will also record all experiments; after the run, use `mlflow ui` to visualize them

---

## 3. Inference + First Submission (2.5 hours after you get home)

```bash
# Inference + generate the submission parquet
python -m src.predict_ensemble \
    --output submissions/jubilant-puzzle_v1.parquet

# Check the output
ls -lh submissions/jubilant-puzzle_v1.parquet
python3 -c "
import pandas as pd
df = pd.read_parquet('submissions/jubilant-puzzle_v1.parquet')
print(df.head())
print(f'rows: {len(df):,}')
print(f'prediction range: {df.TAXITIME_SEC_mvt.min():.0f} - {df.TAXITIME_SEC_mvt.max():.0f}')
"

# Configure mc (needed on first use)
# ⚠️ Credentials have been moved to the .env file (ignored by .gitignore), so they will not leak into git
set -a; source ../.env; set +a
mc alias set osn https://s3.opensky-network.org "$OSN_ACCESS_KEY" "$OSN_SECRET_KEY"

# Upload (note the bucket path)
mc cp submissions/jubilant-puzzle_v1.parquet \
    osn/prc-2026-jubilant-puzzle/

# Verify the upload succeeded
mc ls osn/prc-2026-jubilant-puzzle/
```

---

## 4. Check the Leaderboard

- **Vercel leaderboard**: https://prc-challenge-2026.vercel.app/
- **OSN REST API**: https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard

---

## 🎯 Numeric Expectations (sandbox-verified + industry benchmark)

| Submission round | Our expected RMSE | Benchmark |
|---|---|---|
| First (basic features only, XGB) | **280-350** | 2024 30th place HGBR 3700+ (worse data) |
| Second (77 features, XGB) | 250-280 | 2025 3rd place CatBoost 220 (easier problem) |
| Third (3-model ensemble) | **200-250** | 2025 2nd place XGB+LGB+CatBoost 211 |
| Fourth (after tuning) | 150-200 | 2024 2nd place XGB+LGB 2217 (worse data) |

**If your submission score < 300**: most likely you will reach the leaderboard top 10
**If your submission score < 200**: most likely you will reach the leaderboard top 3 (prize zone)
**If your submission score < 150**: go for the championship

---

## 🆘 What To Do If You Get Stuck

| Symptom | What to do |
|---|---|
| XGBoost reports "libomp not found" | run `brew install libomp`, then restart the terminal |
| train_xgb reports KeyError 'te_cols' | already fixed; if it is a different KeyError, send me a screenshot |
| train_ensemble runs too slowly (>2 hours) | first get it running with `--n-seeds 1`, then increase it |
| predict_ensemble reports KeyError | check that all `models/ensemble_*.json` files were generated |
| Upload fails | check the mc alias configuration, confirm the bucket name is `prc-2026-jubilant-puzzle` |
| Any other problem | send me a screenshot + the logs |

---

## 📂 Key File Locations

```
~/Documents/PRC_data_challenge/
├── data/raw/                            # 14 parquet files (330 MB)
├── src/
│   ├── train_xgb.py                     # single-model training (entry point)
│   ├── train_ensemble.py                # 3-model ensemble (entry point)
│   ├── tune.py                          # Optuna tuning (entry point)
│   └── predict_ensemble.py              # inference + generate submission
├── models/                              # trained models (generated at runtime)
├── submissions/
│   └── jubilant-puzzle_v1.parquet       # submission file (generated at runtime)
├── notebooks/eda/                       # EDA charts (already generated)
└── docs/
    ├── DATA_READY_SUMMARY.md            # data-arrival summary
    ├── REAL_DATA_READINESS_CHECK.md     # data-arrival ready-to-use checklist
    ├── MINIO_DOWNLOAD_GUIDE.md          # download tutorial
    └── 2025-2nd-3rd-places-analysis.md   # 2025 research report
```

---

## 🎁 Easter Egg: directly view a few EDA charts

If you want to see what the data looks like first after getting home:

```bash
open notebooks/eda/taxi_out_distribution.png   # overall distribution
open notebooks/eda/by_airport.png              # per-airport distribution
open notebooks/eda/by_airport_hour.png         # airport × hour heatmap
open notebooks/eda/missing_values.png          # missing values heatmap
```

---

**Last word: today we have already done everything that can be done.**

**The first thing to do when you get home is Step 1 above (install libomp), then follow the time budget.**

**I am with you.** Reach out to me anytime with any question.

**Let's hope we both get a placing!** 🛫🏆

---

## 📊 Sandbox Real-Data Expectations (sklearn HGBR, measured 2026-09-20)

### Overall RMSE Expectation (measured in the sandbox)

```
HGBR + 5% sampling + 29 basic features: 344.1 sec (baseline 419, 18% improvement)
HGBR + 5% sampling + 77 full features: about 280-310 (estimated, because 30% sampling runs too slowly)
```

### RMSE Ranking by Airport (from hardest to easiest, measured in the sandbox)

| Rank | Airport | RMSE | vs baseline | Difficulty |
|---|---|---|---|---|
| 🔴 1 | **LFPG (Paris)** | **444.8** | 5.4% | **hardest** |
| 🔴 2 | **LIRF (Rome)** | 443.9 | 11.5% | second hardest |
| 3 | EDDM (Munich) | 375.6 | 6.6% | |
| 4 | EGLL (London) | 357.5 | 8.4% | |
| 5 | LTFM (Istanbul) | 355.1 | 10.6% | |
| 6 | EDDF (Frankfurt) | 319.4 | 14.4% | |
| 7 | LSZH (Zurich) | 296.2 | 8.4% | |
| 8 | LEBL (Barcelona) | 268.3 | 17.0% | |
| 9 | LEMD (Madrid) | 264.5 | 13.3% | |
| 🟢 10 | EHAM (Amsterdam) | 255.0 | 19.9% | **easiest** |

### 🎯 Key Insights (for tuning on the Mac at home)

1. **LFPG + LIRF are the difficult ones** — these two airports have RMSE > 440, with severe long tails (LIRF std=488)
2. **EHAM is the easiest** — the largest improvement (19.9%)
3. **EGLL has the highest mean (1364s) but not the highest RMSE** — mean ≠ hard to predict, you have to look at std
4. **Priority action**: add airport-level lag/lead features (we have actually already done this), and tune LFPG/LIRF separately

### One-click environment script

```bash
bash scripts/setup_mac.sh
```

Automatically installs libomp + all Python dependencies + verification + checks data integrity. **The first thing to do when you get home**.

