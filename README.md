# PRC Data Challenge 2026 — Taxi-Out Time Prediction

> **Team**: `jubilant-puzzle`
> **Purpose**: Competition entry. **Code + sample data + reproduction scripts.** Full dataset stays local or is pulled from the official MinIO bucket.
> **Challenge**: [PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/) — Taxi-Out Time Prediction
> **Metric**: RMSE (in seconds)

**Final submission**: `jubilant-puzzle_final.parquet` (670,790 rows) — official receipt: `Succeeded`

| Basis | Value | Note |
|---|---:|---|
| Holdout RMSE (4 months) | **324.76** | Internal validation on 2025-01/02/06/07; **−34.1%** vs the 493.05 baseline |
| Best leaderboard score | **360.86** | v22, scored on `truthing.parquet` (immediate feedback) |
| Final official score | pending | The `final` slot is scored on `final_submitting.parquet`, evaluated after the deadline |

> Holdout and leaderboard numbers use different scopes and are **not** directly comparable (v22: holdout 342.40 ↔ leaderboard 360.86).
> Method and full experiment log: `docs/method.md`. Per-version score ledger: `docs/LEADERBOARD_LEDGER.md`.

---

## 1. Repository layout

```
prc_data_challenge/
├── README.md                 This file (entry point)
├── requirements.txt          Python dependencies (key versions pinned)
├── .env.example              Credential template (contains no real secrets)
├── .gitignore                Excludes large files / credentials / logs
│
├── src/                      ★ Shared feature modules (importable right after clone)
│   ├── data_loader.py            Monthly parquet reader + DEP filter + time parsing
│   ├── features.py               30 base features + target rescaling
│   ├── features_aggregate.py     18 lagged congestion aggregates
│   ├── features_laglead.py       18 lead-time future congestion aggregates
│   ├── features_domain.py        3 structural + 8 target-encoded features
│   ├── features_queue.py         Congestion-queue features
│   ├── validation.py             Time-series split + RMSE/MAE/P95
│   ├── train_xgb.py / train_lgb.py / train_ensemble.py
│   ├── predict_xgb.py / predict_ensemble.py
│   └── tune.py                   Optuna hyperparameter search
│
├── scripts/                  ★ One-shot scripts
│   ├── eda.py                    EDA + 8 charts
│   ├── build_sample.py           ★ Build data/sample/ from the full dataset
│   ├── download_data.py          ★ Pull the full dataset from OSN MinIO (needs .env)
│   ├── upload_submission.py      Upload a submission (local use)
│   ├── run_robust.py             Auto-restart for long runs (local use)
│   ├── setup_mac.sh              macOS environment setup (local use)
│   └── gate_check.py             One-shot D0 gate (usable in the cloud)
│
├── data/
│   ├── raw/                    ⚠️ Full parquet data (not in repo, must download)
│   ├── sample/                 ★ Sample data (in repo, ~2 MB, works out of the box)
│   │   ├── training_2025-{01..12}.parquet   12 months, ~70 rows/airport/month (40 DEP + 30 ARR)
│   │   ├── ranking.parquet                   5,000 rows
│   │   ├── submitting.parquet                5,000 rows
│   │   └── sample_meta.json                  Sampling parameters
│   └── external/               Third-party data (derived CSV/JSON from apt.dat are in repo, .dat excluded)
│
├── models/                    Model artifacts (weights excluded, feature lists included)
│   ├── feature_lists/          ★ Feature column names + categorical maps (small, must keep)
│   ├── ensemble_weights.json
│   ├── *.json / *.txt / *.cbm  ⚠️ Model weights (~1.4 GB, excluded, retrain to reproduce)
│
├── docs/                      Design docs / reports
│   ├── method.md                ★ Final method write-up (including negative results)
│   ├── LEADERBOARD_LEDGER.md    ★ Per-version leaderboard ledger
│   └── ...                      Historical design and diagnostic reports
│
├── supplementary/           ★ Supplementary artifacts (not part of the scored file)
│   └── final_uncertainty.parquet   σ + conformal intervals for the final 4 months (8 columns)
│
├── notebooks/                 One-off diagnostic scripts
│   ├── diag/                  Diagnostic history (many contain absolute local paths)
│   ├── eda/                   EDA charts (PNG/CSV)
│   └── holdout/               Holdout validation
│
└── submissions/               Submission files (.gitkeep placeholder; *.parquet excluded)
```

★ = usable immediately after cloning

---

## 2. Quick start

### 2.1 Clone and install

```bash
git clone https://github.com/huangfeng1995/PRC-Data-Challenge-2026.git
cd PRC-Data-Challenge-2026
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# macOS: LightGBM needs libomp
brew install libomp
```

### 2.2 Run on the sample data (default, recommended)

`data/sample/` is **already included** (~2 MB), so this works immediately:

```bash
# Verify the sample data + freeze entry point + feature pipeline
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data_loader import load_month, load_ranking, filter_departures, parse_time_columns
from src.features import build_features
df = load_month('data/sample/training_2025-01-01_2025-02-01.parquet')
df = filter_departures(parse_time_columns(df, cols=['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','IOBT_flt','EOBT_1_flt','AOBT_3_flt','ARVT_3_flt','LOBT_flt']))
print('DEP rows:', len(df))
print('features shape:', build_features(df).shape)
"

# EDA on the sample
python3 scripts/eda.py --data-dir data/sample --output-dir notebooks/eda

# Small baseline run (result within minutes)
python3 -m src.train_xgb --data-dir data/sample --feature-set basic
```

### 2.3 Pull the full dataset (only needed off-local, not on this machine)

```bash
# 1. Copy .env to the project root and fill in the real credentials
cp .env.example .env
# Edit .env and set OSN_ACCESS_KEY / OSN_SECRET_KEY

# 2. Pull the full dataset (~330 MB)
python3 scripts/download_data.py --bucket prc-2026-datasets --dest data/raw

# 3. Full EDA / training (memory hungry; prefer a local machine or a rented GPU)
python3 -m src.train_ensemble --data-dir data/raw --feature-set all --models xgb,lgb,cat --n-seeds 5
```

---

## 3. Security red lines

⚠️ **The following must never be pushed to git**:

- `.env` (OSN credentials) — already in `.gitignore`
- `data/raw/` (full competition data, 330 MB) — already in `.gitignore`
- `models/*.json` / `*.txt` / `*.cbm` / `*.bin` (model weights, ~1.4 GB) — already in `.gitignore`
- `submissions/*.parquet` (competition submission files) — already in `.gitignore`
- Any API token / private key / password

**Pre-commit checklist**:

```bash
# 1. Confirm .env was not added
git status | grep -E '\.env$|OSN_'
# Should print nothing

# 2. Confirm .gitignore covers every large file
git check-ignore data/raw/*.parquet models/*.json submissions/*.parquet
# Should print the filename for each (ignored)
```

---

## 4. Data notes

| Path | Source | In repo? | Size |
|---|---|---|---|
| `data/raw/*.parquet` | OSN MinIO competition data | ❌ Download required | ~330 MB |
| `data/sample/*.parquet` | Locally sampled from the full set (random_state=42) | ✅ Included | ~2 MB |
| `data/external/xplane/*.csv` | X-Plane apt.dat geometry | ✅ Included | ~370 KB |
| `data/external/xplane/*.dat` | Raw X-Plane apt.dat | ❌ Excluded | ~10 MB, re-downloadable |
| `data/external/xplane/stand_rwy_distance.csv` | Full DEP stand→runway distances | ❌ Excluded | 246 MB, regenerate for full runs |
| `models/*.json` | XGBoost / LGB / CatBoost models | ❌ Excluded | ~1.4 GB |

**How to get the full data**: requires OSN MinIO access/secret keys from the official bucket. See `docs/MINIO_DOWNLOAD_GUIDE.md`.

The submitted prediction file itself (`jubilant-puzzle_final.parquet`) is **not** in the repo — it was uploaded directly to the OSN bucket and the submission slot returned `Succeeded`. Regenerate it with `notebooks/diag/d0/final_predict_c_full.py` followed by the Gate checks in `gate_noclip.py`.

---

## 5. Reproduction scripts

| Script | Purpose |
|---|---|
| `python3 scripts/build_sample.py` | Build `data/sample/` from `data/raw/` (needs full data) |
| `python3 scripts/download_data.py` | Pull the full dataset from OSN MinIO (needs `.env`) |
| `python3 scripts/eda.py` | EDA (single XGBoost visualisation) |
| `python3 -m src.train_xgb` | Train a single XGBoost model |
| `python3 -m src.train_ensemble` | XGB + LGB + CatBoost blend, multi-seed |
| `python3 -m src.tune` | Optuna hyperparameter search (50 trials) |
| `python3 -m src.predict_ensemble` | Inference + submission generation |
| `python3 scripts/gate_check.py` | One-shot D0 gate (5 PASS/FAIL checks) |

---

## 6. Final status

| Stage | Status | Note |
|---|---|---|
| Data ingestion | ✅ Full dataset downloaded | OSN credentials in `.env` (not in repo) |
| 77-feature baseline | ✅ | XGB + LGB + CatBoost blend |
| Leaderboard iteration | ✅ | v1 589.91 → v22 **360.86** (best under immediate-scoring) |
| LIRF label-corruption forensics | ✅ | Found `BLOCK` overwritten by `SCHED`; repaired 1484 → 655 |
| Final submission | ✅ Uploaded | noclip E2, holdout **324.76**, official receipt `Succeeded` |
| Uncertainty quantification | ✅ | Physics-constrained second-moment mixture + damage-aware stratified conformal, see `supplementary/` |

**Core finding**: essentially the entire 493.05 → 324.76 gain comes from **detecting and repairing corrupted LIRF labels**, not from new modelling. The remaining non-LIRF tail rows (>1800 s extreme taxi) are real physical phenomena (congestion, de-icing, weather) and were shown to be incompressible without external data.

See `docs/method.md` (method and negative results) and `docs/LEADERBOARD_LEDGER.md` (per-version scores).

---

## 7. Key modelling decisions

**Label corruption on LIRF.** `BLOCK_TIME_UTC_mvt` is overwritten by `SCHED_TIME_UTC_mvt` on damaged LIRF rows, which makes `TAXITIME = MVT − SCHED` an artefact rather than ground truth. Damaged rows have their entire schedule missing (`ADEP_flt` / `BLOCK` / `EOBT_1` / `IOBT` / `AOBT_3` / `LOBT`), verified with 100% consistency across 2025 and 2026. Repairing these rows is the single largest source of improvement.

**No clipping on repaired rows.** On the leaderboard, v22's repaired rows were predicted as exactly `MVT − SCHED` with values up to 111,654 s, and the official scorer accepted them. Ground truth on these rows genuinely reaches 131,167 s. Clipping to 10,800 s inflates their RMSE from 7,918 to 33,313 (4.2×).

**LOBT window projection.** Where `LOBT_flt` is present, predictions are projected into `[MVT − LOBT ± 3606]` s. The 3606 s threshold is a hard property of the data source (constant across December for all 10 airports), not a tuned hyperparameter.

**Uncertainty, not point prediction.** A probabilistic mixture over the two components (corrupted vs normal) produces a usable predictive distribution — σ converges by 10 orders of magnitude, σ-vs-|residual| Spearman 0.4354 / 0.3966, and conformal coverage deviates by ≤ 0.0023 from nominal. The same mixture is **not** usable as a point predictor (RMSE 788.01, worse on 10/10 airports), so uncertainty is reported separately in `supplementary/` and the point prediction stays with the GBDT baseline.

**What was tried and rejected.** Eight families of long-tail methods (EVT/GPD, quantile objectives, log target, Huber, winsorisation, weighted loss, quantile compression, extreme-row rescue) were all falsified on the non-LIRF tail. A quantile objective (P50) was tested on the real leaderboard and came back **+31.72 s worse** than L2, with an ablation showing the objective function itself is the cause — RMSE rewards the conditional mean, and P50 systematically compresses the upper tail.

---

## 8. License

MIT (as required by the competition rules).