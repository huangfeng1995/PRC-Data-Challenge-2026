# 🎉 Sandbox libomp Breakthrough Succeeded (2026-09-20)

## Problem
The macOS sandbox has no Homebrew, so libomp cannot be installed and XGBoost/LightGBM cannot run.

## Solution (complete steps)

1. **Download the precompiled OpenMP** (maintained by R-project):
   ```bash
   curl -L -o /tmp/openmp.tar.gz \
       https://mac.r-project.org/openmp/openmp-19.1.5-darwin20-Release.tar.gz
   tar -xzf /tmp/openmp.tar.gz -C /tmp/openmp-extract
   ```

2. **Copy it to the project's local directory**:
   ```bash
   mkdir -p libomp/include libomp/lib
   cp /tmp/openmp-extract/usr/local/include/* libomp/include/
   cp /tmp/openmp-extract/usr/local/lib/* libomp/lib/
   ```

3. **Key fix**: the internal rpath of the XGBoost and LightGBM dylibs is hardcoded to `/opt/homebrew/opt/libomp/lib`,
   we need to add our libomp path to it:
   ```bash
   # XGBoost
   XGB_PATH="/Users/openclaw/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/vm/tools/lib/python3.10/site-packages/xgboost/lib/libxgboost.dylib"
   install_name_tool -add_rpath "$PWD/libomp/lib" "$XGB_PATH"

   # LightGBM
   LGB_PATH="/Users/openclaw/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/vm/tools/lib/python3.10/site-packages/lightgbm/lib/lib_lightgbm.dylib"
   install_name_tool -add_rpath "$PWD/libomp/lib" "$LGB_PATH"
   ```

4. **MLflow's file store has been deprecated in the new version**, set an environment variable to bypass it:
   ```bash
   export MLFLOW_ALLOW_FILE_STORE=true
   ```

5. **Fix a bug in _encode_categoricals**: `fillna(-1)` fails on category columns, so convert to str first:
   ```python
   X[col] = X[col].astype(str)  # convert to str first
   X[col] = X[col].map(mapping).fillna(-1).astype("int32")
   ```

## Measured XGBoost baseline results (29 basic features)

```
winter_main RMSE: 318.16 seconds
summer_aux RMSE: 304.08 seconds
K-fold mean:      308.12 ± 16.83
```

Top 5 most important features:
1. WK_TBL_CAT_flt (wake turbulence category)
2. ADEP_mvt (airport)
3. AIRCRAFT_TYPE_mvt (aircraft type)
4. RUNWAY_mvt (runway)
5. STAND_mvt (stand)

→ 4 out of these 5 are spatial information, **confirming that the design direction of the congestion/lead aggregate features is correct**.

## Next Steps

- The full-feature baseline (77 features) is running
- Three-model ensemble (XGB+LGB+CatBoost)
- Inference + generate the submission
- Upload to OSN
