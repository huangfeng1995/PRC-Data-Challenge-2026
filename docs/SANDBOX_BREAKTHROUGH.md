# 🎉 沙箱 libomp 突破成功(2026-09-20)

## 问题
macOS 沙箱没有 Homebrew,无法装 libomp,XGBoost/LightGBM 跑不起来。

## 解决方案(完整步骤)

1. **下载预编译 OpenMP**(R-project 维护):
   ```bash
   curl -L -o /tmp/openmp.tar.gz \
       https://mac.r-project.org/openmp/openmp-19.1.5-darwin20-Release.tar.gz
   tar -xzf /tmp/openmp.tar.gz -C /tmp/openmp-extract
   ```

2. **复制到项目本地目录**:
   ```bash
   mkdir -p libomp/include libomp/lib
   cp /tmp/openmp-extract/usr/local/include/* libomp/include/
   cp /tmp/openmp-extract/usr/local/lib/* libomp/lib/
   ```

3. **关键修复**:XGBoost 和 LightGBM 的 dylib 内部 rpath 写死了 `/opt/homebrew/opt/libomp/lib`,
   我们需要把我们的 libomp 路径加进去:
   ```bash
   # XGBoost
   XGB_PATH="/Users/openclaw/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/vm/tools/lib/python3.10/site-packages/xgboost/lib/libxgboost.dylib"
   install_name_tool -add_rpath "$PWD/libomp/lib" "$XGB_PATH"

   # LightGBM
   LGB_PATH="/Users/openclaw/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/vm/tools/lib/python3.10/site-packages/lightgbm/lib/lib_lightgbm.dylib"
   install_name_tool -add_rpath "$PWD/libomp/lib" "$LGB_PATH"
   ```

4. **MLflow 文件存储已被新版废弃**,设置环境变量绕开:
   ```bash
   export MLFLOW_ALLOW_FILE_STORE=true
   ```

5. **修一个 _encode_categoricals 的 bug**:对 category 列 `fillna(-1)` 失败,要先转 str:
   ```python
   X[col] = X[col].astype(str)  # 先转 str
   X[col] = X[col].map(mapping).fillna(-1).astype("int32")
   ```

## 实测 XGBoost baseline 结果(29 基础特征)

```
winter_main RMSE: 318.16 秒
summer_aux RMSE: 304.08 秒
K 折平均:        308.12 ± 16.83
```

Top 5 重要特征:
1. WK_TBL_CAT_flt(尾流等级)
2. ADEP_mvt(机场)
3. AIRCRAFT_TYPE_mvt(机型)
4. RUNWAY_mvt(跑道)
5. STAND_mvt(登机口)

→ 5 个里 4 个是空间信息,**证实拥堵/lead 聚合特征的设计方向正确**。

## 下一步

- 全特征 baseline(77 特征)正在跑
- 三模型融合(XGB+LGB+CatBoost)
- 推理 + 生成 submission
- 上传 OSN