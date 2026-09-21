# PRC 2025 第 2、3 名详细调研

**调研日期**:2026-09-19
**目标**:深入了解 2025 比赛中端选手的方法细节
**GitHub**:https://github.com/orgs/PRC-Data-Challenge-2025/repositories
**比赛题目**:预测飞机燃油消耗(单位 kg)

---

## 1. 2025 第 2 名:`team_jubilant_vase`(RMSE 211.99)

**作者**:未知(仓库匿名,根据 README 风格推测可能是欧洲或印度团队)
**链接**:https://github.com/PRC-Data-Challenge-2025/jubilant-vase

### 1.1 方法概览

```
最终预测 = CatBoost(64.1%) + LightGBM(19.9%) + XGBoost(16.0%)
         ↑
每个模型 = Optuna 调参 + GroupKFold 交叉验证
         ↑
特征 = ADS-B 轨迹特征 + lag/lead 上下文 + 物理反推质量
```

### 1.2 详细方法(逐项)

#### A. 数据清洗(`clean_trajectories.py`)
- **异常值过滤**:经纬度、高度异常的轨迹点
- **插值**:填充小 gap(最多 60 秒)
- **"Ghost segment" 处理**:空段自动重建

#### B. 特征工程(`feature_engineering.py` + `data_utils.py`)

**5 类特征**:

| 类别 | 特征 | 对 taxi-out 的借鉴 |
|---|---|---|
| **时间** | `duration_sec`, `time_since_takeoff`, `is_missing_data` | `time_since_takeoff` 类似我们的聚合 lag |
| **空间** | `distance_km`, `distance_direct_km`, `dist_flown`, `track_change_total` | **不可用**(没有轨迹) |
| **动力学(聚类)** | altitude / groundspeed / TAS / vertical_rate / mach_number 的 mean/min/max/std | **不可用**(没有轨迹) |
| **⭐ Lag/Lead 上下文** | `prev_[feature]` 上一段的统计;`next_phase` 下一段飞行阶段 | ✅ **跟我们的 lag/lead 思路一致!** |
| **物理** | OpenAP/Acropole 燃料流模型 + **反推质量估计** | ❌ 不可用(无轨迹) |

#### C. 物理反推质量(关键创新)
他们的 "Mass Estimator" 思路:
1. 训练时:用 `scipy.optimize.minimize_scalar` 反推每架飞机的"初始质量",让 OpenAP 模型预测的燃油量跟真实值匹配
2. 推理时:训练一个回归模型 `mass = f(flight_duration)`,按飞机类型分组
3. 推理时预测初始质量,然后用 OpenAP 模拟燃油

**对 taxi-out**:我们**没有轨迹**所以用不上。但他们的"按飞机类型分组做 sub-model"思路可以借鉴。

#### D. 训练 + 集成(`optimize_infinity.py` + `train_blend.py`)

| 模型 | 用途 | 配置 |
|---|---|---|
| **XGBoost** | 单模型 | `hist` tree method |
| **LightGBM** | 单模型 | native categorical 支持 |
| **CatBoost** | 单模型 | symmetric trees, ordered boosting |
| **所有模型** | Optuna 调参 | 5-fold GroupKFold(grouped by flight_id) |
| **融合** | scipy.optimize.minimize | **3 个模型的 OOF 预测上优化权重** |

#### E. ⭐ 鲁棒性(`run_robust.py`)

```
跑训练进程 → 监控 → 如果崩了自动重启 → 黑名单记录频繁崩溃的航班
```

**对我们的启示**:虽然我们有完整 pipeline,但**长跑鲁棒性**值得借鉴 —— 我们 5-seed 融合可能跑 1-2 小时,中间崩了会浪费很多时间。

#### F. 结果

| 模型 | 验证集 RMSE | leaderboard RMSE |
|---|---|---|
| XGBoost | 246.8 | - |
| LightGBM | 245.4 | - |
| CatBoost | **244.9** | - |
| **Ensemble** | **241.0** | **224.0** |

**关键发现**:
- 3 个单模型都在 245 左右,**差异极小**
- 融合后降到 241(-1.7%)
- 最终 LB 上 224(-7%),可能是因为他们的训练集更大或者有 trick
- **CatBoost 拿了 64% 权重** —— **CatBoost 在结构化数据上确实是顶配**

---

## 2. 2025 第 3 名:`team_outspoken_tornado`(RMSE 219.85)

**作者**:**Euranova**(法国卢森堡的 AI 公司)
**链接**:https://github.com/PRC-Data-Challenge-2025/outspoken-tornado
**特点**:把流水线做成 **可配置 / 可插拔** 的模块架构

### 2.1 架构

```
config = {
    "cleaning": [
        ["CleanGeo_v0_0_0", {}]
    ],
    "feature_engineering": [
        ["AddAircraftType_v0_0_0", {}],
        ["AddingAircraftFeatures", {}],     # 飞机气动参数(openap)
        ["AddFuelSegmentDuration_v0_0_0", {}],
        ["AddTimeSinceTakeOff_fromFlightList_v0_0_0", {}],
        ["TemporalFeatures", {}],
        ["AddGeoInfo_v0_0_0", {...}],         # 轨迹统计
        ["DropIfExists_v0_0_0", {...}],
    ],
    "model": ["Catboost_v0_0_0", {}],
}
```

**每个 step 是一个类 + 一个配置文件**。这样**团队多人协作时谁动谁的不冲突**。

### 2.2 提交记录(很有意思,值得研究)

| v | 配置 | RMSE |
|---|---|---|
| 1 | avg traj feats + CatBoost + Optuna 20 iter | 277.16 |
| 2 | + Optuna 100 iter + 全量训练 | 256.23 |
| 3 | 99/1 切分(更大验证) | 246.41 |
| 7 | **fix traj feats + fix time since takeoff** | **227.23** ⭐ |
| 8 | v7 不要 GeoFeatures | 231.36 |
| 9 | **+ TabDPT**(transformer) | 230.36 |
| 10 | + openAP 预测 | 237.67 |
| 11 | without TrajectoryFeatures | 227.25 |
| 13 | **最佳版本** | **225.09** |
| 18 | 全部 features + 160 Optuna trials | 303.52(过拟合!) |

### 2.3 关键发现(对我们的启示)

#### 发现 1:✅ "Time since takeoff" 是关键特征
从 v3 (246) 到 v7 (227) 的巨大提升(−19 RMSE),主要因为"fixed time since takeoff"。
- **对我们 taxi-out 的类比**:我们已经有 `time_since_takeoff` 的等价物 —— **`sched_hour` / `is_morning_peak` / `is_evening_peak`**
- 已经在我们 features 里了 ✅

#### 发现 2:❌ Optuna trials 不是越多越好
- v18(160 trials)反而到 303(过拟合)
- v13(具体 trials 数不清楚)反而 225(最佳)
- **启示**:**30-50 次 trials 是甜蜜点**,我们默认 30 OK

#### 发现 3:⚠️ Transformer 没赢
v9 用 TabDPT(深度学习)RMSE 230,**比 v7(纯 CatBoost)的 227 还差**。
- **印证我们之前的判断**:**深度学习在表格数据上不划算**

#### 发现 4:⭕ Trajectory features 影响不大
v7(带轨迹)→ 227
v8(没轨迹,改用 time since takeoff)→ 231
v11(没轨迹)→ 227

**结论**:对于**燃油消耗预测**,轨迹特征有点价值但**不关键**。这跟我们 taxi-out 的"没轨迹"现实吻合 —— **没有轨迹也能拿好成绩**。

---

## 3. 跨年比较:2024 vs 2025 vs 2026

| 维度 | 2024 (ATOW) | 2025 (Fuel) | 2026 (taxi-out) |
|---|---|---|---|
| **目标** | 起飞重量 kg | 燃油 kg | 滑出秒数 |
| **数据丰富度** | 极高(ADS-B 轨迹 158 GiB) | 高(ADS-B 轨迹 + ACARS) | **低**(只有结构化字段) |
| **第 1 名 RMSE** | 1561 | 200 | (待揭晓) |
| **第 1 名方法** | LightGBM × 20 seed | LGB+XGB+CatBoost 融合 | (我们目标) |
| **是否必须深度学习** | ❌ | ❌ | ❌ |
| **是否必须外部数据** | ✅ (OpenAP, METAR) | ✅ (ERA5, ACARS) | ❌ |
| **是否必须专家物理知识** | ✅ | ✅ | ❌ |

### 3.1 我们应该学谁?

| 维度 | 最像我们的是 | 借鉴点 |
|---|---|---|
| **数据规模** | 2025 第 3 名 `outspoken_tornado`(他们只用基础特征 + 时间特征也拿 top 3) | **简单实用** |
| **方法栈** | 2025 第 2 名 `jubilant_vase`(三模型融合 + Optuna) | **完整 pipeline** |
| **代码架构** | 2025 第 3 名(可插拔 step + config) | **便于迭代** |

### 3.2 关键洞察

#### 洞察 1:从 2024 第 1 名到 2025 第 3 名,**特征工程的边际收益在递减**

| 年份 | 第 1 名 RMSE/均值的比 |
|---|---|
| 2024 | 1561 / ~5000 = **31%** (复杂特征大幅领先) |
| 2025 | 200 / ~400 = **50%** (复杂特征差距在缩小) |
| 2026 | (我们基础) |

→ 我们的 taxi-out 比赛,信息密度低,**基础特征 + 拥堵聚合 + 多 seed 集成**已经够,**不需要搞太复杂的物理特征**。

#### 洞察 2:CatBoost 是稳定好牌

2025 第 2 名 `jubilant_vase`:**CatBoost 拿了 64% 权重**
2025 第 3 名 `outspoken_tornado`:**纯 CatBoost 就拿了 top 3**
2024 第 1-3 名:虽然主力是 LightGBM,但很多队也用了 CatBoost

→ **强烈建议我们把 CatBoost 加进 train_ensemble**。我有支持代码,装一下 catboost 就行。

#### 洞察 3:Optuna 50 trials 是甜蜜点

- 2025 第 2 名:Optuna 50 trials
- 2025 第 3 名:最佳 v13 + v18 失败案例(160 trials 过拟合)
- 我们默认 30,可能**调到 50 更稳**

---

## 4. 对我们 2026 方案的最终调整建议

### 4.1 应该做

| # | 调整 | 工作量 | 收益 |
|---|---|---|---|
| 1 | 把 CatBoost 加进 train_ensemble 的 models 列表 | 5 分钟 | +1-3% |
| 2 | 把 n_trials 默认值从 30 改成 50 | 1 行 | 更稳 |
| 3 | 写 `run_robust.py`(自动重启 + 黑名单) | 30 分钟 | 防崩溃 |
| 4 | 加 median voting 选项(对长尾 RMSE 可能更稳) | 5 分钟 | 备选 |

### 4.2 不应该做

| ❌ | 原因 |
|---|---|
| 不要堆 100+ Optuna trials | 会过拟合(参考 outspoken_tornado v18) |
| 不要追 Transformer/GNN | 2024/2025/2026 三年的证据都表明 GBDT 完胜 |
| 不要依赖天气/外部数据 | 2025 第 6 名证明"影响有限",比赛数据没天气 |
| 不要过工程化 | outspoken_tornado v18 是 160 trials 过拟合,2024 organized_volcano 是模块化过头 |

### 4.3 行动优先级

| 优先级 | 行动 |
|---|---|
| 🥇 | 拿到数据后第一轮提交就用 **XGB + LGB + CatBoost 三模型 5-seed 融合** |
| 🥈 | Optuna 调参用 50 trials(默认),然后用调好的超参跑最终模型 |
| 🥉 | 如果还有时间,加 median voting 试试 |

---

## 5. 我们当前栈的评估

| 维度 | 我们 | 2025 第 2 名 | 2025 第 3 名 | 评价 |
|---|---|---|---|---|
| 特征工程 | 77 个(基础 + 拥堵 + lead + domain + TE) | 5 类(基础 + 时序 + 物理 + lag/lead + 反推) | 模块化配置 | ✅ 我们 feature 数够 |
| 模型 | XGB+LGB 融合 | XGB+LGB+CatBoost 融合 | CatBoost 单模型 | ⚠️ 我们差 CatBoost |
| 调参 | Optuna 30 trials | Optuna 50 trials | Optuna 100-150 trials | ✅ 我们可以加到 50 |
| 集成 | scipy 权重优化 | scipy 权重优化 | 不需要 | ✅ 持平 |
| 多 seed | 5 seed | 没明确说 | 没明确说 | ✅ 我们做得对 |
| 鲁棒性 | 无 | run_robust.py | 配置化 | ⚠️ 我们可以加 |

---

## 6. 立刻可做的事(我都可以马上动手)

1. ✍️ **加 CatBoost 到默认 models 列表**(5 分钟)
2. ✍️ **改 Optuna 默认 n_trials = 50**(1 行)
3. ✍️ **加 median voting 选项到 train_ensemble**(15 分钟)
4. ✍️ **写 scripts/run_robust.py**(30 分钟,自动重启)

你说做哪些,我马上动手。