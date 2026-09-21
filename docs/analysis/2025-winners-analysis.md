# PRC 2025 获奖方案分析 + 对 2026 taxi-out 的启示

**调研日期**:2026-09-18
**目的**:从 2025 比赛中学习特征工程、模型架构、工程实践的最佳实践
**数据源**:
- 比赛 outcome 页面: https://ansperformance.eu/study/data-challenge/dc2025/outcome.html
- 前 12 名团队 GitHub: https://github.com/PRC-Data-Challenge-2025

---

## 1. 2025 比赛概况

| 项目 | 详情 |
|---|---|
| 题目 | 预测飞机燃油消耗(kg) |
| 总队伍 | 179 支(48 个国家) |
| 实际提交 | 53 支,共 2,127 次提交 |
| 最佳 RMSE | 200.83(第 1 名) |
| RMSE 区间 | 200–350 之间 |

**对 2026 启示**:
- 2026 比赛比 2025 简单(taxi-out 比 fuel burn 维度少,无 ADS-B 轨迹)
- 我们的目标应该是 RMSE < 250(秒)就能进 top 10,前 3 大概在 150–200 之间
- 比赛池子大,竞争激烈,**模型融合是必须的**

---

## 2. 关键学习:Top 3 团队做了什么

### 第 1 名:resourceful-quiver (RMSE 200.83)
- **核心方法**:领域知识 + LightGBM + Tabular Transformer
- **特征**:用 24 个轨迹特征训练 LGBM 来估计起飞质量(TOW),TOW 作为下游特征
- **物理**:用 `openap` 库做燃油流估算
- **数据清洗**:为 `traffic` 库写了自定义扩展
- **深度模型**:Transformer 单独跑,作为对比
- **文档**:**完整 JOAS 论文** —— 这对复现和评审都是大加分

### 第 2 名:jubilant-vase (RMSE 211.99)
- **核心方法**:**三模型集成** (XGBoost + LightGBM + CatBoost)
- **特征**:上下文 lag/lead 特征(前一段/后一段的统计)
- **物理**:用 ACARS 数据反算 TAS(真速),无需外部天气数据
- **优化**:Optuna 调参 + scipy.minimize 优化加权权重
- **鲁棒性**:`run_robust.py` 自动监控 + 重启 + 黑名单
- **复现性**:流水线分 4 步,清晰

### 第 3 名:outspoken-tornado (RMSE 219.85)
- **核心方法**:纯机器学习,特征偏轨迹统计(mean/max TAS, mean vertical_rate)
- **结构化**:代码模块化最好(`prc_challenge` 模块)

---

## 3. 对 taxi-out 比赛最有价值的 6 个洞察

### 洞察 1:模型融合是前 3 的标配

第 2 名的核心就是 **XGB + LGB + CatBoost 集成**。对 taxi-out 比赛:
- 我们已经写了 XGBoost 路径
- **建议增加**:LightGBM(装好 libomp 后)+ CatBoost(可选)
- **融合策略**:加权平均 / Stacking(用线性回归学权重)

**对我们的影响**:模型融合预计能比单模型提升 5–10% 的 RMSE,这是冲前 3 的关键。

### 洞察 2:上下文 lag/lead 特征很重要

第 2 名明确说"lag/lead features"是关键。具体到 taxi-out 比赛:
- **lag**:同一机场过去 5/15/30 分钟的拥堵统计(我们已经有了)
- **lead**:未来 15/30 分钟的"已确认航班"信息(ranking 数据里也有,可以试试)

### 洞察 3:物理 / 领域知识能加分

第 1 名评委说:"**strongly informed by domain knowledge rather than relying exclusively on machine-learning techniques**"

对 taxi-out 的领域洞察:
- **重型机滑行慢**(B777/B787 跑道分配慢)
- **航司滑行程序**(不同航司的滑行规则不同,Lufthansa 慢,Ryanair 快)
- **跑道长度**(短跑道需要更多 taxi)
- **跑道分配** vs **登机口距离**(这两个距离决定 taxi-out 主要部分)

### 洞察 4:鲁棒性比"最复杂的模型"更重要

第 2 名的 `run_robust.py` 自动重启处理崩溃航班;第 9 名 hardcoded Windows 路径失分;第 11 名 deep domain awareness 加分。

**对我们的影响**:
- 我们的 `data_loader.py` 用 pyarrow 按月读,内存友好
- 但聚合特征 `_vectorized_past_window` 在全量 417 万行上会慢——需要测一次,可能要分机场并行

### 洞察 5:天气数据"试过但价值有限"

第 6 名 `honest-rose` 明确说:"**exploratory use of weather and economic variables, which were ultimately found to have limited influence**"

**对我们的影响**:
- 我们设计文档里就决定**不用天气数据** ✓
- OpenSky Trino 数据库的 ADS-B 轨迹数据也没有
- 这是个正确的判断

### 洞察 6:深度学习不是必须

第 1 名虽然用了 Transformer,但他们的最好成绩是 LightGBM(评委:RMSE 200.83 是 LGBM)。Transformer 单独跑效果不如 LGBM。

**对我们的影响**:
- 我们专心做 LightGBM + XGBoost 集成,先冲到 top 10
- 深度学习(GNN/LSTM)放最后阶段,如果时间够才做
- **不需要为了"用 AI"而用 AI**

---

## 4. 立即可应用的 5 个改进

| # | 改进 | 工作量 | 预期收益 |
|---|---|---|---|
| 1 | **集成 XGBoost + LightGBM + CatBoost** | 1 天 | RMSE -5% |
| 2 | **加 lag/lead 特征**(未来 30 分钟已确认航班) | 半天 | RMSE -2% |
| 3 | **Optuna 自动调参** | 1 天 | RMSE -3% |
| 4 | **加 domain 特征**(登机口区 / 跑道类型 / 航司 × 机场交互) | 半天 | RMSE -2% |
| 5 | **写完整 JOAS 论文 draft** | 1 天 | 名次评审加分(如果前 10) |

---

## 5. 风险点(从前 3 失败案例学到的)

| 风险 | 谁踩了 | 我们怎么避 |
|---|---|---|
| Hardcoded 路径,不能跨平台跑 | 第 8、9 名 | 我们用相对路径 + argparse |
| 缺少 README / 注释 | 第 9、12 名 | 我们从一开始就写 README |
| 没做数据质量检查 | 第 3 名 | 我们有 EDA 脚本 + 异常值过滤 |
| 单模型,没融合 | 第 4、5 名 | 我们计划直接上融合 |
| 没用领域知识 | 第 3 名 | 我们会专门写 domain 特征 |

---

## 6. 参考资料清单

### 必读代码(前 3 名)
- 🥇 [resourceful-quiver](https://github.com/PRC-Data-Challenge-2025/resourceful-quiver) — 第 1 名
- 🥈 [jubilant-vase](https://github.com/PRC-Data-Challenge-2025/jubilant-vase) — 第 2 名(模型融合的标杆)
- 🥉 [outspoken-tornado](https://github.com/PRC-Data-Challenge-2025/outspoken-tornado) — 第 3 名(代码结构最清晰)

### 关键技术栈
- [openap](https://openap.dev/) — 飞机物理模型(燃油流、推力、阻力)
- [traffic](https://traffic-viz.github.io/) — 飞行轨迹处理(2026 数据用不上,但能学其 API 设计)
- [fastmeteo](https://github.com/open-aviation-data/fastmeteo) — ERA5 天气数据(我们决定不用)
- [LightGBM](https://lightgbm.readthedocs.io/) / [XGBoost](https://xgboost.readthedocs.io/) / [CatBoost](https://catboost.ai/) — 三大 GBDT 库

### 学习路径建议
1. 先读第 2 名 README → 看 `data_utils.py` 理解上下文特征怎么生成
2. 再读第 1 名 `make_dataset.py` → 看怎么从原始数据构建最终训练集
3. 最后看第 1 名 `preprocessor.py` → 看物理特征怎么融入(虽然 2026 数据用不上物理)

---

## 7. 我们对 2026 的策略调整

### 原本计划(已写入设计文档)
1. ✅ 基础特征 + 拥堵聚合
2. 分段模型(长尾)
3. 模型融合
4. 论文

### 调整后(吸收 2025 经验)
1. ✅ 基础特征 + 拥堵聚合
2. **加 domain 特征**(登机口区/跑道类型/航司交互)
3. **直接上三模型融合**(XGB+LGB+CatBoost)
4. **Optuna 自动调参**
5. **分段模型**(只在 RMSE 还不够好时做)
6. JOAS 论文 draft(贯穿全程,不要最后才写)

### 时间预算(剩余 23 天)
- Day 1–4:基础 + 聚合 + domain 特征 + 第一次提交
- Day 5–9:三模型集成 + 调参 + 第二次提交
- Day 10–15:查漏补缺,加 lag/lead,鲁棒性测试
- Day 16–20:冲刺,精细化调优 + 第三次提交
- Day 21–23:最终模型 + 论文 + 代码整理
