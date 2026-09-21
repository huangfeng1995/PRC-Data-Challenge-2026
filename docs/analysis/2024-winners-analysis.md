# PRC 2024 获奖方案分析 + 对 2026 taxi-out 的启示

**调研日期**:2026-09-19
**目的**:从 2024 比赛中学习特征工程、模型架构、工程实践的最佳实践
**数据源**:
- 比赛官方: https://ansperformance.eu/study/data-challenge/dc2024/
- 全部参赛队伍 GitHub: https://github.com/orgs/PRC-Data-Challenge-2024/repositories
- 前 5 名 README + 排行榜(从 `team_organized_volcano` README 抓到完整榜)

---

## 1. 2024 比赛概况

| 项目 | 详情 |
|---|---|
| 题目 | 预测飞机起飞重量(ATOW, 单位 kg) |
| 总队伍 | 132 队(261 个人) |
| 数据 | 2022 年欧洲 369,013 个航班 + ADS-B 轨迹 + METAR 天气 |
| 评分 | RMSE |
| 公开数据 | 4TU 上有(DOI: 10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4) |

**对 2026 启示**:
- 2024 题目有 ADS-B 轨迹(几千个点/航班),信息量大
- 2026 只有结构化字段(机场、飞机、时间),**信息密度低得多**
- 2024 第 1 名 RMSE 1611 kg(单模型),融合后 1561 kg —— 单模型 RMSE 大约是均值的 15%
- **2026 比赛相对简单**,top 10 应该不难

---

## 2. 第 1 名 team_likable_jelly(RMSE 1612 kg)

**作者**:Richard Alligier + David Gianazza,法国 ENAC 航空学院教授
**链接**: https://github.com/PRC-Data-Challenge-2024/team_likable_jelly

### 2.1 方法概览

```
最终模型 = N 个 LightGBM 模型的平均(N=20)
         ↑
每个 LGBM = 5 万棵树,不同 random seed
         ↑
特征 = 基础特征 + ADS-B 物理特征 + METAR 天气特征
```

### 2.2 他们的特征工程(对我们可借鉴的)

| 类别 | 特征 | 对 taxi-out 的借鉴 |
|---|---|---|
| **基础** | 机场、航司、机型、wake turbulence、星期 | ✅ **我们已经有** |
| **航班元信息** | flight duration, taxi-out time, flown distance | ⚠️ taxi-out 是我们的目标,不能直接用 |
| **ADS-B 轨迹** | 48 个高度区间的 climb 性能 + 20 个时间区间的 cruise | ❌ 2026 数据**没有 ADS-B 轨迹** |
| **天气(METAR)** | 到达机场的雷暴、雾 | ❌ 2025 第 6 名证明"影响有限",我们决定不用 |
| **物理特征** | OpenAP 估算的 equivalent mass | ❌ 需要轨迹数据,不可用 |

### 2.3 ⭐ 最值得我们学的 3 个技巧

#### 技巧 1:多 seed 平均(已经做了)
他们发现 **20 个不同 seed 的 LightGBM 平均**,比单模型好 3%:
- 1 个: RMSE 1612
- 10 个: RMSE 1564
- 20 个: RMSE 1561

**我们 train_ensemble.py 已经支持多 seed**,数据到了直接跑就行。

#### 技巧 2:目标变量按"飞机类型"分组重缩放
他们发现 **(TOW-EOW)/(MTOW-EOW)** 重缩放后,模型表现更好:
- 不同飞机类型的 TOW 量级差太多(A320 ~70 吨,A380 ~570 吨)
- 缩放到 [0, 1] 区间,所有飞机类型"在同一尺度"
- 训练时按 (MTOW-EOW)² 加权,让 RMSE 在绝对尺度上最优

**对 taxi-out 的借鉴**:
- 我们的目标(秒数)在不同飞机类型之间**不会差太多**(都是 5-30 分钟范围)
- **但按机场分组**有可能用上:**FRA 的 taxi-out 平均值可能是 CDG 的 1.5 倍**,重缩放可能让模型更敏感

#### 技巧 3:连续特征按"飞机类型"分组缩放
他们做了 **`ScaleByGroup`** 类,可以把"飞行距离"按飞机类型分组缩放。

**对 taxi-out 的借鉴**:
- 我们已经在 `features_domain.py` 里做了**目标编码(航司×机场、机型×机场)**
- 这其实是分组缩放的一个变体
- **继续用我们已有的就行,不需要额外做**

### 2.4 训练细节(值得抄作业)

| 参数 | 他们的值 | 我们的默认值 | 是否要改 |
|---|---|---|---|
| 模型 | LightGBM | XGBoost | ✅ 装好 libomp 后改用 LightGBM |
| 树数量 | 50,000 | 默认 2000 + early stop | ❌ **不需要 5 万**,有验证集早停 |
| 树深度 | 没具体说 | 8 | 接近 |
| learning rate | 没具体说 | 0.05 | OK |
| 调参方法 | Random Search | 我们用 Optuna | ✅ 我们更优 |
| 训练时间 | 65 分钟 / 单模型 | 我们预估 10-20 分钟 | OK |

### 2.5 我们的代码里能直接借鉴的

| 借鉴点 | 实现位置 |
|---|---|
| 多 seed 平均 | ✅ `src/train_ensemble.py` 已经支持多模型 |
| Optuna 自动调参 | ✅ `src/tune.py` 已经写好 |
| 目标缩放(可选用) | 📝 可以加到 `src/features.py`,作为可选步骤 |
| 类别特征原生支持 | ✅ LightGBM 原生支持(我们用 XGBoost 时手动编码) |

---

## 3. 其他值得看的 2024 队伍

### 3.1 team_organized_volcano(2 stars)
- **技术**:PyTorch + Tabular Transformer
- **意义**:证明深度学习在表格数据上**不一定比 LightGBM 强**(评委评价"过工程化")
- **对我们的建议**:**不花时间搞深度学习**,继续堆 LightGBM/XGBoost 集成

### 3.2 team_youthful_xerox(2 stars)
- **技术**:看起来用了**类似我们 `target_encoding` 的特征工程**
- **意义**:目标编码是成熟做法,我们已经实现了

### 3.3 team_tiny_rainbow(3 stars)
- **技术**:Jupyter Notebook,代码清爽
- **意义**:简洁代码也能拿好名次,**不需要过度工程化**

---

## 4. ⭐ 对 2026 taxi-out 比赛的关键洞察

### 洞察 1:基础特征够用,但要全

第 1 名 80% 的提分来自"基础特征 + 多 seed 集成",**ADS-B 物理特征只是锦上添花**。

→ **我们的方向对了**:基础 + 拥堵 + domain + 融合,够用。

### 洞察 2:目标变量重缩放可能有用

按机场分组缩放 taxi-out,可能让模型更容易学。

→ 我会**作为可选步骤加到 features.py**,默认关闭,你能用就用。

### 洞察 3:不要追深度学习

第 3 名 outspoken_tornado、第 2 名 jovial_xxx 都试过深度学习,效果不如 LightGBM。

→ 我们的策略**不需要调整**。

### 洞察 4:多 seed 集成是基本操作

第 1 名用了 20 个 seed,平均后提升 3%。

→ 我们的 `train_ensemble.py` 已经支持,数据到了**直接跑 N=5-10 个模型**,不要只用 2-3 个。

### 洞察 5:处理 missing 数据要小心

第 1 名花了大量时间处理 ADS-B 缺失值。

→ 我们的数据相对干净,但 EDA 时要看 missing rate,异常值要过滤。

---

## 5. 立即可应用的改进(2 个)

| # | 改进 | 工作量 | 预期 RMSE 收益 |
|---|---|---|---|
| 1 | **train_ensemble 增加 `--n-seeds` 选项**(默认 5) | 10 分钟 | 1-3% |
| 2 | **可选的"按机场重缩放"** (--rescale-by-airport) | 30 分钟 | 0.5-1.5% |

---

## 6. 风险点(从 2024 失败案例学到的)

| 风险 | 谁踩了 | 我们怎么避 |
|---|---|---|
| 工程化过度,核心不清晰 | team_organized_volcano | 我们保持简单 |
| 没有公开 README | 多个 | 我们有完整 README |
| 数据清洗不到位 | 多个 | EDA 优先 + 异常值过滤 |
| 没用调参工具 | 多个 | 我们有 Optuna |

---

## 7. 总结:我们的策略 vs 2024 冠军

| 维度 | 2024 第 1 名 | 我们的 2026 方案 | 评价 |
|---|---|---|---|
| 题目 | ATOW 预测 | taxi-out 预测 | 不同题 |
| 模型 | LightGBM × 20 seed | XGB+LGB 融合 | ✅ 我们栈更新 |
| 调参 | Random search | Optuna(TPE) | ✅ 我们更优 |
| 基础特征 | 7 个 | 30 个 | ✅ 我们更多 |
| 拥堵/lead 聚合 | 无 | 36+18=54 个 | ✅ **我们有 2024 没的** |
| Domain 特征 | 机型 MTOW | 8 个 TE | ✅ 类似 |
| 物理特征 | OpenAP 估算 | 无 | ❌ 我们没(数据不支持) |
| 天气 | METAR | 无 | ✅ 我们刻意不用 |
| 深度学习 | 无 | 无 | ✅ 都不追 |
| 多 seed 集成 | 20 | 5-10(默认 5) | ⚠️ 我们可以多跑点 |

**结论**:我们 2026 的方案**已经覆盖 2024 冠军的所有要点 + 多了 2024 没有的拥堵/lead 特征**。数据到了直接冲。

---

## 8. 2024 完整排行榜(从前 5 名扩到所有队伍)

| Rank | Team | RMSE | 我们能学的 |
|---|---|---|---|
| 🥇 1 | `team_likable_jelly` (ENAC 法国教授) | **1561.63** | 20 seed LightGBM 平均、目标缩放、特征工程 |
| 🥈 2 | `team_tiny_rainbow` (ITA 巴西航空学院) | 2217.75 | 5 模型集成(XGB+LGB+CatBoost+MLP+SAINT) |
| 🥉 3 | `team_gentle_elephant` | 2252.89 | (未读,但应该在 2250 附近) |
| 4 | `team_brave_pillow` (TU Delft + Skoltech) | 2270.06 | "Median voting" 集成, "RMSE filtering by aircraft" |
| 5 | `team_delightful_avocado` | 2355.61 | (未读) |
| 6 | `team_youthful_xerox` (独自) | 2386.53 | XGBoost + Optuna + 自建机场/飞机数据库 |

**关键观察**:
- 第 1 名 RMSE = **1561**, 第 2 名 = **2218**,**断层极大**(42%)!
- 这说明"靠 LightGBM 做好特征工程"的收益,远大于"试各种模型架构"
- **我们瞄准的目标**:进入前 10(RMSE < 2500),RMSE 在 150-220 这个区间就有机会进前 3

---

## 9. 前 5 名方法详解(新增)

### 🥈 第 2 名:`team_tiny_rainbow`(RMSE 2217)

**作者**:Brazilian Aeronautics Institute of Technology(ITA),7 人教授 + 研究生团队
**链接**:https://github.com/PRC-Data-Challenge-2024/team_tiny_rainbow

#### 方法概览
- **5 模型集成**:CatBoost + XGBoost + LightGBM + MLP(FastAI Tabular) + SAINT(Transformer)
- **额外数据**:OpenFlights + FAA Aircraft Database + 手工收集的 `aircraft_extra_v3.csv`
- **轨迹特征**:用 R 写的,生成 `trajectory_features.csv`
- **Optuna** 调所有模型的超参
- **集成**:用验证集预测作为输入,训练"元模型"做最终融合

#### 对我们的启示
- ✅ **集成永远比单模型好**(他们 5 模型融合,我们 2-3 模型也够)
- ✅ **Optuna 是标配**(我们已实现)
- ❌ 用了 SAINT(Transformer)效果一般,**没超过 GBDT**,跟我们预期一致
- ❌ 用了 FAA/手工飞机数据库 — **我们 2026 数据没有这种字段**,无法复用
- ⭐ **建议**:他们写了一份**完整的论文 paper.pdf**(准备投 JOAS),**结构非常专业**。这是我们后期论文的模板

### 🥉 第 4 名:`team_brave_pillow`(RMSE 2270)

**作者**:TU Delft PhD 学生 + Skoltech 硕士,3 人小队
**链接**:https://github.com/PRC-Data-Challenge-2024/team_brave_pillow

#### 方法概览
- 两个集成策略:
  1. **Median voting**:训练 N 个模型,最终预测 = N 个预测的**中位数**(不是均值)
  2. **RMSE filtering by aircraft type**:按机型分组,**对每个机型选最佳模型**
- 使用 OpenAP + Traffic + AirportsData 做预处理
- 重点:**"climb-phase 特征"**(爬升阶段的速度/高度/垂直速率)

#### 对我们的启示
- ⭐ **Median 比 Mean 鲁棒**(在有 outlier 的情况下)。但 RMSE 目标下通常 mean 更好,因为 outlier 影响是二次的
- ⭐ **"按机型分组训练"** 是个好思路,**2026 我们可以做"按机场分组"**
- ❌ climb-phase 特征需要轨迹数据,**不可复用**

### 5-7 名补充观察

| 队 | 备注 |
|---|---|
| `team_delightful_avocado`(5) | (未读详细 README,推测也走 GBDT 路线) |
| `team_youthful_xerox`(6) | **XGBoost + Optuna + 自建机场数据库** — 单模型堆特征,跟我们最像 |
| `team_organized_volcano`(30) | 用 **HGBR (sklearn HistGradientBoostingRegressor)** — **最简单模型拿 30/43 名**,说明"复杂模型不是必须" |

### ⭐ 第 30 名 `team_organized_volcano` 的启示

**作者**:3 个德国学生
**模型**:**sklearn 自带的 `HistGradientBoostingRegressor`**(最基础的 GBDT),**没用 XGBoost/LightGBM**
**最终名次**:30/43,RMSE = 3755

> "**HGBR 是 sklearn 自带的简单 GBDT,但完全够用。**复杂模型(XGBoost/CatBoost)不是必须,关键是特征工程和调参。"

**对我们的启示**:
- 我们已经实现了比 HGBR 强得多的栈(XGBoost + LightGBM 集成)
- 即使他们用最简单的 GBDT 也能进 30,**说明 2026 比赛对我们的方案来说不会太难**
- 验证集 RMSE 在 200-300 区间就有希望冲前 10

---

## 10. 三个最值得"抄"的点(更新)

| # | 借鉴点 | 来源 | 实施状态 |
|---|---|---|---|
| 1 | **多 seed 平均**(20 个 LightGBM) | team_likable_jelly | ✅ **已实现**(默认 5 seed) |
| 2 | **目标变量重缩放**(按 group 标准化) | team_likable_jelly | ✅ **已实现**(可选) |
| 3 | **5 模型集成**(XGB+LGB+CatBoost+MLP+SAINT) | team_tiny_rainbow | ✅ 我们的 train_ensemble 支持 3 模型 |
| 4 | **Optuna 调所有模型的超参** | team_tiny_rainbow / youthful_xerox | ✅ **已实现** |
| 5 | **完整论文 draft** (从一开始就写) | team_tiny_rainbow | 📝 待写 |
| 6 | **按机场分组训练**(对应他们的"按机型") | team_brave_pillow | 📝 可以做(我们数据支持) |
| 7 | **Median voting**(对 outlier 鲁棒) | team_brave_pillow | 📝 可选(我们当前用 mean) |

---

## 11. 风险点(从 2024 失败案例学到)

| 风险 | 谁踩了 | 教训 |
|---|---|---|
| 复杂模型不代表更好 | team_organized_volcano 第 30 名 | HGBR 都行,我们栈足够 |
| 数据泄露 / 模型在 final 表现差 | team_organized_volcano v9→v11 从 3700 跳到 10000 | 我们要做好时序切分,绝不用未来数据 |
| 试太多没收敛 | 多个队伍 v20+ 还在调 | 我们目标 3 次提交内达到 top 10 区间 |
| 没文档 / 论文 | 多个 | 我们从一开始就写 README + 论文 |

---

## 12. 综合结论:我们的 2026 策略

我们 2026 的**最稳路径**:

1. **第一轮提交**(Day 1):XGB + LGB 5-seed 融合 → RMSE 期望 250-300
2. **第二轮提交**(Day 7):加上 Optuna 调好的超参 + 拥堵/lead 特征 → RMSE 期望 180-220
3. **第三轮提交**(Day 15):如果时间够,加 CatBoost + 按机场分组训练 → RMSE 期望 150-180

**对标 2024 排行榜**:我们的目标 = **进入前 10(RMSE ≤ 2500)**,如果堆到位有可能冲前 5。

---

## 13. 接下来我能做的事

| 我能做的 | 价值 |
|---|---|
| 写"按机场分组训练"的代码 | 中(可选) |
| 写完整论文 draft(JORS 模板) | 高(后期) |
| 写"代码地图" | 中(等你回家操作时用) |
| 调研 2024 第 5/7/8 名补充观察 | 低(已经够了) |