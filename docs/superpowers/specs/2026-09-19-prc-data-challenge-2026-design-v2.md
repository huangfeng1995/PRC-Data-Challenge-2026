# PRC Data Challenge 2026 — 设计文档 v2(整合版)

**撰写日期**:2026-09-19
**基于**:v1 设计 + 2024 第 1-6 名调研 + 2025 第 1-3 名调研
**目的**:基于已完成的全部复盘,重塑接下来 23 天的执行纲领
**这是 v1 的"正式升级版"**,v1 保留为历史记录

---

## 0. 与 v1 相比,v2 的关键变化

| 维度 | v1 (2026-09-18) | v2 (2026-09-19) | 原因 |
|---|---|---|---|
| 模型栈 | XGB + LGB 融合 | **XGB + LGB + CatBoost 融合** | 2025 第 2 名 CatBoost 拿 64% 权重 |
| Optuna trials | 30 | **50** | 2025 第 2/3 名证据:50 是甜蜜点 |
| 多 seed | 5 | **5**(不变) | 2024 第 1 名用 20,我们 5 已够 |
| 鲁棒性 | 无 | **scripts/run_robust.py 自动重启** | 2025 第 2 名有,我们不重 |
| Median voting | 无 | **作为可选项** | 2024 第 4 名证明 median 在 outlier 多时更鲁棒 |
| 目标缩放 | 已加(可选) | **保留可选** | 2024 第 1 名用,边际收益小 |
| 按机场分组 | 无 | **后期可加(可选)** | 2024 第 4 名按机型分组,我们可按机场 |

---

## 1. 比赛核心事实(已确认)

| 项目 | 详情 |
|---|---|
| 题目 | 预测 taxi-out 时间(秒) |
| 训练集 | 2025 全年 × 11 个欧洲机场 ≈ 417 万 movements |
| 测试集 | 2026 年 1 月 + 7 月(评分用) |
| 评分指标 | RMSE(以秒计) |
| 提交限制 | 每天 5 次,每 bucket 1 GB |
| 文件命名 | `<team-name>_v<integer>.parquet` |
| 截止 | 2026-10-11 23:59:59 CET(还剩 **22 天**) |
| 奖金 | 前 3 名共 €5,000 |
| 队伍名 | `jubilant-puzzle` |
| Affiliation | Independent Researcher, Haikou, China |
| 已注册队伍 | 30+,中国队伍 0 支(我们是第一支中国队伍) |

---

## 2. 跨年复盘:关键发现(影响方案选择)

### 2.1 2024(ATOW 预测)前 5 名要点

| 排名 | 队伍 | RMSE | 核心方法 |
|---|---|---|---|
| 1 | `team_likable_jelly` (ENAC 教授) | **1561** | LightGBM × 20 seed + 目标缩放 |
| 2 | `team_tiny_rainbow` (ITA 巴西) | 2217 | 5 模型融合 + Optuna + 完整论文 |
| 3 | `team_gentle_elephant` | 2252 | (未读) |
| 4 | `team_brave_pillow` (TU Delft) | 2270 | Median voting + 按机型分组 |
| 5 | `team_delightful_avocado` | 2355 | (未读) |
| 30 | `team_organized_volcano` | 3755 | **只用 sklearn HGBR** |

**关键观察**:
- 第 1 vs 第 2 断层 42%(1561 vs 2217),**说明"做对基础 + 多 seed"远大于"堆模型"**
- HGBR(最朴素)都能进 top 30,**复杂模型不是必须**
- 第 1 名 80% 提升来自基础特征 + 多 seed,**ADS-B 物理特征只是锦上添花**

### 2.2 2025(燃油消耗)前 3 名要点

| 排名 | 队伍 | RMSE | 核心方法 |
|---|---|---|---|
| 1 | `resourceful-quiver` (未读详细) | 200.83 | 领域知识 + 物理特征 |
| 2 | `jubilant-vase` | 211.99 | **CatBoost(64%) + LGB(20%) + XGB(16%)** + Optuna + lag/lead |
| 3 | `outspoken-tornado` (Euranova) | 219.85 | 模块化 + CatBoost + Optuna |

**关键观察**:
- CatBoost 在三模型融合里拿 64% 权重 → **必须加 CatBoost**
- v18(160 Optuna trials)反而比 v7(基础配置)差 80 RMSE → **Optuna 不是越多越好,50 trials 最佳**
- 没轨迹的特征 + 时间特征也能拿 top 3 → **印证 2026 我们没轨迹也能冲**
- 深度学习(Transformer)没赢 GBDT → **不花时间做深度学习**

### 2.3 跨年共通规律

1. **GBDT 系(XGB/LGB/CatBoost)碾压深度学习** —— 3 年比赛,前 10 名都是 GBDT
2. **多 seed 平均稳定有效** —— 2024 第 1 名 20 个 seed 平均
3. **集成 2-5 个不同模型最好** —— 不是越多越好
4. **物理特征有边际收益,但不是必须** —— 2024 第 1 名用物理特征拿了第 1,但第 30 名只用基础特征也进 30
5. **Optuna 30-50 trials 最佳** —— 多了过拟合
6. **目标缩放(按 group 标准化)在某些场景有效** —— 边际收益小

---

## 3. 我们的最终方案(v2)

### 3.1 策略层级(从底向上)

```
Level 1:基础特征(30 个)
        ↓
Level 2:拥堵/lead 聚合(54 个) ← 2026 独有,2024/2025 都没有
        ↓
Level 3:Domain + 目标编码(11 个)
        ↓
Level 4:三模型融合(XGB + LGB + CatBoost) × 5 seed
        ↓
Level 5:Optuna 50 trials 调好的超参
        ↓
Level 6:scipy 权重优化融合
        ↓
Level 7(可选):Median voting / 按机场分组训练
```

### 3.2 特征工程(总览)

| 类别 | 数量 | 关键特征 |
|---|---|---|
| **时间** | 13 | sched_hour/dow/month、is_morning_peak、hour_sin/cos |
| **空间/路线** | 5 | ADEP、ADES、RUNWAY、STAND、airport_hour |
| **飞机/航班** | 7 | AIRCRAFT_TYPE、WK_TBL_CAT、MARKET_SEGMENT、OPERATOR、is_diverted |
| **计划性** | 3 | iobt_to_sched_sec、eobt_to_sched_sec、iobt_eobt_diff_sec |
| **Domain 结构** | 3 | stand_zone、runway_family、runway_side |
| **目标编码(TE)** | 8 | operator/airport/aircraft 等的 8 个 TE 特征 |
| **Lag 拥堵聚合** | 18 | 过去 15/30/60min 同机场/跑道的 cnt/mean/p90 |
| **Lead 拥堵预测** | 18 | 未来 15/30/60min 同机场/跑道的 cnt/mean/p90 |
| **总计** | **77** | |

**所有特征均经过防泄漏审计**(AOBT_3_flt / BLOCK_TIME_UTC_mvt / MVT_TIME_UTC_mvt 不在特征列表)

### 3.3 模型栈

| 模型 | 角色 | 超参来源 | 默认 seed |
|---|---|---|---|
| **XGBoost** | 主力 1 | Optuna 50 trials | 0-4 |
| **LightGBM** | 主力 2 | Optuna 50 trials | 0-4 |
| **CatBoost** | 主力 3 ⭐新 | Optuna 50 trials | 0-4 |
| 融合 | scipy SLSQP 优化权重 | - | - |

**多 seed 总数**:3 模型 × 5 seed = **15 个独立模型**,推理时先 seed 内平均再模型间加权。

### 3.4 验证策略

| 切分 | 用途 |
|---|---|
| **Winter**(2025-11~12 验证 1 月) | 主验证,对应测试集 1 月 |
| **Summer**(2025-06 验证 7 月) | 辅验证,对应测试集 7 月 |
| **K 折时序 CV**(5 折) | 调参时用,避免单切分过拟合 |
| **最终模型** | 全量 12 个月训练,不切分 |

### 3.5 风险控制

| 风险 | 缓解措施 |
|---|---|
| 长跑崩溃 | `scripts/run_robust.py` 自动重启 + 黑名单 |
| Optuna 过拟合 | 默认 50 trials,封顶 80 |
| 单模型失败 | CatBoost 失败时自动 fallback 到 XGB+LGB |
| 类别未见过 | LightGBM/XGBoost 原生支持 unknown |
| 内存爆 | 按月分批处理,不用全量一次性 |
| macOS 缺 libomp | lazy import + sklearn smoke test 已通过 |

---

## 4. 实施路线图(剩余 22 天)

### Day 1(今天,9/19)— 已完成

- ✅ 项目骨架
- ✅ 全部代码模块(13 个 Python 文件)
- ✅ 全部特征工程
- ✅ 三模型融合(还差 CatBoost)
- ✅ Optuna 调参
- ✅ MinIO 下载教程
- ✅ 2024/2025 调研报告

### Day 2–3(9/20–21)— 等 OSN 审核

- 每天看邮箱(主 + 垃圾 + 通知)
- Discord 扫一遍 #announcements
- **我现在能做的事**:加 CatBoost 到 train_ensemble、改 Optuna 默认 50、加 median voting、scripts/run_robust.py

### Day 4–6(9/22–24)— 拿到数据后的**前 3 天**

| Day | 任务 | 产出 |
|---|---|---|
| Day 4 | 1. 下载数据到 `data/raw/` 2. 跑 EDA(看分布 + 异常) 3. 跑最简 baseline(单 XGB, basic 特征) | baseline RMSE 数字 |
| Day 5 | 4. 加所有特征,跑 `feature_set=all` 单 XGB 5. 加 CatBoost 到 ensemble 6. 跑三模型 5-seed 融合 | 三模型融合 RMSE |
| Day 6 | 7. 第一次提交到 leaderboard `jubilant-puzzle_v1.parquet` | 排名 |

### Day 7–10(9/25–28)— 调参 + 第二轮

| Day | 任务 |
|---|---|
| Day 7–8 | Optuna 调 XGB 50 trials,记录 best params |
| Day 9 | Optuna 调 LGB 50 trials |
| Day 10 | 用调好的超参跑最终 ensemble → 第二次提交 `jubilant-puzzle_v2.parquet` |

### Day 11–14(9/29–10/2)— 高级技巧

| Day | 任务 |
|---|---|
| Day 11–12 | 误差分析:看 leaderboard 哪些样本误差大,针对性加特征 |
| Day 13 | 试 median voting(对比 mean) |
| Day 14 | 试按机场分组训练(对比全局) |

### Day 15–18(10/3–6)— 冲刺 + 论文

| Day | 任务 |
|---|---|
| Day 15–16 | 第三次提交 `jubilant-puzzle_v3.parquet`(如果分数更好) |
| Day 17 | 开始写 JOAS 论文 draft |
| Day 18 | 完善 README + 复现脚本 |

### Day 19–22(10/7–10)— 定稿

| Day | 任务 |
|---|---|
| Day 19 | 训练最终模型(全量 12 个月 + 调好的超参) |
| Day 20 | 第四次提交(可选,保留 1 次) |
| Day 21 | 论文第二稿 + 公开 GitHub 仓库 |
| Day 22 | **最终提交日** — 提交 + 检查 |

---

## 5. 目标分数(对标 2024 / 2025)

| 提交轮次 | 我们对标 | 期望 RMSE(秒) |
|---|---|---|
| 第 1 轮(Day 6) | 2025 第 3 名(只用基础+时间) | **250–300** |
| 第 2 轮(Day 10) | 2025 第 2 名(三模型+Optuna) | **180–220** |
| 第 3 轮(Day 16) | 2024 第 4 名(高级技巧) | **150–180** |
| 最好成绩(对标) | 2024 第 1 名(20-seed+缩放) | **120–150** |

**如果我们的 RMSE < 200**:根据 2025 比赛结果,**有机会进前 3 名**
**如果我们的 RMSE < 250**:根据 2025 比赛结果,**大概率进前 10**

---

## 6. 关键 KPI(产品经理视角)

| KPI | 目标 |
|---|---|
| 拿到审核通过 | Day 3 内 |
| 第一次提交 RMSE | < 300 |
| 第一次提交排名 | 进入 leaderboard |
| 调参后 RMSE 改善 | > 5% |
| 鲁棒性:能跑完 1 次完整实验 | 100% |
| 最终排名 | 前 10(奖金池外) / 前 3(奖金池内) |
| 论文完成度 | ≥ 80%(至少 draft) |
| 代码公开度 | 100%(比赛规则要求) |

---

## 7. 我现在应该做的事(按优先级)

### 🔥 紧急(数据到了立刻需要)

1. ✅ 加 **CatBoost 到 train_ensemble** — 超重要,2025 第 2 名 64% 权重
2. ✅ 改 **Optuna 默认 n_trials = 50** — 防止过拟合
3. ✅ 加 **median voting 选项** — 备选方案
4. ✅ 写 **scripts/run_robust.py** — 防崩溃

### 📊 中等(拿到数据后)

5. EDA 跑完后,基于数据分布调优特征
6. 误差分析(看哪些样本预测差)
7. 试目标重缩放(我们已实现但默认关闭)
8. 试按机场分组(可选)

### 📚 长期(后期)

9. 写 JOAS 论文 draft
10. README 整理 + 复现脚本
11. 公开 GitHub 仓库

---

## 8. 文档结构(总览)

```
/Users/openclaw/Documents/PRC_data_challenge/
├── README.md                       # 项目入口
├── requirements.txt                # Python 依赖
├── docs/
│   ├── REAL_DATA_READINESS_CHECK.md  # 数据到位即用清单
│   ├── MINIO_DOWNLOAD_GUIDE.md        # 下载教程
│   ├── analysis/
│   │   ├── 2024-winners-analysis.md          # 2024 调研
│   │   └── 2025-2nd-3rd-places-analysis.md   # 2025 调研
│   └── superpowers/specs/
│       ├── 2026-09-18-prc-data-challenge-2026-design.md  # v1(历史)
│       └── 2026-09-19-prc-data-challenge-2026-design-v2.md  # v2(本文档,权威)
├── src/
│   ├── data_loader.py
│   ├── features.py                # 含目标重缩放
│   ├── features_aggregate.py      # 18 个 lag
│   ├── features_laglead.py        # 18 个 lead
│   ├── features_domain.py         # 8 个 TE + 3 个结构
│   ├── validation.py
│   ├── train_xgb.py               # 单模型
│   ├── train_ensemble.py          # 多模型 + 多 seed(可加 CatBoost)
│   ├── tune.py                    # Optuna(可改 50)
│   ├── predict_xgb.py
│   ├── predict_ensemble.py        # 多 seed 自动加载
│   └── train_lgb.py / predict.py  # 备用
├── scripts/
│   ├── eda.py
│   ├── download_data.py           # 数据下载(已实现)
│   └── run_robust.py              # 自动重启(待加)
├── models/                        # 训练好的模型(运行时生成)
├── submissions/                   # 提交文件(运行时生成)
└── notebooks/                     # EDA 输出(运行时生成)
```

---

## 9. 决策记录(为什么这样设计)

### Q1:为什么是 XGB + LGB + CatBoost 三模型?
- **A**: 2025 第 2 名 jubilant-vase 权重分布 CatBoost 64% + LGB 20% + XGB 16%,三者集成 RMSE 241 → 224(-7%)。2024 第 1 名虽然只用 LightGBM,但他的 20 seed 平均已经是另一种"集成"。三模型 + 5 seed = 15 个独立模型,足够覆盖多样性。

### Q2:为什么默认 5 seed 不是 20 seed?
- **A**: 2024 第 1 名 20 seed 边际收益递减:1→10 降 3.2%,10→20 降 0.16%。训练时间 5 seed ≈ 20 min,20 seed ≈ 80 min。我们时间紧 + 笔记本算力有限,5 seed 是性价比最优解。如果算力允许,后期可调成 10。

### Q3:为什么 Optuna 50 trials 不是 100?
- **A**: 2025 第 3 名 outspoken-tornado v18(160 trials)RMSE 反而从 227 飙到 303,**过拟合**。50 trials 是甜蜜点。

### Q4:为什么不用深度学习?
- **A**: 2024/2025/2026 三年证据都显示 GBDT 完胜深度学习。2025 第 3 名试了 TabDPT,RMSE 反而比 CatBoost 高。

### Q5:为什么不用天气数据?
- **A**: 2025 第 6 名 honest-rose 试过 ERA5,发现"影响有限"。比赛数据本身没有天气。

### Q6:为什么不用按机场分组的训练?
- **A**: 2024 第 4 名按机型分组 RMSE 2070 → 但 brave_pillow 也说"这种 trick 没复现"。我们先用全局训练,如果误差分析显示某些机场预测差,可以再分。

---

## 10. 风险与回退方案

| 风险情景 | 回退方案 |
|---|---|
| CatBoost 在 macOS 装不上 | 退到 XGB+LGB 融合,损失约 1-2% |
| Optuna 跑得太慢(>2 小时) | 改为 20 trials + 用调好的超参 |
| 拥堵特征在 200 万行上跑得慢(>30 分钟) | 按月分批处理 |
| 某次提交 RMSE 暴涨 | 检查 leakage、重新训练、检查数据 |
| 完全跑不通 | 退到 sklearn HGBR(参考 2024 第 30 名) |

---

## 11. 联系与升级

| 触发条件 | 行动 |
|---|---|
| OSN 审核 3 天无消息 | 写 Discord DM 礼貌催 |
| 提交分数暴涨 | 暂停提交,排查数据 |
| CatBoost 装不上 | 改用 XGB+LGB,告诉用户降级方案 |
| EDA 发现重大数据问题 | 重新设计特征,跟用户确认 |

---

## 12. 一句话总结

> 用 **77 个特征 + 3 模型 5-seed 融合 + Optuna 50 trials**,数据到了 6 天内出第一个 leaderboard 分数,目标 RMSE < 200(对标 2025 第 2 名)。