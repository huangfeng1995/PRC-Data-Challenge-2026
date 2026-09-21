# PRC Data Challenge 2026 — Taxi-Out Time Prediction

**比赛主页**: https://ansperformance.eu/study/data-challenge/dc2026/
**数据字典**: https://prc-data-challenge-2026.netlify.app/data.html
**截止时间**: 2026-10-11 23:59:59 CET(还剩 23 天)
**设计日期**: 2026-09-18

---

## 1. 产品需求(Product Brief)

### 1.1 一句话目标
构建一个回归模型,预测 11 个欧洲主要机场起飞航班的 **taxi-out 时间(秒)**,在 RMSE 指标上拿到名次,奖金池 €5,000。

### 1.2 关键约束(来自比赛方)

| 约束 | 详情 | 影响 |
|---|---|---|
| 评分指标 | RMSE(以秒计) | 平方误差放大长尾异常,长尾必须特别处理 |
| 训练集 | 2025 全年 × 11 个机场,约 417 万条,Parquet,按月切分 | 全量 ~280 MB,单机可下 |
| 测试集 | 2026 年 1 月 + 7 月(评分用) | 冬夏两个时段差很大,必须用全年训练 |
| 提交限制 | 每天 5 次,每 bucket 1 GB 限额 | 提交次数宝贵,本地验证要稳 |
| 目标变量定义 | `TAXITIME_SEC_mvt` = `MVT_TIME_UTC_mvt` − `BLOCK_TIME_UTC_mvt`,**仅对 PHASE=DEP** | 训练时只取 DEP 行 |
| 公开要求 | 代码必须公开到 GitHub,鼓励投 JOAS 论文 | 代码结构要可复现 |

### 1.3 11 个机场(全在欧控核心区)

| ICAO | IATA | 机场 |
|---|---|---|
| EDDF | FRA | Frankfurt Main |
| EDDM | MUC | Munich |
| EGLL | LHR | London Heathrow |
| EHAM | AMS | Amsterdam Schiphol |
| LEBL | BCN | Barcelona-El Prat |
| LEMD | MAD | Madrid-Barajas |
| LFPG | CDG | Charles de Gaulle |
| LIRF | FCO | Rome-Fiumicino |
| LTFM | IST | İstanbul |
| LSZH | ZRH | Zürich |

(注:网站表格里写"11 major European airports",但 Table 1 只列了 10 个 —— 数据里实际可能有 11 个,等拿到数据再核对。)

### 1.4 成功标准
- **MVP**: baseline RMSE < 全局中位数 taxi-out 的标准差(估算约 200–250 秒)
- **目标**: 进入 leaderboard 前 10
- **理想**: 拿到奖金(top 3)

---

## 2. 技术方案(Technical Design)

### 2.1 整体架构

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐
│ 原始 Parquet │ →  │ 特征工程管线  │ →  │ LightGBM   │ →  │ 提交 parquet │
│ (按月 12 个) │    │ (聚合 + lag) │    │ + 分段模型  │    │            │
└─────────────┘    └──────────────┘    └────────────┘    └────────────┘
                            ↓
                  ┌──────────────────┐
                  │ MLflow 实验记录  │
                  │ (代码/参数/分数)  │
                  └──────────────────┘
```

**单脚本起步 → 后续拆模块**。前两周一个文件搞定 baseline,跑通后按下面拆:

```
prc_data_challenge/
├── README.md                  # 比赛说明、运行步骤、复现指令
├── data/                      # 训练/测试 parquet(从 OSN bucket 下载)
├── docs/                      # 设计文档、实验日志
├── src/
│   ├── data_loader.py        # 读取 parquet、按月合并、过滤 DEP
│   ├── features.py            # 特征工程:基础特征 + 拥堵聚合特征
│   ├── validation.py          # 时序切分、TimeSeriesSplit
│   ├── train_lgb.py           # LightGBM 训练脚本
│   ├── train_segment.py       # 分段模型(短/中/长 taxi-out 分别建模)
│   ├── predict.py             # 推理 + 生成提交 parquet
│   └── utils.py               # 公用函数
├── experiments/               # MLflow 自动记录(本地 SQLite 后端)
├── submissions/               # 提交到 OSN 的 parquet 文件
└── notebooks/                 # EDA、可视化、临时分析
```

### 2.2 数据视角

只建模 **PHASE = DEP** 的行。`TAXITIME_SEC_mvt` 是 taxi-out(推出到离地),ARR 行的 `TAXITIME_SEC_mvt` 是 taxi-in,直接过滤掉。

### 2.3 建模方案:A + B 混合

#### A. 全局回归(baseline,先跑通)
- 直接预测 `TAXITIME_SEC_mvt`(秒)
- 模型:**LightGBM** 优先,XGBoost/CatBoost 作备选
- 损失:`regression_l2`(RMSE 直接对应)→ 后期可换 `regression_huber`(对长尾更稳)
- 目标:用最简单特征(airport/runway/hour/dow + aircraft_type) 跑通,看 baseline RMSE 在什么水位

#### B. 分段处理长尾(主要提分点)
观察训练集 taxi-out 分布后,按分位数切三段:
- **短**:`< P25`(比如 < 8 分钟) — 一般正常滑行
- **中**:`P25 ~ P75` — 正常拥堵
- **长**:`> P75`(比如 > 20 分钟) — 异常拥堵,RMSE 误差大头来源

两段式模型:
1. **分类器**:LightGBM 多分类,预测属于哪一段(3 类),概率输出
2. **分段回归**:三个 LightGBM 回归器,各管自己段内的秒数预测
3. **加权融合**:用分类器的 softmax 概率 × 段内预测值,得到最终秒数

(也可以用 single model + log 变换等更轻量的替代,等看到数据分布再定)

### 2.4 特征工程(方案 B)

#### 2.4.1 基础特征(自身信息)
| 类别 | 字段 | 处理 |
|---|---|---|
| 时间 | 计划起飞小时、星期几、月份、是否节假日 | 周期编码(sin/cos)+ 类别 |
| 空间 | 机场 ICAO、跑道 ID、登机口 ID | 类别编码(LightGBM 原生支持) |
| 飞机 | 飞机类型 ICAO、wake turbulence category | 类别编码 |
| 航班 | 航司 ICAO、市场段(mainline/regional/low-cost 等)、航班类型 | 类别编码 |
| 计划性 | `IOBT_flt`(初始计划推出)、`EOBT_1_flt`(FPL 估计推出)、`SCHED_TIME_UTC_mvt`(计划起降) | 时间差分特征 + 类别 |

**关键防 leakage 原则**:
- ✅ 可以用:`IOBT_flt`、`EOBT_1_flt`、`SCHED_TIME_UTC_mvt` —— 计划时间,实时可拿
- ⚠️ 慎用:`AOBT_3_flt`(实际推出时间) —— 这是 taxi-out 的**起点**,会泄漏标签。**只能用于事后聚合或训练目标**,不能直接喂给模型做实时预测
- ⚠️ 慎用:`MVT_TIME_UTC_mvt`(实际起飞) —— 这就是 taxi-out 的**终点**,绝对不能做特征

#### 2.4.2 聚合拥堵特征(方案 B 的核心)

**设计思路**:对每条 DEP 记录,在它"推出时刻"(`AOBT_3_flt` 在训练时,`EOBT_1_flt` 在推理时,先用 `AOBT_3_flt` 跑通)之前的若干时间窗口内,统计同机场/跑道的拥堵状况。

| 特征 | 含义 | 计算 |
|---|---|---|
| `cnt_dep_airport_last_15m` | 过去 15 分钟同机场起飞航班数 | `count where ADEP=this and AOBT in [t-15min, t)` |
| `cnt_dep_airport_last_30m` | 过去 30 分钟同上 | 同上,窗口 30 |
| `cnt_dep_airport_last_60m` | 过去 60 分钟同上 | 同上,窗口 60 |
| `mean_taxi_airport_last_60m` | 过去 60 分钟同机场平均 taxi-out | `mean(TAXITIME_SEC) where ADEP=this and AOBT in [t-60min, t)` |
| `p90_taxi_airport_last_60m` | 过去 60 分钟同机场 taxi-out P90 | `quantile(0.9)`,同上 |
| `cnt_dep_runway_last_30m` | 过去 30 分钟同跑道起飞数 | 同上,加 RUNWAY 条件 |
| `mean_taxi_runway_last_30m` | 过去 30 分钟同跑道平均 taxi-out | 同上,加 RUNWAY 条件 |
| `std_block_delay_airport_last_60m` | 过去 60 分钟同机场平均推出延迟 | `mean(AOBT − IOBT)`,反映整体准点性 |

**计算优化**:每个 (机场, 月) pair 单独排序 + rolling 即可,避免全表 self-join。pandas + pyarrow 比 SQL 灵活。

#### 2.4.3 时间 lag 特征
- 同一机场、同一小时、同一星期几的历史平均 taxi-out(用 2025 全年训练数据的滚动均值)
- 同一机场、同一天(去除星期影响)、过去 7 天同时段的均值

### 2.5 验证策略

**主验证:时间切分**(避免泄漏,贴近比赛)
- 训练:2025-01 ~ 2025-10
- 验证:2025-11 ~ 2025-12(2 个月,对应测试集 2026-01)
- 额外验证:2025-06(对应测试集 2026-07,夏季旺季)

**辅验证:TimeSeriesSplit K 折**
- 按月滚动 5 折,保证每个验证月都有相似月份的训练支撑

**超参调优**:在主验证上做 Optuna 调优,辅验证监控"是否过拟合到冬季"

**最终模型**:训练时用**全部 12 个月**(包括 11、12 月),验证分数只用来选超参,不再做"卡验证集"的过拟合。

### 2.6 评估指标实现
- 训练时 LightGBM 自带 `metric=rmse`
- 验证时:自定义 RMSE = `sqrt(mean((y_pred − y_true)²))`
- 提交前:`assert TAXITIME_SEC_mvt >= 0`,可选 `clip(lower=60)` 去掉异常小值
- 监控指标:`MAE`、`RMSE`、`P95 误差` —— RMSE 高但 MAE 低说明模型在长尾上栽跟头

### 2.7 技术栈
- **语言**:Python 3.10+
- **数据**:pandas, pyarrow, numpy
- **模型**:LightGBM(主), XGBoost / CatBoost(备)
- **实验管理**:MLflow(本地 SQLite 后端,UI 跑 `mlflow ui`)
- **调优**:Optuna(轻量贝叶斯优化)
- **可视化**:matplotlib, seaborn(EDA 用)
- **提交**:pyarrow 写 parquet

### 2.8 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 数据下载需要 OSN 团队审批,可能延迟 | 卡进度 | 先把代码写好,数据一到立刻跑;同时关注 Discord |
| 单机内存不够聚合 417 万行 | 跑崩 | 按月分批聚合 + 用 pyarrow chunked 读 |
| 长尾异常多,RMSE 跑高 | 拿不到名次 | A+B 混合 + 分段模型 + huber 损失 |
| 没有天气数据 | 预测不准 | 用"同日同时段 lag"间接模拟 |
| 提交次数有限(5/天) | 试错成本高 | 本地做扎实,不在 leaderboard 上瞎试 |
| 团队只有 2 人 | 时间紧 | 严守里程碑,不超范围加 feature |

---

## 3. 实施计划(24 天倒计时)

### 阶段 1:数据接入 + Baseline(第 1–4 天,9/18 – 9/21)

| 天 | 任务 | 产出 |
|---|---|---|
| Day 1(今天) | 提交组队申请;下载数据;环境搭建(pandas/pyarrow/lightgbm/mlflow/optuna);写最小 EDA 脚本 | 数据落地,EDA 报告 |
| Day 2 | 完成特征 v1(只有基础特征,无聚合);跑通 LightGBM baseline;MLflow 记录第一次实验 | baseline RMSE 数字 |
| Day 3 | 实现时间切分验证 + TimeSeriesSplit;调 LightGBM 基础超参 | 验证 RMSE + 调参曲线 |
| Day 4 | EDA 深入:看 taxi-out 分布、看长尾、画特征重要性 | EDA notebook + 分段切点确定 |

### 阶段 2:拥堵特征 + 分段模型(第 5–12 天,9/22 – 9/29)

| 天 | 任务 | 产出 |
|---|---|---|
| Day 5–7 | 实现聚合拥堵特征(cnt/mean/p90 各窗口);验证每条特征的 gain | 特征 v2 |
| Day 8 | 实现两段式模型(分类器 + 分段回归);先按 P25/P75 切 | 分段模型 v1 |
| Day 9 | 调优:损失函数(l2 vs huber vs quantile)、分段切点(试 P50/P75/P85) | 分段模型 v2 |
| Day 10–11 | 第一次提交到 leaderboard(基线);开始第二轮迭代 | 第一个提交分数 |
| Day 12 | 周复盘:看 leaderboard,定方向 | 阶段性结论 |

### 阶段 3:冲刺(第 13–20 天,9/30 – 10/7)

| 天 | 任务 | 产出 |
|---|---|---|
| Day 13–15 | 加入 lag 特征 + 周期性特征;XGBoost / CatBoost 对比 | 特征 v3 + 多模型对比 |
| Day 16–17 | 模型融合(stacking / blending);Optuna 自动调参 | 融合模型 |
| Day 18–19 | 鲁棒性检查:对每机场/每时段单独评估,看是否有偏 | 鲁棒性报告 |
| Day 20 | 第二次提交,确认方向 | leaderboard 更新 |

### 阶段 4:定稿 + 公开(第 21–24 天,10/8 – 10/11)

| 天 | 任务 | 产出 |
|---|---|---|
| Day 21 | 训练最终模型(全量 12 个月);最终本地验证 | 最终分数 |
| Day 22 | 最终提交(留 1–2 次备用) | 最佳提交 |
| Day 23 | 代码整理:README、复现步骤、MLflow 导出 | 公开 GitHub repo |
| Day 24 | JOAS 论文 draft(可选,看精力) | 论文草稿 |

### 阶段 5:可能延期(第 25+ 天,可选)
- 如果比赛延期或有余力:深度模型(GNN/LSTM) + 更多外部数据

---

## 4. 立即可执行的动作(等你拍板)

我现在可以马上做的(不需要等数据):
1. **A. 搭项目骨架**:创建 `src/`、`docs/`、`experiments/` 目录,写 `README.md`、空脚本骨架、`requirements.txt`
2. **B. 写数据下载脚本**:从 OSN bucket 拉 12 个 Parquet(需要 OSN access key)
3. **C. 写 EDA 模板**:你拿到数据后直接跑,自动生成报告
4. **D. 写 LightGBM baseline 脚本**:假设数据已经在 `data/` 下,你能直接跑

我建议先做 **A + D**(项目骨架 + 可跑 baseline),数据下载要等 OSN 审批,不强求今天完成。

你想我现在动手做哪些?或者有别的调整?
