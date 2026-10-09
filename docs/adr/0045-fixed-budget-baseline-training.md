# ADR-0045：基线各自固定预算训练与验证隔离

- 状态：已接受
- 日期：2026-09-07
- 修订：取代 ADR-0022/0028/0029/0030/0032/0035/0038/0040 中的早停、验证最优检查点与 ILBERT 验证-driven 调度器条款
- 修订：2026-09-10 取消跨模型统一50-轮预算，改由各模型合同冻结预算
- 修订：2026-09-19 所有按轮训练的基线统一为 10 轮；ILBERT Adam 固定 learning rate 为 `3e-5`

> 2026-09-10：统一基线策略已重新开放。本文仍只约束下列七个基线；
> AIonopedia 的 10轮模型-native 合同由 ADR-0049 定义。

## 背景

Stage 3 基线的五折验证指标本身是正式基准比较结果。若训练同时使用同一
验证折决定停止时刻、学习率或最终检查点，就会让正式报告信号参与模型
拟合与选择。即使训练跑满预算，保存验证最优轮/iteration 仍具有同样问题。

## 决定

1. MLP、ECFP4-XGBoost、D-MPNN、MoLFormer、ILBERT、SPMM 与 LlaSMol 全部固定跑满
   配置预算，并只发布最后一个轮或最后一次 boosting iteration 的模型状态。
2. 验证可在每轮后计算并写入训练历史记录/progress，但只用于审计和报告；不得用于
   早停、检查点选择、学习率调整或其他训练决策。
3. MLP、D-MPNN、MoLFormer、ILBERT、SPMM 与 LlaSMol 统一固定为 10 轮；XGBoost
   保持 1000 trees，因为其训练预算没有轮语义。
4. ILBERT 删除 `ReduceLROnPlateau`，以既有 Adam、恒定 `3e-5` learning rate 完成 10
   轮。其他基线的仅训练集预设调度器保持不变。
5. 所有现役基线 YAML 显式声明
   `training.model_selection: final_training_state`，并禁止 `early_stopping_patience`、
   `early_stopping_rounds` 与 `selection_metric`。ILBERT 固定 `scheduler: constant`。
6. 基线检查点升级为格式 version 2。神经模型清单使用 `final_epoch`、
   `final_valid_*` 与 `epochs_ran`；XGBoost 目标 entry 使用 `trained_rounds` 与
   `final_valid_mae`，评估使用全部已训练 trees。旧 version 1 检查点不迁移。

本决定只覆盖七个论文基线。现役 Stage 3 主模型和 Stage3 Single-任务 MLP 等内部消融
继续遵循各自 ADR，不因本决定改变训练或检查点选择合同。

## 身份、输出与重跑

训练配置与检查点语义不同时，旧基线训练/评估产物不得与新结果混合。
旧输出保持只读；10轮主配置结果写入
`outputs/benchmarks/fixed-budget-10e-v1/<model>/`，split 配置写入
`outputs/benchmarks/fixed-budget-10e-v1/splits/<config-stem>/`。XGBoost 预算未变，继续使用其既有独立输出根。

2026-09-19 的统一 10轮修订改变所有神经基线的训练身份；既有 version 2 产物
不得混合或恢复。每个受影响模型都需要重新执行105个折训练作业及相应验证
评估与汇总；若发布测试集指标，还必须用新检查点重跑五折测试集集成。固定预训练
assets与独立环境可以复用。实现验收不运行正式sweep。

## 后果

- 五折验证只衡量固定训练方案，不再参与该方案的拟合或模型选择。
- 每个作业的实际训练量由各模型配置直接确定，不再要求跨模型等轮预算。
- 检查点 v1 与 v2 的字段和模型语义不同，必须通过独立输出根保持隔离。
