# ADR-0045：Baseline 各自固定预算训练与 Validation 隔离

- 状态：Accepted
- 日期：2026-09-07
- 修订：取代 ADR-0022/0028/0029/0030/0032/0035/0038/0040 中的 early stopping、validation-best checkpoint 与 ILBERT validation-driven scheduler 条款
- 修订：2026-09-10 取消跨模型统一50-epoch预算，改由各模型合同冻结预算

> 2026-09-10：统一 baseline policy 已重新开放。本文仍只约束下列七个 baseline；
> AIonopedia 的 10-epoch model-native 合同由 ADR-0049 定义。

## 背景

Stage 3 baseline 的五折 validation 指标本身是正式 benchmark 结果。若训练同时使用同一
validation fold 决定停止时刻、学习率或最终 checkpoint，就会让正式报告信号参与模型
拟合与选择。即使训练跑满预算，保存 validation 最优 epoch/iteration 仍具有同样问题。

## 决定

1. MLP、ECFP4-XGBoost、D-MPNN、MoLFormer、ILBERT、SPMM 与 LlaSMol 全部固定跑满
   配置预算，并只发布最后一个 epoch 或最后一次 boosting iteration 的模型状态。
2. Validation 可在每轮后计算并写入训练 history/progress，但只用于审计和报告；不得用于
   early stopping、checkpoint selection、学习率调整或其他训练决策。
3. 各 baseline 分别冻结自己的训练预算，不再推断跨模型统一 epoch 数。MLP、D-MPNN、
   MoLFormer、ILBERT 与 SPMM 保持 50 epochs，LlaSMol 固定 10 epochs，XGBoost 固定
   1000 trees。
4. ILBERT 删除 `ReduceLROnPlateau`，以既有 Adam、恒定 `1e-4` learning rate 完成 50
   epochs。其他 baseline 的 train-only 预设 scheduler 保持不变。
5. 所有现役 baseline YAML 显式声明
   `training.model_selection: final_training_state`，并禁止 `early_stopping_patience`、
   `early_stopping_rounds` 与 `selection_metric`。ILBERT 固定 `scheduler: constant`。
6. Baseline checkpoint 升级为 format version 2。神经模型 manifest 使用 `final_epoch`、
   `final_valid_*` 与 `epochs_ran`；XGBoost target entry 使用 `trained_rounds` 与
   `final_valid_mae`，evaluation 使用全部已训练 trees。旧 version 1 checkpoint 不迁移。

本决定只覆盖七个论文 baseline。现役 Stage 3 主模型和 Stage3 Single-task MLP 等内部消融
继续遵循各自 ADR，不因本决定改变训练或 checkpoint selection 合同。

## Identity、输出与重跑

训练配置与 checkpoint 语义不同时，旧 baseline train/evaluate artifact 不得与新结果混合。
旧输出保持只读；新主配置结果写入
`outputs/benchmarks/fixed-budget-v1/<model>/`，split 配置写入
`outputs/benchmarks/fixed-budget-v1/splits/<config-stem>/`。

2026-09-10 的预算修订只改变 LlaSMol 训练身份；其他六个 baseline 的既有 version 2 artifact
保持兼容。LlaSMol 新结果写入
`outputs/benchmarks/fixed-budget-v1/llasmol-10e-bs16-ga2/`，需要重新执行105个fold training
job及相应validation evaluation与汇总；若发布test指标，还必须用新checkpoint重跑五折test
ensemble。固定pretrained assets与独立环境可以复用。实现验收不运行正式sweep。

## 后果

- 五折 validation 只衡量固定训练方案，不再参与该方案的拟合或模型选择。
- 每个 job 的实际训练量由各模型配置直接确定，不再要求跨模型等 epoch 预算。
- checkpoint v1 与 v2 的字段和模型语义不同，必须通过独立输出根保持隔离。
