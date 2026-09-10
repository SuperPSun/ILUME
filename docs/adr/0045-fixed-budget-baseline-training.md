# ADR-0045：Baseline 固定预算训练与 Validation 隔离

- 状态：Accepted
- 日期：2026-09-07
- 修订：取代 ADR-0022/0028/0029/0030/0032/0035/0038/0040 中的 early stopping、validation-best checkpoint 与 ILBERT validation-driven scheduler 条款

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
3. 六个神经 baseline（MLP、D-MPNN、MoLFormer、ILBERT、SPMM、LlaSMol）的主配置与
   split 配置统一训练 50 epochs；XGBoost 固定训练 1000 trees。
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

训练配置与 checkpoint 语义均改变，因此旧 baseline train/evaluate artifact 不得与新结果
混合。旧输出保持只读；新主配置结果写入
`outputs/benchmarks/fixed-budget-v1/<model>/`，split 配置写入
`outputs/benchmarks/fixed-budget-v1/splits/<config-stem>/`。

七个主配置需要重新执行 735 个 fold training job 及相应 validation evaluation 与汇总。
若发布 test 指标，还必须用新 checkpoint 重跑五折 test ensemble。分子 feature cache、固定
pretrained assets 与独立环境不依赖模型选择合同，可以复用。实现验收不运行正式 sweep。

## 后果

- 五折 validation 只衡量固定训练方案，不再参与该方案的拟合或模型选择。
- 每个 job 的实际训练量变为可由配置直接确定，但正式 sweep 的总算力需求会上升。
- checkpoint v1 与 v2 的字段和模型语义不同，必须通过独立输出根保持隔离。
