# ADR-0053：Stage 3 clean hybrid task recipe

- 状态：Accepted
- 日期：2026-09-13
- 修订：ADR-0052 的 pEC50 capacity 与 volume expansion Phase 2 PRIVATE lifetime

## 背景

以五折 macro normalized MAE 约为 0.21173 的 recipe 为基线，对最新一轮结果逐 task
复核后，只有部分定向设置得到明确支持。本轮形成 clean hybrid ablation，不再引入新的
结构或超参数维度。

## 决定

1. pEC50 的 PRIVATE、Tower 与 FiLM hidden ratio 从 `1.0/1.0/1.0` 回滚为
   `0.75/0.75/0.75`；其 LR 与三阶段 epoch 不变。
2. volume expansion 保持 `0.25/0.25/0.25` width 与 Phase 3 PRIVATE 零预算，Phase 2
   PRIVATE epoch 从 3 增至 4，与 thermophysical Phase 2 branch 的完整四个 epoch 对齐。
3. ADR-0052 中 thermal conductivity、refractive index、self diffusion、thermal
   decomposition、melting point、static permittivity、electrical conductivity、surface
   tension、viscosity 及其他 task 的现役 recipe 全部保持不变。六套现役 Stage 3 配置继续
   共享同一 recipe。
4. 继续使用 training identity contract v5 与 resolved-plan format v3。完整 resolved
   recipe 已进入 training identity，因此旧 checkpoint 不得 resume。

## 后果

- 不修改 schema、resolver、模型、三阶段训练器、GLOBAL/GROUP、LR、sampling、loss、
  optimizer、PCGrad、ownership clipping、validation 或 checkpoint 规则。
- pEC50 缩容使 PRIVATE 参数量下降；GLOBAL/GROUP 参数量不变。
- Stage 1、Stage 2 与 Stage 3 prepared artifact 可复用；六套现役 Stage 3 必须在新目录
  重新执行五折 train、validation 与 test。baseline 无需重跑，历史输出保持只读。

## 关联

- [ADR-0048：owner-lifetime 三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：task-specific owner budget 与 PRIVATE capacity](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0052：task recipe 定向回滚与 epoch cleanup](0052-stage3-task-recipe-rollback-and-epoch-cleanup.md)
