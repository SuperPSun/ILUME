# ADR-0055：Stage 3 pEC50 Phase 3 单变量回调

- 状态：Accepted
- 日期：2026-09-14
- 修订：ADR-0054 的 pEC50 Phase 3 PRIVATE lifetime

## 背景

下一轮以当前最佳 Stage 3 recipe 为基线，只回调 pEC50 的 PRIVATE-only calibration
预算，避免同时引入其他超参数变化。

## 决定

1. 六套现役 Stage 3 配置中，pEC50 的 `phase3_private_epochs` 从 5 改为 3。
2. pEC50 的 PRIVATE/Tower/FiLM ratio 保持 `0.75/0.75/0.75`，Phase 1/2 LR 与
   lifetime 保持不变。
3. volume expansion 的 Phase 2/3 PRIVATE epoch 保持 `0/0`，xCO2 的 Phase 3
   PRIVATE epoch 保持 5，viscosity 的 task-specific PRIVATE dropout 保持 0.15；
   其他 task recipe 与全部训练合同不变。
4. training identity contract v5 与 resolved-plan format v3 不升级。resolved recipe
   的变化自然产生新 identity，旧 Stage 3 checkpoint 不可恢复到本合同。

## 后果

- 模型结构、参数量、GLOBAL/GROUP、LR、optimizer、PCGrad、raw sampling、loss、
  ownership-aware clipping、task-gate diagnostics 和 evaluation schema 均不改变。
- Stage 1、Stage 2 与 Stage 3 prepared artifact 可复用；六套现役 Stage 3 需在新目录
  重新执行 train、validation 与 test。baseline 无需重跑，历史输出保持只读。

## 关联

- [ADR-0048：owner-lifetime 三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：task-specific owner budget 与 PRIVATE capacity](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0054：弱任务微调与 task-gate diagnostics](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
