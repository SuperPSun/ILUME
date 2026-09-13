# ADR-0052：Stage 3 task recipe 定向回滚与 epoch cleanup

- 状态：Accepted
- 日期：2026-09-13
- 修订：ADR-0051 的 task-specific PRIVATE capacity、dropout 与 epoch 数值

## 背景

ADR-0051 保留了有效的 three-phase 全局策略，但最新五折与 test 结果显示，部分 task 的定向
specialization 调整没有改善或出现退化。本轮只回滚这些 task 的 PRIVATE recipe，并缩短已明确
越过收益点的 Phase 3；训练结构与算法保持不变。

## 决定

1. volume expansion 保持 `0.25/0.25/0.25` width，Phase 2 PRIVATE 从 0 恢复为 3，
   Phase 3 继续为 0。thermal conductivity 保持 `0.5/0.5/0.5` 与 Phase 2=3，Phase 3
   从 4 缩短为 2。
2. pEC50 恢复 `1/1/1` width。refractive index 恢复 `0.75/0.75/0.75`，保留
   dropout `0.15` 与 Phase 3=3。
3. thermal decomposition 保留 `1.25/1.25/1.0`，显式恢复 dropout `0.10`，Phase 3
   从 7 缩短为 6。viscosity 保留 `0.75/0.75/0.75` 与 Phase 3=6，显式恢复 dropout
   `0.10`。
4. self diffusion Phase 3 从 6 缩短为 4；melting point从 8 缩短为 5。
5. static permittivity、speed of sound、glass transition、electrical conductivity、surface
   tension 及其他 task 的现役设置不变。六套现役 Stage 3 配置继续共享同一 recipe。
6. 继续使用 training identity contract v5 与 resolved-plan format v3。完整 resolved recipe
   已进入 training identity，因此旧 checkpoint 即使版本号相同也不能 resume。

## 后果

- 不修改 schema、训练器、GLOBAL/GROUP、LR、sampling、loss、optimizer、PCGrad、clipping、
  validation 或 checkpoint 规则。
- Stage 1、Stage 2 与 Stage 3 prepared artifact 可复用；六套现役 Stage 3 必须在新目录重新
  执行五折 train、validation 与 test。baseline 无需重跑，历史输出保持只读。

## 关联

- [ADR-0048：owner-lifetime 三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：task-specific owner budget 与 PRIVATE capacity](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0051：弱任务 PRIVATE 定向正则化](0051-stage3-weak-task-private-regularization.md)
