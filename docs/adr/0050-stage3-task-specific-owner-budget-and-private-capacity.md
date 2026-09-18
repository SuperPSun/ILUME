# ADR-0050：Stage 3 task-specific owner budget 与 PRIVATE capacity

- 状态：Accepted
- 日期：2026-09-12
- 修订：ADR-0048 的现役 owner LR、epoch、PRIVATE width 与 task 配置字段

本文集中维护 owner recipe 的字段、默认值、零预算与 identity 合同（含原 ADR-0051 的有效机制）。
现役 task 例外见 [ADR-0055](0055-stage3-pec50-phase3-single-variable-rollback.md) 与正式 YAML；
数值试验过程见 [历史摘要](history.md#adr-0051)，无需串读回滚链。

## 背景

最新五折 validation 与 test benchmark 显示，统一按 `unique_systems` size class 设置 PRIVATE
预算仍会使部分 tiny task 过拟合、部分 medium/large owner 欠训练，并使少数 CV 强但 test 泛化
较弱的任务过度 specialization。本轮保持 ADR-0048 的三阶段结构及全部训练算法，只定向调整
owner LR、训练寿命和 PRIVATE/Tower/FiLM width。

## 决定

1. Phase 1/2 GROUP 的 `LR × epoch` 固定为：biological
   `7.5e-5×8 / 3.75e-5×5`、dielectric_optical `1e-4×8 / 5e-5×3`、
   thermophysical `1.5e-4×10 / 7.5e-5×4`、transport
   `1.5e-4×12 / 7.5e-5×12`、phase_stability `2e-4×15 / 1e-4×20`、
   solvation `2e-4×15 / 1e-4×24`。Phase 2 LR 必须精确等于 Phase 1 terminal LR。
2. tiny/small/medium/large 的 PRIVATE class 默认 width 与 Phase 1 LR×epoch、Phase 2 epoch、
   Phase 3 epoch 分别为：`0.5, 4e-5×6, 3, 2`；`0.75, 6e-5×8, 4, 3`；
   `1.0, 1.2e-4×12, 8, 5`；`1.0, 1.5e-4×15, 12, 8`。Phase 2/3 起始 LR
   由 task 的 resolved Phase 1 LR 连续乘以两个 `0.5` floor 推导，不单独配置。
3. task 顶层允许覆盖 `phase1_private_lr`、`phase1_private_epochs`、
   `phase2_private_epochs`、`phase3_private_epochs`；capacity 继续只在 `model_overrides`
   覆盖 PRIVATE/Tower/FiLM ratio。未覆盖值回退到 size-class 默认。旧 `phase3_epochs`
   字段退役并拒绝加载。
4. `model_overrides.private_dropout` 限定在 `[0, 0.15]`，缺省回退 `model.dropout`；只作用于
   PRIVATE-owned Expert、Tower 和 FiLM。GLOBAL/GROUP 不受 task dropout 覆盖影响。
5. Phase 1 PRIVATE LR 例外为 electrical、refractive index、surface tension `8e-5`，
   viscosity `1e-4`，density `1.25e-4`。Phase 2/3 LR 仍由连续性公式推导，并继续满足
   Phase 1 的严格 `GLOBAL > GROUP > PRIVATE`。
6. Phase 2 PRIVATE epoch 允许为 0：owner 从 branch 开始即冻结且不进入 optimizer，task
   仍参与 raw sampling 并向 GROUP 提供梯度；该 owner 仍进入 delta 与 stitch 完整性校验，
   state 逐 bit 继承 Phase-1 anchor。Phase 3 零预算 task 不创建 optimizer、history 或
   checkpoint，PRIVATE state 逐 bit 继承 Phase-2 anchor，在 final manifest 中审计。
7. `resolved_training_plan.json` format v3 展开最终 task-specific LR、capacity、dropout、
   nominal/effective epoch、freeze epoch 与 update budget；完整 recipe 进入 training identity
   contract v5。相同版本号不代表相同 identity，recipe 不同的 checkpoint 不得交叉 resume。
   prepared identity 与 legacy v1/Capacity identity 不变。

## 兼容边界

六套现役 Stage 3 配置共享同一 recipe。改变 recipe 需要在新目录重新训练；仅整理代码或文档
不产生新的 recipe，也不要求重跑。Stage 1/2 与身份一致的 Stage 3 prepared artifact 可复用。
raw sampling、loss weighting、PCGrad、optimizer、ownership clipping、专家数、固定末轮与
validation reporting-only 合同均不因 recipe 整理改变。

## 关联

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0046：ownership clipping 与 raw sampling](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0048：owner-lifetime 三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
