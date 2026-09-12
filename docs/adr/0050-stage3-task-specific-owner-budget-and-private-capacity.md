# ADR-0050：Stage 3 task-specific owner budget 与 PRIVATE capacity

- 状态：Accepted
- 日期：2026-09-12
- 修订：ADR-0048 的现役 owner LR、epoch、PRIVATE width 与 task 配置字段

> 数值状态：本 ADR 记录上一轮 recipe；现役 task-specific capacity、dropout 与 epoch 修订见
> [ADR-0051](0051-stage3-weak-task-private-regularization.md)。未被 ADR-0051 修订的 LR、
> size-class 默认与三阶段语义继续有效。

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
4. capacity 例外为：static permittivity 与 volume expansion `0.25/0.25/0.25`；thermal
   conductivity 及 electrical conductivity、refractive index、surface tension、viscosity
   `0.75/0.75/0.75`；pEC50 `1/1/1`；glass transition 与 thermal decomposition
   `1.25/1.25/1.0`。
5. Phase 1 PRIVATE LR 例外为 electrical、refractive index、surface tension `8e-5`，
   viscosity `1e-4`，density `1.25e-4`。Phase 2/3 LR 仍由连续性公式推导，并继续满足
   Phase 1 的严格 `GLOBAL > GROUP > PRIVATE`。
6. epoch 例外为：thermal conductivity Phase 2=`6`、Phase 3=`3`；Phase 3 的 electrical=`4`、
   self diffusion=`6`、surface tension=`6`、viscosity=`6`、transfer=`10`、transfer organic=`15`；
   static permittivity 与 volume expansion=`0`。零预算 task 不创建 Phase 3 optimizer、history
   或 checkpoint，PRIVATE state 逐 bit 继承 Phase-2 anchor并在 final manifest 中审计。
7. `resolved_training_plan.json` 展开最终 task-specific LR、capacity、nominal/effective epoch、
   freeze epoch与 update budget。three-phase training identity 升级；prepared identity和 legacy
   v1/Capacity identity不变。

## 后果

- 六套现役 Stage 3 配置使用同一 recipe，必须在新目录重新完成五折 train、validation 与 test。
- Stage 1、Stage 2 与既有 Stage 3 prepared artifact 可复用；旧 three-phase checkpoint不得续训。
- raw sampling、loss weighting、PCGrad、optimizer、ownership clipping、专家数、固定最后 epoch及
  validation reporting-only 合同均保持不变。

## 关联

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0046：ownership clipping 与 raw sampling](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0048：owner-lifetime 三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
