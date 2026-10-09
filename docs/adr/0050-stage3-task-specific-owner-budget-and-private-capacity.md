# ADR-0050：Stage 3 任务特定的owner预算与 PRIVATE容量

- 状态：已接受
- 日期：2026-09-12
- 修订：ADR-0048 的现役 owner LR、轮、PRIVATE 宽度与任务配置字段

本文集中维护 owner 配方的字段、默认值、零预算与身份合同（含原 ADR-0051 的有效机制）。
现役任务例外见 [ADR-0055](0055-stage3-pec50-phase3-single-variable-rollback.md) 与正式 YAML；
数值试验过程见 [历史摘要](history.md#adr-0051)，无需串读回滚链。

## 背景

最新五折验证与测试集基准比较显示，统一按 `unique_systems` size class 设置 PRIVATE
预算仍会使部分 tiny 任务过拟合、部分 medium/large owner 欠训练，并使少数 CV 强但测试集泛化
较弱的任务过度 specialization。本轮保持 ADR-0048 的三阶段结构及全部训练算法，只定向调整
owner LR、训练寿命和 PRIVATE/Tower/FiLM 宽度。

## 决定

1. Phase 1/2 GROUP 的 `LR × epoch` 固定为：biological
   `7.5e-5×8 / 3.75e-5×5`、dielectric_optical `1e-4×8 / 5e-5×3`、
   thermophysical `1.5e-4×10 / 7.5e-5×4`、transport
   `1.5e-4×12 / 7.5e-5×12`、phase_stability `2e-4×15 / 1e-4×20`、
   solvation `2e-4×15 / 1e-4×24`。Phase 2 LR 必须精确等于 Phase 1 terminal LR。
2. tiny/small/medium/large 的 PRIVATE class 默认宽度与 Phase 1 LR×轮、Phase 2 轮、
   Phase 3 轮分别为：`0.5, 4e-5×6, 3, 2`；`0.75, 6e-5×8, 4, 3`；
   `1.0, 1.2e-4×12, 8, 5`；`1.0, 1.5e-4×15, 12, 8`。Phase 2/3 起始 LR
   由任务的 resolved Phase 1 LR 连续乘以两个 `0.5` floor 推导，不单独配置。
3. 任务顶层允许覆盖 `phase1_private_lr`、`phase1_private_epochs`、
   `phase2_private_epochs`、`phase3_private_epochs`；capacity 继续只在 `model_overrides`
   覆盖 PRIVATE/Tower/FiLM 比例。未覆盖值回退到规模类别默认。旧 `phase3_epochs`
   字段退役并拒绝加载。
4. `model_overrides.private_dropout` 限定在 `[0, 0.15]`，缺省回退 `model.dropout`；只作用于
   PRIVATE-owned Expert、Tower 和 FiLM。GLOBAL/GROUP 不受任务 dropout 覆盖影响。
5. Phase 1 PRIVATE LR 例外为 electrical、refractive index、表面张力 `8e-5`，
   viscosity `1e-4`，density `1.25e-4`。Phase 2/3 LR 仍由连续性公式推导，并继续满足
   Phase 1 的严格 `GLOBAL > GROUP > PRIVATE`。
6. Phase 2 PRIVATE 轮允许为 0：owner 从分支开始即冻结且不进入优化器，任务
   仍参与原始样本采样并向 GROUP 提供梯度；该 owner 仍进入 delta 与拼接完整性校验，
   状态逐 bit 继承 Phase-1 锚点。Phase 3 零预算任务不创建优化器、历史记录或
   检查点，PRIVATE 状态逐 bit 继承 Phase-2 锚点，在 final 清单中审计。
7. `resolved_training_plan.json` 格式 v3 展开最终任务-specific LR、capacity、dropout、
   nominal/effective 轮、冻结轮与更新 budget；完整配方进入训练身份
   合同 v5。相同版本号不代表相同身份，配方不同的检查点不得交叉恢复。
   准备产物身份与历史实现 v1/Capacity 身份不变。

## 兼容边界

六套现役 Stage 3 配置共享同一配方。改变配方需要在新目录重新训练；仅整理代码或文档
不产生新的配方，也不要求重跑。Stage 1/2 与身份一致的 Stage 3 准备产物可复用。
原始样本采样、loss weighting、PCGrad、优化器、按参数归属裁剪、专家数、固定末轮与
验证报告-仅合同均不因配方整理改变。

## 关联

- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0046：按参数归属裁剪与原始样本采样](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0048：owner-存续期三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
