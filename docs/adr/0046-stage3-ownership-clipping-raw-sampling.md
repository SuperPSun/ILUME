# ADR-0046：Stage 3 按参数归属裁剪与原始样本采样

- 状态：已接受
- 日期：2026-09-07
- 修订：ADR-0020 的联合训练梯度裁剪与 virtual 采样

> 2026-09-10：现役三阶段训练由
> [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 定义；本文的原始样本采样与
> 按参数归属裁剪继续适用于其全部 phase。ADR-0047 仅保留历史说明。

## 背景

ADR-0020 在联合训练 phase 的分层 PCGrad assembly 后，对全部可训练参数统一执行
global norm 裁剪，并使用 `N'_t=max(N_t,1000)` 构造 virtual 轮。统一裁剪会让单个
任务的大 PRIVATE 梯度同时缩小 GLOBAL、GROUP 和其他 PRIVATE block；virtual 轮
不会增加新的观测，却会让小任务的相同行被反复曝光并提高过拟合风险。

## 决定

1. 现役 v2 Stage 3 Base、全部 v2 split 配置，以及 ADR-0034/0036 的两个对应 Stage 3
   消融使用 `joint_gradient_clip_mode=ownership`。PCGrad assembly 后通过模型公开参数归属
   API 收集参数，分别对 `GLOBAL`、每个 `GROUP:<group>` 和每个 `PRIVATE:<task>` 执行
   `max_grad_norm=1.0`；不得按参数名推断或把 block 拼接后裁剪。无梯度或被冻结的参数不进入
   裁剪。精调原有的逐任务 PRIVATE 裁剪不变。
2. 上述配置使用 `sampling_mode=raw`，每个任务的每条准备产物训练行在每个轮
   恰好使用一次。每个任务独立执行稳定的无放回 shuffle；不补齐、不重复、不 `drop_last`。
   尾 batch 使用真实剩余大小，任务耗尽后不再参与该轮的后续复合步。
3. Raw `B_t` 直接按 `N_t` 比例分配，满足 `1 <= B_t <= N_t`，总 batch 不超过配置的
   `composite_batch_size`；共同 `K=max_t ceil(N_t/B_t)`。联合训练 phase 的分层 PCGrad、
   任务/组权重和按参数归属裁剪只对当步有数据的任务执行。
4. Refinement 使用相同原始 sequence。每个 PRIVATE 优化器的调度器和完成检查使用
   `ceil(N_t/B_t) * refinement_epochs` 个任务局部更新。轮训练 loss 按实际样本数
   累计并以 `N_t` 归一化。
5. Resolved plan 记录原始 algorithm、`N_t`、`B_t`、任务局部步数、共同 `K` 和
   `epoch_exposures=N_t`，不生成 `N'_t`、padded size 或 replication 比例。采样 mode 进入
   训练身份；检查点格式不升级，身份不同即禁止恢复。
6. 历史实现 v1 与 Capacity v1 继续使用 ADR-0020 的虚拟过采样和整模裁剪。
   它们省略 `sampling_mode` 并保留 `virtual_min_size`，既有合法检查点的身份与恢复
   合同不变。

## 后果

- 现役训练不再通过复制观测平衡任务；小任务对 shared 更新的参与次数随其真实 batch 数
  减少，但原有任务/组权重和 PCGrad 数学保持不变。
- Stage 1、Stage 2 与 Stage 3 准备产物身份不变；现役 v2 与两个对应消融的旧
  Stage 3 检查点不能续训，正式比较需要重新训练集和 evaluate，既有输出不得覆盖。
- 不引入 capped oversampling、power-law 采样或可变配置的重复策略。

## 关联

- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：身份与审计合同](0021-identity-audit-contract-v1.md)
- [ADR-0027：后期逐任务精调](0027-late-taskwise-refinement.md)
- [ADR-0034：RDKit 2D HoME 表示消融](0034-rdkit-2d-home-representation-ablation.md)
- [ADR-0036：无Stage1消融](0036-no-stage1-rdkit-stage2-stage3-ablation.md)
