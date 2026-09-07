# ADR-0046：Stage 3 ownership-aware clipping 与 raw sampling

- 状态：Accepted
- 日期：2026-09-07
- 修订：ADR-0020 的 joint gradient clipping 与 virtual sampling

> 2026-09-07：现役四阶段训练由
> [ADR-0047](0047-stage3-deterministic-four-phase-training.md) 定义；本文的 raw sampling 与
> ownership-aware clipping 继续适用于其全部 phase。

## 背景

ADR-0020 在 joint phase 的 hierarchical PCGrad assembly 后，对全部可训练参数统一执行
global norm clipping，并使用 `N'_t=max(N_t,1000)` 构造 virtual epoch。统一裁剪会让单个
task 的大 PRIVATE gradient 同时缩小 GLOBAL、GROUP 和其他 PRIVATE block；virtual epoch
不会增加新的观测，却会让小任务的相同行被反复曝光并提高过拟合风险。

## 决定

1. 现役 v2 Stage 3 Base、全部 v2 split 配置，以及 ADR-0034/0036 的两个对应 Stage 3
   消融使用 `joint_gradient_clip_mode=ownership`。PCGrad assembly 后通过模型公开 ownership
   API 收集参数，分别对 `GLOBAL`、每个 `GROUP:<group>` 和每个 `PRIVATE:<task>` 执行
   `max_grad_norm=1.0`；不得按参数名推断或把 block 拼接后裁剪。无梯度或被冻结的参数不进入
   裁剪。refinement 原有的逐 task PRIVATE clipping 不变。
2. 上述配置使用 `sampling_mode=raw`，每个 task 的每条 prepared training row 在每个 epoch
   恰好使用一次。每个 task 独立执行稳定的无放回 shuffle；不补齐、不重复、不 `drop_last`。
   尾 batch 使用真实剩余大小，task 耗尽后不再参与该 epoch 的后续 composite step。
3. Raw `B_t` 直接按 `N_t` 比例分配，满足 `1 <= B_t <= N_t`，总 batch 不超过配置的
   `composite_batch_size`；共同 `K=max_t ceil(N_t/B_t)`。joint phase 的 hierarchical PCGrad、
   task/group weight 和 ownership clipping 只对当步有数据的 tasks 执行。
4. Refinement 使用相同 raw sequence。每个 PRIVATE optimizer 的 scheduler 和完成检查使用
   `ceil(N_t/B_t) * refinement_epochs` 个 task-local update。epoch training loss 按实际样本数
   累计并以 `N_t` 归一化。
5. Resolved plan 记录 raw algorithm、`N_t`、`B_t`、task-local step 数、共同 `K` 和
   `epoch_exposures=N_t`，不生成 `N'_t`、padded size 或 replication ratio。sampling mode 进入
   training identity；checkpoint format 不升级，identity 不同即禁止 resume。
6. legacy v1 与 Capacity v1 继续使用 ADR-0020 的 virtual oversampling 和整模 clipping。
   它们省略 `sampling_mode` 并保留 `virtual_min_size`，既有合法 checkpoint 的 identity 与恢复
   合同不变。

## 后果

- 现役训练不再通过复制观测平衡 task；小任务对 shared update 的参与次数随其真实 batch 数
  减少，但原有 task/group weight 和 PCGrad 数学保持不变。
- Stage 1、Stage 2 与 Stage 3 prepared artifact identity 不变；现役 v2 与两个对应消融的旧
  Stage 3 checkpoint 不能续训，正式比较需要重新 train 和 evaluate，既有输出不得覆盖。
- 不引入 capped oversampling、power-law sampling 或可变配置的重复策略。

## 关联

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：identity/audit contract](0021-identity-audit-contract-v1.md)
- [ADR-0027：late taskwise refinement](0027-late-taskwise-refinement.md)
- [ADR-0034：RDKit 2D HoME representation ablation](0034-rdkit-2d-home-representation-ablation.md)
- [ADR-0036：No-Stage1 ablation](0036-no-stage1-rdkit-stage2-stage3-ablation.md)
