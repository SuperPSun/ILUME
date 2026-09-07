# ADR-0046：Stage 3 ownership-aware clipping 与 capped virtual sampling

- 状态：Accepted
- 日期：2026-09-07
- 修订：ADR-0020 的 joint gradient clipping 与 virtual sampling

## 背景

ADR-0020 在 joint phase 的 hierarchical PCGrad assembly 后，对全部可训练参数统一执行
global norm clipping，并使用 `N'_t=max(N_t,1000)` 构造 virtual epoch。统一裁剪会让单个
task 的大 PRIVATE gradient 同时缩小 GLOBAL、GROUP 和其他 PRIVATE block；固定最低 1000
则会让极小任务获得过高的名义重复曝光。

## 决定

1. 现役 v2 Stage 3 Base、全部 v2 split 配置，以及 ADR-0034/0036 的两个对应 Stage 3
   消融使用 `joint_gradient_clip_mode=ownership`。PCGrad assembly 后通过模型公开 ownership
   API 收集参数，分别对 `GLOBAL`、每个 `GROUP:<group>` 和每个 `PRIVATE:<task>` 执行
   `max_grad_norm=1.0`；不得按参数名推断或把 block 拼接后裁剪。无梯度或被冻结的参数不进入
   裁剪。refinement 原有的逐 task PRIVATE clipping 不变。
2. 上述配置使用 `virtual_max_replication_ratio=3.0`，名义 virtual size 固定为
   `N'_t=min(max(N_t,1000),3N_t)`。batch allocation 与共同 epoch step 数必须从同一组
   `N'_t` 计算。
3. 3× 只约束名义 `N'_t/N_t`。现有固定 `B_t`、共同 `K` 与整数 padding 保持不变，实际
   exposure `K B_t/N_t` 允许超过 3×；resolved plan 必须分别记录名义比例和实际比例，不能
   将二者混称。
4. `joint_gradient_clip_mode` 与 `virtual_max_replication_ratio` 进入 training identity。
   diagnostics 保留整模裁剪前后 norm 作为只读汇总，并增加逐 ownership block 的裁剪前后
   norm。checkpoint format 不升级；identity 不同即禁止 resume。
5. legacy v1 与 Capacity v1 继续使用 ADR-0020 的整模 clipping 和 uncapped
   `max(N_t,1000)`。对应配置省略两个新字段，序列化也省略 legacy 默认值，既有合法 checkpoint
   的 identity 与恢复合同不变。

## 后果

- 一个 task 的 PRIVATE 梯度只会触发自身 block 的缩放，不再压低 shared 或其他 task 的梯度。
- Stage 1、Stage 2 与 Stage 3 prepared artifact identity 不变；现役 v2 与两个对应消融的旧
  Stage 3 checkpoint 不能续训，正式比较需要重新 train 和 evaluate，既有输出不得覆盖。
- 不引入 power-law sampling、可变 step allocation 或实际 exposure hard cap。

## 关联

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：identity/audit contract](0021-identity-audit-contract-v1.md)
- [ADR-0027：late taskwise refinement](0027-late-taskwise-refinement.md)
- [ADR-0034：RDKit 2D HoME representation ablation](0034-rdkit-2d-home-representation-ablation.md)
- [ADR-0036：No-Stage1 ablation](0036-no-stage1-rdkit-stage2-stage3-ablation.md)
