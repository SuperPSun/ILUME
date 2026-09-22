# ADR-0068：Stage2→Stage3 transfer 下游联合适配

- 状态：Accepted
- 日期：2026-09-22
- 修订：[ADR-0062](0062-stage2-stage3-full-transfer-matrix.md) 与
  [ADR-0065](0065-stage2-stage3-balanced-transfer-matrix.md) 的 Stage 3 下游训练

## 背景

原 transfer matrix 将十个 Stage2 variant 的 ObjectEncoder 输出固化为 1024D
representation bank，再只训练 Stage3 MLP。零 update baseline 因而使用随机初始化且永不更新的
ObjectEncoder，而九个 source 使用经过 physics supervision 更新的 ObjectEncoder。这使矩阵同时混入
“是否获得 physics supervision”和“下游 ObjectEncoder 是否已经可用”两个差异，对 baseline 不公平。

## 决定

1. Stage2 variant 的生成合同保持不变：baseline 仍为零 Stage2 update，九个 source 仍按 full 或
   balanced 合同训练。已有 Stage2 encoder artifact 可以直接复用。
2. representation prepare 不再固化 ObjectEncoder 输出，而是按同一 ObjectKey 顺序保存 Stage1
   entity slots、role、slot count，以及该 variant 的 ObjectEncoder 初始 state/结构。Stage1 backbone
   在下游始终冻结；不同 source 的 slots 可因其 Stage2 source 训练期间的 Stage1 backbone 更新而不同。
3. 每个 `(variant, Stage3 task, fold)` 从自己的 ObjectEncoder state 构造独立模型，并在相同的
   10 个 raw epochs 中同步优化 ObjectEncoder 与 `input→512→256→1` MLP。ObjectEncoder LR 固定复用
   v2 Stage2 authority 的 `3e-5`，MLP LR 保持 `3e-4`；二者共用原 warmup/cosine、AdamW、SmoothL1、
   clipping、seed、shuffle、split、normalization和最终 epoch 发布规则。
4. 相同 `(task, fold)` 的 MLP 初始化仍不包含 variant；representation/ObjectEncoder 初始化是 variant
   之间唯一差异。validation只报告，不使用test，不早停或选择best。
5. transfer representation和model artifact升级为独立v2 kind/format。旧冻结 representation、旧 MLP
   job及其summary保持历史只读，不得与联合适配结果resume或混合汇总。

## 后果

- full与balanced实验均需重新执行representation prepare、1000个下游job和summary；十个Stage2
  encoder无需重训。
- 新矩阵衡量的是“不同Stage2初始化在相同下游联合适配预算后的收益”，不再是冻结表示的线性/MLP
  probing结果。它仍不能单独分离Stage1 backbone更新与ObjectEncoder更新的贡献。
- 现役Stage2、Stage3 HoME、Single-task MLP和benchmark合同不变。
