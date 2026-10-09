# ADR-0068：Stage2→Stage3 transfer 下游联合适配

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-22
- 修订：[ADR-0062](0062-stage2-stage3-full-transfer-matrix.md) 与
  [ADR-0065](0065-stage2-stage3-balanced-transfer-matrix.md) 的 Stage 3 下游训练

> 2026-09-23：等行数变体已由 [ADR-0071](0071-retire-balanced-stage2-stage3-transfer.md)
> 退役；下述联合适配合同现仅用于全量数据 transfer。

## 背景

原 transfer matrix 将十个 Stage2 variant 的 ObjectEncoder 输出固化为 1024D
表示 bank，再只训练 Stage3 MLP。零更新基线因而使用随机初始化且永不更新的
ObjectEncoder，而九个来源使用经过 physics supervision 更新的 ObjectEncoder。这使矩阵同时混入
“是否获得 physics supervision”和“下游 ObjectEncoder 是否已经可用”两个差异，对基线不公平。

## 决定

1. Stage2 variant 的生成合同保持不变：基线仍为零 Stage2 更新，九个来源
   使用全量数据合同训练。等行数合同仅见 ADR-0065 的历史记录。
2. 表示 prepare 不再固化 ObjectEncoder 输出，而是按同一 ObjectKey 顺序保存 Stage1
   实体 slots、角色、槽位 count，以及该 variant 的 ObjectEncoder 初始状态/结构。Stage1 backbone
   在下游始终冻结；不同来源的 slots 可因其 Stage2 来源训练期间的 Stage1 backbone 更新而不同。
3. 每个 `(variant, Stage3 task, fold)` 从自己的 ObjectEncoder 状态构造独立模型，并在相同的
   10 个原始轮中同步优化 ObjectEncoder 与 `input→512→256→1` MLP。ObjectEncoder LR 固定复用
   v2 Stage2 合同依据的 `3e-5`，MLP LR 保持 `3e-4`；二者共用原预热/余弦、AdamW、SmoothL1、
   裁剪、种子、shuffle、split、归一化和最终轮发布规则。
4. 相同 `(task, fold)` 的 MLP 初始化仍不包含 variant；表示/ObjectEncoder 初始化是 variant
   之间唯一差异。验证只报告，不使用测试集，不早停或选择最优。
5. transfer 表示和模型产物升级为独立v2 kind/格式。旧冻结表示、旧 MLP
   作业及其汇总保持历史只读，不得与联合适配结果恢复或混合汇总。

## 后果

- 全量数据实验按需执行表示 prepare、1000 个下游作业和汇总；
  已有效的全量数据 Stage2 编码器无需重训。等行数旧产物保持只读。
- 新矩阵衡量的是“不同Stage2初始化在相同下游联合适配预算后的收益”，不再是冻结表示的线性/MLP
  probing结果。它仍不能单独分离Stage1 backbone更新与ObjectEncoder更新的贡献。
- 现役Stage2、Stage3 HoME、Single-任务 MLP和基准比较合同不变。
