# ADR-0044：当时 v2 Stage2 的任务补偿 teacher 损失

- 状态：正式Stage2范围内已成为历史，由 [ADR-0082](0082-home-mainline-and-core-ablations.md) 取代
- 日期：2026-09-07
- 修订：取代 ADR-0019 中现役 v2 联合训练 phase 的 teacher loss weighting；历史实现 v1、Capacity v1 与 No-Stage1 合同不变

## 背景

ADR-0019 的联合训练 objective 为
`compensation * physics_loss + lambda_teacher * teacher_loss`。其中 physics loss
使用任务-specific compensation，而 teacher loss 不使用。因此，同一个
`lambda_teacher` 在不同任务上并不表示 teacher loss 相对 physics loss 的同一权重。

## 决定

现役 `configs/v2/stage2/base.yaml` 显式使用
`loss.teacher_weighting: task_compensated`。对每个 emitted batch，令
`compensation = w_t * M * batch_rows / task_rows`，联合训练 objective 固定为：

`compensation * (physics_loss + lambda_teacher * teacher_loss)`

因此 `lambda_teacher=0.10` 表示每个任务内 teacher loss 相对 physics loss 的权重为
0.10。第一轮冻结快路的 teacher loss 仍为精确零；teacher MSE 的槽位展开与 reduction
语义不变；精调不加入 teacher loss。

配置同时支持 `uncompensated`，保持原公式，仅供缺省该字段的历史实现 v1 与 Capacity v1
使用。No-Stage1 的 RDKit Stage 2 继续要求 `lambda_teacher=0`，训练路径不变。

## 身份与兼容性

`teacher_weighting` 属于 loss 科研合同并进入训练身份。现役 v2 的显式新策略
使旧 v2 检查点无法恢复；训练必须从新输出目录开始。历史实现配置缺省为
`uncompensated`，且缺省值不写入序列化 payload，因此其既有配置哈希、训练身份
与检查点恢复合同不变。

Prepared 数据、teacher 缓存、检查点、编码器的物理格式均不升级。v2 Stage 2
准备产物数据与 teacher 缓存可复用。新训练发布的 `stage2_encoder.pt` 若 semantic
身份与旧编码器不同，则必须依次重跑 Stage 3 prepare、训练集与评估。

## 后果

现役 v2 的任务权重、数据量与 batch-边界 compensation 同时作用于 physics 和
teacher 两项。旧 v2 检查点不迁移、不覆盖；既有输出保持只读。
