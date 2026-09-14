# ADR-0044：现役 v2 Stage 2 Task-Compensated Teacher Loss

- 状态：Accepted
- 日期：2026-09-07
- 修订：取代 ADR-0019 中现役 v2 joint phase 的 teacher loss weighting；legacy v1、Capacity v1 与 No-Stage1 合同不变

## 背景

ADR-0019 的 joint objective 为
`compensation * physics_loss + lambda_teacher * teacher_loss`。其中 physics loss
使用 task-specific compensation，而 teacher loss 不使用。因此，同一个
`lambda_teacher` 在不同 task 上并不表示 teacher loss 相对 physics loss 的同一权重。

## 决定

现役 `configs/v2/stage2/base.yaml` 显式使用
`loss.teacher_weighting: task_compensated`。对每个 emitted batch，令
`compensation = w_t * M * batch_rows / task_rows`，joint objective 固定为：

`compensation * (physics_loss + lambda_teacher * teacher_loss)`

因此 `lambda_teacher=0.10` 表示每个 task 内 teacher loss 相对 physics loss 的权重为
0.10。第一轮冻结快路的 teacher loss 仍为精确零；teacher MSE 的 slot 展开与 reduction
语义不变；refinement 不加入 teacher loss。

配置同时支持 `uncompensated`，保持原公式，仅供缺省该字段的 legacy v1 与 Capacity v1
使用。No-Stage1 的 RDKit Stage 2 继续要求 `lambda_teacher=0`，训练路径不变。

## Identity 与兼容性

`teacher_weighting` 属于 loss 科研合同并进入 training identity。现役 v2 的显式新策略
使旧 v2 checkpoint 无法 resume；训练必须从新输出目录开始。legacy 配置缺省为
`uncompensated`，且缺省值不写入序列化 payload，因此其既有配置哈希、training identity
与 checkpoint 恢复合同不变。

Prepared data、teacher cache、checkpoint、encoder 的物理格式均不升级。v2 Stage 2
prepared data 与 teacher cache 可复用。新训练发布的 `stage2_encoder.pt` 若 semantic
identity 与旧 encoder 不同，则必须依次重跑 Stage 3 prepare、train 与 evaluation。

## 后果

现役 v2 的 task weight、数据量与 batch-boundary compensation 同时作用于 physics 和
teacher 两项。旧 v2 checkpoint 不迁移、不覆盖；既有输出保持只读。
