# ADR-0027：Stage 2/3 Late Task-wise Refinement

- 状态：Accepted
- 日期：2026-08-26
- 修订：ADR-0019、ADR-0020、ADR-0021、ADR-0026 中的末期训练、最终模型选择、checkpoint 与 HPO 评分合同

## 背景

联合多任务训练能在共享参数中迁移知识，但训练末期继续更新 shared state 与使用跨任务
梯度平衡，会限制单任务参数独立收敛。部分任务因此不能超过对应单任务 MLP baseline。
本决定在总 epoch 预算不变的前提下，把末期训练改为共享表示固定、任务私有参数独立收尾。

## 决定

### 共同训练几何

Stage 2 与 Stage 3 配置显式包含 `training.refinement_ratio: 0.20` 和
`training.refinement_lr_multiplier: 0.10`。`refinement_epochs = ceil(epochs * ratio)`，且
至少保留一个完整 joint epoch。现役 Stage 2 v1 为 4+1，Capacity Stage 2 为 8+2，
20-epoch Stage 3 为 16+4，Capacity formal 为 40+10，Stage 3 v1 为 80+20。两字段进入
training identity，但不进入 HPO 搜索空间。

joint phase 的 warmup+cosine 在 boundary 前的真实 update 数内完整走完。refinement 不
warmup；进入 boundary 后，为每个 task 新建互相独立的 AdamW 与 cosine scheduler，清空
原 optimizer 的一阶、二阶动量。初始 LR 是相应原 LR 的 0.1 倍，Stage 2 cosine 终点为
零，Stage 3 终点沿用 `min_lr_ratio`。每个 scheduler 只按所属 task 的真实 update 数推进。

### Stage 2

boundary 后冻结 backbone 与共享 ObjectEncoder，并将共享模块置为 eval；只将当前 task
head 置为 train。HEAD ownership 来自 registry-backed 模型 API，必须完整覆盖且互不
重叠，不按参数名推断。冻结梯度与 forward 路径分离：所有 object/atom task 都继续运行
boundary 时的 student backbone -> ObjectEncoder，禁止使用 teacher embedding 替代路径。
每个 task 仅对原始 physics loss 反向，不使用 task compensation weight，也不加入 teacher
loss。

`stage2_encoder.pt` 继续只导出 boundary 后不再变化的 backbone/ObjectEncoder，并记录
refinement boundary、shared state hash 与 provenance；head refinement 不改变 Stage 3 表示。

### Stage 3

boundary 后严格通过 ownership API 冻结所有 GLOBAL 与 GROUP，仅允许当前
PRIVATE:<task> 更新。整模先置 eval，仅当前 task 的 PRIVATE 模块置 train。virtual sampling、
batch allocation、microbatch、normalized SmoothL1 与确定性顺序不变；每个 composite step
逐 task 直接更新其 optimizer，不调用 PCGrad，也不使用 task/group weight。diagnostics 记录
`pcgrad_applied=false`、不适用矩阵、task loss、LR、梯度范数与 update count。

### 选择与 artifact

每个 task 的候选为 boundary state 与每个 refinement epoch 结束后的 private state。每个
候选都必须有完整且有限的 validation 主指标；严格变小时才替换，精确并列保留更早候选。
Partial Charge 使用 molecule-macro normalized MAE；HOMO/LUMO 使用 pooled sample-micro
raw MAE；其余 Stage 2 与全部 Stage 3 task 使用 task validation normalized MAE。test 不得
参与选择。

最后一个普通 epoch checkpoint 先保存真实历史状态；随后将同一 frozen shared state 与各
task validation-best private state stitching，重新运行完整 validation，并原子发布：

- `taskwise_refined.pt`：独立 kind、format v1、完整 stitched model、source training
  identity、shared/private hashes、boundary、选择记录与 stitched validation；
- `taskwise_refinement.json`：公开安全 manifest，记录 boundary/best metric、候选、是否严格
  改善、artifact hash 与 stitched validation。

Stage 2 checkpoint 升级到 format v4，Stage 3 checkpoint 升级到 format v2，保存 phase、
per-task optimizer/scheduler、update counters 与 best-state cache。支持 boundary、
mid-refinement 和 finalization 恢复；旧 checkpoint 不迁移。evaluate 默认选择
taskwise-refined artifact；只有显式提供 `--checkpoint-epoch N` 才选择
对应普通历史 checkpoint。正式模型选择与 test 使用 refined artifact，普通 checkpoint
只保留历史、恢复与显式诊断评估语义。

### Capacity 与 HPO

所有 probe、HPO、confirmation、seed robustness 与 formal run 都执行 refinement。HPO
原七维搜索空间不变，trial LR 同时决定 refinement LR。Capacity study/report schema 升级为
v2，删除 `tail_epochs`；probe winner、HPO、confirmation、robustness 和 formal comparison
统一读取 stitched validation 的 `macro_task_equal.normalized_mae.value`。test 只能在
recipe 与 scale 冻结后由显式 evaluate 执行。

refinement 改变 training identity。旧训练、旧 HPO SQLite/study 目录不能续跑；必须使用
不冲突的新输出目录。Stage 2 prepared data、teacher cache、encoder物理格式与 Stage 3
prepared物理格式不因本决定改变，但新的 Stage 2 encoder identity 要求重新准备 Stage 3。

## 后果

- 总 epoch 数不增加，代价是最后 20% 不再改善 shared representation。
- 最终评估 artifact 不对应单一历史 epoch；不同 task 可来自不同 refinement epoch，但共享
  完全相同的 frozen state。
- 普通 checkpoint 与 taskwise-refined artifact 的用途明确分离，恢复、完整性与报告管理更
  复杂，但可以审计每个 task 是否真正改善。
- Stage 2 head refinement 只改善 Stage 2 自身任务，不向 Stage 3 提供额外 representation
  improvement；Stage 3 依靠自己的 PRIVATE refinement 收尾。

## 备选方案

- 拒绝增加 epoch：会改变已注册计算预算。
- 拒绝在 refinement 继续 joint optimizer/PCGrad：残留 optimizer 动量和共享梯度平衡会
  破坏任务独立收尾语义。
- 拒绝只取最后 epoch：不同任务的最佳 private state 不必出现在同一候选 epoch。
- 拒绝把 GROUP 当作 private：GROUP 服务多个 task，继续更新会破坏统一 frozen shared state。

## 参考

- [ADR-0019：Stage 2 Object v3](0019-stage2-catalog-object-v3.md)
- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：Identity / Audit Contract v1](0021-identity-audit-contract-v1.md)
- [ADR-0026：Capacity v1](0026-capacity-v1-pipeline-study.md)
