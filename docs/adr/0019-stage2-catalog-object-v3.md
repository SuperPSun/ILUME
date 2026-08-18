# ADR-0019：Stage 2 Catalog 驱动的 Object v3 Physics Trainer

- 状态：Accepted
- 日期：2026-08-18
- 取代：ADR-0016 的固定任务、head routing 与 batch 调度合同，以及 ADR-0018 的 data/cache/checkpoint 版本、冻结快路径、loss 组合和 accumulation-window 合同。

## 决定

Stage 2 的任务集合由 ILUME-Data `task_catalog.csv` 中 `catalog_schema_version=1`、`stage=2` 的记录唯一决定。Registry 保存规范化的 catalog 事实与纯数据语义，并按完整 `task_id` 字典序固定任务顺序；Python 不再维护任务白名单。`registry_hash` 不包含 Stage 1 维度或 head 派生参数，这些参数单独进入 `model_contract`。配置只定义训练策略：正的 task weight 必须精确覆盖 registry，并在运行时归一化；`gradient_accumulation_steps` 字段保留但必须等于 1。

Stage 1 新增兼容的 `encode_states()` 接口，同时返回 fusion CLS 与 fusion 后、`atom_trunk` 前的 atom states；原 `encode()` 等价地返回其中的 entity CLS。统一 ObjectEncoder 接受 single neutral/cation/anion 或有序 cation/anion。模型按照 `target_level` 与 `topology` 动态创建 object、interaction 和 atom heads；condition 只进入 task head。QM 的三种 role 共享同一个 task/head/scaler，cation 与 anion orbital 是两个完全独立的 task。

Prepare 为每个任务建立 train-only condition/target scaler。普通 object task 要求完整 target，QM 每个 target 独立忽略 missing 并采用 target-macro loss。Partial charge 使用独立 MOL2 子流水线，保留每个 `mol_id`，发布 ragged atom target；scaler 与 loss 均按 molecule 等权。MOL2 mapping 在全部相关 bond type 可可靠归一化时使用 typed graph isomorphism；否则整个样本退化为 element 与 connectivity matching，并在 audit 中记录 mode、未解析类型和原因。全部 model atom 必须映射，额外结构 atom 只允许显式 H；多个合法映射按 model-to-structure tuple 字典序选择第一个。

第一 epoch 使用语义混合冻结快路：object/interaction task 直接使用 teacher CLS，不运行 packer/backbone；atom task运行冻结 Stage 1 获取 atom states，但 object context 使用 teacher CLS。此时 student slot 等于 teacher slot，teacher loss 为精确零。第二至第五 epoch 解冻 Stage 1 encoding backbone。Teacher loss按展开后的每个 slot 计算，不去重且不约束 ObjectEncoder/atom states。

每个 task 独立 shuffle 样本并组 batch；scheduler 每轮确定性打乱 active tasks，各发一个 batch，小任务耗尽后移除且不 cycle。每个 emitted batch 独立执行一次 optimizer step。归一化权重为 `w_t`、epoch batch 总数为 `M` 时，physics compensation 为 `w_t * M * batch_rows / task_rows`，总目标为 `compensation * physics_loss + lambda_teacher * teacher_loss`；teacher loss 不乘 task 权重或 compensation。

Data artifact、teacher cache 和 checkpoint 统一为 v3，旧 v2 明确拒绝且不迁移。完整 epoch checkpoint 保存 registry、model contract、loss/scheduler geometry、optimizer、AMP 与 RNG；固定 epoch 5 为最终 checkpoint。成功保存 epoch 5 后原子导出 format v1 `stage2_encoder.pt`，只包含 Stage 1 encoding state、ObjectEncoder state、配置、内部 state hash 与 provenance，不包含 physics heads 或训练状态。Stage 3 迁移仍延期，并同时拒绝 Object v3 checkpoint 与 encoder artifact。

CUDA 执行继续固定 TF32、fused AdamW、三组学习率、bf16、梯度裁剪和 full validation；不引入 PCGrad、MoE、curriculum、early stopping、best/last 或 step checkpoint。

## 理由

Catalog、数据语义、模型派生维度和训练策略分层后，新增已有结构语义的 simulation task 不再要求修改 Python 白名单，也不会让同名 target 跨任务共享 scaler。Atom supervision 复用 Stage 1 已有 fusion representation，同时保持 reconstruction head 与未来 Stage 3 表示资产的边界。

## 后果

正式运行前必须重新执行 Stage 2 prepare 与 teacher cache；所有 Object v2 artifact/cache/checkpoint 都不可复用。正式 Base 当前包含九个 Stage 2 simulation task。Stage 3 只能在后续专门迁移完成后消费 `stage2_encoder.pt`。
