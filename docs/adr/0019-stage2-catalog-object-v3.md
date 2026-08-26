# ADR-0019：Stage 2 Catalog 驱动的 Object v3 Physics Trainer

- 状态：Accepted
- 日期：2026-08-18
- 取代：ADR-0016 的固定任务、head routing 与 batch 调度合同，以及 ADR-0018 的 data/cache/checkpoint 版本、冻结快路径、loss 组合和 accumulation-window 合同。

> 2026-08-20：本文关于“Stage 3 迁移延期并拒绝 Object v3”的决定已由 [ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md) 取代；其余 Stage 2 Object v3 合同继续有效。
>
> 2026-08-20：本文分散的 identity、lineage 与 audit 规则已由 [ADR-0021](0021-identity-audit-contract-v1.md) 取代；物理格式版本与科研训练合同不变。
>
> 2026-08-25：本文的 cation/anion orbital task 定义已由 [ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 取代；Object v3、训练与 identity 分层合同保持不变。
>
> 2026-08-26：本文的末期 joint training、固定 final checkpoint 与 checkpoint v3 合同已由 [ADR-0027](0027-late-taskwise-refinement.md) 修订；Object v3 数据、teacher 与模型合同保持不变。

## 决定

Stage 2 的任务集合由 ILUME-Data `task_catalog.csv` 中 `catalog_schema_version=1`、`stage=2` 的记录唯一决定。Registry 保存规范化的 catalog 事实与纯数据语义，并按完整 `task_id` 字典序固定任务顺序；Python 不再维护任务白名单。`registry_hash` 不包含 Stage 1 维度或 head 派生参数，这些参数单独进入 `model_contract`。配置只定义训练策略：正的 task weight 必须精确覆盖 registry，并在运行时归一化；`gradient_accumulation_steps` 字段保留但必须等于 1。

Stage 1 新增兼容的 `encode_states()` 接口，同时返回 fusion CLS 与 fusion 后、`atom_trunk` 前的 atom states；原 `encode()` 等价地返回其中的 entity CLS。统一 ObjectEncoder 接受 single neutral/cation/anion 或有序 cation/anion。模型按照 `target_level` 与 `topology` 动态创建 object、interaction 和 atom heads；condition 只进入 task head。QM 的三种 role 共享同一个 task/head/scaler，cation 与 anion orbital 是两个完全独立的 task。

Prepare 为每个任务建立 train-only condition/target scaler。普通 object task 要求完整 target，QM 每个 target 独立忽略 missing 并采用 target-macro loss。Partial charge 使用独立 MOL2 子流水线，保留每个 `mol_id`，发布 ragged atom target；scaler 与 loss 均按 molecule 等权。MOL2 mapping 在全部相关 bond type 可可靠归一化时使用 typed graph isomorphism；否则整个样本退化为 element 与 connectivity matching，并在 audit 中记录 mode、未解析类型和原因。全部 model atom 必须映射，额外结构 atom 只允许显式 H；多个合法映射按 model-to-structure tuple 字典序选择第一个。

第一 epoch 使用语义混合冻结快路：object/interaction task 直接使用 teacher CLS，不运行 packer/backbone；atom task运行冻结 Stage 1 获取 atom states，但 object context 使用 teacher CLS。此时 student slot 等于 teacher slot，teacher loss 为精确零。第二至第五 epoch 解冻 Stage 1 encoding backbone。Teacher loss按展开后的每个 slot 计算，不去重且不约束 ObjectEncoder/atom states。

每个 task 独立 shuffle 样本并组 batch；scheduler 每轮确定性打乱 active tasks，各发一个 batch，小任务耗尽后移除且不 cycle。每个 emitted batch 独立执行一次 optimizer step。归一化权重为 `w_t`、epoch batch 总数为 `M` 时，physics compensation 为 `w_t * M * batch_rows / task_rows`，总目标为 `compensation * physics_loss + lambda_teacher * teacher_loss`；teacher loss 不乘 task 权重或 compensation。

Data artifact 与 teacher cache 保持 v3，checkpoint 按 ADR-0027 升级为 v4，旧 checkpoint 明确拒绝且不迁移。Prepared-data identity 只绑定 source、Stage 1 feature、registry、tensor 与 preparation contract，不包含 Stage 2 `model_contract`。Teacher cache identity 只绑定 entity artifact 与 Stage 1 encoder identity；后者由 encoding-only state、encoding config 与 feature artifact 决定。teacher 的 FP32 dtype 与 math contract 仅记录生成 provenance，不参与 cache identity。完整 epoch checkpoint 保存 registry、model contract、phase、loss/scheduler geometry、joint/per-task optimizer、AMP、refinement cache 与 RNG；epoch 5 是最终普通历史 checkpoint，最终评估模型是独立 taskwise-refined artifact。保存 epoch 5 后原子导出 format v1 `stage2_encoder.pt`，只包含 Stage 1 encoding state、ObjectEncoder state、配置、内部 state hash 与 refinement provenance，不包含 physics heads 或训练状态。

CUDA 执行继续固定 TF32、fused AdamW、三组学习率、bf16、梯度裁剪和 full validation；不引入 PCGrad、MoE、curriculum、early stopping、best/last 或 step checkpoint。

### Execution efficiency refinements

执行优化不得改变 task batch membership、round-robin 顺序、loss、teacher 语义或 optimizer step 顺序。Stage 1 的 `encode()` 直接读取 fusion CLS，只有 `encode_states()` gather fusion atom states；两者仍拒绝 masked batch，并在 eval 下产生相同 entity CLS。

Partial charge 的 CPU packer 只对 Stage 1 entity forward 去重。它发布 molecule offsets、sample-atom 到 unique Stage 1 atom 的 index，以及 sample-atom 到 molecule row 的 index；ObjectEncoder 按 molecule sample 批量运行，AtomHead 按全部 sample atom 单次运行。因此相同 entity 的多个 `mol_id` 共享 Stage 1 states，但不共享 ObjectEncoder/AtomHead 的 sample-level dropout。分子等权 loss 使用 device-side indexed reduction，不在 hot path 逐 molecule 切 ragged tensor。全量 atom target 保持 CPU resident，只有当前 batch 使用 pinned memory 传入 device。

正式 Base 的 execution 参数为 `packing_workers=4`、`packing_prefetch_batches=4`、`cuda_prefetch_batches=1`。ordered CPU packer 的四个逻辑 batch 名额包含正在 H2D 的 batch；completion order 不改变 descriptor order。CUDA 只使用一个 dedicated transfer stream 和一个 lookahead batch，以 event 连接 default stream，不做 per-batch synchronize；CPU 路径完全旁路。训练 loss 与 finite flag 在 device 上累计，只在 logging interval 或 epoch 末一次 materialize；non-finite 最多延迟一个 interval 报错，并且该 epoch 不发布 checkpoint。Validation 复用相同 prefetch 路径及 device-side float64 accumulator，不建立 CLS 或 atom-state cache。

Partial-charge mapping 使用 `spawn` ProcessPool，并保持最多 `2 * workers` 个 outstanding work item；每个 worker 只读取一次 MOL2 bytes，并完成 size/SHA、UTF-8、parse 与 mapping，返回纯 Python/NumPy payload。Parent 按 task、split、source row 顺序消费结果，随后才启动 entity feature pool。Graph DFS 固定按 model atom index 和升序 structure candidate 搜索；第一解即词典序最小解，探测到第二解即停止，并以 `unique|ambiguous` 和 `mapping_count_lower_bound=1|2` 审计。

上述 execution 参数不进入 experiment hash，恢复时允许变化，但 checkpoint 记录实际值作为 provenance。`cuda_prefetch_batches` 目前只接受 1。Data 保持 format v3、checkpoint 为 format v4、encoder 保持 format v1；data signature 额外绑定 preparation contract version，缺少该合同的开发期 v3 artifact 必须重新 prepare。明确不引入 compile、gradient checkpointing、accumulation、batch autotune、OOM fallback、bucketing、多 batch GPU queue或异步 checkpoint。

### Identity boundary refinements

Stage 2 `model_contract` 只属于训练 checkpoint 与 encoder artifact。改变 ObjectEncoder layers、FFN 或 dropout 不使 prepared data 或 teacher cache 失效，但仍改变实验配置与 checkpoint model contract，因此不同模型配置不得互相 resume。Train 启动只用 prepared metadata 验证 Stage 1 feature artifact、registry、tensor contract与实际 dataset tensors，不再比较 data metadata 和当前 Stage 2 model contract。

Teacher cache 的 Stage 1 encoder identity显式覆盖 encoding-only state hash、encoding API contract、实际encoder结构、descriptor schema、role mapping与Stage 1 feature artifact。设备、TF32/CUDA math contract和FP32输出dtype不参与identity，因此cache可跨兼容设备复用；metadata仍保留原生成环境，且不承诺跨硬件重新提取时bitwise一致。Preparation contract升级为3、teacher extraction contract升级为2；更早的开发期v3 artifact/cache不迁移或原地改写。

## 理由

Catalog、数据语义、模型派生维度和训练策略分层后，新增已有结构语义的 simulation task 不再要求修改 Python 白名单，也不会让同名 target 跨任务共享 scaler。Atom supervision 复用 Stage 1 已有 fusion representation，同时保持 reconstruction head 与未来 Stage 3 表示资产的边界。

## 后果

本 identity 边界首次启用时需要一次性重新执行 Stage 2 prepare 与 teacher cache；正式 Base 已于 2026-08-19 完成该刷新。之后调整 ObjectEncoder layers、FFN 或 dropout 可直接复用 data 与 teacher。所有 Object v2 artifact/cache/checkpoint，以及缺少当前 preparation/execution contract 的开发期 Object v3 输出，都不可复用。正式 Base 当前包含九个 Stage 2 simulation task。Stage 3 的后续迁移决定见 [ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md)。
