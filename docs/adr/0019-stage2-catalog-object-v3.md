# ADR-0019：任务目录驱动的 Stage2 Object v3 物理监督训练器

- 状态：部分由 [ADR-0082](0082-home-mainline-and-core-ablations.md) 取代；Object/数据合同保留作历史参考
- 日期：2026-08-18
- 取代：ADR-0016 的固定任务、预测头路由与 batch 调度合同，以及 ADR-0018 的数据/缓存/检查点版本、冻结快路径、loss 组合和 accumulation-window 合同。

> 2026-08-20：本文关于“Stage 3 迁移延期并拒绝 Object v3”的决定已由 [ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md) 取代；其余 Stage 2 Object v3 合同继续有效。
>
> 2026-08-20：本文分散的身份、lineage 与审计规则已由 [ADR-0021](0021-identity-audit-contract-v1.md) 取代；物理格式版本与科研训练合同不变。
>
> 2026-08-25：本文的阳离子/阴离子 orbital 任务定义已由 [ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 取代；Object v3、训练与身份分层合同保持不变。
>
> 2026-08-26：本文的末期联合训练、固定 final 检查点与检查点 v3 合同已由 [ADR-0027](0027-late-taskwise-refinement.md) 修订；Object v3 数据、teacher 与模型合同保持不变。
>
> 2026-09-07：现役 v2 的 teacher loss weighting 已由 [ADR-0044](0044-stage2-v2-task-compensated-teacher-loss.md) 修订；本文的 uncompensated teacher 公式仅继续用于冻结的历史实现 v1 与 Capacity v1 合同。

## 决定

Stage 2 的任务集合由 ILUME-Data `task_catalog.csv` 中 `catalog_schema_version=1`、`stage=2` 的记录唯一决定。Registry 保存规范化的任务目录事实与纯数据语义，并按完整 `task_id` 字典序固定任务顺序；Python 不再维护任务白名单。`registry_hash` 不包含 Stage 1 维度或预测头派生参数，这些参数单独进入 `model_contract`。配置只定义训练策略：正的任务权重必须精确覆盖注册表，并在运行时归一化；`gradient_accumulation_steps` 字段保留但必须等于 1。

Stage 1 新增兼容的 `encode_states()` 接口，同时返回 fusion CLS 与 fusion 后、`atom_trunk` 前的原子状态；原 `encode()` 等价地返回其中的实体 CLS。统一 ObjectEncoder 接受 single 中性分子/阳离子/阴离子或有序阳离子/阴离子。模型按照 `target_level` 与 `topology` 动态创建 object、交互和原子预测头；条件只进入任务预测头。QM 的三种角色共享同一个任务/预测头/scaler，阳离子与阴离子 orbital 是两个完全独立的任务。

Prepare 为每个任务建立仅训练集条件/目标 scaler。普通 object 任务要求完整目标，QM 每个目标独立忽略 missing 并采用目标-macro loss。Partial charge 使用独立 MOL2 子流水线，保留每个 `mol_id`，发布 ragged 原子目标；scaler 与 loss 均按分子等权。MOL2 mapping 在全部相关键 type 可可靠归一化时使用 typed graph isomorphism；否则整个样本退化为 element 与 connectivity matching，并在审计中记录 mode、未解析类型和原因。全部模型原子必须映射，额外结构原子只允许显式 H；多个合法映射按模型-to-structure tuple 字典序选择第一个。

第一轮使用语义混合冻结快路：object/交互任务直接使用 teacher CLS，不运行 packer/backbone；原子任务运行冻结 Stage 1 获取原子状态，但 object context 使用 teacher CLS。此时 student 槽位等于 teacher 槽位，teacher loss 为精确零。第二至第五轮解冻 Stage 1 encoding backbone。Teacher loss按展开后的每个槽位计算，不去重且不约束 ObjectEncoder/原子状态。

每个任务独立 shuffle 样本并组 batch；调度器每轮确定性打乱 active 任务，各发一个 batch，小任务耗尽后移除且不 cycle。每个 emitted batch 独立执行一次优化器步。归一化权重为 `w_t`、轮 batch 总数为 `M` 时，physics compensation 为 `w_t * M * batch_rows / task_rows`，总目标为 `compensation * physics_loss + lambda_teacher * teacher_loss`；teacher loss 不乘任务权重或 compensation。

Data 产物与 teacher 缓存保持 v3，检查点按 ADR-0027 升级为 v4，旧检查点明确拒绝且不迁移。Prepared-数据身份只绑定来源、Stage 1 特征、注册表、张量与 preparation 合同，不包含 Stage 2 `model_contract`。Teacher 缓存身份只绑定实体产物与 Stage 1 编码器身份；后者由 encoding-仅状态、encoding 配置与特征产物决定。teacher 的 FP32 数据类型与 math 合同仅记录生成来源记录，不参与缓存身份。完整轮检查点保存注册表、模型合同、phase、loss/调度器几何定义、联合训练/逐任务优化器、AMP、精调缓存与 RNG；轮 5 是最终普通历史检查点，最终评估模型是独立逐任务精调后产物。保存轮 5 后原子导出格式 v1 `stage2_encoder.pt`，只包含 Stage 1 encoding 状态、ObjectEncoder 状态、配置、内部状态hash 与精调来源记录，不包含物理性质预测头或训练状态。

CUDA 执行继续固定 TF32、fused AdamW、三组学习率、bf16、梯度裁剪和完整验证；不引入 PCGrad、MoE、curriculum、早停、最优/last 或步检查点。

### 执行效率修订

执行优化不得改变任务 batch membership、round-robin 顺序、loss、teacher 语义或优化器步顺序。Stage 1 的 `encode()` 直接读取 fusion CLS，只有 `encode_states()` gather fusion 原子状态；两者仍拒绝带mask batch，并在 eval 下产生相同实体 CLS。

Partial charge 的 CPU packer 只对 Stage 1 实体前向去重。它发布分子 offsets、样本-原子到 unique Stage 1 原子的 index，以及样本-原子到分子行的 index；ObjectEncoder 按分子样本批量运行，AtomHead 按全部样本原子单次运行。因此相同实体的多个 `mol_id` 共享 Stage 1 状态，但不共享 ObjectEncoder/AtomHead 的样本-level dropout。分子等权 loss 使用 device-side indexed reduction，不在 hot 路径逐分子切 ragged 张量。全量原子目标保持 CPU resident，只有当前 batch 使用 pinned memory 传入 device。

正式 Base 的 execution 参数为 `packing_workers=4`、`packing_prefetch_batches=4`、`cuda_prefetch_batches=1`。ordered CPU packer 的四个逻辑 batch 名额包含正在 H2D 的 batch；completion order 不改变描述符 order。CUDA 只使用一个 dedicated transfer stream 和一个 lookahead batch，以 event 连接默认 stream，不做 per-batch synchronize；CPU 路径完全旁路。训练 loss 与有限 flag 在 device 上累计，只在 logging interval 或轮末一次 materialize；非有限最多延迟一个 interval 报错，并且该轮不发布检查点。验证复用相同 prefetch 路径及 device-side float64 accumulator，不建立 CLS 或原子-状态缓存。

Partial-charge mapping 使用 `spawn` ProcessPool，并保持最多 `2 * workers` 个 outstanding work item；每个 worker 只读取一次 MOL2 bytes，并完成 size/SHA、UTF-8、parse 与 mapping，返回纯 Python/NumPy payload。Parent 按任务、split、来源行顺序消费结果，随后才启动实体特征池。Graph DFS 固定按模型原子 index 和升序 structure 候选搜索；第一解即词典序最小解，探测到第二解即停止，并以 `unique|ambiguous` 和 `mapping_count_lower_bound=1|2` 审计。

上述 execution 参数不进入 experiment hash，恢复时允许变化，但检查点记录实际值作为来源记录。`cuda_prefetch_batches` 目前只接受 1。Data 保持格式 v3、检查点为格式 v4、编码器保持格式 v1；数据 signature 额外绑定 preparation 合同 version，缺少该合同的开发期 v3 产物必须重新 prepare。明确不引入 compile、梯度检查点、accumulation、batch autotune、OOM fallback、bucketing、多 batch GPU queue或异步检查点。

### 身份边界修订

Stage 2 `model_contract` 只属于训练检查点与编码器产物。改变 ObjectEncoder 层、FFN 或 dropout 不使准备产物数据或 teacher 缓存失效，但仍改变实验配置与检查点模型合同，因此不同模型配置不得互相恢复。Train 启动只用准备产物元数据验证 Stage 1 特征产物、注册表、张量合同与实际 dataset tensors，不再比较数据元数据和当前 Stage 2 模型合同。

Teacher 缓存的 Stage 1 编码器身份显式覆盖 encoding-仅状态hash、encoding API 合同、实际编码器结构、描述符结构定义、角色 mapping与Stage 1 特征产物。设备、TF32/CUDA math 合同和FP32输出数据类型不参与身份，因此缓存可跨兼容设备复用；元数据仍保留原生成环境，且不承诺跨硬件重新提取时bitwise一致。Preparation 合同升级为3、teacher extraction 合同升级为2；更早的开发期v3 产物/缓存不迁移或原地改写。

## 理由

Catalog、数据语义、模型派生维度和训练策略分层后，新增已有结构语义的模拟任务不再要求修改 Python 白名单，也不会让同名目标跨任务共享 scaler。Atom supervision 复用 Stage 1 已有 fusion 表示，同时保持 reconstruction 预测头与未来 Stage 3 表示资产的边界。

## 后果

本身份边界首次启用时需要一次性重新执行 Stage 2 prepare 与 teacher 缓存；正式 Base 已于 2026-08-19 完成该刷新。之后调整 ObjectEncoder 层、FFN 或 dropout 可直接复用数据与 teacher。所有 Object v2 产物/缓存/检查点，以及缺少当前 preparation/execution 合同的开发期 Object v3 输出，都不可复用。正式 Base 当前包含九个 Stage 2 模拟任务。Stage 3 的后续迁移决定见 [ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md)。
