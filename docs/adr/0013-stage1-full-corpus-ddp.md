# ADR-0013：Stage 1 大规模全量预训练协议

- 状态：Partially Superseded by ADR-0014/0015
- 日期：2026-08-12
- 取代：ADR-0001 的 augmentation multiplier 与 45/45/10 采样、ADR-0004 的单卡限定、ADR-0006、ADR-0008 的覆盖 epoch 与多容量/checkpoint v3 合同

> 2026-08-13：corpus 格式和 prepare 执行合同由 [ADR-0014](0014-stage1-prepare-performance-and-corpus-v2.md) 升级为 corpus v2；训练执行、validation、日志和 checkpoint 合同由 [ADR-0015](0015-stage1-high-throughput-epoch-resume.md) 取代。

## 决定

Stage 1 训练集按数据自然频率采样。一个 epoch 是所有训练实体无放回完整遍历一次；单卡不补齐，DDP 只为 rank 等长在全局序列末尾补最多 `world_size - 1` 个前缀样本。sampler 不读取 role，`drop_last=False`。

original 仍按 role canonical 排序、seed shuffle 后做 95%/5% train/valid。`include_augmentation` 是唯一扩增开关；开启时三份 augmentation CSV 必须同时存在，所有通过 canonical 去重、original overlap、validation-seed descendant 与 QC 检查的实体都进入训练，不做倍率抽样。

三类实体的自然出现频率不变。cation、anion、molecule 的 2:2:1 只作为 element-level loss weight，并分别作用于 masked SMILES token、atom、bond、descriptor scalar 和 fingerprint bit。每路 loss 使用 `sum(weight × element_loss) / sum(weight)`；atom/bond feature 与 fingerprint family 仍先各自归一再等权平均。五路 modality lambda 独立且默认均为 1。

正式 Stage 1 只保留 Base，容量为 `d_model=512`、8 heads、8 层 SMILES、6 层 graph、1024 descriptor hidden、8 层 fusion、FFN 2048。保留 role embedding，默认关闭 gradient checkpointing。训练固定为 AdamW、global batch 256、learning rate `1e-4`、weight decay 0.01、5 epochs、5% warmup 后 cosine decay、梯度累积 1。单卡使用 Python 入口；torchrun 自动启用原生 DDP，global batch 必须整除 world size。

prepare 使用磁盘 SQLite catalog、raw descriptor memmap 和流式 shard 发布。corpus artifact 固定为 `kind="ilume_stage1_corpus"`、`format_version=1`，使用 mmap 紧凑 train/valid index 与独立 shard manifest；index、manifest、audit 和 shard 均做 SHA256 校验，metadata 最后原子发布。其他版本均明确拒绝并要求重新 prepare。

quick validation 每 2000 optimizer step 运行每 role 最多 256 条固定 original validation；epoch 末运行完整 original validation，撞车时只跑完整验证。每个 validation 样本只 forward 一次，同时累计 global、per-role 和 per-modality numerator/denominator。DDP 由 rank 0 验证，其他 rank 在 barrier 等待。

checkpoint 固定为 `kind="ilume_stage1_pretraining"`、`format_version=1`。每 1000 optimizer step 原子覆盖 `last.pt`，每个 epoch 永久保存周期 checkpoint 并刷新 `last.pt`。状态包含模型、optimizer、scheduler、scaler、epoch/cursor、step、所有 rank RNG、world size、有效配置与 artifact/source hashes；mid-epoch resume 从下一批精确继续，且必须保持 world size。Stage 2 只接受该 kind/v1。

## 理由

约 530 万行语料不适合用 Python JSON index、全量对象列表与 role 循环采样。磁盘 catalog、memmap、紧凑 index 和 shard-local shuffle 将常驻内存限制在当前分块，同时保留可审计的去重、泄漏排除、QC 与完整性边界。

自然频率遍历把“数据出现频率”与“科研上希望离子承担更高训练权重”拆成两个独立合同。element-level 2:2:1 可以在不复制小类、不改变 epoch 长度的情况下表达 role 偏好；跨 rank 汇总 numerator/denominator 并补偿 DDP 梯度平均，使单卡与多卡共享同一个 global-batch 优化定义。

## 后果

所有旧 corpus（包括 v3）与 Stage 1 v2/v3 checkpoint 都不能复用，正式训练前必须重新 prepare 并从头训练。Stage 2 教师缓存也必须从新的 Stage 1 Base v1 checkpoint 重建。

单卡与 DDP 各自保证确定性和精确恢复，但不承诺两种模式 bitwise 一致。正式训练前仍需手工 benchmark 吞吐与显存；本决定不增加自动 batch-size/OOM 搜索、warm-start、weights-only 或第三方分布式框架。
