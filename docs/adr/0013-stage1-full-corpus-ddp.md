# ADR-0013：Stage 1 大规模全量预训练协议

- 状态：部分被取代，替代ADR： ADR-0014/0015/0017
- 日期：2026-08-12
- 取代：ADR-0001 的扩增倍率与 45/45/10 采样、ADR-0004 的单卡限定、ADR-0006、ADR-0008 的覆盖轮与多容量/检查点 v3 合同

> 2026-08-13：语料格式和 prepare 执行合同由 [ADR-0014](0014-stage1-prepare-performance-and-corpus-v2.md) 升级为语料 v2；训练执行、验证、日志和检查点合同由 [ADR-0015](0015-stage1-high-throughput-epoch-resume.md) 取代；全局batch 和默认编译模式由 [ADR-0017](0017-stage1-base-runtime-profile.md) 取代。

## 决定

Stage 1 训练集按数据自然频率采样。一个轮是所有训练实体无放回完整遍历一次；单卡不补齐，DDP 只为 rank 等长在全局序列末尾补最多 `world_size - 1` 个前缀样本。sampler 不读取角色，`drop_last=False`。

original 仍按角色 canonical 排序、种子 shuffle 后做 95%/5% 训练集/验证集。`include_augmentation` 是唯一扩增开关；开启时三份 augmentation CSV 必须同时存在，所有通过 canonical 去重、original overlap、验证-种子 descendant 与 QC 检查的实体都进入训练，不做倍率抽样。

三类实体的自然出现频率不变。阳离子、阴离子、分子的 2:2:1 只作为 element-level loss 权重，并分别作用于带mask SMILES token、原子、键、描述符标量和指纹 bit。每路 loss 使用 `sum(weight × element_loss) / sum(weight)`；原子/键特征与指纹 family 仍先各自归一再等权平均。五路 modality lambda 独立且默认均为 1。

正式 Stage 1 只保留 Base，容量为 `d_model=512`、8 注意力头、8 层 SMILES、6 层 graph、1024 描述符隐藏宽度、8 层 fusion、FFN 2048。保留角色embedding，默认关闭梯度检查点。训练固定为 AdamW、全局batch 256、learning rate `1e-4`、权重衰减 0.01、5 轮、5% 预热后余弦 decay、梯度累积 1。单卡使用 Python 入口；torchrun 自动启用原生 DDP，全局batch 必须整除 world size。

prepare 使用磁盘 SQLite 任务目录、原始描述符 memmap 和流式分片发布。语料产物固定为 `kind="ilume_stage1_corpus"`、`format_version=1`，使用 mmap 紧凑训练集/验证集 index 与独立分片清单；index、清单、审计和分片均做 SHA256 校验，元数据最后原子发布。其他版本均明确拒绝并要求重新 prepare。

quick 验证每 2000 优化器步运行每角色最多 256 条固定 original 验证；轮末运行完整 original 验证，撞车时只跑完整验证。每个验证样本只前向一次，同时累计 global、per-角色和 per-modality numerator/denominator。DDP 由 rank 0 验证，其他 rank 在 barrier 等待。

检查点固定为 `kind="ilume_stage1_pretraining"`、`format_version=1`。每 1000 优化器步原子覆盖 `last.pt`，每个轮永久保存周期检查点并刷新 `last.pt`。状态包含模型、优化器、调度器、scaler、轮/cursor、步、所有 rank RNG、world size、有效配置与产物/来源 hashes；轮中途恢复从下一批精确继续，且必须保持 world size。Stage 2 只接受该 kind/v1。

## 理由

约 530 万行语料不适合用 Python JSON index、全量对象列表与角色循环采样。磁盘任务目录、memmap、紧凑 index 和分片-local shuffle 将常驻内存限制在当前分块，同时保留可审计的去重、泄漏排除、QC 与完整性边界。

自然频率遍历把“数据出现频率”与“科研上希望离子承担更高训练权重”拆成两个独立合同。element-level 2:2:1 可以在不复制小类、不改变轮长度的情况下表达角色偏好；跨 rank 汇总 numerator/denominator 并补偿 DDP 梯度平均，使单卡与多卡共享同一个 global-batch 优化定义。

## 后果

所有旧语料（包括 v3）与 Stage 1 v2/v3 检查点都不能复用，正式训练前必须重新 prepare 并从头训练。Stage 2 教师缓存也必须从新的 Stage 1 Base v1 检查点重建。

单卡与 DDP 各自保证确定性和精确恢复，但不承诺两种模式 bitwise 一致。正式训练前仍需手工基准比较吞吐与显存；本决定不增加自动 batch-size/OOM 搜索、warm-start、权重-仅或第三方分布式框架。
