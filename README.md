# ILUME multimodal pretraining

ILUME 面向 cation、anion 和 molecule 三类单分子实体，提供参数完全共享的四模态掩码预训练实现。role 只参与覆盖采样、共享 role embedding 和分组指标，不创建 role 专属 encoder 或 reconstruction head。

```text
SMILES ── tokenizer + Transformer ─────────────┐
Graph ─── shared D-MPNN ──────────────────────┤
Descriptor ── selection + grouped MLP tokens ─┼─ Fusion Transformer ─ reconstruction
Fingerprint ── Morgan/MACCS chunk tokens ─────┘
```

Fusion 序列固定为 `[MM_CLS] + SMILES + atoms + bonds + descriptor groups + fingerprint chunks`。只有 SMILES encoder 使用普通序列位置编码；graph 不使用绝对位置，descriptor 和 fingerprint 分别使用 group identity 与 family/chunk identity。

## 安装

只使用 AIS 时：

```bash
python -m pip install -e ".[dev]"
```

需要 APE、BPE 或 SPE 对照实验时：

```bash
python -m pip install -e ".[dev,tokenizers]"
```

AIS 使用官方 [`atomInSmiles==1.0.2`](https://pypi.org/project/atominsmiles/1.0.2/)。PyPI metadata 将该版本标为 CC BY-NC 4.0，但页面长描述还出现 CC BY-SA 4.0；商业使用或再分发前应向上游确认适用许可。APE 固定到 commit `ff1b3cc00476a8d017d7d54e925681a04475d47f`，SPE 固定为 `SmilesPE==0.0.3`。

## 数据约定

原始数据只读取：

- `data/stage1/cation.csv`
- `data/stage1/anion.csv`
- `data/stage1/molecule.csv`

扩增数据只读取 `data/stage1/augmentation/{cation,anion,molecule}.csv`。`simulation_mol.csv`、`solute.csv`、`solvent.csv` 和 `IL.csv` 均不进入当前准备流程。

原始实体按 role 做 95%/5% train/valid 划分；valid 只含原始实体。准备程序通过扩增表的 `seed_smiles_list` 排除 valid 原始实体的扩增后代。`data.augmentation` 可为每个 role 指定非负倍率或 `all`：`0` 禁用扩增，`1` 最多选取原始训练集等量扩增样本，`4` 最多选四倍，`all` 使用隔离后的全部候选。

```yaml
data:
  augmentation: {cation: 1, anion: 1, neutral: 1}
```

入选实体在任何批量描述符计算前执行统一 QC：含 BCUT2D 不支持键型、Ipc 非有限或平方会超出 float64、以及当前 tokenizer 超过 `max_smiles_tokens` 的实体直接排除。原始和扩增实体使用相同规则；扩增不回填，实际倍率写入 metadata。孤立氢的 RDKit 警告保留，不作为排除条件。

## 准备工件

```bash
ilume-prepare --config configs/smoke.yaml
```

artifact format v3 不再生成单个 `corpus.pt`，而是产生：

- `shards/{split}_{role}_*.pt`：按 split/role 写入，默认约 8192 个样本一个 shard；
- `corpus_index.json`：sample 到 shard/local position 的索引；
- `manifest.csv`：role、split、canonical SMILES、来源与扩增谱系；
- `excluded_entities.csv`：排除原因、bond type、Ipc、token 数和来源审计；
- `tokenizer.json`：只由入选训练集拟合的 tokenizer；
- `descriptor_schema.json`：描述符筛选、删除原因、相关簇和 group 映射；
- `descriptor_scaler.json`：只由入选训练集拟合的有限值均值与标准差；
- `metadata.json`：格式版本、源文件哈希、数据谱系、版本和统计信息。

`PreparedCorpusDataset` 用小型 LRU cache 延迟加载 shard。旧 v2 artifact 和 `corpus.pt` 会被明确拒绝，需重新执行 `ilume-prepare`。

准备过程先完成 QC 并用最终保留训练集拟合 tokenizer，再用磁盘 memmap 分块拟合 descriptor schema/scaler，最后按 split/role shard 逐块构图和计算指纹，不把全语料 graph 同时留在内存。`preparation_state.json` 和 preparation signature 支持中断后复用描述符第一遍及已原子写完的 shard；完整成功后临时 memmap 会删除。每个 shard 的 SHA-256 在首次加载时校验。

## 描述符与指纹

`descriptor.mode` 支持：

- `full`：保留 RDKit 固定顺序的 217 维；
- `clean`：删除全无效、有限值零方差和完全重复维度；
- `pruned`：在 clean 基础上，对共同有限训练行的 `|r| > 0.98` 相关图按连通分量保留代表维度。

筛选只使用当前配置选中的训练集。`descriptor.token_count` 只允许 `1`、`8` 或 `12`；每个语义组独立编码和重建，空组使用可学习 token 但不产生 loss。自然非有限值保持 unavailable，不作为 masked reconstruction target。

`fingerprint.kind` 支持 `none | morgan | maccs | both`。Morgan 默认 2048 bit、radius 2，形成 16 个 128-bit chunk；MACCS 的 167 个有效 bit 补齐为两个 chunk，padding bit 不参与 mask 和 loss。两种 family 的 masked-only BCE 先各自归一化，再等权平均。

## tokenizer 对照

统一 `SmilesTokenizer` 接口支持 `ais | ape | bpe | spe`。默认预算为 2048、最低频率为 2、最大长度为 256（包含 `[CLS]` 和 `[SEP]`）。准备阶段排除超长实体并重新拟合 tokenizer，直到训练集合稳定；直接调用 `encode()` 时仍会对超限输入报错，绝不静默截断。metadata 记录过滤、长度分布、UNK 数、预算/实际词表和后端版本。

## masking 与模型

`MultimodalPacker` 只做确定性组批；`MultimodalMasker` 使用由 epoch 数推导的内部 global step 执行动态 masking、asymmetric masking 和 modality dropout。curriculum 默认行为为：

- 0%–10% 训练进度：dropout 率为 0；
- 10%–60%：线性升至配置概率的一半；
- 60%–100%：线性升至配置概率。

四个顶层模态独立 dropout，但每个样本至少保留一个模态。普通 atom masking 不遮蔽单原子分子；graph modality dropout 仍会整体遮蔽单原子并监督重建。

模型包含 SMILES MLM、atom、bond、descriptor 和 fingerprint 五项重建 loss。graph head 支持 `linear | mlp`；MLP 版本使用共享 residual trunk 后接字段独立分类器。可选共享 role embedding 会加到 CLS 和所有非 padding fusion token 上，不进入各模态 encoder。

## 45/45/10 覆盖型 epoch

正式配置在每个 epoch 内固定 cation 45%、anion 45%、molecule 10% 的全局抽样比例，不要求每个 mini-batch 精确满足。role 内先无放回遍历全部样本，耗尽后再重洗牌补足配额。

```text
required_draws = max(ceil(N_cation/0.45), ceil(N_anion/0.45), ceil(N_molecule/0.10))
effective_batch = batch_size * gradient_accumulation_steps
steps_per_epoch = ceil(required_draws / effective_batch)
draws_per_epoch = steps_per_epoch * effective_batch
```

因此一个 epoch 保证三类入选实体都至少出现一次，并补齐到完整 optimizer step。这不是按数据自然比例遍历一次；为了保持离子实体90%的权重，cation 和 anion 在同一 epoch 内会进入后续无放回循环。

## 运行入口

最小 forward/backward：

```bash
ilume-prepare --config configs/smoke.yaml
ilume-smoke --config configs/smoke.yaml
```

正式单卡训练：

```bash
ilume-prepare --config configs/pretrain_base.yaml
ilume-train --config configs/pretrain_base.yaml
```

XLarge 只改变模型和训练参数，可直接复用 Base artifact：

```bash
CUDA_VISIBLE_DEVICES=0 ilume-train --config configs/pretrain_xlarge.yaml
```

训练器支持 AdamW、BF16/FP16 AMP、梯度累积、梯度裁剪、warmup+cosine、每 epoch 验证、checkpoint/resume、RNG/sampler 状态、stdout 和 JSONL 指标。验证使用固定 mask seed，关闭 modality dropout 与 asymmetric masking，并按验证集自然 role 分布报告总体及 role 分组指标。

Stage 1 的三个 CLI 命令在交互终端中使用 `tqdm`：prepare 分别显示输入加载、实体 QC、tokenizer、descriptor 和 shard 阶段，smoke 显示诊断 step，train 按 `Epoch x/y` 显示当前 epoch 的派生 optimizer step、总 loss、五项模态 loss、学习率和最近 validation loss。输出重定向或运行在非 TTY 作业系统时自动回退为逐步 JSON，其中包含 `epoch`、`epoch_step` 和内部 `global_step`；完整指标始终追加到 `metrics.jsonl`。

checkpoint v3 在 epoch 边界保存 model、optimizer、scheduler、AMP scaler、已完成 epoch、内部 global/micro step、Python/NumPy/PyTorch/CUDA RNG、sampler epoch、config 与 artifact 哈希。恢复时从下一个完整 epoch 开始：

```yaml
training:
  resume_from: artifacts/training/pretrain_base_bs512/checkpoint_epoch_00001.pt
```

旧 step-based 配置和 checkpoint v2 不支持隐式迁移；历史配置仅保存在 `configs/archive/` 供审计。若在 epoch 中途停止，恢复时会重新执行该未完成 epoch。

## 配置与消融

- `configs/smoke.yaml`：两步最小验证；
- `configs/train_test.yaml`：覆盖 prepare 后的训练、验证和 checkpoint 链路；
- `configs/pretrain_base.yaml`：5 epoch Base 正式配置，micro-batch 256；
- `configs/pretrain_large.yaml`：5 epoch Large 配置，micro-batch 256 并启用 gradient checkpointing；
- `configs/pretrain_xlarge.yaml`：5 epoch XLarge 配置，约218.79M参数、640维、10层，micro-batch 128、梯度累积2并启用 gradient checkpointing；
- `configs/ablations/`：Base reference 和九个单因素消融配置；
- `configs/archive/`：不可由当前 trainer 执行的历史 step 配置。

完整配置索引、artifact 复用规则和 OOM 回退方式见 [`configs/README.md`](configs/README.md)。

关键架构决定记录在 [`docs/adr/`](docs/adr/README.md)。

## 验证

```bash
pytest -q
ilume-prepare --config configs/smoke.yaml
ilume-smoke --config configs/smoke.yaml
ilume-train --config configs/train_test.yaml
```

当前正式训练器是单卡实现；不包含 DDP、TensorBoard 或自动实验矩阵调度。

## Stage 2 物性监督对齐

Stage 2 在保持 Stage 1 `forward(batch)` 不变的前提下，通过 `encode(batch)` 读取完整四模态 `fused_cls`。冻结教师的 CLS 先按 checkpoint 哈希写入缓存，训练时仅保留可训练学生：

```text
cation CLS + anion CLS + T ──> density / heat capacity / thermal expansion
molecule CLS ────────────────> 11项 QM 标签
solute CLS + solvent CLS ────> transfer free energy
student entity CLS ──────────> MSE(frozen teacher entity CLS)
```

Stage 2 原始输入固定为 `data/stage2/<task>/{train,valid}.csv`。现有 split 原样保留；温度和标签 scaler 只由训练行拟合。实体特征严格复用 Stage 1 tokenizer、descriptor schema/scaler 和 fingerprint 合同，不重新拟合。无效实体、受影响行和重复条件分别写入审计 CSV；重复 density 观测不会聚合。

完整准备会处理全量 Stage 2 数据并生成教师缓存，因此只在准备正式训练时执行：

```bash
CUDA_VISIBLE_DEVICES=0 ilume-stage2-prepare --config configs/stage2_base.yaml
```

Stage 2 reference 默认按20-step块执行35/20/15/15/15任务比例，使用标准化 SmoothL1 加实体 CLS MSE；正式 Base/Large/XLarge 对比保持有效batch 256。Base reference 可这样启动：

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_base.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage2/training/comparisons/base_reference
```

恢复时必须使用相同有效配置与输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_base.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage2/training/comparisons/base_reference \
  --resume-from artifacts/stage2/training/comparisons/base_reference/checkpoint_step_00001000.pt
```

Stage 2 checkpoint 与 Stage 1 checkpoint v3 是两个显式不同的格式。它保存学生、回归头、优化器、任务游标、RNG、早停状态及 data/teacher/checkpoint 哈希，可从任意保存 step 恢复。采样/容量对比矩阵以及保留原生进度条的单卡串行命令见 [`configs/README.md`](configs/README.md#stage-2-对比矩阵)；详细决定见 ADR-0007。
