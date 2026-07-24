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

`MultimodalPacker` 只做确定性组批；`MultimodalMasker.apply(batch, global_step, max_steps)` 执行动态 masking、asymmetric masking 和 modality dropout。curriculum 默认行为为：

- 0%–10% 训练进度：dropout 率为 0；
- 10%–60%：线性升至配置概率的一半；
- 60%–100%：线性升至配置概率。

四个顶层模态独立 dropout，但每个样本至少保留一个模态。普通 atom masking 不遮蔽单原子分子；graph modality dropout 仍会整体遮蔽单原子并监督重建。

模型包含 SMILES MLM、atom、bond、descriptor 和 fingerprint 五项重建 loss。graph head 支持 `linear | mlp`；MLP 版本使用共享 residual trunk 后接字段独立分类器。可选共享 role embedding 会加到 CLS 和所有非 padding fusion token 上，不进入各模态 encoder。

## 45/45/10 覆盖采样

正式配置固定全训练运行的全局抽样比例：cation 45%、anion 45%、molecule 10%。比例不要求每个 mini-batch 精确满足。role 内先无放回遍历全部样本，再重洗牌进入下一轮；sampler 状态可随 checkpoint 恢复。

给定 `max_steps` 时：

```text
total_draws = max_steps * batch_size * gradient_accumulation_steps
minimum_draws = max(ceil(N_cation/0.45), ceil(N_anion/0.45), ceil(N_molecule/0.10))
```

若最大余数法得到的任一 role 配额小于该 role 训练池，训练在启动时失败，并给出满足覆盖保证所需的最小 `max_steps`。

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

训练器支持 AdamW、BF16/FP16 AMP、梯度累积、梯度裁剪、warmup+cosine、固定频率验证、checkpoint/resume、RNG/sampler 状态、stdout 和 JSONL 指标。验证使用固定 mask seed，关闭 modality dropout 与 asymmetric masking，并按验证集自然 role 分布报告总体及 role 分组指标。

checkpoint 保存 model、optimizer、scheduler、AMP scaler、optimizer/micro step、Python/NumPy/PyTorch/CUDA RNG、sampler 进度、config 与 artifact 哈希。恢复时在配置中设置：

```yaml
training:
  resume_from: artifacts/training/base/checkpoint_step_00001000.pt
```

## 配置与消融

- `configs/smoke.yaml`：两步最小验证；
- `configs/train_test.yaml`：覆盖 prepare 后的训练、验证和 checkpoint 链路；
- `configs/pretrain_base.yaml`：正式参考配置；
- `configs/pretrain_large.yaml`：更大模型并启用 gradient checkpointing；
- `configs/legacy.yaml`：Full/1-token/AIS/无指纹/无 role embedding/linear head；
- `configs/ablations/`：可执行参考模板和逐轴字段说明。

关键架构决定记录在 [`docs/adr/`](docs/adr/README.md)。

## 验证

```bash
pytest -q
ilume-prepare --config configs/smoke.yaml
ilume-smoke --config configs/smoke.yaml
ilume-train --config configs/train_test.yaml
```

当前正式训练器是单卡实现；不包含 DDP、TensorBoard 或自动实验矩阵调度。
