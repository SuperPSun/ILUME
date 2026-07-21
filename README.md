# ILUME multimodal pretraining

本项目为 cation、anion 和 neutral molecule 提供完全共享参数的三模态掩码预训练基线。输入为 Atom-in-SMILES、RDKit molecular graph 和固定顺序的 217 维 RDKit descriptors，融合后分别重构被遮蔽的 SMILES token、atom features、bond features 和 descriptor dimensions。

## 安装

```bash
python -m pip install -e ".[dev]"
```

SMILES 使用官方 [`atomInSmiles==1.0.2`](https://pypi.org/project/atominsmiles/)。PyPI metadata 将该版本标为 CC BY-NC 4.0，但同页长描述又包含 CC BY-SA 4.0 标记；上游许可口径不一致，将本项目用于商业或分发场景前必须向上游确认适用许可。

## 最小运行

```bash
ilume-prepare --config configs/smoke.yaml
ilume-smoke --config configs/smoke.yaml
pytest -q
```

默认 smoke 配置从每个 role 取 20 个实体并执行两个 optimizer step。将 `data.max_samples_per_role` 改为 `null` 可准备当前 Stage 1 的全部语料。

数据来源规则固定如下：

- cation：`data/stage1/cation.csv`
- anion：`data/stage1/anion.csv`
- neutral：合并 `molecule.csv`、`solute.csv`、`solvent.csv`
- `IL.csv` 明确不参与单分子预训练语料

准备命令在 `data.artifacts_dir` 生成：

- `corpus.pt`：token、graph、标准化 descriptor 和 split
- `manifest.csv`：sample、role、canonical SMILES、split 和来源
- `tokenizer.json`：仅用训练集拟合的 AIS vocabulary
- `descriptor_scaler.json`：仅用训练集拟合的均值、scale、有效计数和固定名称顺序
- `metadata.json`：工件格式、RDKit/AIS 版本和数据来源规则

## 模型与 masking

融合序列为 `[MM_CLS] + SMILES + atoms + bonds + descriptor`。SMILES Encoder 内部加入普通 sequence position embedding；graph atom/bond token 在 D-MPNN 和 Fusion 中都不加入绝对位置编码。`FusionLayout` 显式保存各 token 的位置映射。

默认动态 mask ratio 均为 0.15。SMILES 使用 BERT 80/10/10 替换；atom 和 bond 使用可学习的整特征 mask vector；descriptor 输入为 masked value 与 indicator 的拼接。自然产生的非有限 descriptor 值置零并标记 unavailable，不进入重构 loss。

三种 modality dropout 默认均为 0.1，并保证每个样本至少保留一个模态。被丢弃模态保留重构槽位，全部可用内容都作为重构目标。可通过配置启用 asymmetric masking。

## 当前边界

当前 runner 用于可重复的最小 forward/backward 验证，不包含完整 epoch 调度、checkpoint/resume、TensorBoard 或 DDP。模型、batch、sampler 和工件接口彼此独立，可在不改变 `MultimodalPretrainModel.forward(batch)` 的情况下扩展正式训练器。
