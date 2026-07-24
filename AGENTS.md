# ILUME

## 项目定位

为 cation、anion 和 molecule 提供共享参数的四模态掩码预训练实现，模态为可替换 tokenizer 的 SMILES、RDKit molecular graph、经 schema 筛选和语义分组的 RDKit descriptors，以及 Morgan/MACCS fingerprints。

## 运行与验证

```bash
python -m pip install -e ".[dev,tokenizers]"
ilume-prepare --config configs/smoke.yaml
ilume-smoke --config configs/smoke.yaml
pytest -q
```

涉及正式训练器时还应运行 `ilume-train --config configs/train_test.yaml`；该配置会验证 45/45/10 覆盖、validation 和 checkpoint。修改 checkpoint/resume 时必须额外验证从已保存 checkpoint 恢复。

## 技术栈

- Python 3.11+
- PyTorch、RDKit、Atom-in-SMILES、NumPy、PyYAML
- 可选 Hugging Face tokenizers、SmilesPE、固定 commit 的 APE tokenizer
- setuptools `src/` layout，pytest 测试

## 目录与约定

- `src/ilume_pretrain/` 是实现真身，`tests/` 是行为验证。
- `configs/smoke.yaml` 是最小路径；`pretrain_base.yaml`、`pretrain_large.yaml` 是正式 profile；`legacy.yaml` 用于旧架构消融。
- 原始输入只读取 `data/stage1/{cation,anion,molecule}.csv`，扩增只读取 `data/stage1/augmentation/` 同名三表。不得重新引入 `simulation_mol.csv`、`solute.csv`、`solvent.csv` 或 `IL.csv`。
- artifact v3 使用 shard 和索引；旧 v2 和单个 `corpus.pt` 均不得静默迁移，必须重新准备。
- 描述符计算前统一排除 BCUT2D 不支持键型、Ipc 非有限或平方上溢，以及 tokenizer 超过 256 token 的实体，并保留 `excluded_entities.csv` 审计。
- descriptor schema、scaler 和 tokenizer 只能由当前配置选中的训练集拟合；valid 不含扩增实体，并需隔离 valid seed 的扩增后代。
- 正式 sampler 的默认全局比例是 45% cation、45% anion、10% molecule，role 内必须先完成一轮无放回覆盖。
- 保持 `MultimodalPretrainModel.forward(batch)` 接口稳定。
- `data/`、`artifacts/`、缓存和 `*.egg-info/` 是本地输入或生成物，不提交为源码，也不在无关任务中重建或删除。

## 当前边界

项目已包含四模态模型、分片准备、两步 smoke runner 和单卡可恢复正式训练器。当前不实现 DDP、TensorBoard 或自动消融 runner；消融实验通过复制 `configs/ablations/reference.yaml` 并一次改变一个轴完成。
