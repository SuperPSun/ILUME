# ILUME

## 项目定位

为 cation、anion 和 neutral molecule 提供共享参数的三模态掩码预训练基线，模态为 Atom-in-SMILES、RDKit molecular graph 和固定顺序的 217 维 RDKit descriptors。

## 运行与验证

```bash
python -m pip install -e ".[dev]"
ilume-prepare --config configs/smoke.yaml
ilume-smoke --config configs/smoke.yaml
pytest -q
```

默认 smoke 配置每个 role 取 20 个实体并执行两个 optimizer step。修改实现后至少运行相关测试；涉及端到端数据流或模型接口时同时运行完整测试和 smoke。

## 技术栈

- Python 3.11+
- PyTorch、RDKit、Atom-in-SMILES、NumPy、PyYAML
- setuptools `src/` layout，pytest 测试

## 目录与约定

- `src/ilume_pretrain/` 是实现真身，`tests/` 是行为验证。
- `configs/smoke.yaml` 是最小可重复运行配置。
- Stage 1 输入位于 `data/stage1/`：cation/anion 各用同名 CSV，neutral 合并 molecule/solute/solvent；`IL.csv` 不进入单分子预训练语料。
- `data/`、`artifacts/`、缓存和 `*.egg-info/` 是本地输入或生成物，不提交为源码，也不在无关任务中重建或删除。

## 当前状态与下一步

当前只提供可重复的语料准备和两步 forward/backward smoke runner；尚无完整 epoch 调度、checkpoint/resume、TensorBoard 或 DDP。保持 `MultimodalPretrainModel.forward(batch)` 接口稳定；只有任务明确要求时再扩展正式训练器并补相应测试。
