# ILUME

## 项目定位

为 cation、anion 和 molecule 提供共享参数的四模态掩码预训练，以冻结教师 CLS 支持五任务 Stage 2 物性监督对齐，并在冻结 Stage 2 表示上提供 Stage 3 的27任务训练。Stage 3 将21项 IL 任务与6项非 IL 辅助任务分成可训练状态完全隔离的两个域。

## 运行与验证

```bash
python -m pip install -e ".[dev,tokenizers]"
ilume-prepare --config configs/smoke.yaml
ilume-smoke --config configs/smoke.yaml
pytest -q
```

涉及正式训练器时还应运行 `ilume-train --config configs/train_test.yaml`；该配置会验证 45/45/10 覆盖、validation 和 checkpoint。修改 checkpoint/resume 时必须额外验证从已保存 checkpoint 恢复。Stage 2/3 使用临时小数据测试 prepare/train/resume/evaluate 链路；除非用户明确要求，不运行全量教师缓存、正式 Stage 3 prepare、训练矩阵或 test ensemble。

## 技术栈

- Python 3.11+
- PyTorch、RDKit、Atom-in-SMILES、NumPy、PyYAML
- 可选 Hugging Face tokenizers、SmilesPE、固定 commit 的 APE tokenizer
- setuptools `src/` layout，pytest 测试

## 目录与约定

- `src/ilume_pretrain/` 是实现真身，`tests/` 是行为验证。
- `configs/smoke.yaml` 是最小路径；`pretrain_base.yaml`、`pretrain_large.yaml`、`pretrain_xlarge.yaml` 是正式 profile；现役消融配置集中在 `configs/ablations/`。
- 原始输入只读取 `data/stage1/{cation,anion,molecule}.csv`，扩增只读取 `data/stage1/augmentation/` 同名三表。不得重新引入 `simulation_mol.csv`、`solute.csv`、`solvent.csv` 或 `IL.csv`。
- artifact v3 使用 shard 和索引；旧 v2 和单个 `corpus.pt` 均不得静默迁移，必须重新准备。
- 描述符计算前统一排除 BCUT2D 不支持键型、Ipc 非有限或平方上溢，以及 tokenizer 超过 256 token 的实体，并保留 `excluded_entities.csv` 审计。
- descriptor schema、scaler 和 tokenizer 只能由当前配置选中的训练集拟合；valid 不含扩增实体，并需隔离 valid seed 的扩增后代。
- 正式 sampler 在每个覆盖型 epoch 内使用 45% cation、45% anion、10% molecule，role 内必须先完成一轮无放回覆盖。
- 保持 `MultimodalPretrainModel.forward(batch)` 接口稳定。
- Stage 2 只读取 `data/stage2/<task>/{train,valid}.csv`，复用 Stage 1 预处理合同；首版只使用 T，不虚构 P。教师 CLS 必须由 checkpoint 哈希绑定的离线缓存提供，训练时不驻留冻结教师。v2 的三类 IL 在任务内按 cation/anion 体系采样，QM 支持部分缺失标签，训练前10%完整任务块只更新 PairEncoder 和独立回归器。
- Stage 2 v2 正式配置和串行命令以 `configs/README.md` 为准，artifact/output 位于 `artifacts/stage_v2/`，不得覆盖旧 `artifacts/stage2/`。正式配置保留 `shard_cache_size: 10`；首次完整准备后须按实际 shard 数复核，未经性能复核不得调低。
- Stage 3 v2 固定读取 Stage 2 Base reference 的 `best.pt`，训练时只使用离线冻结表示。`il21` 使用 HoME 和 late-solute，`aux6` 使用六个互不共享参数的独立 Head；两域的 optimizer、scheduler、AMP、RNG、BN、早停和选优状态不得共享。正式配置是 `configs/stage3_home.yaml`，优化矩阵入口是 `scripts/run_stage3_matrix.sh`，兼容旧 v2 training checkpoint 只能使用 `configs/stage3_home_legacy.yaml`。
- `.gitignore` 采用 artifact 黑名单：轻量 YAML/JSON/CSV/metrics/audit 可跟踪，模型、tensor、索引和终端日志等重载荷必须忽略。`data/`、重载荷 artifact、缓存和 `*.egg-info/` 不提交，也不在无关任务中重建或删除。

## 当前边界

项目已包含四模态 Stage 1、五任务 Stage 2，以及单阶段双域隔离的27任务 Stage 3。Stage 3 提供准备、单卡可恢复训练、五折汇总、test ensemble 和串行实验矩阵；不实现 DDP、TensorBoard 或并行实验调度，基线仍以独立配置一次改变一个轴。现役 Stage 3 决定以 ADR-0011/0012 为准，ADR-0010 仅保留为已取代的历史记录。
