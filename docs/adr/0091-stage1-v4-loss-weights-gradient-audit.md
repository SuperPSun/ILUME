# ADR-0091：Stage1-v4 损失权重与只读梯度审计

## 状态

已接受（2026-10-07）。仅修订 ADR-0089 的 RDKit/Uni-Mol 系数，并增加执行层面的诊断。ADR-0090 容量和其他数值合同不变；不新增容量配置。

[ADR-0092](0092-stage1-atom-charge-and-frozen-regression-heads.md) 后续将现役审计间隔从5,000改为1,000，并增加可选 CLI 覆盖和原子电荷目标/诊断。以下原始间隔及七目标表在这些新增范围内属于历史；只读审计隔离合同继续有效。

## 决策

后续 2026-10-07 Base 配方更新后，以下系数表属于历史。当前系数见 [Base YAML](../../configs/v4/stage1/base.yaml) 和 [v4 运行手册](../v4-runbook.md#stage1-loss-and-gradient-audit)；审计/归一化合同不变，包括 ADR-0092 规定的现役1,000次更新间隔。

本次决定时 v4 Base 的系数为 SMILES/原子/键=`1/1/1`、alignment=`0.1`、RDKit=`0.5`、Uni-Mol=`0.25`、electronic=`0.1`。共享结构定义默认值和历史 YAML 不变。损失仍先在分子内对有效元素取均值，再按角色2/2/1对分子均值加权。不改变采样器、编码器、模态 dropout、优化器、调度器或固定末轮规则。

`training.gradient_audit_interval_steps` 默认0（关闭）；`gradient_audit_batch_size` 默认32。序列化时省略默认值。启用审计要求双视图 v4。本次 Base 显式设定每完成5,000次优化器更新，用32个验证分子审计。两字段仅属执行设置：记录在运行/检查点配置中，不进入科研配置 hash、训练身份和编码器身份。在轮边界恢复时可以改变审计频率或探针大小，不可改变损失。

## 探针与梯度合同

- 使用独立本地 RNG，以 `data.seed+400000` 为种子，无放回选择 `min(32, validation_size)` 个分子；不按角色/性质/HF分层。复用现有打包器、缓存辅助目标及评估 masker，固定种子 `data.seed+400001`。样本顺序和 mask 在不同步、轮、尝试和 world size 下固定。记录 ID、语料/特征身份、mask hash 和探针 hash。
- 在优化器/调度器更新并清空梯度之后，仅在间隔的正整数倍执行；不额外添加末轮审计。采用 eval 模式与评估 masking（关闭普通及融合模态 dropout），使用训练 AMP 设置和 FP32 平方范数累加。
- 单次即时前向提供全部七个目标。各目标仅对互不重复的 SMILES/Graph/Fusion 参数调用 `autograd.grad`；辅助头不在求导参数内，但保留其对编码器梯度的链式贡献。不调用反向、不写入 `.grad`、不裁剪或更新优化器/调度器。下一目标计算前释放当前梯度集合。
- 原始范数为 `||∇encoder L_i||₂`：保留角色归一化，仅排除外层系数。加权范数为 `abs(lambda_i) * raw_norm`，无需再次反传。它们是各目标的梯度大小，不是向量和的范数，也不衡量目标间的对齐或冲突。
- DDP rank0 评估完整探针，不执行 DDP 前向或训练损失集合通信；其他 rank 通过同步错误/状态广播等待。保留所有模块模式标志及 Python/NumPy/Torch CPU/当前 rank CUDA RNG；不初始化或检查其他 CUDA 设备。正常错误传播到所有 rank；不因 OOM 缩小探针或静默跳过。

## 输出与恢复

追加写入 `gradient_audit.jsonl`，与保持不变的训练指标分开。每行包含轮、全局步、尝试 ID、训练身份、精度、探针元数据、当前系数，以及：

```text
smiles_grad_norm
atom_grad_norm
bond_grad_norm
alignment_grad_norm
rdkit_grad_norm
unimol_grad_norm
electronic_grad_norm
weighted_grad_norms.{smiles,atom,bond,alignment,rdkit,unimol,electronic}
coverage.<objective>.{valid_molecules,valid_role_weight_sum,status}
```

缺失监督记录 `null` 和 `no_valid_targets`，不记录会误导的零。有有效目标但导数为零时记录0。非有限范数明确视为审计错误，绝不静默改变权重或预算。尤其是自然抽取的小探针可能没有电子标签，此时无法诊断电子梯度强度。

恢复保留完整轮/尝试行为。不截断失败尝试的审计行，也不据此选择检查点；重放未完成轮可能追加同一步的另一条观察。审计设置无需升级检查点格式。验证仍仅用于报告：其导数不用于参数更新或自动决策。

## 产物边界与验证

语料、仅训练集统计及已完成 teacher 缓存仍可复用，包括已有失败 mask。新损失权重改变训练身份，因此新 Base 必须在新输出目录从头训练 Stage1，再重新生成/训练 Stage2，并数据准备/训练/评估 Stage3。历史输出只读；实现过程不启动正式运行。

测试将各目标梯度与独立参考比较，覆盖加权范数与缺失目标、开启/关闭审计时模型/梯度/优化器/调度器/RNG一致性、固定探针/mask、仅 rank0 的 DDP/错误传播、轮恢复和身份隔离。梯度审计仅为人工判断提供证据，不新增训练目标或选模机制。
