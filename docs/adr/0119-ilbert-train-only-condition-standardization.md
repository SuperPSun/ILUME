# ADR-0119：ILBERT 独立训练集条件标准化配方

## 决定与范围

已接受（2026-11-08）。ILBERT 黏度 fold4 在相同参数复跑后重现异常；高压样本主导预测误差，原实现将原始物理单位条件直接拼接至预测头。增加独立的条件标准化配方，以验证输入尺度的影响；不宣称该改动已经改善性能，不改写原配方或历史结果。

- `training.condition_transform` 显式支持既有 `raw_physical_units` 与新增 `train_only_zscore`，只扩展 ILBERT；其他模型不变。
- 按注册表条件顺序，用每个任务、每折的训练行以 float64 拟合均值和总体标准差（`ddof=0`）。常量列的 scale 为1；无条件任务使用空统计量。
- 训练、验证、测试使用 `(x - mean) / scale`，送入模型时转为 float32。验证集和测试集不参与统计量拟合；五折测试集成中，每个模型分别使用自身训练折的统计量。
- 仅标准化数值条件。SMILES/AIS、截断、原生 RoBERTa/TextCNN/回归头结构、预训练资产、共享全量微调、目标标准化、seed42、Adam、恒定 `3e-5`、batch16、FP32+TF32及10轮末轮发布均保持原配方。此次不增加梯度裁剪、学习率搜索、早停或验证选模。

## 身份与兼容

新模式的输入合同为 `train_only_zscore_in_registry_order`，`condition_contract.transform` 和 `condition_statistics` 进入训练身份。统计量随完整训练身份保存在 `checkpoint.json` 的 `training_identity.payload.condition_statistics`，由身份自哈希保护。

评估先核对当前训练来源及完整身份，再读取 checkpoint 内的统计量转换评估条件；不在验证或测试行上重新拟合。预测 CSV 继续保存原始物理单位条件。新旧模式互相拒载，统计量篡改或来源变化同样拒绝。

原始物理单位路径不新增身份字段，不改变既有输入合同、张量结构或旧检查点格式。版本仍为 ILBERT checkpoint format2；新模式通过训练身份区分。已有标准化模型的训练审计也必须核对条件统计量，不能将新模式当作旧结果复用。

## 配置与输出

新增自包含配置 [ilbert_condition_zscore.yaml](../../configs/benchmarks/FLftv4_3407/ilbert_condition_zscore.yaml)，与原配置仅在展示名和 `training.condition_transform` 不同。沿用 ADR-0118 的10项变化实验任务、五折、无模拟协议；完整 sweep 为50折训练及对应评估，不代表完整22任务榜单。

独立输出根为 `outputs/benchmarks/FLftv4_3407/condition_zscore/ilbert/`。原 `retrain_22/ilbert/` 及其复跑 attempt 均保留，不覆盖或按验证表现择优拼接。正式采用标准化配方时，各任务五折须使用同一规则；旧原始条件结果不能冒充新配方。其余12任务及模拟的完整比较仍需另外制定配套配置和审计。

本次仅在4090修改、验证及交付；不自动同步远程仓库、不提交Git、不启动正式训练。3090拉取后的命令见[补训手册](../baseline-flft-22-retrain-runbook.md#ilbert-独立条件标准化配方)。

## 验证

临时小数据验证训练行独占拟合、注册表顺序、常量/空条件、验证及测试共用保存统计量、原始物理条件审计、检查点序列化/恢复、跨模式拒载及统计量篡改拒绝。完整 pytest、入口帮助、编译、Markdown链接与diff检查保留记录；正式模型表现由后续独立训练确认。
