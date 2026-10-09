# ADR-0092：Stage1 原子电荷监督与独立冻结回归头

## 状态

已接受（2026-10-07）。为 v4 预训练增加部分原子电荷，将 ADR-0091 现役审计间隔改为1,000次更新，并增加显式选择的后训练命令。编码器架构、输入/输出、自然打乱及下游冻结决定均不变。

## 预训练与审计

以下预训练系数描述本 ADR 的原始配方。后续 2026-10-07 Base 修订见 [Base YAML](../../configs/v4/stage1/base.yaml) 和 [v4 运行手册](../v4-runbook.md#stage1-loss-and-gradient-audit)。目标定义、sidecar 与审计边界不变；以下独立冻结头配方保留自身 batch128 预算。

本次 Base 增加 `lambda_partial_charge=0.1`，独立于 electronic0.1；重建1/1/1、alignment0.1、RDKit0.5及Uni-Mol0.25不变。可丢弃的 `Linear512→1` 用当前图原子状态预测归一化电荷，复用同一次掩码前向。SmoothL1 先在分子内对原子取均值，再按角色2/2/1对有效分子均值加权。只有图编码器接收此目标的表示梯度。缺失分子返回可微零。不增加描述符/角色输入、额外前向或改变采样器。

共享系数默认0，序列化时省略；关闭监督不创建原子头，也不额外调用初始化 RNG。历史配置/状态/数值身份不变。仅 v4 可启用此监督，增加513个辅助参数，不扩增编码器容量。v4 kind/format4保持不变；损失及标签 sidecar 身份拒绝不兼容恢复。完整轮检查点保留所有头；仅编码器导出仍只包含 SMILES 编码器、图编码器和 Fusion。

Base 审计间隔为1,000；`--gradient-audit-interval-steps` 可覆盖 YAML，含0表示关闭。审计设置仍仅属执行参数。启用原子监督增加 `partial_charge_grad_norm` 及其系数、加权范数和覆盖率。此前七字段、固定32个验证探针、mask、模式/RNG恢复、仅 rank0 求导及同步错误不变。无有效原子目标记录 null；有效但导数为零记录0。

## 仅训练来源的原子标签 sidecar

预训练标签仅来自 partial-charge 训练集行及其引用并通过验证的 MOL2 结构。复用现有带类型图同构/连通性回退、氢处理策略及确定性映射合同，通过 `common.atom_targets` 供 Stage1/2 共享；保留原 Stage2 公开导入及映射身份命名空间。

共享 MOL2 解析器仅识别已核验的来源头别名：`MOLEMOLE/MOLMOLLE/MOMOLULE/MMOLCULE/MOLECMOL→MOLECULE`、`MOLM/AMOL→ATOM` 和 `MOLD/BMOL→BOND`。也接受已核验前缀 `@<TRMOLS>`、`@<TRIMOL>`、`@<TMOLOS>` 和 `@<MOLPOS>` 替代 `@<TRIPOS>`。不做模糊纠错。不改写原始字节或资源 SHA；原子/键计数、数值、引用及映射校验保持严格。Stage1 电荷来源身份包含完整段名/前缀别名表，因此旧 sidecar/检查点不能静默复用不同的已接受监督集合。标准段解析及图映射算法不变。未消费的 SUBSTRUCTURE 头拼写错误无需别名。

Stage1 仅跳过已验证/解析但两次图同构尝试均失败的记录。这些记录不进入电荷目标或归一化，但其语料分子与其他目标不变。`mapping_audit.json` 记录已映射/跳过状态、split、CSV行号、mol_id、canonical SMILES、资源文件名/SHA及 `no_graph_isomorphism` 原因。元数据报告尝试、保留、跳过的观察数。资源缺失、SHA/大小不匹配、解析错误、无效 SMILES 和角色不匹配仍致命失败。显式跳过策略进入电荷来源身份；旧 sidecar/检查点不能交叉恢复。共享 Stage2 调用方仍通过兼容 ValueError 的类型化异常拒绝无同构记录；不改变 Stage2 映射或选择。

独立电荷头训练对训练集/验证集使用相同策略，并在输出根写入 `partial_charge_train_mapping_audit.json` 和 `partial_charge_valid_mapping_audit.json`。不读取测试集；保留的训练集为空仍报错。

精确关联语料中已有的 canonical 结构，不沿种子/augmentation 祖先关系关联。保留每条来源 CSV 行及来源 ID 作为独立电荷观察，包括有冲突的 canonical 重复行；绝不平均标签或选择首/末行。对匹配 Stage1 训练结构的全部观察拟合电荷均值/总体标准差（零方差→1）；匹配验证集 split 的标签仅用于报告。语料分子只编码一次；每条观察独立对原子损失取均值，并在电荷目标中按角色加权。其他目标及语料采样不变。sidecar format2和显式观察策略进入身份，拒绝旧 sidecar/检查点。没有匹配训练原子时，mask掉全部电荷监督并报告覆盖率。

独立 partial-charge 头训练同样把来源行保留为独立样本，复用同一冻结原子表示。所有目标仍拒绝 canonical 训练集/验证集重叠。保留观察不能消除映射歧义，也不增加构象输入：相同结构输入无法区分来源构象。

对于 partial charge，现有梯度审计覆盖字段 `valid_molecules` 统计独立监督观察，而非唯一结构。sidecar的 `matched_molecules`/`source_molecules` 统计保留的唯一 canonical 结构；`source_observations` 统计成功映射来源行，scaler的 `observation_count` 统计匹配训练观察。`attempted_observations` 包含跳过行；`skipped_observations` 统计同构失败行。

独立 sidecar 绑定语料/清单、训练集 CSV、结构清单及引用 MOL2 的 SHA/大小、映射、统计和目标状态hash。保存归一化目标与映射审计；来源/语料损坏、不完整或变化立即失败。不改写语料/统计/teacher缓存。普通 CLI prepare也准备独立受管 sidecar 运行；`--partial-charge-only` 从已有语料准备，不重建五百万分子特征。现役缓存路径指向该运行的 `artifacts/` 目录。启用时训练必须有有效 sidecar。dataset/packer按实际原子数拼接不等长标签和 mask。

## 独立冻结头训练

[ADR-0093](0093-stage1-configurable-frozen-predictors.md) 为本节扩展可配置 MLP/残差 MLP 预测器及 format2 产物；以下 Linear 初始化和数据/冻结/优化边界仍有效。

`scripts/stage1/regression.py` 必须显式运行，不在预训练后自动调用。接受完整末轮 v4 预训练检查点、自包含回归 YAML、新输出、可选目标及仅执行用途的设备。不接受仅编码器导出或未完成轮。默认目标为 HOMO/LUMO、十一项 HF 标量及部分原子电荷；`q_max/min/std/pos_frac` 是标量摘要，不是原子电荷。

Encoder与Fusion冻结并设为eval。无mask地一次编码完整模拟训练集/验证集结构，不使用模态dropout或描述符输入；缓存learned实体/原子状态并绑定来源及张量 hash。无需teacher推理。各目标有独立预测器优化器。Linear从原电子头行或原子头初始化；非线性按ADR-0093采用任务局部随机初始化。旧v4检查点可选择已有标量任务；缺失原子头报错，绝不以随机头替代。

各任务后训练归一化仅用完整训练集拟合。通过目标坐标的精确仿射变换转换初始头权重/偏置，在浮点容差内保留原单位预测。默认10轮原始轮、batch128、AdamW1e-4、betas0.9/0.999、eps1e-8、WD0.01、裁剪1、BF16、恒定LR。保留角色2/2/1分子均值；原子loss/metrics在分子内平均。仅训练集更新头；来源验证集在初始化和每轮报告MAE/RMSE；绝不读取测试集。canonical 训练集/验证集重叠或冲突的标量电子标签报错；partial-charge观察分别保留。缺失标量标签跳过，空训练集失败，空验证集指标为null。不论验证表现均发布固定末轮。失败头运行在新输出重启；不支持中途恢复或覆盖。

输出包含冻结表示库、逐任务指标、`regression_head.pt/json` 及最终汇总。独立kind/身份绑定基础检查点SHA、基础训练身份、编码器 hash、来源训练集/验证集 hash、特征身份、归一化、任务/预算/种子及最终头hash。加载检查来源SHA、自身hash和产物/状态/清单一致性。不更新原检查点、仅编码器产物、其他头或下游Stage2/3；回归产物不是Stage1编码器替代品或正式Stage3报告模型。

## 验证与产物边界

测试覆盖可选审计CLI、关闭时的历史结构定义/状态/初始化、分子/角色加权、缺失标签、原子梯度、CPU/CUDA BF16/DDP审计隔离、来源/映射/hash拒载、语料/teacher身份不变、末轮恢复和仅编码器导出。后训练测试验证仿射初始化、冻结/非目标状态、训练集/验证集/测试集边界、恒定LR、来源绑定，以及人为恶化验证结果不能改变最终头状态。

新的电荷预训练需要新的Stage1和下游Stage2/3训练输出；准备sidecar后已有语料及teacher缓存仍可复用。仅独立冻结头训练无需重跑下游。实现过程不执行正式prepare、teacher生成、训练或后训练。本地电子/电荷来源缺失须明确视为数据问题，不回退或放宽校验。
