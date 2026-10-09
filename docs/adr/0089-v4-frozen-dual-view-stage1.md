# ADR-0089：v4 冻结双视图 Stage1 与解耦 HoME 学习

## 状态

已接受。仅在当时新增 `configs/v4/` 主线表示和Stage1冻结边界范围内替代ADR-0039/0082。v3、历史实现和Capacity产物/配置保留原合同，不迁移或重标身份。

[ADR-0090](0090-stage1-v4-residual-encoder-capacity.md) 后续将编码器容量改为十二层SMILES和八个独立残差图块。该纯架构修订复用已有v4语料/统计/teacher缓存；以下其他决定不变。

[ADR-0091](0091-stage1-v4-loss-weights-gradient-audit.md) 后续将当时RDKit/Uni-Mol系数改为0.5/0.25，并增加只读梯度审计。以下原始系数表的这两项属于历史；目标和归一化不变。

[ADR-0092](0092-stage1-atom-charge-and-frozen-regression-heads.md) 增加仅训练来源的部分原子电荷、可丢弃原子头及独立冻结回归头后训练。已有语料、编码器维度及下游冻结边界不变；以下五类目标列表早于这项原子监督。

## 背景

后续2026-10-07 Base配方采用全局batch512及更新损失系数；当前值由 [Base YAML](../../configs/v4/stage1/base.yaml) 和 [v4运行手册](../v4-runbook.md#stage1-loss-and-gradient-audit) 定义。以下系数/batch值描述原始决定，不是当前运行配方。架构、目标定义及下游永久冻结不变。

Stage1应学习与任务无关的分子结构；Stage2/3优化多任务HoME而不改变Stage1编码器。这是完整配方比较，不能归因于单一机制：架构、辅助监督、10轮预算及永久冻结同时改变。

## 决策

- Stage1独立编码SMILES CLS和图原子状态均值（各512D），不输入角色或描述符。残差MLP为 `LN(x + Linear1024(GELU(Linear1024(x))))`；learned输出1024D，原子状态512D。不使用token融合、cross-attention、指纹或专家。
- SMILES/原子/键结构重建系数各为1。alignment采用独立 `512→512→256` 投影层和 `256→128→256` 预测器（隐藏层LayerNorm/GELU），双向负余弦并detach对侧投影；不使用负样本或EMA。RDKit217 MSE、Uni-Mol768余弦和带mask的13目标电子SmoothL1仅作辅助；alignment/RDKit/3D/electronic系数为 `0.1/1/0.1/0.1`。
- 各目标先在分子内对有效元素/目标取均值，再计算 `sum(role_weight * molecule_loss) / sum(valid_role_weight)`。阳离子/阴离子/中性角色为**损失权重2/2/1**，不是采样概率。缺失目标返回可微零；DDP全局汇总有效分母并补偿梯度平均。
- 保留完整语料轮打乱及成员/QC/split。不做角色平衡、重复或性质加载器。仅Fusion采用80/10/10双视图/仅SMILES/仅图dropout；局部结构/alignment状态仍计算两个编码器。
- Stage1固定10轮、全局batch128、AdamW1e-4/WD0.01、5%预热/现有余弦调度、BF16、裁剪1、完整轮恢复。训练检查点保留可丢弃头；`stage1_encoder.pt` 仅导出两个编码器及融合层、最终状态hash和来源身份。
- 电子目标精确按canonical SMILES关联模拟**仅训练集**：HOMO/LUMO PBE/TZVP及十一项HF目标。统计仅用匹配的唯一Stage1训练结构。不用模拟验证集/测试集标签，不沿augmentation 种子传播标签。RDKit统计也仅拟合Stage1 训练集。
- Uni-Mol2 84M仅离线运行于独立 `unimol_tools==0.1.3.post1` 环境。必须有本地 `modelzoo/84M/checkpoint.pt`，不隐式下载。尝试完整语料中每个唯一canonical结构；由种子确定的ETKDGv3，优先MMFF否则UFF；缺少力场参数须审计为失败。保留电荷/立体化学。不支持/失败分子保留其他监督并mask掉3D；绝不替换为2D/零坐标或裁剪原子。审计失败、来源检查点SHA、teacher/RDKit版本、语料身份、排序结构索引和分片 hash。严格按分片恢复；完整生成前通过独立 `--limit/--output` 审计报告吞吐、成功率和存储。部分审计不能用于Stage1训练。
- Teacher的 `--workers` 仅为执行参数（默认1）：有界spawn CPU进程池按排序顺序准备构象/特征，仅父进程负责GPU推理。特征只计算一次。不改变缓存配方、分子种子、batch边界、失败mask或分片恢复合同。非TTY运行在处理中和分片提交时刷新进度JSON；同一缓存根只能有一个写入者。
- 下游拼接冻结learned1024和标准化RDKit217，得到1241。ObjectEncoder独占 `Linear1241→1024 + LayerNorm`，之后接原有1024D transformer/HoME接口。旧默认输入1024不创建投影，也不额外消耗RNG。Stage1在Stage2/3始终eval、无梯度、不进入优化器。
- Stage2 prepare缓存冻结实体/原子张量，绑定原编码器hash。九任务仅物理监督 HoME仍采用逻辑/微batch256、固定10轮，ObjectEncoder含入口投影的LR为3e-5。partial-charge用冻结512原子状态及1024上下文。严格校验状态/owner/shape hash后迁移GLOBAL和匹配thermophysical/solvation GROUP。
- Stage3保存冻结1241实体槽位；仅ObjectEncoder在Phase1以1.5e-5适配15轮，之后冻结并物化最终状态绑定的表示缓存。当时实验/PRIVATE配方、二十任务合同（含hydration）、Phase2/3五项模拟辅助、原始样本暴露、`weighted_owner_raw_v1`、裁剪、分支锚点、拼接及验证仅报告均保留；不使用PCGrad。
- 版本隔离的核心消融：w/o Stage1为固定种子随机冻结同结构编码器加显式描述符；w/o Stage2为配对零更新ObjectEncoder及二十实验任务；w/o Stage3-HoME为正式准备产物冻结表示及既有末轮单任务MLP。后两项也省略模拟辅助训练，因此不是纯路由/权重效应。
- v4语料/训练/编码器格式及缓存身份独立；Stage2 final/编码器有独立kind和format4。Stage3 准备产物 kind/format4、final/delta/full kind/format4、resolved-plan11和训练 contract15拒绝v3交叉恢复。单任务MLP保留注册的训练后端/配方，但合同为v4时使用v4 检查点 kind/格式和状态命名空间。旧路径和默认行为不变。

## 后果与验证

需要新的v4 Stage1、teacher缓存、Stage2和Stage3输出；可复用来源数据和split，不能复用旧准备产物/检查点。基线代码不变；复用要求比较身份兼容。实现不下载权重、不生成正式缓存或启动正式训练/评估。真实teacher GPU兼容及小样本成功率/吞吐是用户完整生成的前提；合同测试使用显式注入的假teacher，不是科研替代品。

回归测试覆盖描述符/角色无关编码、分子加权/DDP损失、缺失标签、fusion dropout、缓存teacher失败/完整性、末轮导出/轮恢复、永久冻结Stage1、可训练ObjectEncoder投影、含partial charge的九任务推理、来源/Stage3 final加载及旧合同回归。命令和环境隔离见 [v4运行手册](../v4-runbook.md)。
