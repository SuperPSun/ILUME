# ADR-0113：v4 描述符适用性、环计数变换与严格复审

## 状态与范围

已接受（2026-11-03）。定向修订 ADR-0112 的数值处理，保留原始217列及显式211列选择；不新增删列，不改变分子成员、split、自然shuffle、角色权重、loss系数、编码器、专家容量或训练预算。此配方基于训练集审计提出，尚未证明会改善下游性能。

现役 Base 显式设置 `descriptor.value_policy: phi_mask_ring_log1p_v1`。共享默认 `raw`，默认序列化省略；旧配置、schema format2和旧身份继续按原值处理。baseline、ChemBERTa及历史消融配方不批量改写。

## 正式预处理

1. 继续排除 `Ipc/Kappa3/SMR_VSA8/SlogP_VSA9/fr_Nhpyrrole/fr_amide`。保留四项partial-charge、八项BCUT2D及此前17项高相关描述符。数据QC不因本次调整删除分子，不自动删除自由基、多片段或大宏环。
2. `Phi` 仅在**单个连通片段、重原子数至少3、键数至少2且结果有限**时有效。这是明确的工程适用性规则，不是对全部其他结构作化学错误判定。不适用时只将该项设为无效：它不进入scaler拟合、不参与该分子的描述符loss；其他监督照常使用。
3. 下列八项环计数统一变为 `log1p(x)`：`RingCount`、`NumAliphaticCarbocycles`、`NumAliphaticHeterocycles`、`NumAliphaticRings`、`NumHeterocycles`、`NumSaturatedCarbocycles`、`NumSaturatedHeterocycles`、`NumSaturatedRings`。有限的负计数视为合同错误并失败。零仍为零，合法大值不因超过阈值而屏蔽；不再叠加截尾、log变换其他列或修改loss。
4. 按上述处理后的train有限值拟合均值/总体标准差；valid/test只应用同一统计。缺失项标准化后填0，Stage1继续逐维mask、分子内有效维均值及角色2/2/1归一化；下游显式输入使用同一标准化零填充，不增加mask特征。

原始计算、结构适用性判断和列选择/变换由Stage1公开函数与同一DescriptorSchema提供。prepare矩阵只预先应用Phi适用性mask，scaler分块选择时应用一次log1p；分片和Stage2/3样本构造同样从原始值经该schema处理，禁止双重取log或在下游重新拟合。learned1024、atom512、211维监督、1235维实体和2478维双槽L1输入均不变。

新policy使用schema format3并绑定policy名称；未知policy/不一致来源拒绝加载。descriptor配置进入corpus/training身份，schema与scaler SHA进入feature及下游来源身份；维度相同也不能混用未经变换的旧211维产物。沿用Stage训练/checkpoint格式，通过现有来源与状态校验严格隔离。

## 只读复审工具

现役 `configs/v4/stage1/descriptor_selection.yaml` 设置 `audit_mode: prepared_review`，只接受具有新policy、SHA及corpus来源一致的已完成prepared。

- 仍只遍历train，保留未改动的float64 `raw_descriptors.npy`（217列），另存 `processed_descriptors.npy`（schema选择及处理后的211列）。不从标准化float32逆变换，不读取valid/test目标。
- 在处理后的分布上严格使用 `std/IQR > 10`、`max_abs/P99(abs) > 1000`、单条记录占总体平方偏差 `> 0.05`。任一命中标记review；IQR或绝对值P99为零时跳过相应比值，不因IQR零而删稀疏特征。
- 非有限比例 `> 0.01`、共同有效样本至少100且 `abs(Pearson)>0.98`也只标记review。全无效、平方数值不安全、零方差与完全重复列仍给出drop候选。任何候选均不自动改写正式schema，17项高相关列和12项部分缺失列保持原选择。
- `descriptor_statistics.csv`记录处理后的统计、单条方差贡献、keep/review/drop；`descriptor_selection.json`保留正式schema的排除记录及复审候选，非自动加载的训练产物。新review输出format2，独立审计身份纳入mode、阈值、schema SHA和policy。
- `descriptor_groups.json`按role×原始/扩增记录有效数量、Phi适用性屏蔽数及相对全train中心的平方偏差贡献；为九项关注列及仍有重尾/单条支配信号的列保存标准化绝对值最高20条的结构、sample ID与seed字符串。seed聚合只覆盖这些极端记录，不声称是全语料结构族统计，不改变采样或权重。
- 共享默认 `legacy_selection` 保留ADR-0110原始审计算法和默认序列化；不重写旧审计结果。

## 证据与解释边界

2026-11-03只读复核的1298个不同结构全部能解析，九项关注描述符全部复现。Phi最大值来自无键的`[O].[O].[O]`；环计数>100的1187条记录中1164条为扩增。新版RDKit的对称环枚举在部分宏环上产生组合增长，不等价于独立环数；不自动回退legacy环算法。详见独立审计输出，不把本次工程处理等同于化学审批或下游性能结论。

## 运行与验证

需要在新输出重做Stage1 prepare、匹配corpus的teacher及电荷sidecar并从头训练，再重新prepare/train Stage2和prepare/train/evaluate Stage3；旧输出保持只读，不手改SHA或迁移缓存，不自动归档/删除。审计必须在新prepared完成后运行，原始数据、split与Uni-Mol源权重可复用。实现阶段仅临时小数据，不启动正式准备、全量审计或训练。

验证覆盖Phi结构边界/单项mask、八列log1p恰好一次、train-only统计、统一下游样本构造、schema format2/3兼容及新旧policy拒绝、真实loss缺失梯度、完整轮恢复和只读梯度审计。复审覆盖阈值边界、重尾/缺失/相关不自动删除、原始与处理后矩阵、role/扩增/seed报告、串并行一致及SHA损坏拒绝；旧配置和消融不随新Base数值处理迁移。
