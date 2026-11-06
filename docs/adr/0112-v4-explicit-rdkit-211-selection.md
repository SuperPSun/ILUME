# ADR-0112：v4 Base 显式选择 211 维 RDKit 描述符

## 状态与决定

后续修订：[ADR-0113](0113-v4-descriptor-validity-ring-log-review.md) 为现役Base增加Phi适用性mask与八项环计数log1p；本篇“不做对数变换”的数值处理保留为上一版历史，211列选择和宽度继续有效。

已接受（2026-11-02）。根据 ADR-0110 的全量 train 审计和用户审阅，现役 `configs/v4/stage1/base.yaml` 在原始 217 列中仅排除 `Ipc`、`Kappa3`、`SMR_VSA8`、`SlogP_VSA9`、`fr_Nhpyrrole`、`fr_amide`。前两列为重尾，后四列为零方差或完全重复；最终保留顺序沿用原始 RDKit 顺序。

保留存在少量非有限值的四项 partial-charge 和八项 BCUT2D 描述符，也保留审计中标记的 17 项高相关描述符。本决定不自动采用审计工具的 182 列候选，不做截尾、对数变换、重新采样或分子删除，不声称该选择已证明可改善下游性能。原独立审计工具于2026-11-06退役（ADR-0110）；该退役不修改本篇的211列选择，后续人工建议也不自动接入schema。

## 模型与数值边界

- `data.descriptor_dim` 和 RDKit 原始名称清单仍为 217。新增 `descriptor.excluded_names`，默认空且序列化省略；只允许 v4 双视图的 `full` 模式，拒绝未知、重复或排除全部列的配置。
- 统一 `DescriptorSchema` 同时决定 Stage1 RDKit 辅助目标和下游显式描述符的名称、顺序与宽度，不分别筛选。Base 的描述符预测头输出从 217 改为 211；learned1024、atom512、编码器结构及其他辅助头不变。
- 归一化只用 train 的有限值拟合。非有限项继续在标准化后填零，并使用逐维有效 mask 排除其监督；loss 继续先做分子内有效维均值，再用角色 2/2/1 归一化。下游显式输入沿用原有标准化零填充，不新增 mask 特征。
- 实体宽度为 `1024+211=1235`，双槽位、两组三维角色及两个有效位的 L1 输入为 `2×1235+8=2478`。Stage2/3 从来源 schema/identity 解析宽度，按同一宽度构造 GLOBAL/GROUP L1 输入投影、缓存、prepared slots、推理路径和 resolved plan；不改变专家数量、隐藏宽度、owner、LR、预算、loss 或训练顺序。
- 未配置排除的历史配置继续使用 217/1241/2490，默认序列化、状态键及身份计算保持原样。ChemBERTa 固定补零表示仍使用其原 1241 维合同；baseline 和历史消融配置不批量改写。

## 身份与运行边界

排除列表进入 corpus/训练配置身份，保留列及统计进入 feature identity。非默认实体宽度进入冻结来源身份、Stage2/3 输入合同和训练计划。加载严格校验 schema、宽度、状态形状与来源 hash；217 与 211 不能交叉恢复，也不能手改 metadata/SHA 来兼容。沿用现有 artifact 格式，无须仅为该配方增加版本号。

三个正式入口仍为 `configs/v4/stage*/base.yaml`。需要重新 prepare Stage1、生成匹配新 corpus 身份的 teacher/电荷 sidecar、从头训练 Stage1，继而重新 prepare/train Stage2 和 prepare/train/evaluate Stage3。原始数据、split authority 与 Uni-Mol 源权重可复用。虽然 teacher 数学输入没有新增描述符，当前 teacher 缓存绑定 corpus 身份，仍不得直接混用旧缓存。

既有全 217 维 prepared、teacher、审计输出和模型保持只读；本次不移动、删除、覆盖或重标记。使用标准输出路径前须由用户另行确认归档/清理，或使用新目录并同步下游来源路径。实现只运行临时小数据测试，不启动正式准备、缓存或训练。

## 验证

覆盖排除名称与顺序、配置 round-trip 和默认省略、211 维目标及缺失 mask 梯度、编码器参数不变、217/211 来源拒绝；小数据贯通 Stage1 训练/导出及 Stage2 冻结表示与 final 加载。双宽度测试验证 1241/2490 和 1235/2478 的 GLOBAL/GROUP 输入、角色/槽位、梯度与错误宽度拒绝。既有数据 catalog 缺项和历史配置缺失单独报告，不放宽合同。
