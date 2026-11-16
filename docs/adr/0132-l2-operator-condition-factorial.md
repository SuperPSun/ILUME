# ADR-0132：I2的L2交互算子与条件通路四格对照

## 状态与范围

已接受（2026-11-16，用户确认执行）。以[ADR-0127](0127-partner-injection-and-typed-encoding-ablations.md)的I2为参照，仅比较L2交互算子与条件接入方式。新增A1／A2／A3三套配方、六份完整Stage2／Stage3 YAML；不修改现役Base、I2或其他消融，不追加训练seed，不启用HPO。

| 设置 | L2交互算子 | 条件通路 |
|---|---|---|
| 既有I2 | 注意力 | 条件token |
| A1 | MLP-Mixer式残差交互 | 条件token |
| A2 | 注意力 | 后置FiLM |
| A3 | MLP-Mixer式残差交互 | 后置FiLM |

配置位于`configs/ablations/l2_factorial_3407/<编号>/stage{2,3}.yaml`，输出位于`outputs/ablations/l2_factorial_3407/<编号>/stage{2,3}/`。运行见[专用手册](../l2-factorial-ablation-runbook.md)。正式训练由用户启动。

## 保持的合同

固定当前共享冻结Stage1、learned1024＋RDKit211表示及既有prepare、data/3407、训练seed42及原折派生、22实验＋两模拟、G2_C01专家数、007/032训练预算。L1注意力、主体路由、三槽顺序、1024D接口、原预测塔与Task Gate不变。

沿用I2：GLOBAL L2仅主体，GROUP／PRIVATE L2接收第三成分；PRIVATE的第三成分表示仍由原GROUP投影与主体GROUP路由加权得到。不存在新增旁路。两模拟PRIVATE隔离、owner样本加权、loss、裁剪、学习率、冻结日历、完整轮恢复与固定末轮发布均按原合同。

新模式只允许`expert_architecture: entity_token_attention_v1`、`partner_injection: group_private_l2`且无类型／体系隔离，不组合其他消融。未实现本轮之外的条件感知路由、特征分流或预测头改动。

## MLP交互

L1仍使用原注意力；只将GLOBAL／GROUP／PRIVATE的L2堆栈替换为残差MLP混合模块。固定七个位置：三个实体和四个条件位置。

每层为：

1. 对通道Pre-LN后清零无效位置，逐通道执行`7→14→7`的token混合MLP，再做Dropout、残差和无效位置清零。
2. 执行通道Pre-LN与`1024→原owner隐藏宽度→1024`的FFN，再做Dropout、残差和无效位置清零。

层数沿用两层，激活与Dropout沿用原owner／PRIVATE任务配方，最后LayerNorm，输出保留三个实体位置。不是逐实体互不交互的MLP，也不是旧2490D/2478D扁平专家。

A3在混合堆栈中将四个条件位置全部mask，而非删除这些位置，因此A1／A3的MLP堆栈形状和随机初值完全一致。无效实体／条件在每次跨槽混合前清零，不能经偏置泄漏。

## FiLM条件通路

A2／A3不将条件token送入L2交互堆栈。保留原GLOBAL／GROUP／PRIVATE各自的`ConditionTokens`、物理单位恢复与四类型编码，不拟合新统计，不改变温度／压力／频率／波长的缩放。

各L2专家新增一个归属原owner的`Linear(4×1024+4, 64)→SiLU→Linear(64, 2×1024)`，输入是按固定类型顺序拼接的四个条件表示及有效位，输出`s,b`。固定64维瓶颈，避免直接大矩阵显著增加容量，不新增搜索变量。对L2最终归一化后的实体表示应用`h'=(1+s)⊙h+b`，各实体共用本样本的仿射参数，再清零无效实体。

FiLM末层权重及偏置均初始化为零，即初始FiLM为恒等映射；末层开始学习后，梯度传入瓶颈及原条件编码器。所有条件缺席时显式保持恒等，不把缺席当作数值零。该通路替换条件token参与交互，不与之叠加；不恢复历史MLP的ConditionFiLM产物或加载合同。

## 配对初始化与身份

先按I2消耗原owner初始化流，新增／替换模块在独立的owner＋专家索引种子域中构造，不推进外层RNG。因此保留的L1、路由、条件编码器、归一化、Task Gate与Tower逐tensor保持I2初值；A2的注意力权重也与I2相同。Dropout轨迹与后续训练状态不要求相同。

`model.l2_operator`可为`attention`／`mlp_mixer`，`model.condition_mode`可为`tokens`／`film`。缺省分别为`attention`／`tokens`，序列化省略，旧配置、检查点合同和默认数值路径不变。新值及混合、FiLM、初始化定义进入模型与训练身份。

Stage3变体分别为`l2_mlp_tokens`、`l2_attention_film`、`l2_mlp_film`；加载研究配置时严格核对变体与两字段。沿用现有token产物kind／format及严格加载器；相同kind、format或张量形状不代表身份兼容。

各项必须从头训练自己的Stage2，并迁移到同项Stage3。既有I2 Stage2 final不能用于A1／A2／A3；跨格模型、来源、恢复或评估均拒载。Stage1及prepare只读复用，不改SHA、数据metadata或既有产物。

## 成本、比较与验证

新增工作为3次Stage2和15个Stage3折次，Stage1新增训练为零。MLP替换和FiLM会改变参数量、计算量及优化轨迹；保留原隐藏宽度和预算，不宣称等参数或等计算。实际参数、owner更新步数、GPU时间与显存沿用现有性能日志记录。

按生产1024D、五模拟Stage2及22实验＋两模拟Stage3，在meta设备实例化统计的下游参数如下，不含冻结Stage1；未运行模型前向或正式训练：

| 设置 | Stage2 | Stage3 |
|---|---:|---:|
| I2 | 227,886,685 | 616,739,218 |
| A1 | 152,319,391 | 339,659,140 |
| A2 | 231,446,941 | 629,793,490 |
| A3 | 155,879,647 | 352,713,412 |

MLP若改善，也可能包含容量减少的贡献；本轮只解释固定配方对照，不自动增加容量匹配实验。

主比较为22实验等权、完整五折nMAE。分别比较I2→A1、I2→A2，以及A1→A3／A2→A3，报告四格交互差值和全部任务退化；两模拟独立报告。不按任务拼接赢家，不根据测试选配方，不以五折离散度宣称多seed显著性。

复用临时小模型检查三槽交互、GLOBAL第三成分隔离、条件梯度、缺席mask、恒等FiLM、保留参数初值、owner归属及冻结、严格共享状态迁移、序列化与四格交叉拒载。复用既有主链执行小数据Stage2→Stage3三阶段、完整轮恢复、final严格重载及独立评估验收，不增加默认重型参数化。运行完整pytest、受影响入口帮助、compileall及文档链接检查；不启动正式prepare、训练或评估。
