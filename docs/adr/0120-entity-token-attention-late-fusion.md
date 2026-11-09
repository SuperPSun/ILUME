# ADR-0120：实体token注意力、L2条件与末端拼接

- 状态：已接受
- 日期：2026-11-09
- 替代范围：现役Base的HoME专家、条件注入和对象融合；ADR-0096/0097的梯度、owner预算与数据合同不变。历史MLP配置保持原身份。

## 决定

Stage1不变且永久冻结。Base继续G2_C01专家数量、Stage2 007 / Stage3 032优化超参、data/3407及22实验＋两模拟。正式两份Base显式使用`expert_architecture: entity_token_attention_v1`、8头、2层；注意力宽度1024。此前已训练结果不得作为新架构性能或迁移来源。

每个样本保存三个1024D实体token，直至任务预测塔前都不做实体均值池化。槽位固定为：单分子`[分子,空,空]`；IL为`[阳离子,阴离子,空]`；IL+partner为`[阳离子,阴离子,partner]`；溶质+溶剂为`[溶剂,空,溶质]`。prepared仍按原primary/partner对象保存，模型按显式合同重排，不修改来源列、角色或已有SHA。角色取形式电荷，槽位位置另有embedding；无效槽位在每层及拼接前置零。

GLOBAL和GROUP L1独立接收全部实体。每个专家独有1235→1024投影、角色/位置embedding及两层Pre-LN自注意力/FFN。L1仅允许同一对象内部交互：IL的阳阴离子相互注意，partner独立编码；单分子走同一网络。GLOBAL/GROUP并行，不增加ObjectEncoder。L1 gate对primary与partner分别读取三槽mask后的结构输入、role one-hot与有效位，不读条件，不让另一对象通过路由权重提前改变本对象表示；primary两槽共用其路由，按专家维度逐槽求和，保留实体维度。取消旧GROUP实体均值残差，保留逐token注意力残差与LayerNorm。

L2的GLOBAL、GROUP、PRIVATE各专家采用两层1024D/8头Pre-LN自注意力及FFN。输入为三个实体token加最多四个条件token，允许跨对象及结构—条件交互；输出保留三个实体位置，不对条件或实体做池化。原压缩对象的PartnerInteraction及后置ConditionFiLM仅在新模式移除；其交互功能由L2注意力承担。PRIVATE接收GROUP L1序列，GLOBAL L2接收GLOBAL L1序列。FFN宽度继续按各专家原hidden_ratio解析，激活/dropout及预测塔隐藏宽度继承原配方。

条件按物理类型独立`1→64→1024`、SiLU和类型embedding编码。温度K、压力kPa、频率MHz、波长nm均除以1000；不取log、不截断、不拟合新统计。模型绑定原任务/训练折scaler，从prepared z-score还原物理值后统一缩放；相同物理工况按浮点容差一致。未知类型、缺失统计和非有限值失败。目标归一化、原单位预测与来源数据不变。缺席条件mask，数值0仍可有效。

GLOBAL、各GROUP、各PRIVATE分别拥有条件编码器，同owner内专家共享，不跨owner共享参数。GLOBAL/GROUP固定声明四类型以确保Stage2→Stage3完整迁移；PRIVATE仅声明自身条件。Stage2未见的条件投影不能宣称已预训练。

Task Gate读取GLOBAL和GROUP的结构token拼接及三有效位（6147D），不直接读条件；flat softmax候选顺序不变。按每个样本的专家权重逐槽位求和；原L2残差改为逐token`LN(local+mixed)`，无效槽位清零。最后拼接三槽1024D和三有效位得到3075D，送入任务预测塔。实体交互不等于实体压缩；只在最终预测塔形成体系输出。

## owner、产物与兼容

两阶段共享同一模型实现。Stage2五模拟参与GLOBAL/GROUP/PRIVATE；Stage3 Phase1实验全owner、Phase2冻结GLOBAL、Phase3仅PRIVATE；两模拟Phase1冻结、Phase2/3仅自身PRIVATE，绝不进入共享梯度分子/分母/预算。全部新模块明确归属owner，锚点/分支/末轮拼接/恢复及梯度诊断复用现有机制。

迁移完整GLOBAL、thermophysical/solvation GROUP及两模拟PRIVATE，含条件编码器，逐tensor验证。新Stage2 kind为`ilume_stage2_token_home_final_v4`；Stage3为`ilume_stage3_token_home_three_phase_final_v4`，周期/分支kind也独立。外层format4保留，架构、注意力层数/头数、物理条件合同、三槽顺序、融合方式和owner集合进入训练身份。旧MLP默认字段省略，初始化/配置身份及读取行为保留；禁止跨架构恢复、部分加载或改名复用。

Stage1与身份匹配的prepared可复用，不重算/篡改metadata和SHA。新Stage2/3需从头训练；Base新输出根为`outputs/v4/stage{2,3}/token_attention/`，历史`base/`只读。prepare仍用现有入口，运行前必须配对当前211维DGT来源；本次不启动正式运行、不归档或删除旧产物。公开冻结Predictor、实验/模拟评估共用同一模型条件转换，不改生成模块的训练。

## 容量与验证

同一当前G2_C01/22+2任务集合的下游参数（不含冻结Stage1）：

| 模型 | 历史MLP | token注意力 |
|---|---:|---:|
| Stage2 | 82,163,409 | 227,886,685 |
| Stage3 | 172,093,584 | 616,739,218 |

Stage1 DGT encoder为77,986,987参数，正式维度模型总计分别305,873,672及694,726,205。参数统计使用meta设备实例化真实结构，不启动训练；正式内存、吞吐及预测质量必须由实际运行测量。同轮数/样本暴露不代表同计算量，不承诺拼接或注意力必然提升精度，不自动缩batch或改预算。

复用临时fixture验证：三槽顺序与mask、条件只影响L2、实体内部/跨对象交互、逐槽位专家融合、末端拼接、梯度及owner冻结；完整Stage2 prepare/train/export/reload、Stage3三阶段/恢复/拼接/迁移、实验/模拟预测和五模型集成；历史MLP及消融兼容。完整pytest、入口help、compileall、配置往返及Markdown链接检查。正式训练、五折及完整产物回传仍由用户执行。
