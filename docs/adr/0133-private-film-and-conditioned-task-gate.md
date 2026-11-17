# ADR-0133：I2的PRIVATE FiLM与工况Task Gate独立对照

## 状态与范围

已接受（2026-11-17，用户要求先增加两项，不等待10轮C0／I2／S3）。以现有5轮Stage1来源的I2为共同参照，新增F1和G1两个独立对照，二者不组合。新配置在`configs/ablations/condition_routing_3407/{F1,G1}/stage{2,3}.yaml`，输出在`outputs/ablations/condition_routing_3407/<编号>/`。命令见[专用手册](../condition-routing-ablation-runbook.md)。

固定共享冻结Stage1 `outputs/v4/stage1/base/train/stage1_encoder.pt`（batch84／5轮）及匹配prepare、data/3407、训练seed42与折派生、G2_C01、007/032预算、22实验＋两模拟。沿用[ADR-0127](0127-partner-injection-and-typed-encoding-ablations.md)的I2接入：GLOBAL只读主体，GROUP／PRIVATE L2接入溶质；L1、候选数量、残差、预测塔、owner样本加权梯度、loss、采样、冻结、完整轮恢复与末轮发布不变。

本轮只替代这两个独立研究模式涉及的条件／门控定义，不修改Base、既有I／S／R／A或10轮配置，不恢复推理期路由干预、门控后校准、均匀门控、早停或HPO，不追加seed。

## F1：仅PRIVATE使用FiLM

`model.condition_mode: private_film`，Stage3变体`private_condition_film`。

| owner | I2 | A2（已有） | F1 |
|---|---|---|---|
| GLOBAL L2 | 条件token | FiLM | 条件token |
| GROUP L2 | 条件token | FiLM | 条件token |
| PRIVATE L2 | 条件token | FiLM | FiLM |

全部L2仍为原注意力。仅PRIVATE复用[ADR-0132](0132-l2-operator-condition-factorial.md)的FiLM实现：条件不进入PRIVATE注意力，而由原PRIVATE条件编码器及`4×1024+4→64→2×1024`网络输出逐特征缩放／偏移，作用于PRIVATE注意力输出。末层全零使初始FiLM为恒等，条件缺席时保持恒等，mask与物理单位恢复不变。GLOBAL／GROUP仍将条件token送入原注意力。

FiLM归属对应PRIVATE，包括两模拟PRIVATE；沿用原冻结预算。Task Gate仍只读主体结构，不读物理条件或溶质。新增PRIVATE模块使用ADR-0132的owner／专家索引随机域，保留参数初值逐tensor与I2一致；对应PRIVATE完整初值与A2相同。

## G1：Task Gate读取物理工况

`model.task_gate_conditioned: true`，Stage3变体`condition_task_gate`。所有专家及条件通路保持I2，即全部使用原注意力＋条件token。

保留Task Gate原6147D结构输入及其线性参数，增加固定八维工况输入：四种原物理坐标加四个有效位。温度K、压力kPa、频率MHz、波长nm按原合同恢复并除以1000，使用本折训练scaler；缺席值零填并显式mask，有效零值与缺席区分。不新增统计、归一化、条件编码器或条件MLP。

门控定义为`softmax(W_struct·h + b + W_condition·[physical4,mask4])`。`W_condition`形状为候选数×8，初始全零；原结构权重／偏置与I2完全一致，新增参数不消耗初始化RNG。起始门控与I2一致，后续从原训练loss学习，不修改专家权重温度或强迫等权。

Task Gate仍逐任务归属PRIVATE；新增参数随PRIVATE寿命训练／冻结。相同结构在不同工况下可学得不同混合权重；L1与主体路由仍不读取工况。溶质不通过门控结构输入旁路进入GLOBAL。原共享梯度分母、裁剪、调度及诊断仅报告规则不变。参数在训练和推理两侧一致使用，不是推理期干预或后训练校准。

## 来源与兼容

`condition_mode: private_film`仅允许I2的未分流注意力结构。`task_gate_conditioned`必须为严格bool，仅允许未分流I2注意力＋条件token，不与FiLM、MLP混合或体系／电荷类型隔离组合。研究配置同时核对变体与两字段，其他已有变体拒绝携带新模式。

默认`task_gate_conditioned: false`省略序列化；既有`tokens`／`film`行为、初始化、模型合同与产物保持不变。F1作用范围与G1条件坐标／门控初值定义进入模型及训练身份，kind／format沿用现有严格token合同；相同张量形状不能跨模式迁移、恢复或评估。

两项均从头训练自己的Stage2，再训练Stage3五折；匹配的两模拟PRIVATE也从本项Stage2迁移。不能直接复用I2或A2的Stage2 final，不提供部分加载、改hash或跳过owner校验。Stage1与数据身份不变的既有prepare只读共用，不重复编码、不修改数据metadata或既有SHA。

## 比较、成本与验收

新增两次Stage2及十个Stage3折，不新增Stage1训练。F1与I2比较PRIVATE条件通路，辅以A2区分全owner替换；G1只与I2比较门控新增工况。完整22任务等权、五折nMAE为主，逐任务收益／退化与两模拟独立报告。不按任务拼装赢家，不按测试选型，不用五折离散度宣称多训练seed显著性。

F1增加PRIVATE容量，G1增加少量门控参数，均记录实际参数量和训练成本，不宣称等参数／计算量。门控占比不是因果重要性；不得根据GLOBAL占比高就强制改权重。

生产1024D在meta设备实例化的下游参数如下，不含冻结Stage1，未执行正式模型前向或训练：

| 配方 | Stage2 | Stage3 |
|---|---:|---:|
| I2 | 227,886,685 | 616,739,218 |
| F1 | 229,864,605 | 626,233,234 |
| G1 | 227,886,813 | 616,739,866 |

G1仅增加Stage2的128个及Stage3的648个参数；F1的Stage3增加约1.54%。完整权重仍需分别保留，不跨架构复用来源。

验收复用现有小模型和主链：原参数及RNG不变、仅PRIVATE增加FiLM、G1零初值门控等价与工况梯度、L1／GLOBAL的溶质隔离、缺席条件／实体mask、owner冻结及无样本边界、严格Stage2迁移与两模拟PRIVATE来源、状态重载、变体及默认序列化、跨模式恢复拒载。两项另用临时小数据调用既有完整主链检查训练／完整轮恢复／末轮重载／独立评估，不增加默认重型链参数化。

执行完整pytest、受影响入口help、compileall、diff和文档链接检查；不启动正式prepare、训练或评估，不修改运行中的实验。完整产物仍归档4090，其他设备代码由用户自行git pull。
