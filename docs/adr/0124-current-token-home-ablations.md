# ADR-0124：现役DGT与实体token HoME配对消融

- 状态：已接受
- 日期：2026-11-11
- 范围：更新已有消融，不新增实验；原C6 Uniform Task Gate、C12/C13 ChemBERTa-2已取消并删除配置、注册及专用实现。既有输出、模型资产和环境保留。
- 替代：ADR-0100的活动套件；ADR-0107仅保留历史科研记录，复现使用对应Git版本。

## 共同参照

C0为`configs/v4/stage2/base.yaml`和`configs/v4/stage3/base.yaml`：G2_C01、007/032训练配方、ADR-0120的1024D/8头/2层实体token注意力、当前data/3407的22实验＋两模拟。solvation使用当前3684体系来源；不恢复已退役的两项任务，不将旧24任务macro与新22任务macro直接比较。

Stage1固定现有`outputs/v4/stage1/base/train/stage1_encoder.pt`。本轮配对来源是已完成的`base-100h.yaml`配方：SMILES12层＋双路DGT8层，全局batch84、5轮、seed42、训练图上限160；不是标准Base的batch256/10轮。需要重训的单视图及辅助监督只改变指定部分，保留相同训练索引过滤、teacher/电荷/拓扑来源、mask、loss权重及预算。Stage1仍永久冻结；learned1024＋RDKit211为1235D实体。五折与原始行、角色、mask、物理条件、目标归一化、样本加权共享梯度、PRIVATE原始梯度和模拟隔离保持一致。Stage3训练seed按42+fold派生。

科研配置位于`configs/ablations/token_attention_3407/`，共31份完整YAML；输出位于`outputs/ablations/token_attention_3407/`。原`v4/`与`v4_3407/`的其余配置仅作历史追溯，来源和结果不改写，不可作为新版运行入口。已取消四项容量对照及追加seed测试不恢复。

## 编号与实验定义

| 新编号 | 变体 | 旧编号 | 唯一指定变化与预算 |
|---|---|---|---|
| C0 | Base | C0 | 完整三阶段；复用匹配的新Base |
| C1 | no_stage1 | C1 | 同结构随机SMILES＋DGT冻结，保留显式RDKit211；重新准备表示、Stage2与五折Stage3 |
| C2 | no_stage2 | C2 | 当前配对Stage2的零更新初始化；不读取任何Stage2训练后参数，保留Stage3两模拟PRIVATE训练 |
| C3 | no_global | C3 | 删除GLOBAL的L1/L2注意力专家、L1 gate及无消费者的GLOBAL条件编码器；仅重训Stage3 |
| C4 | no_group | C4 | 删除GROUP的L1/L2注意力专家、L1 gate及无消费者的GROUP条件编码器；保留GROUP逐token LayerNorm；仅重训Stage3 |
| C5 | no_private | C5 | 删除PRIVATE注意力专家及无消费者的PRIVATE条件编码器；Task Gate、Tower、任务归一化和PRIVATE三阶段预算保留 |
| C6 | single_task_mlp | C8 | 当前三实体槽、角色one-hot、mask及标准化条件输入独立MLP，22任务×5折，无模拟训练/报告 |
| C7 | no_smiles | C9 | 删除SMILES编码器、重建头与一致性监督；保留DGT及其双路读出；完整Stage1→Stage2→Stage3重训 |
| C8 | no_graph | C10 | 删除DGT及原子/键/电荷/一致性监督；保留SMILES；完整重训 |
| C9 | phase1 | C11 | 只读新Base固定Phase1末轮完整状态，无新增训练 |
| C10 | phase2 | C7 | 只读新Base Phase2拼接末状态，无新增训练 |

删除专家前，先按完整来源进行严格加载和配对初始化。C3的GLOBAL结构token改为对应有效实体的learned1024；C4的GROUP结构token改为对应learned1024再经过原GROUP LayerNorm。三实体槽保持逐槽位独立，不作均值、不新增可训练投影；partner信息、角色和有效位不丢失。剩余L2专家继续接收完整实体及对应owner的物理条件token；Task Gate对剩余候选重新归一化，其输出行从原初始化按原顺序截取。保留参数逐tensor一致。删除参数不参与前向或优化，不缩短阶段日历或剩余owner预算。

C4仍保留GROUP归一化，不能称为完全取消GROUP共享。C5只检验PRIVATE专家及其依赖条件通路，不能称为取消全部任务独立参数。新模式本来没有FiLM或压缩式PartnerInteraction，不为消融恢复它们。

C6按ADR-0120重排三槽：IL为阳离子/阴离子/空，IL＋partner为阳离子/阴离子/partner，溶质＋溶剂为溶剂/空/溶质。每槽1235维，额外角色one-hot9维和mask3维，合计3717维，再拼接现有标准化条件；输入不读取HoME表示。MLP仍为input→1024→512→1、SiLU/dropout0.1、AdamW3e-4/WD0.01、batch128、10轮末轮，原warmup/cosine与loss配方不变。独立注册名`ilume_stage3_single_task_mlp_entity_token_v4`，新输入/状态/kind拒绝旧实体MLP权重。

C7/C8缺失分支的512D在Fusion前固定填零，剩余分支宽度和dropout不变；无图不运行DGT。分子级Uni-Mol、电子及RDKit监督和下游显式RDKit211都保留；无图的原子表示为行对齐零宽张量。AIS分词与语料准备不变，不将C8称为完全无图衍生信息。

## 扩展监督

E1～E5编号不变，分别只置零`lambda_unimol`、`lambda_electronic`、`lambda_partial_charge`、`lambda_alignment`、`lambda_descriptor`。每项使用自己的Stage1、表示缓存、Stage2与五折Stage3。E5仍提供下游显式RDKit211。可复用corpus、teacher和电荷/拓扑sidecar的前提是来源身份一致；训练不改变它们，不自动重prepare。

Stage1重训共7次：两项单视图＋五监督；C1只导出零更新随机编码器。C0已完整且来源一致时，新Stage2共8次、HoME Stage3共60个折、独立MLP共110个任务折；阶段状态对照只评估。成本不包含Base训练，不称等计算量。

## 身份、验证与报告

实现保持在`ablations/`，Stage仅提供公开L1 token混合入口；Stage不得反向导入消融。新Stage3周期/阶段/最终产物使用`ilume_stage3_token_home_ablation_v1_*` / `ilume_stage3_token_home_ablation_final_v1`；架构、来源、消融、owner、状态hash仍严格绑定。禁止宽松strict、改SHA、混用旧Stage2、旧表示或历史24任务来源。

复用临时数据验证：保留参数和gate行配对、删除专家不参与前向/优化、partner/条件有效、零更新来源无训练后参数、原始PRIVATE梯度与模拟共享隔离、DGT单视图梯度和导出、完整Phase2只读重载、交叉恢复拒绝、实体MLP槽位/条件，以及实际Stage2→Stage3训练/验证/预测和严格权重重载。检查31份配置、相关入口、pytest、编译及链接。

主表为22实验任务等权macro normalized MAE，逐任务MAE/RMSE/R²、相对C0差值及胜负；六GROUP、规模类别分层，两模拟单列。固定末轮/阶段末状态，验证只报告，不选checkpoint、不追加seed。五折标准差是划分差异。删除组件减少容量、MLP改变架构与预算、阶段对照改变训练量，只解释总体贡献，不单独证明机制或等预算最优。

正式prepare、训练、评估由用户启动。新输出不可覆盖；所有设备的完整权重、配置、日志、身份和恢复产物回传4090。运行见[消融手册](../v4-ablation-runbook.md)。
