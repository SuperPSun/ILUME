# ADR-0100：现役 v4 Entity-HoME 独立消融

- 状态：已接受
- 日期：2026-10-15
- 范围：独立消融研究；正式 Base、历史产物、基线预算不变。

## 科研协议

以 ADR-0095/0096/0097 的现役 Base 为唯一配对参照：训练 seed42、`data/42` 五折，24实验性质，hydration 为 random/cv1、无测试集；两项模拟保持 PRIVATE 隔离并独立报告。保留原始样本覆盖、条件、目标归一化、SmoothL1、owner 裁剪、学习率和剩余 owner 预算。Stage1 在下游永久冻结；验证只报告；预先指定末轮/阶段末状态，禁止验证选模或按结果选择扩展实验。

| 核心对照 | 变化 | 解释边界 |
|---|---|---|
| C0 Base | 正式完整模型 | 配对参考 |
| C1 no_stage1 | 固定 seed42 同结构随机冻结编码器，保留显式 RDKit217；重新准备实体、训练 Stage2/3 | Stage1 预训练总体贡献 |
| C2 no_stage2 | 用完整 Stage2 自包含 YAML 构造训练前参数；零优化器更新；仍保留两项模拟 PRIVATE | Stage2 模拟预训练及迁移总体贡献 |
| C3 no_global | 删除 GLOBAL L1/L2 专家及 L1 gate；Task Gate 的 z_global 输入为有效 learned1024 均值 | 同时减少容量及迁移参数 |
| C4 no_group | 删除 GROUP L1/L2 专家及 L1 gate；保留 GROUP LayerNorm、partner 交互与原 owner 预算 | 领域专家通路贡献，不是完全取消 GROUP 共享 |
| C5 no_private | 删除 PRIVATE expert，保留 Task Gate/FiLM/Tower/归一化及原三阶段 PRIVATE 配方 | 独立专家贡献，不是所有独立参数贡献 |
| C6 uniform_gate | 从训练开始固定最终候选等权，冻结 Task Gate；L1 gate 正常训练 | 最终样本相关选择的贡献 |
| C7 phase2 | 只读 Base Phase2 stitched 全状态 | 包含减少训练量，不能证明三阶段等预算最优 |
| C8 single_task_mlp | primary2490、声明 partner2490、现有标准化条件按序拼接；独立1024→512→1 MLP | 同时改变架构和预算的整体系统对照 |
| C9 no_smiles | 删除 SMILES 编码器、重建头与双视图一致性头；从头训练 Stage1/2/3 | 图分支保留；编码分支及其依赖监督总体贡献 |
| C10 no_graph | 删除图编码器、原子/键重建头、电荷头与双视图一致性头；从头训练 Stage1/2/3 | SMILES分支保留；同时减少监督与容量 |

C3～C5删除专家通路前先完整加载并校验来源；保留参数逐张量一致，Task Gate 保留候选对应输出行按原顺序截取。删除 GLOBAL 不缩短 Phase1 日历；没有训练梯度的任务仍保留原始采样与 loss 报告，不创建虚假梯度或优化器更新。GROUP 删除后 primary/partner 的 L1 delta 均为零，保留现有 LayerNorm、FiLM、partner 路径及 L2 residual。

MLP 保留 AdamW3e-4/WD0.01、batch128、BF16 CUDA、5%预热/余弦、10轮末轮、SiLU/dropout0.1，只训练24实验任务，不生成模拟榜单；与历史 Object 表示 MLP 使用不同模型注册名、输入合同、kind及状态 hash 命名空间。

C9/C10保持原1024D残差融合，缺失分支的512D槽固定填零；剩余分支宽度、masking和dropout不变。不执行缺失分支前向或优化。无图模型无原子表示，缓存为行对齐零宽原子张量；learned1024＋RDKit217接口不变。两项都保留分子级RDKit/Uni-Mol/电子监督与显式描述符，双视图一致性关闭；原子电荷只在图分支存在时保留。AIS分词、图衍生的准备特征及RDKit统计仍按原数据合同生成，因此不能称为完全不使用图衍生信息。各项使用独立Stage1权重、Stage2/3缓存与新训练产物；不复用完整模型的迁移权重。新增成本2次Stage1、2次Stage2、10个Stage3折训练。

## 扩展实验

- Stage1 五项监督分别只置零 `lambda_unimol`、`lambda_electronic`、`lambda_partial_charge`、`lambda_alignment`、`lambda_descriptor`。不变更语料/teacher/原子标签来源与筛选；各自重新训练 Stage1/2/3 和新实体缓存。关闭 RDKit loss 仍保留下游显式 RDKit217。
- 容量四对照全部为零更新 Stage2：删除 GLOBAL、删除 GROUP，以及各自完整 HoME 参数量匹配模型。只对所有 Expert 隐藏宽度使用同一比例；枚举舍入后不同整数宽度组合，按参数差最小选择，不读取标签或验证指标，差异上限1%，参数计数只包含下游HoME，不计冻结Stage1。Gate/Tower/FiLM/partner及专家数量不变。新宽度使用固定源训练 seed 和模块名独立初始化，非专家参数保留。
- Base Phase1 固定最后 epoch 检查点、Phase2 stitched、Phase3 final 只读比较。Phase1/2 selector 不能用于训练、恢复或挑选轮数。

## 身份、接口与隔离

实现位于 `ablations/entity_home/`；Stage 包仅通过公开 Stage1 配置扩展注册/模型工厂和 `Stage3Config.runtime_backend` 调用入口提供的研究后端，禁止反向导入消融包。36份完整 YAML 位于 `configs/ablations/v4/`，不修改正式 YAML，不恢复旧 ObjectEncoder。

`ablation` 写入运行配置和训练身份 plan：变体、零更新初始化 YAML、所移除监督、完整初始模型状态 hash，以及参数量匹配的比例和计数。checkpoint/scope/final 使用独立 `ilume_stage3_entity_home_ablation_*_v1` kind；保留 format4、plan14/training18，不跨消融/正式或历史合同恢复。来源、数据、owner、SHA、完整张量校验不放宽。C1 导出独立 `ilume_stage1_random_encoder_v4`、零更新身份及固定 epoch0，不冒充预训练末轮。C2 从真正的 Stage2 初始模型构造来源，不读取已训练 Stage2 final，也不伪造正式 final。

单视图 Stage1 配置含 `ablation.variant`，训练与编码器身份均绑定视图；训练检查点/编码器导出使用独立 `ilume_stage1_single_view_pretraining_v4` / `ilume_stage1_single_view_encoder_v4` kind及format4。入口注册研究配置，普通Stage API不导入消融包；研究API调用者先导入 `ablations.entity_home.stage1`。Stage2完整来源合同与实体缓存记录Stage1视图，Stage3后端核对声明与真实来源，禁止交叉恢复。保留来源语料/teacher身份和完整strict张量加载。

阶段评估直接读取 Base 检查点，不复制或重标来源身份。Phase2 校验 stitched SHA、清单、owner集合及 Base 完整训练身份；Phase1 固定配置最后 epoch并校验其训练身份/状态 hash。两者独立评估身份绑定 selector与来源 SHA。

## 报告与验收

现役 v4汇总快照结构不变。独立分析报告包含逐任务 MAE/RMSE/R²/normalized MAE、消融减 Base 差值、相对变化、胜负/平局、七 GROUP 与规模类别分层，以及 GPU小时、峰值显存、参数量、实际 owner 更新和样本暴露。五折标准差仅表示划分差异，不是独立训练重复；本轮不追加 seed 稳定性测试。可选配对 bootstrap按 split 分组单位保持整体系条件行，seed42、10000次、95%区间，解释为固定预测上的不确定性。bootstrap 的点估计和区间使用原单位 pooled-row MAE差，与主表五折等权指标明确区分。

小数据测试另覆盖单视图分支与依赖头不存在、缺失输入不影响表示、监督与梯度、训练/导出重载及来源隔离。小数据测试覆盖保留参数与候选顺序、partner有效、门控冻结、零更新模拟 PRIVATE、冻结/耗尽任务、严格来源与交叉身份拒载、Phase2完整重载、恢复、MLP槽位合同及报告配对。运行完整 pytest、入口 help、compileall、diff及Markdown链接检查。实现不启动正式训练/prepare/评估；完整结果与模型权重按4090归档规则回传，旧产物不覆盖。

命令见 [消融运行手册](../v4-ablation-runbook.md)。
