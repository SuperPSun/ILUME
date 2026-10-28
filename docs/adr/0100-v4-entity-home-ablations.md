# ADR-0100：现役 v4 Entity-HoME 独立消融

- 状态：已接受
- 日期：2026-10-15
- 更新：2026-10-23，新增配对最终优化配方的3407套件；原split42配置与产物保留。
- 更新：2026-10-26，取消3407套件四项容量对照；仅Phase1评估正式编号为C11，与C7、C0比较完整Base的阶段增量。
- 范围：独立消融研究；正式 Base、历史产物、基线预算不变。

## 科研协议

现役实验统一使用 `data/3407` 五折，以 `configs/v4/stage2/optimized_3407.yaml` 与 `configs/v4/stage3/optimized_3407.yaml` 的完整模型为唯一配对参照（HoME G2_C05 / Stage2 007 / Stage3 032，见[最终配方](../hpo-results.md)）。训练seed42、Stage3按42+fold派生；24实验性质，hydration为random/cv1、无测试集；两项模拟保持PRIVATE隔离并独立报告。除所声明的消融外，模型、任务归组、原始样本覆盖、条件、目标归一化、SmoothL1、owner裁剪、学习率和预算全部与优化配方一致。优化配方将两项电位任务归入phase_stability，共六个GROUP；不能混用原Base的七GROUP分层。Stage1在下游永久冻结；验证只报告；发布预先指定末轮/阶段末状态。

原始 `configs/ablations/v4/` 的36份split42配置与 `outputs/ablations/v4/` 保留历史身份；`configs/ablations/v4_3407/` 共保留40份YAML，其中36份现役、4份容量对照已取消并仅供追溯，输出只写 `outputs/ablations/v4_3407/`。Stage1仍共享 `data/stage1` 和原语料prepare，不能将其语料seed误改为3407。3407 Stage2/3 prepare仅在数据、来源和完整性校验通过时复用；不存在时按配套配置重新prepare，既有不匹配产物不覆盖、不改SHA。

| 核心对照 | 变化 | 解释边界 |
|---|---|---|
| C0 Base | 正式完整模型（Phase1＋Phase2＋Phase3） | 配对参考 |
| C1 no_stage1 | 固定 seed42 同结构随机冻结编码器，保留显式 RDKit217；重新准备实体、训练 Stage2/3 | Stage1 预训练总体贡献 |
| C2 no_stage2 | 用完整 Stage2 自包含 YAML 构造训练前参数；零优化器更新；仍保留两项模拟 PRIVATE | Stage2 模拟预训练及迁移总体贡献 |
| C3 no_global | 删除 GLOBAL L1/L2 专家及 L1 gate；Task Gate 的 z_global 输入为有效 learned1024 均值 | 同时减少容量及迁移参数 |
| C4 no_group | 删除 GROUP L1/L2 专家及 L1 gate；保留 GROUP LayerNorm、partner 交互与原 owner 预算 | 领域专家通路贡献，不是完全取消 GROUP 共享 |
| C5 no_private | 删除 PRIVATE expert，保留 Task Gate/FiLM/Tower/归一化及原三阶段 PRIVATE 配方 | 独立专家贡献，不是所有独立参数贡献 |
| C6 uniform_gate | 从训练开始固定最终候选等权，冻结 Task Gate；L1 gate 正常训练 | 最终样本相关选择的贡献 |
| C7 phase2 | 只读 Base Phase2 stitched 全状态（Phase1＋Phase2） | 包含减少训练量，不能证明三阶段等预算最优 |
| C8 single_task_mlp | primary2490、声明 partner2490、现有标准化条件按序拼接；独立1024→512→1 MLP | 同时改变架构和预算的整体系统对照 |
| C9 no_smiles | 删除 SMILES 编码器、重建头与双视图一致性头；从头训练 Stage1/2/3 | 图分支保留；编码分支及其依赖监督总体贡献 |
| C10 no_graph | 删除图编码器、原子/键重建头、电荷头与双视图一致性头；从头训练 Stage1/2/3 | SMILES分支保留；同时减少监督与容量 |
| C11 phase1 | 只读 Base Phase1 固定末轮全状态（仅Phase1） | 与C7、C0比较阶段增量，无新增训练 |
| C12 chemberta2_mtr | 冻结ChemBERTa-2 MTR替换Stage1，无显式RDKit | 外部预训练表示整体替换 |
| C13 chemberta2_mlm | 冻结ChemBERTa-2 MLM替换Stage1，无显式RDKit | 外部预训练表示整体替换 |

C3～C5删除专家通路前先完整加载并校验来源；保留参数逐张量一致，Task Gate 保留候选对应输出行按原顺序截取。删除 GLOBAL 不缩短 Phase1 日历；没有训练梯度的任务仍保留原始采样与 loss 报告，不创建虚假梯度或优化器更新。GROUP 删除后 primary/partner 的 L1 delta 均为零，保留现有 LayerNorm、FiLM、partner 路径及 L2 residual。

MLP 保留 AdamW3e-4/WD0.01、batch128、BF16 CUDA、5%预热/余弦、10轮末轮、SiLU/dropout0.1，只训练24实验任务，不生成模拟榜单；与历史 Object 表示 MLP 使用不同模型注册名、输入合同、kind及状态 hash 命名空间。

C9/C10保持原1024D残差融合，缺失分支的512D槽固定填零；剩余分支宽度、masking和dropout不变。不执行缺失分支前向或优化。无图模型无原子表示，缓存为行对齐零宽原子张量；learned1024＋RDKit217接口不变。两项都保留分子级RDKit/Uni-Mol/电子监督与显式描述符，双视图一致性关闭；原子电荷只在图分支存在时保留。AIS分词、图衍生的准备特征及RDKit统计仍按原数据合同生成，因此不能称为完全不使用图衍生信息。各项使用独立Stage1权重、Stage2/3缓存与新训练产物；不复用完整模型的迁移权重。新增成本2次Stage1、2次Stage2、10个Stage3折训练。

C12/C13的冻结源、补零、严格加载和无描述符合同见[ADR-0107](0107-chemberta2-frozen-home-ablations.md)。新增4份Stage2/3配置，均重新准备实体及训练下游，不训练Stage1。

## 扩展实验

- Stage1 五项监督分别只置零 `lambda_unimol`、`lambda_electronic`、`lambda_partial_charge`、`lambda_alignment`、`lambda_descriptor`。不变更语料/teacher/原子标签来源与筛选；各自重新训练 Stage1/2/3 和新实体缓存。关闭 RDKit loss 仍保留下游显式 RDKit217。
- 四项容量对照（`capacity_no_global`、`capacity_match_global`、`capacity_no_group`、`capacity_match_group`）已取消，不再纳入3407套件的训练、评估或主表；本轮减少20个Stage3折训练。配置和历史split42实现保留，3407容量配置在加载时明确拒绝，不能用旧批量命令重新启动。C3/C4仍保留，但不再通过容量匹配对照区分结构与参数量效应。
- 阶段增量按C11（仅Phase1）→C7（Phase1＋Phase2）→C0（Phase1＋Phase2＋Phase3）比较，无新增训练。C11复用现有 `core/phase1/stage3.yaml` 和 `outputs/ablations/v4_3407/phase1/`，内部variant与model_id保持原身份。五折分别按同一协议评估Phase1末轮检查点、Phase2 stitched和Phase3 final；不同阶段训练量不同，结果只说明阶段增量，不能据此宣称三阶段在等预算下最优。Phase1/2 selector不能用于训练、恢复或挑选轮数。

## 身份、接口与隔离

实现位于 `ablations/entity_home/`；Stage包仅通过公开Stage1配置扩展注册/模型工厂和 `Stage3Config.runtime_backend` 调用入口提供的研究后端，禁止反向导入消融包。现役36份完整YAML及4份已取消容量配置位于 `configs/ablations/v4_3407/`，历史split42的36份位于 `configs/ablations/v4/`，不修改正式YAML，不恢复旧ObjectEncoder。MLP复用同一3407实体prepare和authority；sweep允许显式GPU设备槽。汇总通过 `--ablation-authority-config configs/v4/stage3/optimized_3407.yaml` 绑定分组及bootstrap来源，独立分析的split_seed读取authority配置，不能写死42。

`ablation` 写入运行配置和训练身份 plan：变体、零更新初始化 YAML、所移除监督、完整初始模型状态 hash，以及参数量匹配的比例和计数。checkpoint/scope/final 使用独立 `ilume_stage3_entity_home_ablation_*_v1` kind；保留 format4、plan14/training18，不跨消融/正式或历史合同恢复。来源、数据、owner、SHA、完整张量校验不放宽。C1 导出独立 `ilume_stage1_random_encoder_v4`、零更新身份及固定 epoch0，不冒充预训练末轮。C2 从真正的 Stage2 初始模型构造来源，不读取已训练 Stage2 final，也不伪造正式 final。

单视图 Stage1 配置含 `ablation.variant`，训练与编码器身份均绑定视图；训练检查点/编码器导出使用独立 `ilume_stage1_single_view_pretraining_v4` / `ilume_stage1_single_view_encoder_v4` kind及format4。入口注册研究配置，普通Stage API不导入消融包；研究API调用者先导入 `ablations.entity_home.stage1`。Stage2完整来源合同与实体缓存记录Stage1视图，Stage3后端核对声明与真实来源，禁止交叉恢复。保留来源语料/teacher身份和完整strict张量加载。

阶段评估直接读取 Base 检查点，不复制或重标来源身份。Phase2 校验 stitched SHA、清单、owner集合及 Base 完整训练身份；Phase1 固定配置最后 epoch并校验其训练身份/状态 hash。两者独立评估身份绑定 selector与来源 SHA。

## 报告与验收

现役 v4汇总快照结构不变。独立分析报告包含逐任务 MAE/RMSE/R²/normalized MAE、消融减 Base 差值、相对变化、胜负/平局、七 GROUP 与规模类别分层，以及 GPU小时、峰值显存、参数量、实际 owner 更新和样本暴露。五折标准差仅表示划分差异，不是独立训练重复；本轮不追加 seed 稳定性测试。可选配对 bootstrap按 split 分组单位保持整体系条件行，seed42、10000次、95%区间，解释为固定预测上的不确定性。bootstrap 的点估计和区间使用原单位 pooled-row MAE差，与主表五折等权指标明确区分。

小数据测试另覆盖单视图分支与依赖头不存在、缺失输入不影响表示、监督与梯度、训练/导出重载及来源隔离。小数据测试覆盖保留参数与候选顺序、partner有效、门控冻结、零更新模拟 PRIVATE、冻结/耗尽任务、严格来源与交叉身份拒载、Phase2完整重载、恢复、MLP槽位合同及报告配对。运行完整 pytest、入口 help、compileall、diff及Markdown链接检查。实现不启动正式训练/prepare/评估；完整结果与模型权重按4090归档规则回传，旧产物不覆盖。

命令见 [消融运行手册](../v4-ablation-runbook.md)。
