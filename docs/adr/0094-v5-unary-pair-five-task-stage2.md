> 历史合同：Stage2/3 已由 [ADR-0095](0095-v4-entity-home-without-object-encoder.md) 替代。对应配置与实现请从历史 Git 版本读取；产物不得用于新 v4。

# ADR-0094：v5 Unary + Pair Stage2 及配对 Stage3 迁移

## 状态

已由 ADR-0095 取代（2026-10-08）。历史决定于2026-10-08接受。仅对当时新增v5协议替代ADR-0082/0083/0084/0086/0089的Stage2/3主线架构、任务集合及模拟报告。Stage1 v4与独立预测器不变。更早配置、产物及基线训练配方保留原合同。

## 决策

Stage1永久eval/冻结，不进入下游优化器。实体输入为learned1024加训练集标准化RDKit217。共享Unary通过LayerNorm、三角色embedding（正态标准差0.02）及预归一化残差MLP1024→512→1024（GELU/dropout0.10）将1241映射到1024。单实体返回LN(Unary)。有序阳/阴离子Unary状态构成 `[Uc, Ua, abs(Uc-Ua), Uc*Ua]`；Pair为LN4096→512→1024（GELU/dropout0.10）。IL输出为LN(均值(Uc,Ua)+gamma*Pair)，gamma初始化为零。Pair权重采用模块正常初始化，因此gamma首步可学习，gamma变化后Pair可学习。Pair建模IL内部，PartnerInteraction仍处理不同对象。正式维度下ObjectEncoder为4,962,305参数，配对Transformer为19,131,392（约减少74%）；这是容量比较，不证明预测改善。

`forward(entity_inputs, entity_roles)` 接口保持 `[B,1|2,1241]`→`[B,1024]`。`formal_charge_v1` 下每个单实体角色由canonical SMILES净形式电荷决定，包括带电solute/solvent。IL槽位验证先正阳离子、后负阴离子。prepare、缓存、inference共用此策略；历史槽位保留旧语义。

Stage2 Base仅训练thermophysical的density、heat_capacity、thermal_expansion、heat_of_vaporization，以及solvation的transfer_organic。不创建electronic GROUP、原子 adapter或电子任务参数。保留损失权重1.0/0.8/0.8/1.0/0.5、SmoothL1和仅训练集统计。保留完整原始行、batch/microbatch256、每逻辑batch一次更新、10轮和固定末轮导出。没有独立Stage2 evaluator。直接导出实际训练编码器、完整末轮HoME及SHA绑定清单。

当时仅保留 `configs/v5/stage2` 和 `configs/v5/stage3` 下自包含Base配置。用户于2026-10-08决定移除三组新增架构/任务集合对照。历史Transformer/九任务实现及v4配置保留原合同。共享HoME初始化继续采用独立SHA256(种子,owner) CPU RNG上下文。任务ID缺失/重复或loss权重不一致立即失败。

Stage3采用任务目录的22实验任务，将x_co2替换为不同的gas_solubility目标，并增加water_activity_coefficient和enthalpy_of_vaporization_or_sublimation。其余19任务保留PRIVATE配方。气体溶解度为IL primary + gas partner、487体系、solvation/small、IL-solute/cv1；水活度为IL、167体系、solvation/small，条件为水摩尔分数/T/P；焓为IL、137体系、thermophysical/small。hydration仍为random/cv1、无测试集。其他任务目录 system split原样消费；prepare不重切。

焓任务展示名为**汽化焓（Enthalpy of vaporization）**；任务目录中的任务ID、目录及目标列保留 `enthalpy_of_vaporization_or_sublimation` 以维持来源。恒定来源 `phase` 列不作为模型条件；仅从任务目录条件声明式选择 `temperature_K`。Base不采用类别编码。所选数值条件及目标统计仅用四个训练折。所选条件列表和形式电荷角色进入准备产物/模型身份；忽略的来源列仍受文件SHA校验覆盖。

严格校验owner/来源/张量后迁移ObjectEncoder、完整GLOBAL及匹配thermophysical/solvation GROUP。仅heat_of_vaporization和thermal_expansion模拟PRIVATE owner迁入Stage3。Phase1用22实验任务适配ObjectEncoder，模拟PRIVATE冻结；Phase2/3冻结整个ObjectEncoder，含gamma和Pair。Phase2模拟梯度仅更新自身PRIVATE（含门控、FiLM及tower）；GLOBAL/GROUP仅接收实验梯度。模拟任务不参与共享聚合分母，也不扩展GROUP 调度器更新预算。计划记录 `simulation_private_only_v1`。保留原始owner聚合 `weighted_owner_raw_v1`、large模拟PRIVATE、共同锚点、owner存续期/冻结、末轮拼接和严格恢复。

新的final/编码器/检查点 kind及format5标识v5。Stage3 plan13/training17绑定表示、角色/条件定义、任务/owner集合、来源SHA和张量 hash。通过配置/产物合同显式选择v5，不从参数名猜测。拒绝跨架构或任务集合恢复。历史v4身份由原实现读取。

模拟评估仅报告两任务，将五个final模型的原单位预测平均。汇总显式校验两任务v5或四任务历史协议，采用任务等权均值及比较身份。不同任务/数据协议不共用榜单。基线训练不变，不暗示扩展电子基线或重映射任务。

## 验证与运行

使用临时CPU 测试数据覆盖Unary/Pair梯度激活、角色/顺序/partner路径、五任务注册表、模拟仅PRIVATE更新、完整数据准备/训练/导出/重载、22+2 owner迁移、三阶段/拼接/恢复、训练折归一化、条件选择/SHA拒载及预测/集成报告。运行完整测试、入口help、compileall和文档检查。实现验收不生成正式数据输出或启动GPU作业。

命令及前提见历史v5运行手册（须从对应Git版本读取；现役手册为 [v4运行手册](../v4-runbook.md)）。正式运行需要配置指定的完整v4 Stage1 编码器/特征、新的身份绑定prepare及未使用的 `outputs/v5/` 路径。过期元数据必须由正常prepare重新生成，不手动重算hash。本地电荷资源已通过大小/SHA审计，但不能替代正式prepare验证。
