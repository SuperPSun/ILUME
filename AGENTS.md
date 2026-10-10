# ILUME 工作区规则

## 使用顺序与修改边界

- 仓库维护的说明、手册和 ADR 统一使用中文；命令、路径、配置键、API、模型名称及科研身份标识保留原文。第三方原始文档和历史生成报告不随文档翻译改写。
- 运行步骤读 [README](README.md)，科研合同按 [ADR 索引](docs/adr/README.md) 查对应主题，再读正式 YAML 与当前实现。旧设计只从 [历史摘要](docs/adr/history.md) 追溯，不作为现役入口。
- 整理、重构不得改变数据筛选/split、分词器、描述符/指纹、masking、模型、loss、优化顺序、验证/选模及任何 Stage 数值行为。科研问题单独记录，不顺手修改。
- 现役为 `configs/v4/stage1`、`configs/v4/stage2`、`configs/v4/stage3`（ADR-0095）；Stage1合同不变。Stage2/3取消ObjectEncoder，24实验+两模拟。新输出使用 `outputs/v4/stage{2,3}/entity_home/`；旧Object/v5加载需历史Git版本，历史产物只读，禁止跨合同加载。


## 科学红线

修改对应 Stage 前，必须阅读下表列出的 ADR；详细机制与数值以这些合同及 YAML 为准。

| 范围 | 必须保持的边界 | ADR |
|---|---|---|
| Stage 1 | 仅SMILES/Graph输入，独立512D 编码器、12层SMILES Transformer/8独立残差图block、CLS/原子均值、1024残差MLP；learned1024、atom512，不输入角色/描述符。结构重建、停止梯度一致性、RDKit217、离线Uni-Mol768、13电子目标及仅训练集原子电荷Linear512→1；现役系数和执行配置只读Base YAML及v4手册，sidecar不改语料/teacher。各项分子均值再按角色loss权重2/2/1；自然shuffle不变。fusion-only80/10/10，10轮、完整轮恢复；训练format4及仅编码器导出。每1000步rank0固定32分子梯度审计只报告，可CLI覆盖，不更新/调权/选模，不进入科研身份；独立14任务后训练只更新回归头，不替换编码器 | 0089/0090/0091/0092；执行0013/0014/0015/0017；0039仅历史v3 |
| Stage 2 | Stage1永久冻结/eval；缓存1241D实体、形式电荷角色及有效位。GLOBAL/GROUP L1并行读取2490D有序槽位拼接，各专家内部投影，无独立ObjectEncoder。仅五项模拟、thermophysical/solvation GROUP，无电子GROUP/原子 adapter | 0095；Stage1表示0089 |
| Stage 2 训练与产物 | 五模拟每联合step各取一个未耗尽batch，GLOBAL跨任务/GROUP组内按真实batch样本数加权，PRIVATE原始平均梯度，不乘任务补偿；`batch_sample_weighted_owner_raw_v1`及联合步调度进入身份，旧final拒绝迁移。256逻辑/微批、10轮末轮、完整stage2_final.pt/json；无仅编码器导出、精调/最优/last/evaluator。迁移GLOBAL、匹配GROUP及两模拟PRIVATE，严格身份和张量 hash | 0095/0097；历史0082/0083/0085 |
| Stage 3 | 24实验+两模拟；electrochemical继承phase_stability预算，两电位限small/仅温度。Phase1实验GLOBAL/GROUP/PRIVATE；Phase2冻结GLOBAL；Phase3仅PRIVATE。模拟Phase1冻结，Phase2/3仅自身PRIVATE，不进共享梯度/分母/调度预算。共同锚点与末轮拼接 | 0095；机制0050/0070 |
| Stage 3 owner | 现役Entity-HoME Phase1/2的GLOBAL跨实验任务、GROUP组内按当前真实batch样本数加权平均；PRIVATE保持原始平均Loss梯度，模拟不进共享分母。Phase3单任务更新；`batch_sample_weighted_owner_raw_v1`及权重规则进入训练身份。历史非Entity保留0070合同。owner LR/存续期与规模类别默认、任务覆盖项的宽度/dropout进入身份；提前结束用`requires_grad=False`，不删除任务；零预算PRIVATE逐bit继承锚点。Base训练集/验证集microbatch上限1024 | 0096；预算0050/0055；诊断0054 |
| Stage 3 诊断 | 只读门控权重占比、归一化熵、PRIVATE 权重占比分位数；测试集 aggregate 合并折-样本。不得新增前向、进入 loss/选择或改预测 CSV；历史实现不输出。pEC50 Phase 3 为 3 轮 | 0054/0055 |

禁止恢复：Stage1 描述符/角色输入、重型token融合、指纹、角色平衡/重复采样、扩增倍率、多容量正式配置、轮中途恢复；Stage2/3不得反传Stage1，禁止Stage2渐进解冻、早停、最优/last、PCGrad或累加窗口；Stage3 四阶段、路由干预、门控校准或HPO/Optuna。v4 alignment/RDKit/Uni-Mol/electronic辅助头只用于预训练，不进入仅编码器部署。ADR-0087候选保持历史v3身份，不增加正式Base数量。

ADR-0093仅扩展独立回归头后训练：YAML支持默认Linear及逐目标MLP/残差 MLP；Linear继承预训练头并做等价标准化换算，非线性采用任务局部随机初始化/dropout。编码器/Fusion永久冻结、entity1024/atom512输入不变，验证只报告；format2内嵌结构并兼容旧Linear format1，不改变Stage1/2/3身份或正式训练。

ADR-0092原子电荷同canonical源行分别保留，不平均、不首末行选择；只扩展电荷监督，不复制语料或其他loss。sidecar format2及观察策略进入身份，旧sidecar/检查点不得交叉续训；无图同构映射仅跳过电荷记录并审计，策略进入身份；损坏资源/解析错误仍失败，来源与split检查不放宽，机制见ADR-0092与v4手册。

### 历史合同、消融与基线

- 历史实现/Capacity 保持五模态格式 v2、Stage 2 仅补偿 physics 的 loss 与既有精调；Stage 3 保持整模裁剪、`max(N_t,1000)` 虚拟过采样、80/20 精调和 `taskwise_refined`（ADR-0026/0027）。Capacity 是端到端预注册研究，不是正式多容量主线、严格缩放定律或仅编码器效应；只用 Stage 1 Base prepare 一次，共享 `outputs/experiments_v1/stage1/prepare/artifacts`。Stage 3 正式输入只用 `formal/*.yaml`，选择只读拼接后的验证，不读末轮均值或测试集。
- v4消融配置和旧ObjectEncoder实现/加载已移除；历史消融与历史实现/Capacity复现须使用对应Git版本，输出保留。
- ADR-0034/0036/0062/0068/0072/0073/0079/0080/0081 及 Stage3 候选仅作历史追溯，不作为活跃入口；旧输出只读，不跨身份加载。v1/Capacity 冻结合同仍有效。
- ADR-0086：十个正式基线对汽化热、热膨胀、HOMO、LUMO 独立单次训练，任务目录原训练集/验证集/测试集；不训练 partial charge，不更新实验任务模型。神经模型 10 轮、XGBoost 1000 trees 保持各自末轮配方。单任务 MLP 消融和历史 split 配置不扩展。Stage3 模拟验证集/测试集使用五个 final 原单位预测均值，比较身份绑定完整行集合、来源 SHA 与原始训练集总体标准差（零方差为 1）；模拟榜单不得进入实验 wins/radar/scatter。
- 基线只复用注册表、split、canonical SMILES、条件/目标与评估口径，不改变 Stage 数值合同。按 ADR 索引读取各模型合同，不能把一个模型的预算推及其他模型。
- ADR-0045：MLP/D-MPNN/MoLFormer/ILBERT/SPMM/LlaSMol 均为 10 轮，XGBoost 为 1000 trees；全部发布末轮状态。AIonopedia、ILTransR 与 AIFC 也使用各自现役的 10轮配方。除模型 ADR 明定外，验证不驱动早停、选检查点、调度器或训练决策。
- AIonopedia（0049）只用通用已发布多模态检查点、完整官方下游拓扑和独立条件路径，禁止 property-specific 资源或 pure-Qwen。ILTransR（0057）只用通过一致性验证的通用预训练转换与共享全量微调编码器/TextCNN，禁止监督训练 property 检查点；条件使用全量 Stage 3 协变量的传导式总体z-score。
- D-MPNN（0042）所有注册表槽位共用唯一消息传递编码器，不跨任务/折共享。高级基线环境独立 hash-lock，不改主环境、不自动安装、不静默回退。基线不支持恢复，失败在新尝试完整重跑。

## 结构与入口

- Stage 实现仅在 `src/common`、`src/stage1`～`src/stage3`；基线在 `benchmarks/`，消融在 `ablations/`，二者禁止被 Stage 导入。消融代码不得放回 `benchmarks/`。
- `common` 只收至少两个 Stage 实际复用的原子功能；不建 `utils.py`。跨 Stage 只导入公开合同，禁止 `_private_symbol` 或 `import *`。
- 运行入口仅 `scripts/stage{1,2,3}/*.py`、`scripts/benchmarks/*.py`，不恢复 console CLI、shell/smoke/matrix/折 runner 或搜索入口。
- 科研与 prepare 参数写完整自包含 YAML；`preparation` 不进入实验身份。`--output`、`--resume`、Stage 3 折/评估 selector 是运行参数。
- Stage 3 训练入口必填 `--fold`，可多折；`--output` 是共同根目录，运行位于 `foldN/`。并发只在该入口用 spawn worker 和显式设备槽调度，不下沉到 `src/stage3`。

## 数据、身份、输出与恢复

现役hydration仍使用random/cv1五折、无测试集；gas_solubility不是旧x_co2的同义任务。焓保留任务目录原标识，展示汽化焓，仅温度。基线历史20实验/四模拟配方不扩展。运行见 [v4手册](docs/v4-runbook.md)。

- 数据不进 Git；prepare 写 `data/stage*/metadata.json`。每次操作冻结 `run_config.yaml`、公开安全的 `metadata.json`，成功后写 `summary.json`；禁止用户名、hostname、私有绝对路径。
- 新训练/评估输出不可覆盖。可复用 Stage 1/2 prepare 只在数据身份不变时刷新允许忽略的执行/模型字段（ADR-0019）。Stage 2 准备产物数据不绑定 HoME 模型；正式 HoME 不生成 teacher 缓存，模型合同属于训练检查点/final。
- 保留准备产物 SHA 和完整性检查；跨 Stage 保留 ADR-0021 语义身份、必要状态hash；正式 HoME transfer 还绑定 Stage2 final 产物及Stage1源SHA。
- 显式恢复严格校验阶段、折、配置、步/轮、优化器/调度器/AMP。Stage 1 只从完整轮恢复，可改变 world size；Stage 2 HoME 另校验 owner、注册表/模型、RNG、数据/任务规模、数学精度及优化器实现；旧 Object v2/v3、实验 HoME 不迁移到正式身份。
- Stage 3 另校验解析后的计划、参数归属、Stage 2 SHA、数据/归一化；原始随机排列与分支 RNG 由种子/折/phase/scope/轮重建，历史实现重建虚拟采样器。布尔 `--resume` 仅跳过身份一致且汇总/final 产物/清单完整的折；恢复必须匹配检查点、metrics/诊断尾部、owner 更新/冻结状态，历史实现匹配两份根历史记录，不截断、不猜测。
- 周期检查点不可覆盖。Stage1合同不变；Stage2完整五任务final/清单；Stage3三阶段锚点/owner delta及末轮final。新entity_home kind/format4、plan14/training18；绑定架构、数据、Stage1来源、owner与张量 hash，旧Object/v5明确拒载。
- Stage2独立evaluator/报告继续退役；完整模型与SimulationHoME.predict保留。Stage3只预测两模拟辅助，模拟验证独立记录；Stage1电子和电荷功能不变。
- 结果报告发布 Stage 3 实验与独立标量模拟验证/测试集（ADR-0031/0061/0085/0086），不读取或发布 Stage 2 结果；汇总快照结构定义为 v4。榜首 Stage 3 ILUME 的验证五折与测试集集成 SVG 只由 summarizer 从完整性验证通过的预测生成，不改结构定义、指标或模型选择。 现役两任务/历史四任务模拟协议按显式任务集校验并算等权均值，不同协议/数据身份不混排；基线训练配方不扩展（0095）。

## 验证与清理

- Uni-Mol2 独立环境、权重获取/迁移与审计顺序以 [v4 手册](docs/v4-runbook.md) 为准。RDKit 比较导入后的运行时与准备产物特征合同，不只比较 pip 元数据；完成全量 teacher 缓存后才切回训练环境。
- 修改后运行 `pytest -q`；按风险检查十个Stage入口（含离线`stage1/teacher.py`和独立`stage1/regression.py`）与四个基准比较脚本的`--help`、compileall、diff/Markdown链接检查。只用临时小数据；不自动下载teacher权重、生成全量3D缓存或执行正式prepare/训练/评估/回归头后训练。先用独立审计输出验证teacher成功率、吞吐、存储，再由用户正式运行。
- 优先复用/修改现有测试。只有此前未覆盖且会造成实质损失的科研、恢复、产物/身份、CLI/报告或高风险调度合同，才新增最小行为测试；不为私有辅助函数、搬家、简单重构或覆盖率扩测试。`tests/` 按 Stage/基准比较/common/architecture 集中组织，`conftest.py` 只放跨文件复用的小测试数据。
- `trash/` 不进 Git。移动旧产物/YAML/未消费数据或删除机器缓存前，报告精确文件数、大小、目标和冲突策略，等用户明确确认。不得覆盖、重排或删除既有 `trash/`。
