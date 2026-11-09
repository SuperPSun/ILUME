# ILUME 工作区规则

## 使用顺序与修改边界

- 仓库维护的说明、手册和 ADR 统一使用中文；命令、路径、配置键、API、模型名称及科研身份标识保留原文。第三方原始文档和历史生成报告不随文档翻译改写。
- 运行步骤读 [README](README.md)，科研合同按 [ADR 索引](docs/adr/README.md) 查对应主题，再读正式 YAML 与当前实现。旧设计只从 [历史摘要](docs/adr/history.md) 追溯，不作为现役入口。
- 整理、重构不得改变数据筛选/split、分词器、描述符/指纹、masking、模型、loss、优化顺序、验证/选模及任何 Stage 数值行为。科研问题单独记录，不顺手修改。
- 现役为 `configs/v4/stage1`、`configs/v4/stage2`、`configs/v4/stage3`（ADR-0095/0114）；Stage2/3的Base采用G2_C01分组/专家数及Stage2 007 / Stage3 032训练超参，data/3407、训练seed42；历史optimized_3407/HPO选择与消融保持G2_C05身份。Stage1现役DGT图架构见ADR-0116。Stage2/3取消ObjectEncoder，Base按ADR-0117为22实验+两模拟。运行路径以正式 YAML 和 [v4手册](docs/v4-runbook.md) 为准；旧Object/v5加载需历史Git版本，历史产物只读，禁止跨合同加载。


## 科学红线

[ADR-0120](docs/adr/0120-entity-token-attention-late-fusion.md)替代下表旧Base的MLP专家、2478D扁平专家入口和条件/池化合同：新Base为1024D/8头/2层实体token注意力，L1仅对象内部交互，L2的GLOBAL/GROUP/PRIVATE联合实体与物理条件；三槽保留至Task Gate后拼接3075D。溶剂槽位0、溶质槽位2。移除新模式的均值残差、压缩式PartnerInteraction及FiLM，保留逐token残差。Stage1、owner/样本加权/预算不变，新输出`outputs/v4/stage{2,3}/token_attention/`；旧MLP配置及产物保持历史身份。

修改对应 Stage 前，必须阅读下表列出的 ADR；详细机制与数值以这些合同及 YAML 为准。

| 范围 | 必须保持的边界 | ADR |
|---|---|---|
| Stage 1 | 仅SMILES/Graph输入，独立512D 编码器、12层SMILES Transformer/8层独立双路2D DGT、CLS/原子与键均值拼接后投影512、1024残差MLP；learned1024、atom512，不输入角色/描述符。结构重建、停止梯度一致性、RDKit211（显式排除6项、Phi适用性mask、八环计数log1p、Gasteiger失败屏蔽14项PEOE）、离线Uni-Mol768、13电子目标及仅训练集原子电荷Linear512→1；现役系数和执行配置只读Base YAML及v4手册，sidecar不改语料/teacher。各项分子均值再按角色loss权重2/2/1；自然shuffle不变。fusion-only80/10/10，10轮、完整轮恢复；训练format4及仅编码器导出。每1000步rank0固定32分子梯度审计只报告，可CLI覆盖，不更新/调权/选模，不进入科研身份；独立14任务后训练只更新回归头，不替换编码器 | 0089/0090/0091/0092/0112/0113/0115/0116；执行0013/0014/0015/0017；0039仅历史v3 |
| Stage 2 | Stage1永久冻结/eval；缓存1235D实体、形式电荷角色及有效位。GLOBAL/GROUP L1并行读取1235D实体token，各专家内部投影到1024D并作对象内注意力；L2三个owner层级联合实体与物理条件，Task Gate后按三槽拼接3075D，无独立ObjectEncoder。仅五项模拟、thermophysical/solvation GROUP，无电子GROUP/原子 adapter | 0120/0095/0112；Stage1表示0089 |
| Stage 2 训练与产物 | 五模拟每联合step各取一个未耗尽batch，GLOBAL跨任务/GROUP组内按真实batch样本数加权，PRIVATE原始平均梯度，不乘任务补偿；`batch_sample_weighted_owner_raw_v1`及联合步调度进入身份，旧final拒绝迁移。256逻辑/微批、10轮末轮、完整stage2_final.pt/json；无仅编码器导出、精调/最优/last/evaluator。迁移GLOBAL、匹配GROUP及两模拟PRIVATE，严格身份和张量 hash | 0095/0097；历史0082/0083/0085 |
| Stage 3 | Base为22实验+两模拟（0117排除equilibrium_pressure/water_activity_coefficient、solvation更新为3684体系）；两电位归入phase_stability、限small/仅温度，G2_C01容量按0114。Phase1实验GLOBAL/GROUP/PRIVATE；Phase2冻结GLOBAL；Phase3仅PRIVATE。模拟Phase1冻结，Phase2/3仅自身PRIVATE，不进共享梯度/分母/调度预算。共同锚点与末轮拼接 | 0095/0117；机制0050/0070 |
| Stage 3 owner | 现役Entity-HoME Phase1/2的GLOBAL跨实验任务、GROUP组内按当前真实batch样本数加权平均；PRIVATE保持原始平均Loss梯度，模拟不进共享分母。Phase3单任务更新；`batch_sample_weighted_owner_raw_v1`及权重规则进入训练身份。历史非Entity保留0070合同。owner LR/存续期与规模类别默认、任务覆盖项的宽度/dropout进入身份；提前结束用`requires_grad=False`，不删除任务；零预算PRIVATE逐bit继承锚点。Base训练集/验证集microbatch上限1024 | 0096；预算0050/0055；诊断0054 |
| Stage 3 诊断 | Entity-HoME在diagnostics.jsonl按整轮记录owner梯度前后范数mean/max、实际梯度step分母的裁剪率及真实共享样本贡献，冻结/缺失不混入均值、模拟不进共享分母；仅累计既有CPU标量/batch长度，不改训练身份，旧行保留。只读门控权重占比、归一化熵、PRIVATE 权重占比分位数；测试集 aggregate 合并折-样本。不得新增前向、进入 loss/选择或改预测 CSV；历史实现不输出。pEC50 Phase3沿用032为2轮；旧Base的3轮仅历史 | 0054/0055/0098/0114 |

禁止恢复：Stage1 描述符/角色输入、重型token融合、指纹、角色平衡/重复采样、扩增倍率、多容量正式配置、轮中途恢复；Stage2/3不得反传Stage1，禁止Stage2渐进解冻、早停、最优/last、PCGrad或累加窗口；Stage3 四阶段、路由干预、门控校准；正式 Base 不启用 HPO/Optuna。独立静态 HPO 按 [ADR-0099](docs/adr/0099-simple-static-hpo.md) 和 [HPO手册](docs/hpo-simple-runbook.md) 执行，不恢复旧 HPO_v4。v4 alignment/RDKit/Uni-Mol/electronic辅助头只用于预训练，不进入仅编码器部署。ADR-0087候选保持历史v3身份，不增加正式Base数量。

ADR-0093仅扩展独立回归头后训练：YAML支持默认Linear及逐目标MLP/残差 MLP；Linear继承预训练头并做等价标准化换算，非线性采用任务局部随机初始化/dropout。编码器/Fusion永久冻结、entity1024/atom512输入不变，验证只报告；format2内嵌结构并兼容旧Linear format1，不改变Stage1/2/3身份或正式训练。

ADR-0110的独立描述符阈值筛选/复审入口已退役；正式列清单与数值策略只按ADR-0112/0113/0115及YAML执行。人工评审建议未自动接入训练；prepare/QC、有效mask、SHA检查、gradient audit及teacher/电荷审计继续保留。

ADR-0111将主项目及新teacher环境固定为包 `rdkit==2026.9.1` / runtime `2026.09.1`；原始217维清单不变（现役Base选择与接口见ADR-0112），旧版本prepare/teacher/sidecar/checkpoint仍严格隔离。新版本须重新prepare与训练，不改SHA、不混用旧来源；三个Stage继续使用 `configs/v4/stage*/base.yaml`，不设临时版本后缀Base。复用标准 `outputs/v4/stage*/base/` 前须完成用户按精确清单确认的旧运行清理/归档，不能覆盖遗留metadata/log或复用旧sidecar。主环境仍名 `ilume`，备份后仅清理旧RDKit并升级，不直接覆盖混装包，不改Python/PyTorch/CUDA/NumPy；基线独立环境和hash lock不变。

ADR-0112：现役Stage1 Base在`descriptor.excluded_names`仅排除`Ipc/Kappa3/SMR_VSA8/SlogP_VSA9/fr_Nhpyrrole/fr_amide`，监督及显式输入统一211维，保留12项有非有限值及17项高相关描述符；有限值拟合、标准化零填充和逐维有效mask不变。learned1024/atom512不变，实体1235、双槽L1输入2478由来源schema/identity解析。默认空排除仍为历史217/1241/2490且默认序列化省略；旧配置身份、ChemBERTa与baseline合同不批量修改。新Base须重新prepare/teacher/sidecar及三个Stage训练，旧输出只读，不自动迁移或清理。

ADR-0113：Base的`descriptor.value_policy: phi_mask_ring_log1p_v1`对非单连通/重原子少于3/键少于2的结构仅屏蔽Phi；八环计数统一log1p后按train有限值拟合scaler。Stage1监督和下游共用同一schema，零填充/mask/分子及角色权重不变，不双重log、不删分子、不截尾；211/1235/2478维不变。新schema format3，默认raw省略字段且兼容format2；数值policy进入corpus/训练/feature来源身份，新旧211也不能混用。独立复审工具已退役，正式数值策略保持有效。重新准备匹配teacher/sidecar并重训三个Stage，旧输出只读，实施不自动启动正式运行。

ADR-0115：现役Base在保留0113处理的基础上启用`phi_mask_ring_log1p_gasteiger_v2`，副本上严格检查Gasteiger参数及原子/隐式氢电荷；失败时屏蔽整组14项PEOE_VSA，不进scaler或描述符监督，下游填0。其他特征、维数、loss和采样不变；旧raw/v1行为保留，policy身份拒绝交叉复用。需新prepare/teacher/sidecar及三个Stage训练，旧产物只读，不自动运行或清理。

ADR-0116：现役Stage1 Base使用512D/8头/8层双路2D DGT、每路FFN1024，encoder77,986,987参数。仅最短路径≤8与长度≤18无弦环拓扑，逐通道pair score/value attention；原子/键双池化Linear1024→512，learned1024/atom512不变。新增独立DGT拓扑sidecar与prepare的`--dgt-topology-only`，绑定corpus/分片/结构/hash；不修改原corpus/teacher/电荷sidecar。固定query32非重入重计算；Base另启用CUDA局部`dgt_attention_compile`，保持dropout在编译区外、原score/value定义及FP32 softmax，CPU即时、错误不回退。梯度审计临时使用未编译attention以支持逐目标保留图求导，结束（含异常）恢复原开关，正常训练继续编译。此开关共享默认false/省略，不进入科研身份，和另两项DGT执行值一样严格核对恢复；旧缺省按false，不能切换开关续训，不自动缩batch/截断。旧D-MPNN默认序列化、state与预测保留，不能跨架构恢复；新Base从头训练Stage1并重跑下游，独立benchmark和历史消融不变。

ADR-0117：用户确认现役Base采用22实验＋两模拟辅助，显式移除equilibrium_pressure和water_activity_coefficient，solvation unique_systems按更新catalog为3684。六组、剩余任务recipe、训练超参及严格校验不变；不改原始数据/catalog/SHA，不批量更新历史optimized_3407、HPO、消融或基线。新Stage3身份拒绝旧24任务/旧solvation来源；不同任务集合macro不可直接比较。

ADR-0114：主模型Stage2/3 Base采用G2_C01，一个GLOBAL，GROUP专家数transport/thermophysical/phase_stability/dielectric_optical/solvation/biological为1/1/2/1/2/1；PRIVATE宽度不变，训练超参沿用007/032，3407与当前211维Stage1来源严格配对。旧HPO生成读取冻结G0_C05模板，selected.yaml及optimized_3407、WeakOpt、既有消融配置/结果保留历史身份；不能用旧权重或消融替代新主模型，需要重新训练。

ADR-0092原子电荷同canonical源行分别保留，不平均、不首末行选择；只扩展电荷监督，不复制语料或其他loss。sidecar format2及观察策略进入身份，旧sidecar/检查点不得交叉续训；无图同构映射仅跳过电荷记录并审计，策略进入身份；损坏资源/解析错误仍失败，来源与split检查不放宽，机制见ADR-0092与v4手册。

### 历史合同、消融与基线

- 历史实现/Capacity 保持五模态格式 v2、Stage 2 仅补偿 physics 的 loss 与既有精调；Stage 3 保持整模裁剪、`max(N_t,1000)` 虚拟过采样、80/20 精调和 `taskwise_refined`（ADR-0026/0027）。Capacity 是端到端预注册研究，不是正式多容量主线、严格缩放定律或仅编码器效应；只用 Stage 1 Base prepare 一次，共享 `outputs/experiments_v1/stage1/prepare/artifacts`。Stage 3 正式输入只用 `formal/*.yaml`，选择只读拼接后的验证，不读末轮均值或测试集。
- 旧Object消融配置和旧ObjectEncoder实现/加载已移除；现役独立Entity-HoME消融按 [ADR-0100](docs/adr/0100-v4-entity-home-ablations.md) 与 [消融手册](docs/v4-ablation-runbook.md) 执行。现役36份配置与4份已取消容量配置为 `configs/ablations/v4_3407/`，配对 `optimized_3407`，输出 `outputs/ablations/v4_3407/`；原split42的 `configs/ablations/v4/` 与产物保留。核心包含移除SMILES/分子图的完整重训，C11为仅Phase1，阶段增量按C11→C7→C0比较；C12/C13以冻结ChemBERTa-2 MTR/MLM替换Stage1，不提供显式RDKit描述符（[ADR-0107](docs/adr/0107-chemberta2-frozen-home-ablations.md)）；固定训练seed42、不追加seed稳定性测试。历史消融与历史实现/Capacity复现须使用对应Git版本，输出保留。
- ADR-0034/0036/0062/0068/0072/0073/0079/0080/0081 及 Stage3 候选仅作历史追溯，不作为活跃入口；旧输出只读，不跨身份加载。v1/Capacity 冻结合同仍有效。
- ADR-0086：十个正式基线对汽化热、热膨胀、HOMO、LUMO 独立单次训练，任务目录原训练集/验证集/测试集；不训练 partial charge，不更新实验任务模型。神经模型 10 轮、XGBoost 1000 trees 保持各自末轮配方。历史单任务 MLP 消融和历史 split 配置不扩展；现役实体 MLP 按 ADR-0100 独立注册。Stage3 模拟验证集/测试集使用五个 final 原单位预测均值，比较身份绑定完整行集合、来源 SHA 与原始训练集总体标准差（零方差为 1）；模拟榜单不得进入实验 wins/radar/scatter。
- 基线只复用注册表、split、canonical SMILES、条件/目标与评估口径，不改变 Stage 数值合同。按 ADR 索引读取各模型合同，不能把一个模型的预算推及其他模型。
- ADR-0045：MLP/D-MPNN/MoLFormer/ILBERT/SPMM/LlaSMol 均为 10 轮，XGBoost 为 1000 trees；全部发布末轮状态。AIonopedia、ILTransR 与 AIFC 也使用各自现役的 10轮配方。除模型 ADR 明定外，验证不驱动早停、选检查点、调度器或训练决策。
- AIonopedia（0049）只用通用已发布多模态检查点、完整官方下游拓扑和独立条件路径，禁止 property-specific 资源或 pure-Qwen。ILTransR（0057）只用通过一致性验证的通用预训练转换与共享全量微调编码器/TextCNN，禁止监督训练 property 检查点；条件使用全量 Stage 3 协变量的传导式总体z-score。
- D-MPNN（0042）所有注册表槽位共用唯一消息传递编码器，不跨任务/折共享。高级基线环境独立 hash-lock，不改主环境、不自动安装、不静默回退。基线不支持恢复，失败在新尝试完整重跑。

## 结构与入口

- Stage 实现仅在 `src/common`、`src/stage1`～`src/stage3`；基线在 `benchmarks/`，消融在 `ablations/`，二者禁止被 Stage 导入。消融代码不得放回 `benchmarks/`。
- `common` 只收至少两个 Stage 实际复用的原子功能；不建 `utils.py`。跨 Stage 只导入公开合同，禁止 `_private_symbol` 或 `import *`。
- 运行入口仅 `scripts/stage{1,2,3}/*.py`、`scripts/benchmarks/*.py`，不恢复 console CLI、shell/smoke/matrix/折 runner 或自动搜索入口；允许 ADR-0099 的独立 `scripts/HPO/*.py` 静态配置批量入口和 ADR-0106 的独立 `scripts/generation/*.py` 专用生成研究入口。生成实现在 `generation/`，按单个完整请求训练，移除通用目标编码与跨请求重标记；只读 ADR-0105 的公开冻结 Predictor，Stage 禁止反向依赖生成模块；Stage 科研算法仍在对应 Stage。
- 科研与 prepare 参数写完整自包含 YAML；`preparation` 不进入实验身份。`--output`、`--resume`、Stage 3 折/评估 selector 是运行参数。
- Stage 3 训练入口必填 `--fold`，可多折；`--output` 是共同根目录，运行位于 `foldN/`。正式折并发只在该入口用 spawn worker 和显式设备槽调度，不下沉到 `src/stage3`；独立HPO外层使用设备队列，max-parallel任务槽轮流分配到设备，允许一卡多任务；底层折并行固定1，禁止嵌套放大并发。

Stage1训练DataLoader在W>0时按实际最大训练分片S、全局batch B和每rank worker数W使用`prefetch_factor=2+ceil(S/(B×W))`，提前排到下一分片；验证仍为2、W=0同步不变。不重排batch、不改变mask/RNG或放宽SHA；科研/缓存身份不变，新进程生效。额外占用CPU/pinned内存及在途句柄；pin线程须把CPU局部键索引/拓扑复制为主进程非共享storage，避免深队列保留IPC句柄导致ancdata错误。不承诺隐藏首分片或消除worker重复解包，执行细节见v4手册。

ADR-0122：Stage1 Base及H100配置在prepare实体QC启用`data.max_graph_nodes: 160`，原子数或无向键数严格超限才排除完整结构，等于上限保留；当前新增排除14条扩增训练样本，现有已入库原始样本不变。记录原因/大小审计，不截断、不改原始文件、不在训练中跳过。默认null/序列化省略以保留历史身份；新上限进入corpus身份并在训练时核对，旧prepared/teacher/电荷/拓扑及checkpoint不得直接混用。H100专用`base-100h.yaml`为全局batch84/5轮，双rank每卡42；标准Base仍为256/10轮。其余模型、loss和训练策略不变，不自动启停、归档或覆盖旧运行。

## 数据、身份、输出与恢复

- 4090是统一训练产物归档设备。无论在3090、其他服务器或本机训练，训练结果和完整模型权重都必须回传至4090对应项目目录；保留运行配置、日志、身份清单及恢复所需检查点，禁止仅同步报告或排除权重。回传与完整性核验遵循 [v4手册](docs/v4-runbook.md#跨设备训练产物回传)。

现役hydration仍使用random/cv1五折、无测试集；gas_solubility不是旧x_co2的同义任务。焓保留任务目录原标识，展示汽化焓，仅温度。基线历史20实验/四模拟配方不扩展。运行见 [v4手册](docs/v4-runbook.md)。

- 数据不进 Git；Stage1 共用 `data/stage1`。2026-10-20起，后续新实验统一采用 `data/3407`；本次HPO的共同训练seed仍为42（Stage3按fold派生），不因此更改基线各自的训练seed合同。已有正式Base、基线、消融配置及产物保留原split与身份，不批量改写或跨split复用。新实验须新建配套YAML及输出，Stage2/3来源、catalog和缓存严格配对；决定、关键报告保留策略及归档状态见 [HPO结果与最终配方](docs/hpo-results.md)。其他 split 的路径配对、缓存和汇总见 [v4手册](docs/v4-runbook.md#数据-split-与路径)。默认prepare将数据审计 `metadata.json` 写入配置的数据目录；独立HPO通过 `preparation.data_metadata_path` 写入输出审计目录，运行元数据引用同一位置；不同 split 使用独立 YAML 与输出，并保持身份校验。每次操作冻结 `run_config.yaml`、公开安全的 `metadata.json`，成功后写 `summary.json`；禁止用户名、hostname、私有绝对路径。
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

## 当前基线数据协议

ADR-0121新增 `flft_v4_3407_22_completion`：MLP/XGBoost/标准化ILBERT各其余12实验五折＋热膨胀/HOMO/LUMO三模拟，共189作业，含无条件ILBERT从头重训。配置 `configs/benchmarks/FLftv4_3407/completion/`、输出 `outputs/benchmarks/FLftv4_3407/completion_22/`；精确绑定Base22任务，电子模拟仍用data/stage1/properties。原结构/种子/10轮或1000trees不变，MLP/XGBoost当前RDKit独立新缓存，ILBERT训练行z-score。3090每卡独立任务、无DDP，用户启动；旧产物保留，汇总器本次未扩展，不得把12任务子集当完整22任务榜单。

ADR-0119为ILBERT增加独立 `train_only_zscore` 条件标准化：每任务/折仅训练行拟合均值和总体标准差，常量scale1；验证/测试使用checkpoint训练身份内保存的统计量，原始条件继续写入预测。旧raw配方与身份保持兼容，新旧互拒载；新配置为 `configs/benchmarks/FLftv4_3407/ilbert_condition_zscore.yaml`，输出 `outputs/benchmarks/FLftv4_3407/condition_zscore/ilbert/`，覆盖10变化任务五折50作业，不与旧attempt混合。原结构、资产和10轮固定配方不变，本次仅交付不自动训练。

ADR-0118新增 `configs/benchmarks/FLftv4_3407/` 的 `flft_v4_3407_22_retrain` 独立补训协议，authority为当前Base22任务，仅显式训练10项变化任务、五折共500作业，不运行模拟。原结构、资产、种子和配方不变，输出 `outputs/benchmarks/FLftv4_3407/retrain_22/<model>/`；每卡独立折次、不使用DDP。现役调度为szx其余9模型450作业，H100保留已启动LlaSMol的50作业；szx新sbatch位于 `/home/sunp/slurm/baseline/`，明确排除LlaSMol并保留Slurm的GPU分配，3090旧计划仅备用。旧根已迁至FLftv4_3407，过期任务及完整旧sweep记录按用户确认清单归档trash。历史12任务和三模拟未完成复用审计，不直接形成完整22任务榜单；正式训练仍由用户执行。

ADR-0117本轮只调整Base；以下基线仍保留24任务协议，尚未迁移到当前22任务catalog，重跑配对基线须另行建立协议与身份。

ADR-0102为现役基线入口：`configs/benchmarks/catalog_v4_3407`使用data/3407的24实验和HOMO/LUMO/热膨胀三模拟，十模型全量重训共1,230作业，输出outputs/benchmarks/v4_3407；各模型原配方与训练种子不变，不复用split42结果或清单。旧outputs/benchmarks按先报告精确数量/字节/目标并确认的规则在4090归档trash、szx删除，预训练资产和环境保留。ADR-0101及catalog_v4配置保留split42历史合同；旧训练身份不改写，正式训练仍由用户启动。
