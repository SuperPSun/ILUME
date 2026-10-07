# ILUME workspace rules

## 使用顺序与修改边界

- 运行步骤读 [README](README.md)，科研合同按 [ADR 索引](docs/adr/README.md) 查对应主题，再读正式 YAML 与当前实现。旧设计只从 [历史摘要](docs/adr/history.md) 追溯，不作为现役入口。
- 整理、重构不得改变数据筛选/split、tokenizer、descriptor/fingerprint、masking、模型、loss、优化顺序、验证/选模及任何 Stage 数值行为。科研问题单独记录，不顺手修改。
- `configs/v4/stage1`、`configs/v4/stage2`、`configs/v4/stage3` 是现役 Base，三个核心消融在 `configs/v4/ablations`（ADR-0089）。v2 Stage1 与 v3 HoME 配置/产物保持历史身份；`configs/v1` 与 `configs/experiments_v1` 冻结 legacy 合同。禁止跨合同加载 artifact/checkpoint。

## 科学红线

修改对应 Stage 前，必须阅读下表列出的 ADR；详细机制与数值以这些合同及 YAML 为准。

| 范围 | 必须保持的边界 | ADR |
|---|---|---|
| Stage 1 | 仅SMILES/Graph输入，独立512D encoder、12层SMILES Transformer/8独立residual图block、CLS/atom mean、1024残差MLP；learned1024、atom512，不输入role/descriptor。结构重建、stop-gradient一致性、RDKit217、离线Uni-Mol768、13电子目标；原系数1/0.1/0.5/0.25/0.1，另有train-only原子电荷Linear512→1及独立系数0.1，标签sidecar不改语料/teacher。各项分子均值再按角色loss权重2/2/1；自然shuffle不变。fusion-only80/10/10，10轮、batch128、完整epoch恢复；训练format4及encoder-only导出。每1000步rank0固定32分子gradient audit只报告，可CLI覆盖，不更新/调权/选模，不进入科研身份；独立14任务后训练只更新回归头，不替换encoder | 0089/0090/0091/0092；执行0013/0014/0015/0017；0039仅历史v3 |
| Stage 2 | Stage1永久eval/冻结/不进optimizer，缓存learned1024+标准化RDKit217及atom512。ObjectEncoder内部1241→1024投影属于Object优化块；原1024接口不创建投影。九task physics-only HoME，无Stage2 teacher loss；HOMO/LUMO、QM mask、Partial Charge分子等权保留 | 0089/0082；机制0019/0025 |
| Stage 2 训练与产物 | 256 逻辑 batch、256 微批、每逻辑 batch 一次 optimizer/scheduler update；10 epochs 末轮发布完整 `stage2_final.pt`、manifest 与 `stage2_encoder.pt`，不做 refinement、best/last。不提供独立 evaluate 入口或榜单；实验任务初始化只迁移表示、GLOBAL 与匹配的 thermophysical/solvation GROUP，simulation 预测支路额外从完整产物初始化电子 GROUP、五项 PRIVATE 和 atom adapter | 0082/0083/0084/0085；配方来源 0079 |
| Stage 3 | v4正式Stage2-HoME来源，严格SHA/owner/tensor hash；永久冻结Stage1、保存1241D slots。Phase1联合更新ObjectEncoder（含入口投影）与20-task Flat HoME；Phase2/3冻结ObjectEncoder并加入五项simulation的thermophysical/electronic GROUP/PRIVATE。共享GROUP模拟权重0.1、实验1，纯模拟电子GROUP/Phase3不降权；共同anchor、25个PRIVATE、固定末轮stitch、实验/模拟validation只读分开 | 0089/0084；机制0020/0046/0048/0067/0075 |
| Stage 3 owner | Phase 1/2 使用原始梯度的 task/group 加权聚合，Phase 3 单任务更新；`weighted_owner_raw_v1` 进入训练身份。owner LR/lifetime 与 size-class 默认、task override 的 width/dropout 进入 identity；提前结束用 `requires_grad=False`，不删除 task；零预算 PRIVATE 逐 bit 继承 anchor。Base train/valid microbatch 上限 1024 | 0050/0055/0070；诊断 0054 |
| Stage 3 diagnostics | 只读 gate mass、normalized entropy、PRIVATE mass 分位数；test aggregate 合并 fold-sample。不得新增 forward、进入 loss/selection 或改 prediction CSV；legacy 不输出。pEC50 Phase 3 为 3 epochs | 0054/0055 |

禁止恢复：Stage1 descriptor/role输入、重型token fusion、fingerprint、角色平衡/重复采样、augmentation multiplier、多容量正式配置、mid-epoch恢复；Stage2/3不得反传Stage1，禁止Stage2渐进解冻、early stopping、best/last、PCGrad或accumulation window；Stage3 four-phase、routing intervention、gate calibration或HPO/Optuna。v4 alignment/RDKit/Uni-Mol/electronic heads只用于预训练，不进入encoder-only部署。ADR-0087候选保持历史v3身份，不增加正式Base数量。

现役v4 Stage1 Base数值以YAML及v4手册为准：SMILES/atom/bond/alignment/RDKit/Uni-Mol/electronic/partial-charge系数为0.20/1/1/0.10/0.25/0.50/0.25/0.25，全局batch512、每rank32 workers；上表与ADR中的先前系数/batch128为历史数值，loss定义与审计边界不变。

ADR-0093仅扩展独立回归头后训练：YAML支持默认Linear及逐目标MLP/residual MLP；Linear继承预训练头并做等价标准化换算，非线性采用task-local随机初始化/dropout。encoder/Fusion永久冻结、entity1024/atom512输入不变，validation只报告；format2内嵌结构并兼容旧Linear format1，不改变Stage1/2/3身份或正式训练。

ADR-0092原子电荷同canonical源行分别保留，不平均、不首末行选择；只扩展电荷监督，不复制语料或其他loss。sidecar format2及观察策略进入身份，旧sidecar/checkpoint不得交叉续训；无图同构映射仅跳过电荷记录并审计，策略进入身份；损坏资源/解析错误仍失败，来源与split检查不放宽，机制见ADR-0092与v4手册。

### Legacy、消融与 baseline

- legacy/Capacity 保持五模态 format v2、Stage 2 仅补偿 physics 的 loss 与既有 refinement；Stage 3 保持整模 clipping、`max(N_t,1000)` virtual oversampling、80/20 refinement 和 `taskwise_refined`（ADR-0026/0027）。Capacity 是端到端预注册研究，不是正式多容量主线、strict scaling law 或 encoder-only effect；只用 Stage 1 Base prepare 一次，共享 `outputs/experiments_v1/stage1/prepare/artifacts`。Stage 3 正式输入只用 `formal/*.yaml`，选择只读 stitched validation，不读末轮均值或 test。
- 核心消融仅有w/o Stage1、w/o Stage2、w/o Stage3-HoME（0089/0082/0084），v4版本隔离：w/o Stage1固定seed随机、永久冻结的同结构encoder并保留显式RDKit；w/o Stage2配对零更新ObjectEncoder，20实验任务；MLP用v4正式prepared冻结表示，20任务、1024→512、10轮末轮。后两项还缺少模拟辅助训练，不能单独解释routing/Stage2权重。
- ADR-0034/0036/0062/0068/0072/0073/0079/0080/0081 及 Stage3 候选仅作历史追溯，不作为活跃入口；旧输出只读，不跨身份加载。v1/Capacity 冻结合同仍有效。
- ADR-0086：十个正式 baseline 对 heat of vaporization、thermal expansion、HOMO、LUMO 独立单次训练，catalog 原 train/valid/test；不训练 partial charge，不更新实验任务模型。神经模型 10 epochs、XGBoost 1000 trees 保持各自末轮配方。单任务 MLP 消融和历史 split 配置不扩展。Stage3 simulation valid/test 使用五个 final 原单位预测均值，比较身份绑定完整行集合、来源 SHA 与原始 train population std（零方差为 1）；模拟榜单不得进入实验 wins/radar/scatter。
- Baseline 只复用 registry、split、canonical SMILES、condition/target 与评估口径，不改变 Stage 数值合同。按 ADR 索引读取各模型合同，不能把一个模型的预算推及其他模型。
- ADR-0045：MLP/D-MPNN/MoLFormer/ILBERT/SPMM/LlaSMol 均为 10 epochs，XGBoost 为 1000 trees；全部发布 final state。AIonopedia、ILTransR 与 AIFC 也使用各自现役的 10-epoch recipe。除模型 ADR 明定外，validation 不驱动早停、选 checkpoint、scheduler 或训练决策。
- AIonopedia（0049）只用 generic released multimodal checkpoint、完整官方 downstream topology 和独立 condition path，禁止 property-specific 资源或 pure-Qwen。ILTransR（0057）只用 parity-validated generic pretraining 转换与共享 full-fine-tuning encoder/TextCNN，禁止 supervised property checkpoint；condition 使用全量 Stage 3 covariates 的 transductive population z-score。
- D-MPNN（0042）所有 registry slot 共用唯一 message-passing encoder，不跨 task/fold 共享。高级 baseline 环境独立 hash-lock，不改主环境、不自动安装、不静默回退。Baseline 不支持 resume，失败在新 attempt 完整重跑。

## 结构与入口

- Stage 实现仅在 `src/common`、`src/stage1`～`src/stage3`；baseline 在 `benchmarks/`，消融在 `ablations/`，二者禁止被 Stage 导入。消融代码不得放回 `benchmarks/`。
- `common` 只收至少两个 Stage 实际复用的原子功能；不建 `utils.py`。跨 Stage 只导入公开 contract，禁止 `_private_symbol` 或 `import *`。
- 运行入口仅 `scripts/stage{1,2,3}/*.py`、`scripts/benchmarks/*.py`，不恢复 console CLI、shell/smoke/matrix/fold runner 或搜索入口。
- 科研与 prepare 参数写完整自包含 YAML；`preparation` 不进入实验身份。`--output`、`--resume`、Stage 3 fold/evaluation selector 是运行参数。
- Stage 3 train 必填 `--fold`，可多 fold；`--output` 是共同 root，run 位于 `foldN/`。并发只在该入口用 spawn worker 和显式设备槽调度，不下沉到 `src/stage3`。

## 数据、身份、输出与恢复

现役 Stage3 以 `experiment/hydration` 替代 `experiment/transfer`（[ADR-0088](docs/adr/0088-stage3-hydration-replaces-transfer.md)）：单 solute + temperature_K、151 个体系、solvation GROUP、small 默认 PRIVATE 配方、random 五折、无 test。旧 transfer 产物保持历史身份；更换后的任务集合需要新的 Stage3 与 baseline 产物，禁止覆写既有输出。

- 数据不进 Git；prepare 写 `data/stage*/metadata.json`。每次操作冻结 `run_config.yaml`、公开安全的 `metadata.json`，成功后写 `summary.json`；禁止用户名、hostname、私有绝对路径。
- 新 train/evaluate 输出不可覆盖。reusable Stage 1/2 prepare 只在数据身份不变时刷新允许忽略的执行/模型字段（ADR-0019）。Stage 2 prepared data 不绑定 HoME model；正式 HoME 不生成 teacher cache，模型合同属于训练 checkpoint/encoder。
- 保留 prepared SHA 和完整性检查；跨 Stage 保留 ADR-0021 semantic identity、必要 state hash；正式 HoME transfer 还绑定 Stage2 final artifact 与 encoder SHA。
- 显式 resume 严格校验阶段、fold、配置、step/epoch、optimizer/scheduler/AMP。Stage 1 只从完整 epoch 恢复，可改变 world size；Stage 2 HoME 另校验 owner、registry/model、RNG、数据/任务规模、数学精度及 optimizer implementation；旧 Object v2/v3、实验 HoME 不迁移到正式身份。
- Stage 3 另校验 resolved plan、ownership、Stage 2 SHA、数据/normalization；raw permutation 与 branch RNG 由 seed/fold/phase/scope/epoch 重建，legacy 重建 virtual sampler。布尔 `--resume` 仅 skip identity 一致且 summary/final artifact/manifest 完整的 fold；恢复必须匹配 checkpoint、metrics/diagnostics 尾部、owner update/freeze state，legacy 匹配两份根 history，不截断、不猜测。
- 周期checkpoint不可覆盖。Stage1 `last.pt`保留辅助头用于恢复，另导出冻结encoder-only `stage1_encoder.pt`；Stage2末轮完整九任务 `stage2_final.pt/json` 与 `stage2_encoder.pt`；Stage3 Phase1全量/Phase2/3 anchor-owner delta，最终`three_phase_final.pt/json`。v4使用独立kind/format4、Stage3 plan11/training15；不得跨v3加载。Stage2/3不生成best/last，legacy仍用taskwise_refined。
- Stage 2 独立 evaluator/reporting 已退役（ADR-0085）；完整九任务模型与 `SimulationHoME.predict` 保留，五项模拟任务由 Stage3 final 预测，simulation validation 单独记录。simulated QM electrostatic/HF 权重仍保存在 Stage2 完整产物中。
- Reporting 发布 Stage 3 experimental 与独立 scalar simulation validation/test（ADR-0031/0061/0085/0086），不读取或发布 Stage 2 结果；summary snapshot schema 为 v4。榜首 Stage 3 ILUME 的 validation 五折与 test ensemble SVG 只由 summarizer 从完整性验证通过的 prediction 生成，不改 schema、指标或模型选择。

## 验证与清理

- Uni-Mol2 独立环境、权重获取/迁移与 audit 顺序以 [v4 手册](docs/v4-runbook.md) 为准。RDKit 比较导入后的 runtime 与 prepared feature contract，不只比较 pip metadata；完成全量 teacher cache 后才切回训练环境。
- 修改后运行 `pytest -q`；按风险检查十个Stage入口（含离线`stage1/teacher.py`和独立`stage1/regression.py`）与四个benchmark script的`--help`、compileall、diff/Markdown链接检查。只用临时小数据；不自动下载teacher权重、生成全量3D缓存或执行正式prepare/训练/evaluation/回归头后训练。先用独立audit输出验证teacher成功率、吞吐、存储，再由用户正式运行。
- 优先复用/修改现有测试。只有此前未覆盖且会造成实质损失的科研、resume、artifact/identity、CLI/reporting 或高风险调度合同，才新增最小行为测试；不为 private helper、搬家、简单重构或 coverage 扩测试。`tests/` 按 Stage/benchmark/common/architecture 集中组织，`conftest.py` 只放跨文件复用的小 fixture。
- `trash/` 不进 Git。移动旧 artifact/YAML/未消费数据或删除机器缓存前，报告精确文件数、大小、目标和冲突策略，等用户明确确认。不得覆盖、重排或删除既有 `trash/`。
