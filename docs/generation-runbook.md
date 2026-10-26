# 专用离子液体逆向设计手册

每份 Anion-first GFlowNet 只针对一个完整用户请求训练，可同时优化多个性质。请求固定后，通过同一奖励训练图编码器、前向策略和标量 `logZ`。性质/工况编码MLP、随机任务采样、跨请求重标记和组合泛化划分已移除。Stage1–3和五折Predictor保持冻结，合同见 [ADR-0106](adr/0106-dedicated-inverse-design.md)；[ADR-0105](adr/0105-universal-inverse-design.md)保留冻结推理及化学资源合同。

## 固定请求与训练配置

先编辑完整配置 [v1_3407.yaml](../configs/generation/v1_3407.yaml) 的 `request`，内容格式可参考 [request.example.yaml](../configs/generation/request.example.yaml)。正式科学参数全部内嵌在训练YAML，不再通过运行时 `--request` 切换。示例中的电导率/黏度、阈值和工况只说明格式，未作为用户正式选定任务。

一个固定请求包括：所选性质、各自方向/单位/阈值/权重、逐性质工况、必要溶质、重原子数/骨架家族/必须与禁止SMARTS、是否固定某个阴离子。改变其中任一字段，须建立新YAML、新协议和新训练输出；不能使用原权重直接响应新请求。

当前示例预算保持10,000次更新、每步8条在线轨迹和最多8条同请求回放；图编码宽度64、深度3，Adam学习率0.0003。这是独立生成研究的初始配方，未经过正式效果验证。阴离子自由选择时仍影响阳离子动作策略；用户固定阴离子时仅保留库中该成员。

## 化学资源与协议

从仓库根目录、现有 `ilume` 环境执行。资源提取不加载Predictor，也不训练：

```bash
conda run -n ilume python scripts/generation/prepare.py --config configs/generation/v1_3407.yaml --kind resources --output outputs/generation/dedicated_v1_3407/resource_candidates
```

`resources.draft.json` 包含阴离子、已知阳离子、BRICS候选骨架/片段和排除记录，记录来源文件SHA与行号。`pretraining_seen` 及 `pretraining_scope: source_corpus_only` 只表示位于源语料；实际训练曝光须核对绑定编码器的准备索引。不得将自动提取结果直接视为化学或合成可行性已审定。

选择实际研究使用的资源，填写 `review.chemistry` 和 `review.synthesis` 具体依据后，以新版本 `status: approved` 保存到配置的 `resources.library_path`。保留原草案和来源，不覆盖旧审定文件。JSON格式仍为：

```json
{
  "format_version": 1,
  "status": "draft",
  "review": {"chemistry": "", "synthesis": ""},
  "anions": [{"id": "chloride", "smiles": "[Cl-]", "family": "halide", "sources": ["填写可追溯来源"], "pretraining_seen": true}],
  "cores": [{"id": "ammonium", "smiles": "[1*][N+](C)(C)C", "family": "ammonium", "sources": ["填写可追溯来源"], "pretraining_seen": false}],
  "fragments": [{"id": "methyl", "smiles": "[1*:1]C", "sources": ["填写可追溯来源"]}]
}
```

示例保留draft，不能直接用于训练。连接位点用dummy原子的正整数isotope标记类型，片段恰有一个`:1`入口映射；仅相同类型位点单键连接，其他位点保留供后续生长。BRICS原标签不等于反应兼容表，需审定连接类型。片段中性、增加重原子且可完成所有端口，不隐式氢封端。每个候选只有一种阳离子和一种阴离子，允许多价，不展开化学计量；结构合法不证明实际为液态或可合成。

填写真实请求后冻结专用协议：

```bash
conda run -n ilume python scripts/generation/prepare.py --config configs/generation/v1_3407.yaml --kind protocol --output outputs/generation/dedicated_v1_3407/protocol
```

该步骤严格重载现有五折Predictor，仅为所选性质收集开发结构/工况参考，不读取正式测试标签。`protocol.json`的 `format_version: 2`、`kind: ilume_dedicated_gfn_protocol_v1` 绑定完整请求、所选能力表和Predictor来源。不再生成任务掩码划分或16组工况上下文。

## 训练、恢复与候选导出

确认资源与请求后，由用户启动训练：

```bash
conda run -n ilume python scripts/generation/train.py --config configs/generation/v1_3407.yaml --device cuda:0 --output outputs/generation/dedicated_v1_3407/train
```

完整轨迹通过同一请求的冻结Predictor评分，采用普通 Trajectory Balance；`logZ`为可训练标量，反向概率为合法父状态上的均匀概率。回放只复用同请求轨迹和预测，不改变目标或工况。权重几何满意度、五折分歧和适用域惩罚保持原公式。硬性质阈值仅用于交付筛选，训练仍使用平滑奖励。

输出含 `run_config.yaml`、`metadata.json`、`request.json`、资源/协议快照、`metrics.jsonl`、预测缓存、周期完整检查点以及 `gfn_final.pt/json`。检查点 `kind: ilume_dedicated_gfn_v1` 保存图/策略参数、标量logZ、优化器、RNG、回放、缓存及历史，绑定请求和全部科研身份。发布预定预算末状态，验证或测试不参与选模。

同请求恢复读取旧完整检查点，写新尝试目录：

```bash
conda run -n ilume python scripts/generation/train.py --config configs/generation/v1_3407.yaml --device cuda:0 --resume outputs/generation/dedicated_v1_3407/train/checkpoint_step_0000100.pt --output outputs/generation/dedicated_v1_3407/train_retry
```

恢复要求全部请求、资源、协议、配方及原执行身份一致。通用版检查点和旧协议明确拒载，不改身份迁移。旧产物不删除、不覆盖；全部结果与完整模型遵循4090归档规则。

使用同一配置采样：

```bash
conda run -n ilume python scripts/generation/sample.py --config configs/generation/v1_3407.yaml --checkpoint outputs/generation/dedicated_v1_3407/train/gfn_final.pt --device cuda:0 --output outputs/generation/dedicated_v1_3407/design_001
```

`sample`与`evaluate`都只接受配置内的固定请求，没有 `--request` 或 `--partition`。`evaluate.py`使用同一组配置和检查点参数，在新的输出目录统计该请求的计算研究指标。修改 `sampling` 的尝试次数、候选数量和评分预算可以复用原模型，修改科学请求则不允许。

## 性质、工况与可靠性

下表的 `T/P/f/λ/x` 分别对应 `temperature_K/pressure_kPa/frequency_MHz/wavelength_nm/x_water_unitless`，必须使用完整配置键；空白工况使用 `{}`。实际能力表以协议内正式解析结果为准。

| task（实验项前缀为 `experiment/`） | 用户单位 | 必需工况/额外结构 |
|---|---|---|
| electrical_conductivity | S/m | T、P |
| self_diffusion_coefficient | 10^-9*m^2/s | T、P |
| viscosity | mPa*s | T、P |
| thermal_conductivity | W/m/K | T、P |
| density | g/cm^3 | T、P |
| heat_capacity | J/mol/K | T、P |
| speed_of_sound | m/s | T、P |
| surface_tension | mN/m | T |
| equilibrium_pressure | kPa | T |
| glass_transition_temperature | K | 无 |
| melting_point | K | 无 |
| thermal_decomposition_temperature | K | 无 |
| dynamic_relative_permittivity | unitless | T、P、f |
| static_relative_permittivity | unitless | T、P |
| refractive_index | unitless | T、P、λ |
| solvation | kcal/mol | T；`solutes[task]` |
| pec50 | pEC50 | 无 |
| gas_solubility | unitless | T、P；`solutes[task]` |
| water_activity_coefficient | unitless | x、T、P |
| enthalpy_of_vaporization_or_sublimation | kJ/mol | T |
| anodic_potential_limit | V | T |
| cathodic_potential_limit | V | T |
| simulation/heat_of_vaporization | kJ/mol | T |
| simulation/thermal_expansion | K^-1 | T |

不推断缺失工况，不支持临时新增预测头。每个目标必须给出正权重及匹配单位；阈值可省略，此时用开发目标均值作平滑优化参照，交付不增设该阈值。log10性质的用户阈值按物理单位填写，内部转回原目标空间；物理显示是原目标空间五折均值的逆变换，不改成各折物理值均值。实验与模拟保持独立来源。

适用域逐性质使用开发离子对参考，综合同一参考行阴阳离子（及必要溶质）Morgan相似度、工况越界和电荷组合覆盖。五折分歧与适用域仅为风险筛查，不是校准置信区间或真实误差保证。

Predictor权重驻留CPU、按折移至指定设备，不同时占据五份GPU模型。预测缓存绑定模型、执行身份、结构、性质、工况和溶质；改变设备/精度/运行时版本会使用新缓存，恢复训练要求原执行身份。大规模化学空间的吞吐和内存须实测。

## 输出与研究验证

原始样本保存在 `raw_samples_0000.jsonl`，含重复、失败、轨迹、固定请求、每折预测、原单位结果、风险分项及来源。交付候选 `candidates_0000.json` 先严格筛选预测阈值、按离子对去重，再作Pareto分层与结构最远点选择。数量不足记录 `shortfall`，不放宽阈值。固定偏好的候选不代表全空间Pareto前沿覆盖。

`summary.json`报告固定 `request`、原始/交付多样性、达标率、候选不足、实际评分数和耗时，不再输出任务组合partition。无环阳离子的Murcko类别记 `acyclic`，不作为完整家族分类。精确超体积仅对一至三目标计算，参考点为各方向原目标空间开发z-score的−2；更多目标明确记未计算。

`sampling.max_property_queries` 限制新增唯一“离子对×性质×工况×溶质”的五折评分次数，完整组合预算不足时停止。所有对照使用相同请求、化学空间和后处理，报告实际预算、在线及回放训练成本。现有研究对照如下：

| 对照 | 执行方式 | 评价要求 |
|---|---|---|
| 无阴离子结构条件 | 新完整配置设置 `model.anion_conditioned: false` 并重训 | 固定请求，比较达标、结构分布及阴离子家族覆盖 |
| 独立重组 | `evaluate.py`追加 `--independent-pairing` | 独立抽取阴离子后重新评分；保留原生成阴离子身份 |
| 随机合法组装 | `evaluate.py`追加 `--random-policy` | 同一已锁定请求及相同评分预算 |
| 无分歧/无适用域惩罚 | 新完整配置将对应reward penalty置0并重训 | 独立新身份、新输出；比较性能与风险 |

本次不执行跨任务组合泛化实验；当前实现也不提供通用模型微调或跨请求恢复。若要研究另一个任务，明确新建配套请求、协议和模型。

## 当前状态与验收

实现验收使用临时小数据，验证可枚举图目标分布、多路径与固定阴离子、同请求完整恢复、跨请求/旧产物拒载、冻结推理、单位/工况、评分预算与候选不足；运行项目pytest、入口help、编译、diff与文档链接检查。正式资源准备、训练和评估仍由用户启动。

| 研究项 | 当前状态 |
|---|---|
| 化学与合成规则审定 | 待审定资源及依据 |
| 实际用户请求的专用GFN训练 | 待用户确定配置并运行 |
| 协同生成、随机组装与可靠性对照 | 待相同请求/预算的正式计算结果 |
| 真实误差与实际候选性质验证 | 待独立标签、模拟或实验 |

单元测试、示例阈值和同一Predictor评分均不替代正式研究证据。
