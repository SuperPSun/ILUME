# ADR 导航

正式 YAML 定义运行参数，ADR 定义科研与兼容合同；后写 ADR 只在明确重叠范围内优先。先按主题查下表，不必顺序阅读全部文件。历史正文里的“现役”指决策当时，不能覆盖本索引的替代关系。

## 主线与公共合同

| 主题 | ADR（按修订顺序） | 阅读重点 |
|---|---|---|
| v4 实体HoME Stage2/3 主线 | [0095](0095-v4-entity-home-without-object-encoder.md) | 无ObjectEncoder；GLOBAL/GROUP直接并行处理实体；五模拟、24实验+两模拟仅PRIVATE更新；旧0094及Object合同退役 |
| v4 冻结双视图Stage1 | [0089](0089-v4-frozen-dual-view-stage1.md)、[0090](0090-stage1-v4-residual-encoder-capacity.md)、[0091](0091-stage1-v4-loss-weights-gradient-audit.md)、[0092](0092-stage1-atom-charge-and-frozen-regression-heads.md) | Stage1 learned1024、自然shuffle与2/2/1 loss权重；约60.73M 编码器，仅训练集原子电荷、重复结构独立观察/sidecar format2及1000步只读审计；现役系数/batch/workers读[Base YAML](../../configs/v4/stage1/base.yaml)和[v4手册](../v4-runbook.md)，正文旧数值为历史；独立冻结回归头后训练不替换编码器；下游永久冻结Stage1 |
| 历史 v2/v3 表示与隔离 | [0039](0039-global-rdkit-v2-mainline.md) | 三模态 Stage 1、1024D 实体/Object/HoME；仅约束历史配置，现役表示由 0089 取代 |
| Stage 1 执行 | [0013](0013-stage1-full-corpus-ddp.md)、[0014](0014-stage1-prepare-performance-and-corpus-v2.md)、[0015](0015-stage1-high-throughput-epoch-resume.md)、[0017](0017-stage1-base-runtime-profile.md) | 全量轮、prepare/运行时、DDP 与完整轮恢复 |
| Stage1 独立回归头 | [0092](0092-stage1-atom-charge-and-frozen-regression-heads.md)、[0093](0093-stage1-configurable-frozen-predictors.md) | 冻结 entity1024/atom512；Linear/MLP/残差 MLP 默认与逐目标覆盖；独立 format2，兼容旧 Linear format1；验证只报告 |
| 历史 v3/v4 Stage2-HoME | [0089](0089-v4-frozen-dual-view-stage1.md)（v4 表示与冻结边界）、[0082](0082-home-mainline-and-core-ablations.md)、[0083](0083-stage2-home-full-artifact-evaluation.md)、[0085](0085-retire-stage2-home-evaluation.md)、[0079](0079-stage2-home-stage3-home-transfer-ablation.md)（历史配方来源）、[0025](0025-stage2-homo-lumo-scalar-tasks.md) | 九任务仅物理监督 HoME、完整模型产物与 GLOBAL/GROUP 迁移；独立评估由 0085 退役；正式数值读 YAML |
| Stage 3 训练机制与历史任务集 | [0089](0089-v4-frozen-dual-view-stage1.md)（v4 表示与冻结边界）、[0084](0084-stage3-simulation-phase2-phase3.md)、[0020](0020-stage3-v1-sparse-home-pcgrad.md)、[0046](0046-stage3-ownership-clipping-raw-sampling.md)、[0048](0048-stage3-owner-lifetime-three-phase-training.md)、[0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md)、[0070](0070-stage3-retire-pcgrad.md)、[0075](0075-stage2-zero-update-stage3-object-phase1.md) | 历史20项实验+五项模拟；历史v5任务集见0094；现役v4由0095替代；稀疏 HoME、原始样本采样/裁剪、三阶段 owner 存续期/容量、ObjectEncoder Phase 1 |
| Stage 3 配方 | [0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md)、[0055](0055-stage3-pec50-phase3-single-variable-rollback.md) | owner 默认/覆盖/零预算与现役任务例外，完整数值读 YAML；0051～0053 已并入历史摘要 |
| Stage 3 诊断 | [0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md) | 只读任务门控统计、折-样本聚合与输出边界 |
| 历史 Stage3 二十任务目录 | [0088](0088-stage3-hydration-replaces-transfer.md)、[0067](0067-stage3-twenty-task-catalog.md) | hydration 替换 transfer、移除 volume expansion；主线/核心消融/基线的任务合同依据与产物边界 |
| 身份与历史实现精调 | [0021](0021-identity-audit-contract-v1.md)、[0027](0027-late-taskwise-refinement.md) | 语义身份/审计；0027 精调只约束历史实现/Capacity |
| 模拟性质基线/评估 | [0095](0095-v4-entity-home-without-object-encoder.md)、[0094](0094-v5-unary-pair-five-task-stage2.md)、[0086](0086-scalar-simulation-baselines-and-reporting.md) | 基线维持历史四项；现役v4 Stage3两项/历史四项五模型原单位集成按协议隔离，独立验证集/测试集榜单；Stage2 evaluate 保持退役 |
| 结果报告 | [0023](0023-unified-evaluation-reporting.md)、[0031](0031-stage3-summary-normalization-relaxation.md)、[0043](0043-retire-stage2-evaluation-and-v2-refinement.md)（历史）、[0061](0061-ilume-task-scatter-summary.md)、[0085](0085-retire-stage2-home-evaluation.md) | Stage3 实验与 0086 独立模拟榜单；Stage2 evaluator/报告由 0085 退役 |

## 基线与内部消融

共同预算与末轮状态原则见 [0045](0045-fixed-budget-baseline-training.md)；后加入的模型以自己的 ADR 为准。实现分别位于 `benchmarks/` 与 `ablations/`，操作步骤见 [README](../../README.md#baselines-and-ablations)。

| 模型/实验 | ADR |
|---|---|
| MLP / ECFP-XGBoost | [0022](0022-mlp-ecfp-xgboost-baselines.md) |
| D-MPNN | [0028](0028-chemprop-dmpnn-baseline.md)、[0042](0042-dmpnn-shared-component-encoder.md) |
| MoLFormer | [0029](0029-molformer-baseline.md)、[0030](0030-molformer-throughput-contract.md) |
| ILBERT | [0032](0032-ilbert-baseline.md) |
| SPMM | [0035](0035-spmm-baseline.md)、[0037](0037-spmm-wordpiece-character-limit.md)、[0038](0038-spmm-throughput-contract.md) |
| LlaSMol | [0040](0040-llasmol-mistral-7b-baseline.md) |
| AIonopedia | [0049](0049-aionopedia-multimodal-baseline.md) |
| ILTransR | [0057](0057-iltransr-stage3-baseline.md) |
| AIFC | [0060](0060-aifc-stage3-baseline.md) |
| 核心三项消融 | [0089](0089-v4-frozen-dual-view-stage1.md)（历史 v4）、[0082](0082-home-mainline-and-core-ablations.md)、[0084](0084-stage3-simulation-phase2-phase3.md)、[0077](0077-stage3-single-task-mlp-v2-ablation.md)（历史配方来源） |

## 冻结合同与历史

| 范围 | 入口与状态 |
|---|---|
| v3 Stage2/Stage3 HoME 候选 | [0087](0087-stage2-stage3-home-base1-candidates.md)：base1-1～base1-10 保留原配对来源、容量与预算合同，不是 v4 入口 |
| 旧迁移/微调/跨域 HoME 实验 | [0079](0079-stage2-home-stage3-home-transfer-ablation.md)、[0080](0080-stage2-home-transfer-full-finetune.md)、[0081](0081-stage2-stage3-cross-domain-home.md)：历史；正式身份由 0082 替代 |
| 旧 RDKit / transfer / 路由 / 分组候选 | [0034](0034-rdkit-2d-home-representation-ablation.md)、[0036](0036-no-stage1-rdkit-stage2-stage3-ablation.md)、[0062](0062-stage2-stage3-full-transfer-matrix.md)、[0072](0072-stage3-transfer-knowledge-hierarchy-ablation.md)、[0073](0073-stage3-encoder-full-finetune-ablation.md)、[0063](0063-stage3-knowledge-graph-grouping-candidate.md)、[0064](0064-stage3-knowledge-graph-budget-candidates.md)、[0066](0066-stage3-knowledge-graph-targeted-small-experiments.md)、[0074](0074-stage3-base-global-group-capacity-candidates.md)：历史，不是活跃入口 |
| Capacity v1 | [0026](0026-capacity-v1-pipeline-study.md)、[0027](0027-late-taskwise-refinement.md)；历史实现端到端研究，HPO 已退役。固定运行见 [手册](../capacity-v1-runbook.md) |
| AIonopedia 128维宽度标量预测头 | [0076](0076-aionopedia-scalar-head-capacity-comparison.md)：已退役；现役配置不再支持 128 宽头 |
| AIonopedia 新增图侧条件通路消融 | [0078](0078-aionopedia-extra-graph-conditions-ablation.md)：已退役；现役模型保留完整条件通路 |
| Stage 1 基础 | [0001](0001-data-and-role-sampling.md)、[0002](0002-descriptor-schema-and-tokens.md)、[0003](0003-smiles-tokenizers.md)、[0004](0004-fourth-modality-and-training.md)、[0005](0005-exclude-invalid-pretraining-entities.md)；按各篇顶部区分有效数据/预处理规则与被 0039 取代的模态设计 |
| 早期 Stage 1/2/3、Object v1/v2、四阶段训练 | [合并历史摘要](history.md)：0006～0012、0016、0018、0047，均已被取代；保留原编号、理由和 Git 原文定位 |
| 旧 Stage 2 评估 | [0023](0023-unified-evaluation-reporting.md)、[0024](0024-stage2-partial-charge-benchmark-suite.md)、[0025](0025-stage2-homo-lumo-scalar-tasks.md)、[0027](0027-late-taskwise-refinement.md) 的旧榜单由 [0043](0043-retire-stage2-evaluation-and-v2-refinement.md) 退役；[0083](0083-stage2-home-full-artifact-evaluation.md) 的正式 HoME 五任务独立评估也由 [0085](0085-retire-stage2-home-evaluation.md) 退役 |
| PRIVATE 配方试验 | [0051～0053 历史摘要](history.md#adr-0051)；有效机制见 0050，最终任务设置见 0055 |
| v2 Stage 3 HPO | [0041](0041-stage3-v2-three-phase-hpo.md)：搜索入口退役；准备产物身份的数据/训练分离修订仍需按当前实现核对，不能据此恢复搜索 |
| 路由 / 门控校准 | [0056](0056-stage3-inference-only-routing-ablation.md)、[0059](0059-stage3-gate-only-post-training-calibration.md)：已退役，保留问题、负结果与不恢复边界 |
| Stage2→Stage3 等行数迁移矩阵 | [0065](0065-stage2-stage3-balanced-transfer-matrix.md)、[0071](0071-retire-balanced-stage2-stage3-transfer.md)：已退役；旧输出只读，现役主线见 0095 |

历史文件数不代表现役方案数。需要复现旧决定时查对应 Git 版本；日常运行只从正式 YAML 和上方现役合同进入。

- [ADR-0095](0095-v4-entity-home-without-object-encoder.md)：现役 v4 HoME 直接消费冻结实体、24实验与两项仅PRIVATE更新模拟。
