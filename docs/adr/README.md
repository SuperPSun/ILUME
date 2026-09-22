# ADR 导航

正式 YAML 定义运行参数，ADR 定义科研与兼容合同；后写 ADR 只在明确重叠范围内优先。先按主题查下表，不必顺序阅读全部文件。历史正文里的“现役”指决策当时，不能覆盖本索引的替代关系。

## 主线与公共合同

| 主题 | ADR（按修订顺序） | 阅读重点 |
|---|---|---|
| v2 表示与隔离 | [0039](0039-global-rdkit-v2-mainline.md) | 三模态 Stage 1、1024D entity/Object/HoME；v1 隔离 |
| Stage 1 执行 | [0013](0013-stage1-full-corpus-ddp.md)、[0014](0014-stage1-prepare-performance-and-corpus-v2.md)、[0015](0015-stage1-high-throughput-epoch-resume.md)、[0017](0017-stage1-base-runtime-profile.md) | 全量 epoch、prepare/runtime、DDP 与完整 epoch 恢复 |
| Stage 2 | [0019](0019-stage2-catalog-object-v3.md)、[0025](0025-stage2-homo-lumo-scalar-tasks.md)、[0043](0043-retire-stage2-evaluation-and-v2-refinement.md)、[0044](0044-stage2-v2-task-compensated-teacher-loss.md) | Object v3、HOMO/LUMO、v2 joint-only、task-compensated teacher |
| Stage 3 训练 | [0020](0020-stage3-v1-sparse-home-pcgrad.md)、[0046](0046-stage3-ownership-clipping-raw-sampling.md)、[0048](0048-stage3-owner-lifetime-three-phase-training.md)、[0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md) | sparse HoME、raw sampling/clipping、三阶段 owner lifetime/capacity |
| Stage 3 recipe | [0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md)、[0055](0055-stage3-pec50-phase3-single-variable-rollback.md) | owner 默认/覆盖/零预算与现役 task 例外，完整数值读 YAML；0051～0053 已并入历史摘要 |
| Stage 3 diagnostics | [0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md) | 只读 task gate 统计、fold-sample 聚合与输出边界 |
| Stage 3 分组候选 | [0063](0063-stage3-knowledge-graph-grouping-candidate.md) | 知识图谱六组、YAML 驱动分组与 Base 隔离 |
| Stage 3 分组预算候选 | [0064](0064-stage3-knowledge-graph-budget-candidates.md) | base1_1～base1_5：四个诊断与一个组合，固定分组与独立身份 |
| Stage 3 分组定向小实验 | [0066](0066-stage3-knowledge-graph-targeted-small-experiments.md) | base2_1～base2_5：singleton、大组容量与 task lifetime |
| Stage 3 二十任务 catalog | [0067](0067-stage3-twenty-task-catalog.md) | 移除 volume expansion；现役 v2/消融/baseline/transfer target 同步与 artifact 边界 |
| 身份与 legacy refinement | [0021](0021-identity-audit-contract-v1.md)、[0027](0027-late-taskwise-refinement.md) | semantic identity/audit；0027 refinement 只约束 legacy/Capacity |
| Reporting | [0023](0023-unified-evaluation-reporting.md)、[0031](0031-stage3-summary-normalization-relaxation.md)、[0043](0043-retire-stage2-evaluation-and-v2-refinement.md)、[0061](0061-ilume-task-scatter-summary.md) | Stage 3 schema、comparison、Stage 2 reporting 退役、task scatter |

## Baseline 与内部消融

共同预算与 final-state 原则见 [0045](0045-fixed-budget-baseline-training.md)；后加入的模型以自己的 ADR 为准。实现分别位于 `benchmarks/` 与 `ablations/`，操作步骤见 [README](../../README.md#baselines-and-ablations)。

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
| Single-task MLP | [0033](0033-stage3-single-task-mlp-ablation.md) |
| RDKit-HoME | [0034](0034-rdkit-2d-home-representation-ablation.md) |
| No-Stage1 | [0036](0036-no-stage1-rdkit-stage2-stage3-ablation.md) |
| Stage2→Stage3 全迁移矩阵 | [0062](0062-stage2-stage3-full-transfer-matrix.md) |
| Stage2→Stage3 等行数迁移矩阵 | [0065](0065-stage2-stage3-balanced-transfer-matrix.md) |
| Stage2→Stage3 下游联合适配 | [0068](0068-stage2-stage3-joint-downstream-adaptation.md) |

## 冻结合同与历史

| 范围 | 入口与状态 |
|---|---|
| Capacity v1 | [0026](0026-capacity-v1-pipeline-study.md)、[0027](0027-late-taskwise-refinement.md)；legacy 端到端研究，HPO 已退役。固定运行见 [手册](../capacity-v1-runbook.md) |
| Stage 1 基础 | [0001](0001-data-and-role-sampling.md)、[0002](0002-descriptor-schema-and-tokens.md)、[0003](0003-smiles-tokenizers.md)、[0004](0004-fourth-modality-and-training.md)、[0005](0005-exclude-invalid-pretraining-entities.md)；按各篇顶部区分有效数据/预处理规则与被 0039 取代的模态设计 |
| 早期 Stage 1/2/3、Object v1/v2、四阶段训练 | [合并历史摘要](history.md)：0006～0012、0016、0018、0047，均已被取代；保留原编号、理由和 Git 原文定位 |
| Stage 2 evaluation | [0023](0023-unified-evaluation-reporting.md)、[0024](0024-stage2-partial-charge-benchmark-suite.md)、[0025](0025-stage2-homo-lumo-scalar-tasks.md)、[0027](0027-late-taskwise-refinement.md) 的相关段落已由 [0043](0043-retire-stage2-evaluation-and-v2-refinement.md) 退役；未涉及的训练/通用 reporting 合同保留 |
| PRIVATE recipe 试验 | [0051～0053 历史摘要](history.md#adr-0051)；有效机制见 0050，最终 task 设置见 0055 |
| v2 Stage 3 HPO | [0041](0041-stage3-v2-three-phase-hpo.md)：搜索入口退役；prepared identity 的数据/训练分离修订仍需按当前实现核对，不能据此恢复搜索 |
| Routing / gate calibration | [0056](0056-stage3-inference-only-routing-ablation.md)、[0059](0059-stage3-gate-only-post-training-calibration.md)：Retired，保留问题、负结果与不恢复边界 |

历史文件数不代表现役方案数。需要复现旧决定时查对应 Git 版本；日常运行只从正式 YAML 和上方现役合同进入。
