# Architecture Decision Records

正式 YAML 定义运行参数，ADR 定义会影响数据谱系、实验可比性或模型接口的稳定合同。后写 ADR 在重叠范围内优先；历史正文保留当时理由，不作为现役入口。

## 现役权威

- v2 主线：[ADR-0039](0039-global-rdkit-v2-mainline.md) 定义三模态 Stage 1、1024 维 entity/Object/HoME 表示和 v1 隔离边界。
- Stage 1 执行合同：[ADR-0013](0013-stage1-full-corpus-ddp.md)、[0014](0014-stage1-prepare-performance-and-corpus-v2.md)、[0015](0015-stage1-high-throughput-epoch-resume.md)、[0017](0017-stage1-base-runtime-profile.md)；与 ADR-0039 重叠的五模态决定只保留给 legacy v1/Capacity v1。
- Stage 2：[ADR-0019](0019-stage2-catalog-object-v3.md) 定义 Object v3；[ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 定义现役 HOMO/LUMO 与 reporting v2；[ADR-0027](0027-late-taskwise-refinement.md) 修订末期训练和最终评估 artifact；表示宽度由 ADR-0039 修订。
- Stage 3：[ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md) 定义 sparse-label HoME 与 hierarchical PCGrad；末期 PRIVATE-only refinement 由 [ADR-0027](0027-late-taskwise-refinement.md) 定义，主线宽度由 ADR-0039 修订。
- 跨 Stage identity/audit：[ADR-0021](0021-identity-audit-contract-v1.md)、[ADR-0027](0027-late-taskwise-refinement.md)。
- Baseline、消融与 reporting：baseline 实现位于 `benchmarks/`，内部消融位于 `ablations/` 或使用隔离的 Stage backend；合同见 [ADR-0022](0022-mlp-ecfp-xgboost-baselines.md)、[0023](0023-unified-evaluation-reporting.md)、[0024](0024-stage2-partial-charge-benchmark-suite.md)、[0025](0025-stage2-homo-lumo-scalar-tasks.md)、[0028](0028-chemprop-dmpnn-baseline.md)、[0029](0029-molformer-baseline.md)、[0030](0030-molformer-throughput-contract.md)、[0031](0031-stage3-summary-normalization-relaxation.md)、[0032](0032-ilbert-baseline.md)、[0033](0033-stage3-single-task-mlp-ablation.md)、[0034](0034-rdkit-2d-home-representation-ablation.md)、[0035](0035-spmm-baseline.md)、[0036](0036-no-stage1-rdkit-stage2-stage3-ablation.md)、[0037](0037-spmm-wordpiece-character-limit.md)、[0038](0038-spmm-throughput-contract.md)。

## 预注册实验

- [ADR-0026：Capacity v1](0026-capacity-v1-pipeline-study.md) 定义独立的端到端容量研究；其 refinement/HPO 评分修订见 [ADR-0027](0027-late-taskwise-refinement.md)，运行步骤见 [操作手册](../capacity-v1-runbook.md)。

## 历史与基础记录

- Stage 1 基础决定：[ADR-0001](0001-data-and-role-sampling.md)、[0002](0002-descriptor-schema-and-tokens.md)、[0003](0003-smiles-tokenizers.md)、[0004](0004-fourth-modality-and-training.md)、[0005](0005-exclude-invalid-pretraining-entities.md)。每篇顶部注明仍有效与已取代范围。
- 已取代的早期训练设计：ADR-0006～0012；现役替代分别由上方 Stage 1/2/3 权威给出。
- 已取代的 Stage 2 Object 设计：[ADR-0016](0016-stage2-universal-object-modeling.md)、[0018](0018-stage2-object-v2-throughput.md)。

查现役行为先读本页对应权威和正式 YAML；只有追溯设计理由或迁移边界时再读历史 ADR。
