# Architecture Decision Records

正式 YAML 定义运行参数，ADR 定义会影响数据谱系、实验可比性或模型接口的稳定合同。后写 ADR 在重叠范围内优先；历史正文保留当时理由，不作为现役入口。

## 现役权威

- Stage 1：[ADR-0013](0013-stage1-full-corpus-ddp.md)、[0014](0014-stage1-prepare-performance-and-corpus-v2.md)、[0015](0015-stage1-high-throughput-epoch-resume.md)、[0017](0017-stage1-base-runtime-profile.md)。基础数据、descriptor、tokenizer、fingerprint 与 QC 决定仍见 ADR-0001～0005 各自的有效范围。
- Stage 2：[ADR-0019](0019-stage2-catalog-object-v3.md) 定义 Object v3；[ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 定义现役 HOMO/LUMO 与 reporting v2；[ADR-0027](0027-late-taskwise-refinement.md) 修订末期训练和最终评估 artifact。
- Stage 3：[ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md) 定义 sparse-label HoME 与 hierarchical PCGrad；末期 PRIVATE-only refinement 由 [ADR-0027](0027-late-taskwise-refinement.md) 定义。
- 跨 Stage identity/audit：[ADR-0021](0021-identity-audit-contract-v1.md)、[ADR-0027](0027-late-taskwise-refinement.md)。
- Baseline 与 reporting：[ADR-0022](0022-mlp-ecfp-xgboost-baselines.md)、[0023](0023-unified-evaluation-reporting.md)、[0024](0024-stage2-partial-charge-benchmark-suite.md)、[0025](0025-stage2-homo-lumo-scalar-tasks.md)、[0028](0028-chemprop-dmpnn-baseline.md)、[0029](0029-molformer-baseline.md)、[0030](0030-molformer-throughput-contract.md)、[0031](0031-stage3-summary-normalization-relaxation.md)、[0032](0032-ilbert-baseline.md)。

## 预注册实验

- [ADR-0026：Capacity v1](0026-capacity-v1-pipeline-study.md) 定义独立的端到端容量研究；其 refinement/HPO 评分修订见 [ADR-0027](0027-late-taskwise-refinement.md)，运行步骤见 [操作手册](../capacity-v1-runbook.md)。

## 历史与基础记录

- Stage 1 基础决定：[ADR-0001](0001-data-and-role-sampling.md)、[0002](0002-descriptor-schema-and-tokens.md)、[0003](0003-smiles-tokenizers.md)、[0004](0004-fourth-modality-and-training.md)、[0005](0005-exclude-invalid-pretraining-entities.md)。每篇顶部注明仍有效与已取代范围。
- 已取代的早期训练设计：ADR-0006～0012；现役替代分别由上方 Stage 1/2/3 权威给出。
- 已取代的 Stage 2 Object 设计：[ADR-0016](0016-stage2-universal-object-modeling.md)、[0018](0018-stage2-object-v2-throughput.md)。

查现役行为先读本页对应权威和正式 YAML；只有追溯设计理由或迁移边界时再读历史 ADR。
