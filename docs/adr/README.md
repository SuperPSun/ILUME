# Architecture Decision Records

本目录记录会改变数据谱系、实验可比性或模型接口的决定。状态为 `Accepted` 的 ADR 是当前实现约束；如需反转，新增 ADR 取代旧决定，不直接改写历史理由。

- [ADR-0001：数据边界与 45/45/10 覆盖采样（采样已取代）](0001-data-and-role-sampling.md)
- [ADR-0002：描述符 schema 与分组 token](0002-descriptor-schema-and-tokens.md)
- [ADR-0003：SMILES tokenizer 后端](0003-smiles-tokenizers.md)
- [ADR-0004：指纹、role embedding 与单卡训练器（训练器部分已取代）](0004-fourth-modality-and-training.md)
- [ADR-0005：准备阶段直接排除异常实体（artifact 版本已取代）](0005-exclude-invalid-pretraining-entities.md)
- [ADR-0006：覆盖型 epoch 与正式实验配置（已取代）](0006-coverage-epochs-and-experiment-configs.md)
- [ADR-0007：Stage 2 多任务物性监督与冻结教师对齐（已取代）](0007-stage2-property-alignment.md)
- [ADR-0008：Base 正式训练 profile 收敛到 effective batch 256（已取代）](0008-base-training-profile.md)
- [ADR-0009：Stage 2 体系采样、PairEncoder 与渐进解冻（已取代）](0009-stage2-system-sampling-and-progressive-unfreezing.md)
- [ADR-0010：Stage 3 HoME 与 late-solute 多任务训练（已取代）](0010-stage3-home-and-late-solute.md)
- [ADR-0011：Stage 3 单阶段双域完全隔离](0011-stage3-single-stage-isolated-domains.md)
- [ADR-0012：Stage 3 高吞吐与预算守恒执行合同](0012-stage3-throughput-and-budget.md)
- [ADR-0013：Stage 1 大规模全量预训练协议（部分已取代）](0013-stage1-full-corpus-ddp.md)
- [ADR-0014：Stage 1 Prepare 并行化与 Corpus v2](0014-stage1-prepare-performance-and-corpus-v2.md)
- [ADR-0015：Stage 1 高吞吐与 Epoch 边界恢复合同（部分已取代）](0015-stage1-high-throughput-epoch-resume.md)
- [ADR-0016：Stage 2 统一 ObjectEncoder 与全覆盖训练](0016-stage2-universal-object-modeling.md)
- [ADR-0017：Stage 1 Base 的 global batch 与 eager 执行](0017-stage1-base-runtime-profile.md)
