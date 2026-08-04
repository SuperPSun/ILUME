# Architecture Decision Records

本目录记录会改变数据谱系、实验可比性或模型接口的决定。状态为 `Accepted` 的 ADR 是当前实现约束；如需反转，新增 ADR 取代旧决定，不直接改写历史理由。

- [ADR-0001：数据边界与 45/45/10 覆盖采样（训练预算部分已取代）](0001-data-and-role-sampling.md)
- [ADR-0002：描述符 schema 与分组 token](0002-descriptor-schema-and-tokens.md)
- [ADR-0003：SMILES tokenizer 后端](0003-smiles-tokenizers.md)
- [ADR-0004：指纹、role embedding 与单卡训练器](0004-fourth-modality-and-training.md)
- [ADR-0005：准备阶段直接排除异常实体](0005-exclude-invalid-pretraining-entities.md)
- [ADR-0006：覆盖型 epoch 与正式实验配置（Base profile 部分已取代）](0006-coverage-epochs-and-experiment-configs.md)
- [ADR-0007：Stage 2 多任务物性监督与冻结教师对齐](0007-stage2-property-alignment.md)
- [ADR-0008：Base 正式训练 profile 收敛到 effective batch 256](0008-base-training-profile.md)
