# ADR-0079：Stage2-HoME → Stage3-HoME 独立迁移消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-26
- 范围：`configs/ablations/stage2_home_transfer.yaml`；不修订正式 Stage2/Stage3 Base

## 决定

在现有九任务 Stage2 准备产物数据上，另建仅物理监督 HoME 源模型。Stage1 表示编码器从第 1 轮起、连同 ObjectEncoder 和 HoME 一起训练十轮；不使用 teacher、distillation 或浅层回归头。沿用 Stage2 的任务权重、逐任务原始 batch 调度、256 行逻辑 batch、AdamW、BF16 和末轮发布。每个逻辑 batch 只执行一次优化器/调度器更新；微批大小纳入独立身份。初版使用 8 行微批；当前 `stage2_microbatch_size` 允许 1–256 的整数，YAML 默认 256。改变微批可能改变 dropout、实体去重范围和浮点累积，因此必须在新目录重跑，不交叉恢复；显存不足时显式修改为 128/64 等值，不自动缩小。

训练使用单个 CPU 打包线程，按原 schedule 顺序最多预取两个待消费逻辑 batch。CUDA 路径对 CPU batch 使用 pinned memory 和非阻塞传输；不缓存可训练编码器的输出。detached loss 在 GPU 上按 float64 累计，在逻辑 batch 边界检查有限性并读取；原普通任务 mask 在训练前统一校验，QM mask 与分子等权归一化保留。`performance.jsonl` 是独立的只读执行日志，记录逐任务的打包时间、等待时间、训练调度时间，以及整轮训练、验证、检查点耗时和训练显存峰值；打包与训练重叠，不能把各字段相加作为整轮耗时。性能日志不进入训练身份或恢复历史记录校验。

Stage2 的 density、heat capacity、热膨胀和汽化热使用与正式 Base 同形状的 `thermophysical` GROUP，transfer organic 使用同形状的 `solvation` GROUP。HOMO、LUMO、QM electrostatic 和 partial charge 使用仅限源模型的 `electronic_structure` GROUP。QM 的 11 个目标仍共同构成原模拟任务的带mask 目标-macro physics loss；partial charge 仍按分子等权，原子状态与 ObjectEncoder 上下文共同进入其不迁移的适配路径。

独立的 `stage2_home_transfer.pt` 只允许迁移表示编码器、GLOBAL 及上述两个可映射 GROUP 的完整 owner 状态。Stage2 模拟 PRIVATE、电子 GROUP、原子适配和预测模块不迁移；逐张量名称、形状、数据类型、owner 集合与 hash 均需严格校验。配套编码器文件只为现有 Stage3 ObjectEncoder/slots 准备接口提供相同的表示状态，不代表正式 Stage2 浅层头曾参与训练。

Stage3 在独立准备产物/输出根目录使用正式 Base 的 20-任务 split、归一化、Flat HoME 和 three-phase 配方。Stage1 slots 冻结；ObjectEncoder 在 Phase 1 继续适配、Phase 2/3 冻结。GLOBAL、thermophysical 和 solvation 从源模型初始化；其余 GROUP 与全部实验 PRIVATE/任务门控/FiLM/归一化/tower按正式种子重新初始化。Phase 1/2 沿用现役 `weighted_owner_raw_v1`，而非历史 PCGrad。验证只报告，固定末轮拼接；先看五折任务等权 macro NMAE，再报告测试集集成，不能按测试集选模型。

## 边界与解释

Stage3 消融评估使用统一 `open_run_directory()` 发布 `run_config.yaml`、`metadata.json`、`attempts.jsonl` 与 `summary.json`。元数据标记 `stage3/evaluate`、报告结构定义、折/split 和完成/失败状态；运行身份绑定相应折的 final 产物 SHA、训练身份与模型状态hash。原有汇总报告和预测合同保留，study 为 `ilume-stage2-home-transfer-v1`，显示名为 `ILUME (Stage2-HoME transfer)`。统一 summarizer 可发现五折验证和测试集集成，不读取 Stage2 源训练指标；旧缺少元数据的目录仍只读，不自动迁移或覆盖。

正式 `stage2_encoder.pt`、Stage3 Base、检查点、准备产物和评估入口保持原合同，旧输出只读。消融使用独立训练身份、检查点/final kind，不能与正式 Base 交叉恢复或加载。由于新源模型同时移除了 teacher loss 并取消首轮 Stage1 冻结，结果衡量的是完整预训练方案差异，不能严格单独归因于 HoME 与浅层预测头的架构差异。
