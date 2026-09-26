# ADR-0079：Stage2-HoME → Stage3-HoME 独立迁移消融

- 状态：Experimental
- 日期：2026-09-26
- 范围：`configs/ablations/stage2_home_transfer.yaml`；不修订正式 Stage2/Stage3 Base

## 决定

在现有九任务 Stage2 prepared 数据上，另建 physics-only HoME 源模型。Stage1 表示编码器从第 1 epoch 起、连同 ObjectEncoder 和 HoME 一起训练十轮；不使用 teacher、distillation 或浅层回归头。沿用 Stage2 的 task weight、逐任务 raw batch 调度、256 行逻辑 batch、AdamW、BF16 和末轮发布。每个逻辑 batch 只执行一次 optimizer/scheduler update；微批大小纳入独立身份。初版使用 8 行微批；当前 `stage2_microbatch_size` 允许 1–256 的整数，YAML 默认 256。改变微批可能改变 dropout、实体去重范围和浮点累积，因此必须在新目录重跑，不交叉 resume；显存不足时显式修改为 128/64 等值，不自动缩小。

训练使用单个 CPU 打包线程，按原 schedule 顺序最多预取两个待消费逻辑 batch。CUDA 路径对 CPU batch 使用 pinned memory 和非阻塞传输；不缓存可训练编码器的输出。detached loss 在 GPU 上按 float64 累计，在逻辑 batch 边界检查有限性并读取；原普通任务 mask 在训练前统一校验，QM mask 与分子等权归一化保留。`performance.jsonl` 是独立的只读执行日志，记录逐 task 的打包时间、等待时间、训练调度时间，以及整轮训练、validation、checkpoint 耗时和训练显存峰值；打包与训练重叠，不能把各字段相加作为整轮耗时。性能日志不进入训练 identity 或恢复 history 校验。

Stage2 的 density、heat capacity、thermal expansion 和 heat of vaporization 使用与正式 Base 同形状的 `thermophysical` GROUP，transfer organic 使用同形状的 `solvation` GROUP。HOMO、LUMO、QM electrostatic 和 partial charge 使用仅限源模型的 `electronic_structure` GROUP。QM 的 11 个目标仍共同构成原模拟任务的 masked target-macro physics loss；partial charge 仍按分子等权，原子状态与 ObjectEncoder 上下文共同进入其不迁移的适配路径。

独立的 `stage2_home_transfer.pt` 只允许迁移表示编码器、GLOBAL 及上述两个可映射 GROUP 的完整 owner 状态。Stage2 simulation PRIVATE、电子 GROUP、原子适配和预测模块不迁移；逐 tensor 名称、形状、dtype、owner 集合与 hash 均需严格校验。配套编码器文件只为现有 Stage3 ObjectEncoder/slots 准备接口提供相同的表示状态，不代表正式 Stage2 浅层头曾参与训练。

Stage3 在独立 prepared/output root 使用正式 Base 的 20-task split、normalization、Flat HoME 和 three-phase recipe。Stage1 slots 冻结；ObjectEncoder 在 Phase 1 继续适配、Phase 2/3 冻结。GLOBAL、thermophysical 和 solvation 从源模型初始化；其余 GROUP 与全部实验 PRIVATE/task gate/FiLM/normalization/tower按正式种子重新初始化。Phase 1/2 沿用现役 `weighted_owner_raw_v1`，而非历史 PCGrad。Validation 只报告，固定末轮 stitch；先看五折 task-equal macro NMAE，再报告 test ensemble，不能按 test 选模型。

## 边界与解释

Stage3 消融 evaluation 使用统一 `open_run_directory()` 发布 `run_config.yaml`、`metadata.json`、`attempts.jsonl` 与 `summary.json`。metadata 标记 `stage3/evaluate`、reporting schema、fold/split 和完成/失败状态；运行身份绑定相应 fold 的 final artifact SHA、training identity 与 model state hash。原有 summary reporting 和 prediction 合同保留，study 为 `ilume-stage2-home-transfer-v1`，显示名为 `ILUME (Stage2-HoME transfer)`。统一 summarizer 可发现五折 validation 和 test ensemble，不读取 Stage2 源训练指标；旧缺少 metadata 的目录仍只读，不自动迁移或覆盖。

正式 `stage2_encoder.pt`、Stage3 Base、checkpoint、prepared artifact 和评估入口保持原合同，旧输出只读。消融使用独立训练身份、checkpoint/final kind，不能与正式 Base 交叉恢复或加载。由于新源模型同时移除了 teacher loss 并取消首轮 Stage1 冻结，结果衡量的是完整预训练方案差异，不能严格单独归因于 HoME 与浅层 head 的架构差异。
