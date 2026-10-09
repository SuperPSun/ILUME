# ADR-0015：Stage 1 高吞吐与 Epoch 边界恢复合同

- 状态：部分被取代，替代ADR： ADR-0017
- 日期：2026-08-13
- 局部取代：ADR-0013 的训练执行、验证、日志与检查点 v1 合同

> 2026-08-13：全局batch 256 与默认 `compile: true` 由 [ADR-0017](0017-stage1-base-runtime-profile.md) 取代；其余执行、验证、日志和检查点 v2 合同仍然有效。

## 决定

Stage 1 保持自然频率全量轮、原有 batch 成员、全局batch 256、五模态目标、element-level 2/2/1 角色权重、AdamW、预热/余弦和梯度裁剪。禁止 length bucketing。Fusion layout 在 collate 阶段生成，GPU 前向使用向量化 gather/scatter；动态 masking 可改变 RNG 调用顺序，但概率、schedule、种子边界和监督定义不变。

正式训练默认 `compile: true`，编译异常直接终止且不静默回退；`compile` 是冻结到运行配置、检查点和元数据的执行参数，但不进入科研实验身份。CUDA 启用 pinned custom batch、non-blocking H2D、persistent workers、固定 prefetch 和 TF32；这些均不进入科研配置。

DDP 每步将五路 global weighted loss denominator 打包归约，训练日志每 10 步写一次。quick 验证每 5000 步运行，轮末仍运行完整验证；所有 rank 无重复地分摊同一固定验证集并一次归约完整统计。验证使用训练相同的 AMP 数据类型。

检查点保持 `kind="ilume_stage1_pretraining"`，升级为 `format_version=2`，只在完整验证成功后的轮边界发布。`checkpoint_epoch_00003.pt` 表示 Epoch 1–3 已完整完成，`last.pt` 始终指向最新完整边界。状态保存模型、优化器、调度器、AMP scaler、已完成轮数/全局步数、完整配置及产物/来源身份；不保存轮 cursor、sampler cursor、rank RNG 或轮中途状态。中断后从上个完整轮重跑。

每个轮根据种子、轮、rank 和 world size 重新设定 Python、NumPy、Torch 与 CUDA RNG。同种子、同 world size、同 compile 设置和同环境的新版运行可从轮边界复现；允许改变 world size 恢复，但记录新的尝试，且不承诺后续轨迹一致。旧检查点 v1 明确拒绝，不做迁移。

## 理由

现有训练热点来自 Fusion 逐样本 GPU 标量同步、CPU masking、自定义 batch 未 pin、逐样本分片 fetch、细粒度 collective、rank-0-仅验证、逐步 JSON I/O 和轮中途精确恢复状态。上述执行优化不改变样本覆盖、batch 组成或训练目标，同时删除与完整轮训练不成比例的恢复复杂度。

## 后果

Stage 2 必须使用 Stage 1 检查点 v2 重新准备教师缓存。训练失败不会覆盖上个完整 `last.pt`；`metrics.jsonl` 通过 `attempt_id` 保留不同尝试，不截断失败尝试。旧检查点、旧 FP32 验证指标和新版 AMP 验证指标不得按 bitwise 方式比较。
