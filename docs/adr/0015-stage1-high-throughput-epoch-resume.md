# ADR-0015：Stage 1 高吞吐与 Epoch 边界恢复合同

- 状态：Partially Superseded by ADR-0017
- 日期：2026-08-13
- 局部取代：ADR-0013 的训练执行、validation、日志与 checkpoint v1 合同

> 2026-08-13：global batch 256 与默认 `compile: true` 由 [ADR-0017](0017-stage1-base-runtime-profile.md) 取代；其余执行、validation、日志和 checkpoint v2 合同仍然有效。

## 决定

Stage 1 保持自然频率全量 epoch、原有 batch 成员、global batch 256、五模态目标、element-level 2/2/1 role weight、AdamW、warmup/cosine 和梯度裁剪。禁止 length bucketing。Fusion layout 在 collate 阶段生成，GPU forward 使用向量化 gather/scatter；dynamic masking 可改变 RNG 调用顺序，但概率、schedule、seed 边界和监督定义不变。

正式训练默认 `compile: true`，编译异常直接终止且不静默回退；`compile` 是冻结到 run config、checkpoint 和 metadata 的执行参数，但不进入科研实验 identity。CUDA 启用 pinned custom batch、non-blocking H2D、persistent workers、固定 prefetch 和 TF32；这些均不进入科研配置。

DDP 每步将五路 global weighted loss denominator 打包归约，训练日志每 10 step 写一次。quick validation 每 5000 step 运行，epoch 末仍运行 full validation；所有 rank 无重复地分摊同一固定 validation 集并一次归约完整统计。validation 使用训练相同的 AMP dtype。

checkpoint 保持 `kind="ilume_stage1_pretraining"`，升级为 `format_version=2`，只在 full validation 成功后的 epoch 边界发布。`checkpoint_epoch_00003.pt` 表示 Epoch 1–3 已完整完成，`last.pt` 始终指向最新完整边界。状态保存 model、optimizer、scheduler、AMP scaler、completed epoch/global step、完整配置及 artifact/source identity；不保存 epoch cursor、sampler cursor、rank RNG 或 mid-epoch 状态。中断后从上个完整 epoch 重跑。

每个 epoch 根据 seed、epoch、rank 和 world size 重新设定 Python、NumPy、Torch 与 CUDA RNG。同 seed、同 world size、同 compile 设置和同环境的新版运行可从 epoch 边界复现；允许改变 world size 恢复，但记录新的 attempt，且不承诺后续轨迹一致。旧 checkpoint v1 明确拒绝，不做迁移。

## 理由

现有训练热点来自 Fusion 逐样本 GPU scalar 同步、CPU masking、自定义 batch 未 pin、逐样本 shard fetch、细粒度 collective、rank-0-only validation、逐 step JSON I/O 和 mid-epoch 精确恢复状态。上述执行优化不改变样本覆盖、batch 组成或训练目标，同时删除与完整 epoch 训练不成比例的恢复复杂度。

## 后果

Stage 2 必须使用 Stage 1 checkpoint v2 重新准备教师缓存。训练失败不会覆盖上个完整 `last.pt`；`metrics.jsonl` 通过 `attempt_id` 保留不同尝试，不截断失败 attempt。旧 checkpoint、旧 FP32 validation 指标和新版 AMP validation 指标不得按 bitwise 方式比较。
