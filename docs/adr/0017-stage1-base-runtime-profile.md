# ADR-0017：Stage 1 Base 的 global batch 与 eager 执行

- 状态：Accepted
- 日期：2026-08-13
- 局部取代：ADR-0013/0015 中 global batch 256 与默认 `compile: true` 的决定

## 决定

Stage 1 正式 Base 的 global batch 固定为 128，梯度累积为 1。单卡每步处理 128 条；DDP 按 rank 平分该 global batch，因此 global batch 必须能被 world size 整除。epoch 仍按自然频率完整覆盖训练集，optimizer step 数、warmup 和 cosine 时钟由实际行数与 global batch 128 推导。

正式 Base 默认 `training.compile: false`，直接使用 eager 执行。`compile` 继续写入 run config、checkpoint 和 metadata，但不进入科研实验 identity；恢复时允许显式切换。若将其设为 `true`，编译异常仍必须直接终止，不得静默回退 eager。

ADR-0013/0015 中其他模型、数据、loss、optimizer、validation、checkpoint v2 和 epoch-boundary resume 合同保持不变。

## 理由

正式 YAML 和已启动运行均已使用 global batch 128 与 eager 执行。将这两项明确为现役合同，可消除配置、测试、运行指南和恢复认知的分叉，且不扩张到 Stage 1 其他科研决定。

## 后果

相同 epoch 现在包含更多 optimizer step，不应将 global batch 128 训练与旧 global batch 256 训练视为可直接继续的同一运行。Stage 1 checkpoint kind 和 `format_version=2` 不变；checkpoint 中的完整配置、config hash、每 epoch step 数与 resume 校验会阻止错误混接。Stage 2 仍只消费完成的 Stage 1 v2 checkpoint。
