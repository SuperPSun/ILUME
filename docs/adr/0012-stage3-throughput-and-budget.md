# ADR-0012：Stage 3 高吞吐与预算守恒执行合同

- 状态：Accepted
- 日期：2026-08-05
- 扩展：ADR-0011 的双域训练执行方式

## 背景

初始 Stage 3 v2 在 RTX 4090 上只占约 1.6 GiB 显存、4%–9% SM 和约 74 W，但训练进程使用约 38–43 个 CPU 核。原因是 76/76 个 PyTorch CPU 线程、每个任务一次 backward 和 loss 标量同步、CPU 常驻表示的多次小拷贝，以及训练热路径中的 Python pair lookup。提高利用率不能以扩大总样本预算或破坏 IL21/Aux6 隔离为代价。

## 决定

1. Stage 3 三个 Python 入口在导入运行实现前调用 `stage3.config.configure_process_runtime()`，将 CPU intra-op/inter-op 线程限制为 4/1，并将 OMP、MKL 和 OpenBLAS 设置为 4 线程。
2. 当前 fold 的冻结表示、条件、目标和预计算 embedding index 在 CUDA 上常驻。SystemCursor 与审计字段保留在 CPU；resident 分配失败时直接报错，不静默回退。
3. 正式配置使用 `domain` backward：按确定性任务顺序完成全部 forward，IL21/Aux6 loss 分别除以 21/6，每域只 backward 一次。loss 和验证统计只在域边界同步到 CPU。
4. IL21 使用 batch 128、5000 blocks、每 50 blocks 验证；Aux6 使用 batch 256、2500 blocks、每 25 blocks 验证。两域每任务总预算仍为 640,000 行，验证间隔仍为 6,400 行，LR 保持 `3e-4`。
5. artifact 与 checkpoint schema 继续使用 v2。线程数、resident 模式不进入语义哈希；batch、blocks 和 `domain` backward 进入，因此优化前 training checkpoint 不能恢复到正式优化配置。
6. 正式自包含配置为 `configs/v1/stage3/reference.yaml`，新训练写入 `outputs/v1/stage3/reference/checkpoints/fold<fold>`。旧 v2 checkpoint 配置只由私有兼容解析器读取，不恢复 legacy YAML 或旧运行入口。

## 后果

- 单域内部的梯度求和浮点顺序与旧实现不同，因此优化训练必须从头开始；双域参数、RNG、BN、optimizer、scheduler、scaler 与早停隔离不变。
- 若正式 batch 的短 CUDA 验证发生 OOM，IL21 固定降级到 `64 × 10000/validation 100`，Aux6 固定降级到 `128 × 5000/validation 50`，仍使用 domain backward。
- `torch.compile` 不在本次决定中。正式五折与对照实验由用户逐配置运行，不提供 matrix runner；代码验收只做短 benchmark 和临时链路。
