# ADR-0011：Stage 3 单阶段双域完全隔离

- 状态：Accepted
- 日期：2026-08-05
- 取代：ADR-0010 中的 Phase 1/Phase 2、27 项共享 HoME 与阶段扩展决定

## 背景

transfer organic、四个离子 HOMO/LUMO 和 charge 不涉及 IL 体系。若把这六项放入与 IL 任务共享的 HoME 或 optimizer block，它们会通过共享参数、BatchNorm、loss 归一化、调度或随机数状态改变 21 项 IL 任务的更新。Stage 3 的目标因此从“27 项共享多任务模型”调整为“一个命令协调两个完全隔离的训练域”。

## 决定

1. 删除 Phase 1/Phase 2 和 `--init-from`。Stage 3 v2 一次运行协调 `il21` 与 `aux6`；旧 v1 artifact/checkpoint 只保留审计，不能恢复或迁移到 v2。
2. `il21` 保留 19 个直接 IL 任务以及 solvation/transfer。HoME 仍使用两层 global/group/private experts；solute CLS 只在第二层进入 solvation group、private expert 与 gate，第二层 global expert 和全部直接 IL 任务不接收 solute。
3. `aux6` 的六项任务分别使用独立 `IndependentTaskHead`。每个 head 私有地包含 embedding adapter、条件与 phase fusion、LayerNorm、一个固定 Expert block 和回归 tower；六项之间不共享任何可训练参数，也不进入 HoME。
4. artifact 分隔在 `artifacts/stage3_v2/data/il21` 与 `artifacts/stage3_v2/data/aux6`。两个域独立生成冻结缓存、fold scaler、任务 tensor、哈希和排除审计；IL pair 泄漏与跨任务曝光只审计 `il21`。
5. 每个外层 cycle 最多执行一个 21-task IL block 和一个 6-task aux block。两个域分别持有 optimizer、scheduler、AMP scaler、梯度裁剪、sampling cursor、任务顺序 RNG、Torch/CUDA dropout RNG、block 计数、验证、早停和最佳指标。执行 aux block 不得改变任何 IL 参数、梯度、BatchNorm buffer 或随机状态。
6. 两域分别按 macro normalized MAE 选优，并保存 `best_il21.pt` 与 `best_aux6.pt`。训练结束后组装评估专用 `best.pt`；只有包含双方完整运行状态的 v2 training checkpoint 可以用于 `--resume-from`。
7. Shared-bottom、MMoE、early-solute、without Feature-gate 和 without Self-gate 是 `il21`-only 对照，不创建也不重复训练 `aux6`。

## 兼容性与解释限制

- 两域共享同一进程、GPU和总体运行时间，因此进程级故障仍可能同时中断；数值训练状态则必须完全隔离。
- v2 loader 校验 artifact version、domain/task registry、Stage 2 checkpoint SHA-256、来源数据哈希和具体 artifact 哈希，并明确拒绝 v1。
- `best.pt`、`best_il21.pt` 和 `best_aux6.pt` 不保存 optimizer 等运行状态，不能用于恢复。
- `il21` 继续采用 task-local fold；跨任务曝光审计必须随结果保留，五折结果不能解释为所有任务联合意义上的 unseen-IL 泛化。

## 后果

主线仍可用一个 CLI 和一份输出汇总 27 项任务，但六项非 IL 任务的任何反向传播、AMP overflow、调度或早停都不会进入 21 项 IL 任务的数值更新。正式 v2 prepare、五折训练、基线与 test ensemble 由用户执行；代码验收仅使用临时小数据和短验证。
