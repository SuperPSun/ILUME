# ADR-0025：Stage 2 HOMO/LUMO 独立标量任务与 reporting v2

- 状态：Accepted
- 日期：2026-08-25
- 取代：ADR-0019 的 role-oriented orbital task，以及 ADR-0023/0024 的 Stage 2 Core/Full 单元与子合同版本

## 决定

Stage 2 registry 以 `simulation/homo` 和 `simulation/lumo` 取代 cation/anion orbital task。每个任务只有一个 target 和一个独立 scalar head，同时自然混合 cation 与 anion 样本；各任务分别用 pooled train rows 拟合 global target scaler，不做 role balancing 或 role-specific normalization。两项 task weight 均为 `1.0`，九任务逐行完整覆盖、一个 batch 一个 optimizer step和五 epoch 合同不变。

Producer 的 `ion_role`、`provenance_source_file`、`provenance_source_row` 必须通过 role、formal-charge 和来源校验。Prepare 的 tensor artifact 不保留这些审计字段，ILUME、MLP 与 ECFP+XGBoost 的 prediction CSV 保留它们。MLP 继续以 normalized-target MSE 训练；XGBoost 继续拟合 raw scalar target。

HOMO/LUMO 各自的 headline 是 pooled sample-micro raw MAE，单位 eV。另报 cation/anion 的 count 与 raw MAE，但诊断值不参与 aggregate 或 wins。Core 是 heat of vaporization、HOMO、LUMO 三个 task 的 normalized MAE 等权平均；Full 是这三个 Core task 加 Partial Charge 的四单元等权平均，仍禁止跨 run 拼接。

通用 reporting envelope 保持 schema v1，Stage 2 子合同升级为 `stage2-core-evaluation-v2` 与 `stage2-benchmark-suite-v2`。Core CSV 增加 `subset=pooled|cation|anion`，只有 pooled 行参加排名；leaderboard 使用 `valid_tasks`、`total_tasks` 和 `per_task_wins`。缺少 v2 子合同的旧结果只进入 health。MLP 与 ECFP+XGBoost 的 Partial/Full 仍为 unsupported。

## 兼容性与后果

Registry、catalog 与 source identity 均改变。旧 prepared artifact、checkpoint、encoder 和 run 不可恢复或迁移，物理 format version 不因纯语义 breaking change升级。Teacher cache 只在既有 identity gate 确认 entity artifact 与 Stage 1 encoder identity 完全一致时复用，否则明确失败并要求重建；不增加 fallback。

现有 Stage 2 Base 输出在单独确认归档或删除方案前保持只读且不得覆盖。本变更不运行正式 prepare、teacher extraction、train 或 evaluate；当前 Stage 2 v2 榜单因此为空，旧结果作为 legacy health 保留。Stage 3 代码与旧结果不变；未来采用新 encoder 时必须建立新的 Stage 3 identity/run。
