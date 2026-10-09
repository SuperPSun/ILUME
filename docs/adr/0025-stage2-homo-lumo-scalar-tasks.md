# ADR-0025：Stage 2 HOMO/LUMO 独立标量任务与报告 v2

- 状态：已接受
- 日期：2026-08-25
- 取代：ADR-0019 的角色-oriented orbital 任务，以及 ADR-0023/0024 的 Stage 2 Core/完整单元与子合同版本

## 决定

Stage 2 注册表以 `simulation/homo` 和 `simulation/lumo` 取代阳离子/阴离子 orbital 任务。每个任务只有一个目标和一个独立标量预测头，同时自然混合阳离子与阴离子样本；各任务分别用 pooled 训练集行拟合 global 目标 scaler，不做角色 balancing 或角色-specific 归一化。两项任务权重均为 `1.0`，九任务逐行完整覆盖、一个 batch 一个优化器步和五轮合同不变。

Producer 的 `ion_role`、`provenance_source_file`、`provenance_source_row` 必须通过角色、正式-charge 和来源校验。Prepare 的张量产物不保留这些审计字段，ILUME、MLP 与 ECFP+XGBoost 的预测 CSV 保留它们。MLP 继续以归一化-目标 MSE 训练；XGBoost 继续拟合原始标量目标。

HOMO/LUMO 各自的 headline 是 pooled 样本-micro 原始 MAE，单位 eV。另报阳离子/阴离子的 count 与原始 MAE，但诊断值不参与 aggregate 或 wins。Core 是汽化热、HOMO、LUMO 三个任务的归一化MAE 等权平均；完整是这三个 Core 任务加 Partial Charge 的四单元等权平均，仍禁止跨运行拼接。

通用报告 envelope 保持结构定义 v1，Stage 2 子合同升级为 `stage2-core-evaluation-v2` 与 `stage2-benchmark-suite-v2`。Core CSV 增加 `subset=pooled|cation|anion`，只有 pooled 行参加排名；榜单使用 `valid_tasks`、`total_tasks` 和 `per_task_wins`。缺少 v2 子合同的旧结果只进入 health。MLP 与 ECFP+XGBoost 的 Partial/完整仍为 unsupported。

## 兼容性与后果

Registry、任务目录与来源身份均改变。旧准备产物、检查点、编码器和运行不可恢复或迁移，物理格式 version 不因纯语义 breaking change升级。Teacher 缓存只在既有身份门控确认实体产物与 Stage 1 编码器身份完全一致时复用，否则明确失败并要求重建；不增加 fallback。

现有 Stage 2 Base 输出在单独确认归档或删除方案前保持只读且不得覆盖。本变更不运行正式 prepare、teacher extraction、训练集或 evaluate；当前 Stage 2 v2 榜单因此为空，旧结果作为历史实现 health 保留。Stage 3 代码与旧结果不变；未来采用新编码器时必须建立新的 Stage 3 身份/运行。
