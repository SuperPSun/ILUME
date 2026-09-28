# ADR-0084：Stage3 Phase2/3 保留五项 simulation 预测能力

- 状态：Accepted
- 日期：2026-09-28
- 修订：扩展 [ADR-0082](0082-home-mainline-and-core-ablations.md) 的 Stage3 20-task 训练合同；Stage2 完整产物仍按 [ADR-0083](0083-stage2-home-full-artifact-evaluation.md)。

> 最终四项 scalar simulation 评估与 baseline 比较由 [ADR-0086](0086-scalar-simulation-baselines-and-reporting.md) 扩展；partial charge 不进入该榜单。

## 决定

正式 Full ILUME 和 w/o Stage1 的 Stage3 final 包含 20 项 experimental task 与 heat of vaporization、thermal expansion、HOMO、LUMO、partial atomic charge 五项 simulation task 的预测头。Stage3 Phase1 仍只训练 20 项实验任务并适配 ObjectEncoder；simulation 的电子 GROUP、PRIVATE 和 atom adapter 在 Phase1 冻结。Phase2 从 Stage2 完整 `stage2_final.pt` 初始化五项 simulation PRIVATE、电子 GROUP 和 atom adapter，并将五项任务按所属 GROUP 加入训练；Phase3 为五项分别建立 PRIVATE scope。原有实验分支、Stage1 冻结和 Phase1 ObjectEncoder 适配顺序不变。实验任务的 Stage2→Stage3 初始化仍只迁移 GLOBAL 与 thermophysical、solvation GROUP；电子 GROUP 与 simulation PRIVATE 仅服务于模拟预测支路。

Phase2 的 thermophysical GROUP 继续使用 4 epochs、`7.5e-5`；新增 electronic GROUP 使用相同预算。五项 simulation task 每轮按 Stage2 prepared train 原始行数无放回覆盖，逻辑 task batch 为 256，不采用 virtual oversampling。五项均按 train 规模归入 Stage3 `large` PRIVATE 类：Phase2 PRIVATE 的有效 4 epochs、初始 LR `7.5e-5`，Phase3 PRIVATE 8 epochs、初始 LR `3.75e-5`。simulation loss 沿用 Stage2 physics SmoothL1，partial charge 按分子等权；GROUP/PRIVATE 更新仍使用 Stage3 `weighted_owner_raw_v1` owner 聚合、clipping、AdamW 与对应 LR 衰减。Phase2 的共享 thermophysical GROUP 中，每项模拟任务 `task_weight=0.1`，实验任务保持 `1.0`；沿用 Phase1 的组内权重归一聚合，GROUP 和 PRIVATE 梯度均遵循现有归一权重。仅模拟任务的 electronic GROUP 保持 `task_weight=1.0`；Phase3 仍按原单任务 loss 更新，不应用这项降权。该系数进入 YAML、resolved registry、训练身份和恢复/评估校验。simulation 加入后 GROUP 每轮 update 数按其组内最长 task 原始覆盖计算，进入 resolved plan 和身份。

simulation 特征由完整 Stage2 final 的 Stage1 backbone 与第 1 阶段适配后的 ObjectEncoder 构建，partial charge 继续使用 Stage2 atom adapter。Stage3 final 保存完整模型状态、simulation 数据与来源身份、owner manifest、state hash 和独立的 simulation validation；正式 Stage3 leaderboard/evaluator 仍只汇总 20 项实验任务，模拟 validation 单独记录。thermal expansion test split 已存在；Stage2 独立 evaluator 和榜单已由 [ADR-0085](0085-retire-stage2-home-evaluation.md) 退役。

w/o Stage1 采用相同五项 simulation Phase2/3 合同，但来源是独立随机 Stage1 的 Stage2 final。按用户决定，w/o Stage2 与 w/o Stage3-HoME 维持原 20 项实验任务合同；与 Full ILUME 的差异因此包含 simulation 辅助训练，不能把对照结果解释成单一 owner 或 routing 的因果效应。新 Stage3 final/checkpoint kind 与训练合同版本拒载旧 20-task HoME 产物，不迁移、不覆盖旧输出。

## 验证边界

只用临时小数据检查 Stage2 source owner 逐 tensor 初始化、simulation 原子/标量梯度、Phase2/3 owner 更新与 stitch、resume/final 身份和榜单隔离。不启动正式训练或评估。
