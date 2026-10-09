# ADR-0084：Stage3 Phase2/3 保留五项模拟预测能力

- 状态：已接受
- 日期：2026-09-28
- 修订：扩展 [ADR-0082](0082-home-mainline-and-core-ablations.md) 的 Stage3 20-任务训练合同；Stage2 完整产物仍按 [ADR-0083](0083-stage2-home-full-artifact-evaluation.md)。

> 最终四项标量模拟评估与基线比较由 [ADR-0086](0086-scalar-simulation-baselines-and-reporting.md) 扩展；partial charge 不进入该榜单。

## 决定

正式完整 ILUME 和 w/o Stage1 的 Stage3 final 包含 20 项实验任务与汽化热、热膨胀、HOMO、LUMO、partial atomic charge 五项模拟任务的预测头。Stage3 Phase1 仍只训练 20 项实验任务并适配 ObjectEncoder；模拟的电子 GROUP、PRIVATE 和原子 adapter 在 Phase1 冻结。Phase2 从 Stage2 完整 `stage2_final.pt` 初始化五项模拟 PRIVATE、电子 GROUP 和原子 adapter，并将五项任务按所属 GROUP 加入训练；Phase3 为五项分别建立 PRIVATE scope。原有实验分支、Stage1 冻结和 Phase1 ObjectEncoder 适配顺序不变。实验任务的 Stage2→Stage3 初始化仍只迁移 GLOBAL 与 thermophysical、solvation GROUP；电子 GROUP 与模拟 PRIVATE 仅服务于模拟预测支路。

Phase2 的 thermophysical GROUP 继续使用 4 轮、`7.5e-5`；新增 electronic GROUP 使用相同预算。五项模拟任务每轮按 Stage2 准备产物训练集原始行数无放回覆盖，逻辑任务 batch 为 256，不采用虚拟过采样。五项均按训练集规模归入 Stage3 `large` PRIVATE 类：Phase2 PRIVATE 的有效 4 轮、初始 LR `7.5e-5`，Phase3 PRIVATE 8 轮、初始 LR `3.75e-5`。模拟 loss 沿用 Stage2 physics SmoothL1，partial charge 按分子等权；GROUP/PRIVATE 更新仍使用 Stage3 `weighted_owner_raw_v1` owner 聚合、裁剪、AdamW 与对应 LR 衰减。Phase2 的共享 thermophysical GROUP 中，每项模拟任务 `task_weight=0.1`，实验任务保持 `1.0`；沿用 Phase1 的组内权重归一聚合，GROUP 和 PRIVATE 梯度均遵循现有归一权重。仅模拟任务的 electronic GROUP 保持 `task_weight=1.0`；Phase3 仍按原单任务 loss 更新，不应用这项降权。该系数进入 YAML、resolved 注册表、训练身份和恢复/评估校验。模拟加入后 GROUP 每轮更新数按其组内最长任务原始覆盖计算，进入解析后的计划和身份。

模拟特征由完整 Stage2 final 的 Stage1 backbone 与第 1 阶段适配后的 ObjectEncoder 构建，partial charge 继续使用 Stage2 原子 adapter。Stage3 final 保存完整模型状态、模拟数据与来源身份、owner 清单、状态hash 和独立的模拟验证；正式 Stage3 榜单/evaluator 仍只汇总 20 项实验任务，模拟验证单独记录。热膨胀测试集 split 已存在；Stage2 独立 evaluator 和榜单已由 [ADR-0085](0085-retire-stage2-home-evaluation.md) 退役。

w/o Stage1 采用相同五项模拟 Phase2/3 合同，但来源是独立随机 Stage1 的 Stage2 final。按用户决定，w/o Stage2 与 w/o Stage3-HoME 维持原 20 项实验任务合同；与完整 ILUME 的差异因此包含模拟辅助训练，不能把对照结果解释成单一 owner 或路由的因果效应。新 Stage3 final/检查点 kind 与训练合同版本拒载旧 20-任务 HoME 产物，不迁移、不覆盖旧输出。

## 验证边界

只用临时小数据检查 Stage2 来源 owner 逐张量初始化、模拟原子/标量梯度、Phase2/3 owner 更新与拼接、恢复/final 身份和榜单隔离。不启动正式训练或评估。
