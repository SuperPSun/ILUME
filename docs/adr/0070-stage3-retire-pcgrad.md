# ADR-0070：Stage 3 主线退役 PCGrad

状态：现役。覆盖 ADR-0020、0048、0050 中关于现役 Stage 3 梯度投影的规定；其余模型、数据、预算及 owner 生命周期保持各自 YAML 合同。ADR-0069 作为切换前的消融记录保留。

## 决定

现役 v2 三阶段 Stage 3 在 Phase 1 和 Phase 2 使用原始梯度的 owner 加权聚合。组内 task 梯度仍按 `task_weight` 归一聚合；GLOBAL 将各组结果按 `group_weight` 聚合；PRIVATE 沿用组内归一 task 权重。Phase 3 仍是单任务更新。训练计划以 `math.gradient_aggregation: weighted_owner_raw_v1` 标识此合同。

Base、三种 split、base1/base2 候选、RDKit-HoME 和 No-Stage1 Stage 3 共用此算法，各自 prepared 数据、模型和训练预算不变。Base 复用 `outputs/v2/stage3/base/prepare/artifacts`，新训练与评估写入 `outputs/v2/stage3/base_no_pcgrad/`；其他实验沿用原输出根并添加 `_no_pcgrad` 后缀。旧输出只读，不移动、不覆盖。

旧 PCGrad 三阶段 checkpoint 与新训练及评估身份不兼容。三阶段 resolved plan 格式升级为 4、训练身份合同升级为 6、周期 checkpoint 格式升级为 2。旧 v1/Capacity 的训练和恢复入口明确拒绝；其历史最终 `taskwise_refined.pt` 可继续只读评估。历史 ADR 与结果保留为科研记录，可执行 PCGrad、配置开关及新产物中的专用状态和诊断字段退役。

## 依据与范围

切换接受此前消融观察到的小幅指标差异：五折 validation task-equal macro NMAE 增加约 0.06%，test ensemble 增加约 0.27%。这两个数来自旧消融对照，不代表新身份下重训结果。validation 的五折 task-equal macro normalized MAE 和逐任务配对差值仍是主要比较；test ensemble 独立报告，不用于调参。

完整 Base 命令及准备产物边界见 [README Stage 3](../../README.md#stage-3)。本 ADR 不授权在代码实施或测试阶段启动正式五折训练或评估。
