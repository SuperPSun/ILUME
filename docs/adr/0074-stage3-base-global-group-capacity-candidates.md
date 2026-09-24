# ADR-0074：Stage 3 Base GLOBAL/GROUP 容量候选

- 状态：Experimental
- 日期：2026-09-24
- 配置：`configs/v2/stage3/base3_1.yaml` 至 `base3_5.yaml`

## 决定

五份自包含配置均以现役 20-task `base.yaml` 为共同锚点，保留其六组任务归属、PRIVATE 容量与 task override、Flat routing、three-phase owner LR/epoch、raw sampling、原始梯度加权聚合、optimizer、clipping、loss 和固定末轮发布。只增加 GLOBAL 或多任务 GROUP 的 expert 数与 hidden width：

| 配置 | 相对 Base 的容量改动 |
|---|---|
| base3_1 | GLOBAL experts `2→3` |
| base3_2 | GLOBAL expert hidden ratio `2.0→2.5` |
| base3_3 | transport、thermophysical experts `2→3`；phase_stability、solvation `3→4` |
| base3_4 | 上述四个 GROUP 的 expert hidden ratio `1.5→2.0` |
| base3_5 | 合并 base3_1～base3_4 的全部容量改动 |

dielectric_optical 与 biological 保持原容量；前者包含两个极小任务，后者只有 pEC50。`model.expert_hidden_ratio` 在这批显式 GROUP 与 task-specific PRIVATE recipe 下改变 GLOBAL expert width，不改变 GROUP 或 PRIVATE width。增加 expert 数会同步改变相关 gate 的输出宽度，这是模型结构变化的一部分。

## 比较边界

五份配置共享 Base prepared artifact 与 Stage 2 encoder，但各自使用独立 training identity 与 `outputs/v2/stage3/base3_N/` 输出根，不加载或恢复 Base、base1/base2 或其他候选的 Stage 3 checkpoint。首先比较五折 validation task-equal macro NMAE 与逐任务变化，test ensemble只作候选确定后的报告；base3_5 为组合实验，不用于解释任一单项改动的因果效果。实现本配置时不启动正式训练或评估。
