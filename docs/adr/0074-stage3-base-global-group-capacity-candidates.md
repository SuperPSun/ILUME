# ADR-0074：Stage 3 Base GLOBAL/GROUP 容量候选

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-24
- 配置：`configs/v2/stage3/base3_1.yaml` 至 `base3_5.yaml`

## 决定

五份自包含配置均以现役 20-任务 `base.yaml` 为共同锚点，保留其六组任务归属、PRIVATE 容量与任务覆盖项、Flat 路由、three-phase owner LR/轮、原始样本采样、原始梯度加权聚合、优化器、裁剪、loss 和固定末轮发布。只增加 GLOBAL 或多任务 GROUP 的专家数与隐藏宽度宽度：

| 配置 | 相对 Base 的容量改动 |
|---|---|
| base3_1 | GLOBAL 专家 `2→3` |
| base3_2 | GLOBAL 专家隐藏宽度比例 `2.0→2.5` |
| base3_3 | transport、thermophysical 专家 `2→3`；phase_stability、solvation `3→4` |
| base3_4 | 上述四个 GROUP 的专家隐藏宽度比例 `1.5→2.0` |
| base3_5 | 合并 base3_1～base3_4 的全部容量改动 |

dielectric_optical 与 biological 保持原容量；前者包含两个极小任务，后者只有 pEC50。`model.expert_hidden_ratio` 在这批显式 GROUP 与任务-specific PRIVATE 配方下改变 GLOBAL 专家宽度，不改变 GROUP 或 PRIVATE 宽度。增加专家数会同步改变相关门控的输出宽度，这是模型结构变化的一部分。

## 比较边界

五份配置共享 Base 准备产物与 Stage 2 编码器，但各自使用独立训练身份与 `outputs/v2/stage3/base3_N/` 输出根，不加载或恢复 Base、base1/base2 或其他候选的 Stage 3 检查点。首先比较五折验证任务等权 macro NMAE 与逐任务变化，测试集集成只作候选确定后的报告；base3_5 为组合实验，不用于解释任一单项改动的因果效果。实现本配置时不启动正式训练或评估。
