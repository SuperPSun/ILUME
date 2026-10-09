# ADR-0063：Stage 3 知识图谱分组候选配置

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-19
- 适用配置：`configs/v2/stage3/base1.yaml`

## 背景

知识图谱给出的性质关系与现役 Base 的六组划分不同。需要在不改变 Flat HoME、three-phase
训练、任务配方或数据划分的条件下，单独检验新的 GROUP 共享边界。同时，任务归组必须是
YAML 训练侧配置，而不是写死在模型或训练器中的分类表。

## 决定

1. 新增独立候选 `base1.yaml`，现役 `base.yaml` 及其 split、RDKit-HoME、No-Stage1 配置均不变。
2. 六组定义为：
   - `transport_dynamics`：电导率、黏度、自扩散；
   - `thermophysical_interfacial_response`：密度、热容、声速、
   表面张力、热导率、折射率、动态介电常数、xCO2；
   - `phase_stability`：玻璃化转变、熔点、平衡压力、热分解；
   - `solvation_transfer`：溶剂化、迁移、向有机溶剂迁移；
   - `biological`：pEC50；
   - `static_dielectric`：静态介电常数。
3. 为隔离变量，GROUP 配方分别继承旧 `transport`、`thermophysical`、`phase_stability`、
   `solvation`、`biological`、`dielectric_optical`；GLOBAL、PRIVATE及全部训练数值逐项沿用Base。
4. 显式YAML的`groups`完整定义GROUP，`tasks.<task>.meta_group`定义任务归属。模型、参数归属、
   PCGrad和Phase 2分支只消费解析后的配置；`BASE_GROUP_TASKS`仅保留为省略配置时的历史实现默认，
   不约束该候选。
5. `base1.yaml`复用Base 准备产物与同一Stage 2 编码器。`meta_group`不进入准备产物
   身份，但会改变GROUP 参数归属、部分TaskGate 候选 count、解析后的计划及训练
   身份，因此必须在独立输出目录重新训练，不能加载或恢复 Base 检查点。

## 边界

- 本实验不修改专家结构、GROUP 配方、任务-specific capacity/dropout、LR、轮、采样、
  PCGrad、优化器、loss、裁剪、验证或固定-末轮状态协议。
- `base1.yaml`是可运行候选，不取代现役Base；结果确认前不得覆盖正式Base输出。
- Stage 1、Stage 2和现有Stage 3 准备产物均可复用，不执行新的prepare。

## 关联

- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0048：owner-存续期三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：任务特定的owner预算与PRIVATE容量](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0054：任务门控诊断](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
