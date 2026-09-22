# ADR-0063：Stage 3 知识图谱分组候选配置

- 状态：Accepted（独立候选实验）
- 日期：2026-09-19
- 适用配置：`configs/v2/stage3/base1.yaml`

## 背景

知识图谱给出的性质关系与现役 Base 的六组划分不同。需要在不改变 Flat HoME、three-phase
训练、task recipe 或数据划分的条件下，单独检验新的 GROUP 共享边界。同时，任务归组必须是
YAML 训练侧配置，而不是写死在模型或训练器中的分类表。

## 决定

1. 新增独立候选 `base1.yaml`，现役 `base.yaml` 及其 split、RDKit-HoME、No-Stage1 配置均不变。
2. 六组定义为：
   - `transport_dynamics`：electrical conductivity、viscosity、self diffusion；
   - `thermophysical_interfacial_response`：density、heat capacity、speed of
   sound、surface tension、thermal conductivity、refractive index、dynamic permittivity、xCO2；
   - `phase_stability`：glass transition、melting point、equilibrium pressure、thermal decomposition；
   - `solvation_transfer`：solvation、transfer、transfer organic；
   - `biological`：pEC50；
   - `static_dielectric`：static permittivity。
3. 为隔离变量，GROUP recipe分别继承旧 `transport`、`thermophysical`、`phase_stability`、
   `solvation`、`biological`、`dielectric_optical`；GLOBAL、PRIVATE及全部训练数值逐项沿用Base。
4. 显式YAML的`groups`完整定义GROUP，`tasks.<task>.meta_group`定义任务归属。模型、ownership、
   PCGrad和Phase 2分支只消费解析后的配置；`BASE_GROUP_TASKS`仅保留为省略配置时的legacy默认，
   不约束该候选。
5. `base1.yaml`复用Base prepared artifact与同一Stage 2 encoder。`meta_group`不进入prepared
   identity，但会改变GROUP ownership、部分TaskGate candidate count、resolved plan及training
   identity，因此必须在独立输出目录重新训练，不能加载或resume Base checkpoint。

## 边界

- 本实验不修改专家结构、GROUP recipe、task-specific capacity/dropout、LR、epoch、sampling、
  PCGrad、optimizer、loss、clipping、validation或fixed-final-state协议。
- `base1.yaml`是可运行候选，不取代现役Base；结果确认前不得覆盖正式Base输出。
- Stage 1、Stage 2和现有Stage 3 prepared artifact均可复用，不执行新的prepare。

## 关联

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0048：owner-lifetime三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：task-specific owner budget与PRIVATE capacity](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0054：task-gate diagnostics](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
