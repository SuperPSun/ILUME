# ADR-0066：Stage 3 知识图谱分组定向小实验

- 状态：Accepted（独立候选实验，未替代 Base/base1 系列）
- 日期：2026-09-21
- 配置：`configs/v2/stage3/base2_1.yaml` 至 `base2_5.yaml`

## 背景

[ADR-0064](0064-stage3-knowledge-graph-budget-candidates.md) 的 `base1_5` 同时采用了
thermophysical/interfacial 大组的第三个 expert、15 个 Phase 1 epoch、较高 GROUP LR，
以及 static singleton 的保守 LR/lifetime。新一轮实验固定该完整状态为共同锚点，只检验
static GROUP hidden width、两个 task 的 Phase 3 lifetime，以及大组第三个 expert 是否必要。

## 决定

五份配置均为 `base1_5.yaml` 的自包含副本，不能按编号逐级继承：

| 配置 | 相对 `base1_5` 的固定改动 | 检验假设 |
|---|---|---|
| base2_1 | `static_dielectric.expert_hidden_ratio` 0.75→0.25 | singleton GROUP 降容能否减少错误 local specialization |
| base2_2 | `speed_of_sound.phase3_private_epochs` 0→2 | 大组表示固定后，两轮低 LR PRIVATE calibration 是否有益 |
| base2_3 | `self_diffusion_coefficient.phase3_private_epochs` 4→2 | Phase 3 后两轮是否属于过度 specialization |
| base2_4 | `thermophysical_interfacial_response.experts` 3→2 | 强化预算下第三个 GROUP expert 是否仍有必要 |
| base2_5 | 精确合并以上四项 | 组合能否保留单项收益 |

共同边界如下：

1. thermophysical/interfacial GROUP 固定为 Phase 1 `2e-4 × 15`、Phase 2
   `1e-4 × 4`。static GROUP 固定为 Phase 1 `5e-5 × 8`、Phase 2
   `2.5e-5 × 1`；static PRIVATE 固定为 Phase 1 LR `2e-5`、epochs
   `4/1/0` 与 ratio `0.25/0.25/0.25`。
2. `base2_2/5` 中 speed of sound 的 Phase 3 LR 仍由既有 recipe 解析为
   `1.5e-5`；`base2_3/5` 中 self diffusion 的 Phase 3 LR 仍为 `1e-5`。
   只改变固定训练寿命，不改变 optimizer、scheduler 或选择规则。
3. `base2_1/5` 只缩小 static GROUP expert 的 hidden width，candidate count不变。
   `base2_4/5` 删除大组第三个L1/L2 expert，并使该组八个task gate宽度由6恢复为5。
4. 六个知识图谱GROUP及20个task归属、Flat routing、three-phase、GLOBAL、其他
   GROUP/PRIVATE、dropout、raw sampling、PCGrad、ownership clipping、AdamW、loss、
   validation reporting-only和fixed-final-state协议均保持 `base1_5`。

## 比较与产物

五个候选复用同一Stage 3 prepared artifact和Stage 2 encoder，但完整resolved recipe、
模型形状或两者之一不同，因此training identity彼此不同。每个候选必须从头训练并隔离写入
`outputs/v2/stage3/base2_N`，不得加载或resume Base/base1系列checkpoint。

主选择指标为system-split五折task-equal macro NMAE。定向审计static、speed of sound、
self diffusion、thermophysical/interfacial八任务，以及 `base2_5` 是否保留 `base1_5` 在
refractive index和thermal conductivity上的表现。五个候选都可以生成
test ensemble，但test只作探索性报告，不得用于反向选配置或继续调参。实现验收不执行正式训练
或evaluation。

## 关联

- [ADR-0048：三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：owner recipe与LR连续性](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0054：gate diagnostics](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
- [ADR-0063：知识图谱分组](0063-stage3-knowledge-graph-grouping-candidate.md)
- [ADR-0064：base1预算候选](0064-stage3-knowledge-graph-budget-candidates.md)
