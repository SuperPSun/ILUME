# ADR-0054：Stage 3 task-gate diagnostics

- 状态：Accepted
- 日期：2026-09-14
- 数值修订：当时同时调整四项 task recipe；最终设置统一见 [ADR-0055](0055-stage3-pec50-phase3-single-variable-rollback.md)

## 决定

为观察 task gate 对 GLOBAL、GROUP 和 PRIVATE candidates 的使用情况，three-phase validation
与独立 evaluator 复用同一次 forward 已返回的 `task_gate`，不增加 forward 或参与训练决策。

- 按实际 task-specific candidate count，将权重依序聚合为 GLOBAL、GROUP、PRIVATE mass。
- 每个 task 报告三类 mean mass、PRIVATE mass 的 p10/p50/p90，以及逐样本
  `-sum(p*log(p))/log(candidate_count)` 后取均值的归一化 entropy。
- validation 按 fold/task 报告；test 同时报每 fold 与 aggregate，aggregate 合并全部
  fold-sample 后计算，不对不同 fold 的 expert 索引作语义对齐。
- `gate_diagnostics` 进入每轮 three-phase validation history、stitched/final validation
  manifest 与独立 validation/test `summary.json`；legacy v1/Capacity 不输出。

prediction CSV、原 metrics、`reporting` block、common reporting schema、validation reporting-only
语义均不变；不改变预测、loss、optimizer、PCGrad、sampling、clipping、模型结构或 checkpoint
selection。owner 与恢复合同见 [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 和
[ADR-0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md)。

当年的 recipe 调整为 volume expansion Phase 2 PRIVATE `4→0`、pEC50/xCO2 Phase 3 `3→5`、
viscosity PRIVATE dropout `0.10→0.15`；pEC50 后由 0055 回调。数值变化进入 training identity，
诊断字段不改变 reporting schema；不要将当时要求新目录重跑的说明套用于纯文档整理。
