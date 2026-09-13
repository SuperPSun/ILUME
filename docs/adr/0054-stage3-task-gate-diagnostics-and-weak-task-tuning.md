# ADR-0054：Stage 3 弱任务微调与 task-gate diagnostics

- 状态：Accepted
- 日期：2026-09-14
- 修订：ADR-0053 的四项 task recipe

## 背景

最新训练曲线表明，volume expansion 的 Phase 2 PRIVATE 更新持续恶化，而 pEC50 与 xCO2
在 Phase 3 结束时仍有收益。与此同时，需要在不改变模型行为的前提下观察 task gate 对
GLOBAL、GROUP 和 PRIVATE candidates 的使用情况，为后续是否拆分 PRIVATE expert 与
calibration lifetime 提供证据。

## 决定

1. volume expansion 的 Phase 2 PRIVATE epoch 从 4 改为 0；pEC50 与 xCO2 的 Phase 3
   PRIVATE epoch从 3 改为 5；viscosity 的 task-specific PRIVATE dropout从 0.10 改为
   0.15。其他现役 recipe 不变，六套现役 Stage 3 配置继续同步。
2. three-phase validation 与独立 evaluator 复用模型同一次 forward 已返回的 `task_gate`。
   按实际 task-specific candidate count，把权重依序聚合为 GLOBAL、GROUP 与 PRIVATE mass。
3. 每个 task 报告三类 mean mass、PRIVATE mass 的 p10/p50/p90，以及逐样本
   `-sum(p*log(p))/log(candidate_count)` 后取均值的归一化 task-gate entropy。
4. test 的五折 aggregate 将全部 fold-sample 观测合并后计算统计，不对不同 fold 的 expert
   索引作语义对齐。validation 按 fold/task 报告；test 同时报告每 fold 与 aggregate。
5. 新字段只加入 three-phase validation/evaluation 的 `gate_diagnostics`。prediction CSV、
   原 metrics、`reporting` block、common reporting schema 和 validation reporting-only
   语义均保持不变；legacy v1/Capacity 不输出该字段。
6. training identity contract v5 与 resolved-plan format v3 保持不变。task recipe 已进入
   identity，因此旧 checkpoint 不能 resume。

## 后果

- 不增加 forward，不改变预测、loss、optimizer、PCGrad、sampling、clipping、模型结构或
  checkpoint selection。
- gate diagnostics 会进入每轮 three-phase validation history、stitched/final validation
  manifest，以及独立 validation/test `summary.json`。
- Stage 1、Stage 2 与 Stage 3 prepared artifact 可复用；六套现役 Stage 3 必须在新目录
  重新执行 train、validation 与 test。baseline 无需重跑，历史输出保持只读。

## 关联

- [ADR-0048：owner-lifetime 三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：task-specific owner budget 与 PRIVATE capacity](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0053：clean hybrid task recipe](0053-stage3-clean-hybrid-task-recipe.md)
