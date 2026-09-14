# ADR-0056：Stage 3 inference-only task-gate routing ablation

- 状态：Accepted
- 日期：2026-09-14
- 关联：ADR-0054 的 task-gate diagnostics

## 背景

现役 Stage 3 的若干任务在五折 validation 较强但 test/OOD 泛化较弱。需要在不重新训练、
不修改 checkpoint 或模型结构的前提下，判断 L2 task gate 是否过度依赖 local candidates。

## 决定

1. three-phase evaluator 支持 `learned_gate`、`no_private`、`global_only`、
   `global_floor_025` 和 `global_floor_050`。默认 `learned_gate` 继续执行原始数值路径。
2. forced mode 只改变 GLOBAL、当前 GROUP 和当前 PRIVATE candidates 的最终 mixture
   weights；candidate tensors、task gate网络和后续 normalization/tower保持不变。
3. no-private/global-only 在保留集合内维持原始相对比例；GLOBAL floor 只修改低于阈值的
   样本，并分别保持 GLOBAL 内与GROUP+PRIVATE内的相对比例。零mass使用允许集合内均匀
   fallback。
4. forced mode与learned reference在同一次forward中复用相同candidate tensors。
   evaluator按fold和test ensemble报告MAE/NMAE差异，并复用ADR-0054统计intervention后
   gate diagnostics。
5. routing mode只属于evaluation identity；`learned_gate`省略该identity字段以保持历史
   hash，非默认mode写入identity、summary、run metadata与reporting protocol，并使用隔离
   study/output。checkpoint与training identity不变。
6. legacy v1/Capacity拒绝forced routing。prediction CSV列、common reporting schema、
   checkpoint selector与默认训练/评估行为保持不变。

## 后果

- 现有prepared artifact和Stage 3 checkpoint可直接复用，只需执行evaluation。
- forced routing结果是诊断证据，不得按test表现直接选为正式模型。
- 非默认mode多执行一次mixture、normalization和tower，但不重复representation或expert
  计算，也不增加模型forward次数。

## 关联

- [ADR-0048：owner-lifetime三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0054：弱任务微调与task-gate diagnostics](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
- [ADR-0055：pEC50 Phase 3单变量回调](0055-stage3-pec50-phase3-single-variable-rollback.md)
