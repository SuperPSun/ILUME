# ADR-0059：Stage 3 仅门控的后训练校准

- 状态：已退役
- 日期：2026-09-15

## 问题与结论

检验 OOD 弱项是否可通过门控-仅后训练改善。各任务从同一 three_phase_final 锚点分叉，只用折训练集行更新门控；固定预算发布独立产物，不用验证/测试集选模。

实验结果不足以支持后训练校准进入正式 Stage 3；正式产物未被覆盖。

## 现役边界

实验 CLI、配置、身份扩展和加载支持已移除，历史输出只读保留且不能由当前代码续跑。Stage 3 保持 Flat learned 任务门控，以 `three_phase_final.pt` 为现役 three-phase 评估产物；只读诊断见 [ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)，训练合同见 [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md)。

原实验的逐项设计可从 Git 提交 `1edf5d8` 中的同名文件追溯。
