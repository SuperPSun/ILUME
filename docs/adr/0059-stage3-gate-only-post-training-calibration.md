# ADR-0059：Stage 3 gate-only post-training calibration

- 状态：Retired
- 日期：2026-09-15

## 问题与结论

检验 OOD 弱项是否可通过 gate-only 后训练改善。各 task 从同一 three_phase_final anchor 分叉，只用 fold train rows 更新 gate；固定预算发布独立 artifact，不用 validation/test 选模。

实验结果不足以支持后训练校准进入正式 Stage 3；正式 artifact 未被覆盖。

## 现役边界

实验 CLI、配置、identity 扩展和加载支持已移除，历史输出只读保留且不能由当前代码续跑。Stage 3 保持 Flat learned task gate，以 `three_phase_final.pt` 为现役 three-phase evaluation artifact；只读 diagnostics 见 [ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)，训练合同见 [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md)。

原实验的逐项设计可从 Git 提交 `1edf5d8` 中的同名文件追溯。
