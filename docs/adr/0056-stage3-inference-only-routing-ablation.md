# ADR-0056：Stage 3 inference-only task-gate routing ablation

- 状态：Retired
- 日期：2026-09-14

## 问题与结论

检验 test/OOD 弱项是否来自 task gate 过度依赖 local candidates。曾在同一次 forward 中固定 candidate tensor，仅干预最终 mixture；与 learned gate 成对比较。

no-private、global-only 和统一 hard GLOBAL floor 的结果说明 local specialization 必要，不进入正式方案。

## 现役边界

实验 CLI、配置、identity 扩展和加载支持已移除，历史输出只读保留且不能由当前代码续跑。Stage 3 保持 Flat learned task gate，以 `three_phase_final.pt` 为现役 three-phase evaluation artifact；只读 diagnostics 见 [ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)，训练合同见 [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md)。

原实验的逐项设计可从 Git 提交 `1edf5d8` 中的同名文件追溯。
