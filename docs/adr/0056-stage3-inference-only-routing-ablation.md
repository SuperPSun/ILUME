# ADR-0056：Stage 3 仅推理时的任务门控路由消融

- 状态：已退役
- 日期：2026-09-14

## 问题与结论

检验测试集/OOD 弱项是否来自任务门控过度依赖 local 候选。曾在同一次前向中固定候选张量，仅干预最终 mixture；与 learned 门控成对比较。

no-PRIVATE、global-仅和统一 hard GLOBAL floor 的结果说明 local specialization 必要，不进入正式方案。

## 现役边界

实验 CLI、配置、身份扩展和加载支持已移除，历史输出只读保留且不能由当前代码续跑。Stage 3 保持 Flat learned 任务门控，以 `three_phase_final.pt` 为现役 three-phase 评估产物；只读诊断见 [ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)，训练合同见 [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md)。

原实验的逐项设计可从 Git 提交 `1edf5d8` 中的同名文件追溯。
