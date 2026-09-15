# ADR-0059：Stage 3 gate-only post-training calibration

- 状态：Retired
- 日期：2026-09-15
- 关联：ADR-0048 的 three-phase final、ADR-0054 的 task-gate diagnostics

## 背景

现役 Flat Stage 3 已发布固定末轮的 `three_phase_final.pt`。为判断 OOD 弱任务是否主要
需要 routing 再校准，需要在不重训 experts、不改变正式 baseline 和不使用 validation/test
选模的前提下，增加一个独立的 gate-only 后训练实验。

## 决定

1. `gate_calibration` 是默认关闭、只适用于现役 three-phase 的配置段；现役六套配置显式
   使用固定 3 epoch 和 `lr_scale=0.25`。它不进入 three-phase resolved plan 或 training
   identity，因此既有 Flat final artifact 保持兼容。
2. 每个 fold 的所有 task branch 均从同一个 `three_phase_final.pt` anchor 分叉。每个 branch
   只解冻当前 task 的 `task_gate.weight/bias`，其余参数保持 eval、冻结且逐 bit 不变；现有
   ownership manifest 不修改。
3. 每个 task 使用当前 fold 的 train rows，按既有 `B_t` 做严格 raw、无放回 epoch。LR 为
   Phase 3 nominal PRIVATE LR 的 0.25 倍，使用 constant LR、AdamW、SmoothL1、BF16 和
   gate-only gradient clipping；不使用 PCGrad 或 scheduler，固定发布最后 epoch。
4. 每个 branch 保存绑定 anchor 与 calibration identity 的 gate delta、optimizer、RNG、
   update count 和连续 history。所有 gate delta 校验完整、互不重叠后 stitch；非 gate tensor
   发生任何变化都失败。零 epoch 直接继承 anchor，且不建立 task history/checkpoint。
5. 独立 artifact 为 `gate_calibrated.pt/json`，记录 base artifact SHA、base model hash、base
   training identity、calibration recipe/seed、每 task gate hash 和最终 model hash。Object 与
   RDKit-HoME 使用不同 kind，不覆盖正式 `three_phase_final.pt`。
6. calibration validation 只报告每 task 的 pre/post MAE、NMAE、原 gate diagnostics、gate
   参数 delta norm 和 `D_KL(anchor_gate || calibrated_gate)`，不参与训练或选择。paired
   evaluator 对 validation、test fold 和五折 ensemble 同时报告 base/calibrated 差异；CSV
   只保留 calibrated prediction 且列结构不变。
7. 默认 evaluator 仍选择 `three_phase_final`。只有显式传入 `--gate-calibration-dir` 才选择
   calibrated artifact；该模式禁止叠加 forced `--routing-mode`，test 不得用于参数选择。

## 后果

- 实验已经完成，结果不足以支持 gate-only post-training calibration 进入正式 Stage 3。
- calibration CLI、配置、训练模块、artifact加载、resume与paired evaluator支持均已从现役
  代码移除；历史 `gate_calibrated.pt` 与summary只读保留，当前代码不再加载或续跑。
- 正式 Stage 3 继续以 Flat learned gate 的 `three_phase_final.pt` 为唯一现役three-phase
  evaluation artifact。

## 关联

- [ADR-0048：owner-lifetime三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0054：弱任务微调与task-gate diagnostics](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
- [ADR-0056：inference-only routing ablation](0056-stage3-inference-only-routing-ablation.md)
