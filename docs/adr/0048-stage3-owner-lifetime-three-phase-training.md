# ADR-0048：Stage 3 owner-lifetime deterministic 三阶段训练

- 状态：Accepted
- 日期：2026-09-10
- 取代：ADR-0047 的现役 v2 Stage 3 四阶段 schedule、checkpoint 与 final artifact

## 背景

Stage 3 的 21 个实验任务在 `unique_systems` 上相差两个数量级；raw rows 会把同一体系的多条件
观测误判为更多独立信息。统一 owner 训练寿命使小任务 PRIVATE 过拟合，而大任务及 shared
owner 尚未充分训练。需要在不改变 raw sampling、loss、weighting、PCGrad 或专家数量的前提下，
只调整 owner-specific LR、训练寿命和预注册的 PRIVATE width。

## 决定

1. 现役 v2 Base、random/system/individual split，以及 ADR-0034/0036 的 Stage 3 消融使用
   `schedule_mode=three_phase`。GLOBAL 固定训练 Phase 1 的 15 个全任务 epoch；六个 Phase 2
   GROUP branch 从同一个 Phase-1 hash 分叉并 stitch；21 个 Phase 3 PRIVATE branch 从同一个
   Phase-2 stitched hash 分叉并 stitch。validation 只记录，最终始终使用固定最后 epoch。
2. Phase 1 全部 task 始终 raw sampling。GLOBAL 使用 `2.5e-4`、5% warmup、cosine 到 0.1；
   GROUP 与 PRIVATE 使用 YAML owner/class recipe、无 warmup、cosine 到 0.5。owner 达到自身
   epoch 后以 `requires_grad=False` 冻结，但 task 不移除：被冻结的 PRIVATE/GROUP 仍作为
   differentiable forward path，继续向尚未冻结的 GROUP/GLOBAL 提供梯度。Phase 1 保持完整
   hierarchical PCGrad、task/group weight 与 ownership-aware clipping。
3. Phase 2 冻结 GLOBAL。GROUP 按 group 固定 epoch；PRIVATE nominal epoch 按 size class，实际
   `min(private_epochs, group_epochs)`。PRIVATE 提前冻结后 task 仍向 GROUP 提供梯度。只在当前
   GROUP block 做组内 PCGrad；PRIVATE 不投影，task weight 保持，单组 branch 不重复施加 group
   weight。各 owner 无 warmup并 cosine 到 0.5。
4. Phase 3 冻结 GLOBAL/GROUP，每个 branch 只训练一个 PRIVATE，关闭 PCGrad 与 task/group
   weight，无 warmup并 cosine 到 0.2。各 task 的固定 epoch（含五个显式例外）由 YAML 记录。
5. PRIVATE size class 与 `unique_systems` 显式写入 YAML；catalog 数值必须一致，但程序不自动
   分桶。tiny/small/medium/large 的 private/tower/FiLM ratio 固定为 0.5/0.75/1.0/1.0，且
   `private_experts=1`。GLOBAL/GROUP expert 数和 hidden ratio 保持 ADR-0047 的容量设计。
   GROUP 的 Phase 1/2 `LR × epoch` 分别固定为：biological `7.5e-5×8 / 3.75e-5×5`、
   dielectric_optical `1e-4×8 / 5e-5×3`、thermophysical
   `1.25e-4×10 / 6.25e-5×5`、transport `1.25e-4×12 / 6.25e-5×12`、
   phase_stability `1.5e-4×15 / 7.5e-5×20`、solvation
   `1.5e-4×15 / 7.5e-5×24`。PRIVATE tiny/small/medium/large 的 Phase 1、Phase 2 与
   Phase 3 LR 分别为 `4e-5×6 / 2e-5×3 / 1e-5`、
   `6e-5×8 / 3e-5×4 / 1.5e-5`、`8e-5×12 / 4e-5×8 / 2e-5`、
   `1e-4×15 / 5e-5×12 / 2.5e-5`。Phase 3 默认 epoch 为 2/3/5/8；
   self diffusion、surface tension、thermal decomposition、transfer、transfer organic
   分别覆盖为 8/8/10/10/15。
6. 每阶段/branch 新建 AdamW。owner scheduler 只在对应 owner 实际持有 gradient 且 optimizer
   update 成功时推进；计划按 GLOBAL 的 `K`、GROUP 的 `K_g` 与 PRIVATE 的 `K_t` 展开更新预算。
   跨阶段起始 LR 不得高于上一阶段 terminal LR。冻结不得以 LR=0 模拟。
7. Phase 1 checkpoint 保存完整模型、optimizer、owner scheduler、RNG、owner update 和 freeze
   state；Phase 2/3 保存绑定 immutable anchor 的 owner delta及同等恢复状态。history 尾、更新
   数、freeze epoch、anchor、identity 与 state hash 必须完全一致，才允许从 epoch 边界恢复。
8. 最终 artifact 为 `three_phase_final.pt/json`，Phase 2 另发布 `phase_2/stitched.pt/json`。
   training identity、checkpoint kind、tensor hash namespace 和 final kind 与 four-phase、legacy
   均隔离。当前代码拒绝 four-phase config，也不 resume/evaluate `four_phase_final`；历史文件
   保持只读。
9. `resolved_training_plan.json` 展开 GLOBAL、每个 GROUP/PRIVATE 的 nominal/terminal LR、
   nominal/effective epoch、freeze epoch、实际 update budget、size class、`unique_systems` 和
   capacity。legacy v1 与 Capacity v1 继续受 ADR-0027 约束并使用 `taskwise_refined`。

## 后果

- 小任务可以停止自身参数拟合，而其观测仍用于更高层共享表示；大任务和大 group 得到更长
  的固定训练预算。
- Stage 1、Stage 2 与 Stage 3 prepared artifact 可复用；六份现役配置必须在新目录重新执行
  Stage 3 train、validation 与 test evaluation，既有 four-phase 输出不得覆盖或迁移。
- 分支执行顺序不影响 stitch 结果；任何越权 owner 变化、anchor 不一致、delta 缺失或重叠都
  必须失败。

## 备选方案

- 拒绝按 raw rows 自动分桶：多条件重复观测不能代表独立体系数。
- 拒绝 owner 冻结时删除 task：会同时剥夺 shared owner 所需的监督信号。
- 拒绝 early stopping 或 validation-best stitching：与固定预算、reporting-only 合同冲突。
- 拒绝同时修改 sampling、loss weighting、PCGrad 或专家数量：会混淆本轮实验变量。

## 关联

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：identity/audit contract](0021-identity-audit-contract-v1.md)
- [ADR-0027：late taskwise refinement](0027-late-taskwise-refinement.md)
- [ADR-0046：ownership clipping 与 raw sampling](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0047：deterministic 四阶段训练（历史）](0047-stage3-deterministic-four-phase-training.md)
