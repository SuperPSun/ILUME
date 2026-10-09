# ADR-0048：Stage 3 按owner存续期确定性执行的三阶段训练

- 状态：已接受
- 日期：2026-09-10
- 取代：ADR-0047 的现役 v2 Stage 3 四阶段 schedule、检查点与 final 产物
- 数值修订：ADR-0050 取代本文的 owner LR、轮、PRIVATE 宽度与任务字段

## 背景

Stage 3 的 21 个实验任务在 `unique_systems` 上相差两个数量级；原始行会把同一体系的多条件
观测误判为更多独立信息。统一 owner 训练寿命使小任务 PRIVATE 过拟合，而大任务及 shared
owner 尚未充分训练。需要在不改变原始样本采样、loss、weighting、PCGrad 或专家数量的前提下，
只调整 owner-specific LR、训练寿命和预注册的 PRIVATE 宽度。

## 决定

1. 现役 v2 Base、random/system/individual split，以及 ADR-0034/0036 的 Stage 3 消融使用
   `schedule_mode=three_phase`。GLOBAL 固定训练 Phase 1 的 15 个全任务轮；六个 Phase 2
   GROUP 分支从同一个 Phase-1 hash 分叉并拼接；21 个 Phase 3 PRIVATE 分支从同一个
   Phase-2 拼接后的 hash 分叉并拼接。验证只记录，最终始终使用固定最后轮。
2. Phase 1 全部任务始终原始样本采样。GLOBAL 使用 `2.5e-4`、5% 预热、余弦到 0.1；
   GROUP 与 PRIVATE 使用 YAML owner/class 配方、无预热、余弦到 0.5。owner 达到自身
   轮后以 `requires_grad=False` 冻结，但任务不移除：被冻结的 PRIVATE/GROUP 仍作为
   differentiable 前向路径，继续向尚未冻结的 GROUP/GLOBAL 提供梯度。Phase 1 保持完整
   分层 PCGrad、任务/组权重与按参数归属裁剪。
3. Phase 2 冻结 GLOBAL。GROUP 按组固定轮；PRIVATE nominal 轮按 size class，实际
   `min(private_epochs, group_epochs)`。PRIVATE 提前冻结后任务仍向 GROUP 提供梯度。只在当前
   GROUP block 做组内 PCGrad；PRIVATE 不投影，任务权重保持，单组分支不重复施加组
   权重。各 owner 无预热并余弦到 0.5。
4. Phase 3 冻结 GLOBAL/GROUP，每个分支只训练一个 PRIVATE，关闭 PCGrad 与任务/组
   权重，无预热并余弦到 0.2。各任务的固定轮（含五个显式例外）由 YAML 记录。
5. 以下数值配方是 2026-09-10 的历史基线，现役值由 ADR-0050 修订。PRIVATE size class 与
   `unique_systems` 显式写入 YAML；任务目录数值必须一致，但程序不自动
   分桶。tiny/small/medium/large 的 PRIVATE/tower/FiLM 比例固定为 0.5/0.75/1.0/1.0，且
   `private_experts=1`。GLOBAL/GROUP 专家数和隐藏宽度比例保持 ADR-0047 的容量设计。
   GROUP 的 Phase 1/2 `LR × epoch` 分别固定为：biological `7.5e-5×8 / 3.75e-5×5`、
   dielectric_optical `1e-4×8 / 5e-5×3`、thermophysical
   `1.25e-4×10 / 6.25e-5×5`、transport `1.25e-4×12 / 6.25e-5×12`、
   phase_stability `1.5e-4×15 / 7.5e-5×20`、solvation
   `1.5e-4×15 / 7.5e-5×24`。PRIVATE tiny/small/medium/large 的 Phase 1、Phase 2 与
   Phase 3 LR 分别为 `4e-5×6 / 2e-5×3 / 1e-5`、
   `6e-5×8 / 3e-5×4 / 1.5e-5`、`8e-5×12 / 4e-5×8 / 2e-5`、
   `1e-4×15 / 5e-5×12 / 2.5e-5`。Phase 3 默认轮为 2/3/5/8；
   自扩散、表面张力、热分解、transfer、transfer organic
   分别覆盖为 8/8/10/10/15。
6. 每阶段/分支新建 AdamW。owner 调度器只在对应 owner 实际持有梯度且优化器
   更新成功时推进；计划按 GLOBAL 的 `K`、GROUP 的 `K_g` 与 PRIVATE 的 `K_t` 展开更新预算。
   跨阶段起始 LR 不得高于上一阶段 terminal LR。冻结不得以 LR=0 模拟。
7. Phase 1 检查点保存完整模型、优化器、owner 调度器、RNG、owner 更新和冻结
   状态；Phase 2/3 保存绑定 immutable 锚点的 owner delta及同等恢复状态。历史记录尾、更新
   数、冻结轮、锚点、身份与状态hash 必须完全一致，才允许从轮边界恢复。
8. 最终产物为 `three_phase_final.pt/json`，Phase 2 另发布 `phase_2/stitched.pt/json`。
   训练身份、检查点 kind、张量 hash namespace 和 final kind 与四阶段、历史实现
   均隔离。当前代码拒绝四阶段配置，也不恢复/evaluate `four_phase_final`；历史文件
   保持只读。
9. `resolved_training_plan.json` 展开 GLOBAL、每个 GROUP/PRIVATE 的 nominal/terminal LR、
   nominal/effective 轮、冻结轮、实际更新 budget、size class、`unique_systems` 和
   capacity。历史实现 v1 与 Capacity v1 继续受 ADR-0027 约束并使用 `taskwise_refined`。

## 后果

- 小任务可以停止自身参数拟合，而其观测仍用于更高层共享表示；大任务和大组得到更长
  的固定训练预算。
- Stage 1、Stage 2 与 Stage 3 准备产物可复用；六份现役配置必须在新目录重新执行
  Stage 3 训练集、验证与测试集评估，既有四阶段输出不得覆盖或迁移。
- 分支执行顺序不影响拼接结果；任何越权 owner 变化、锚点不一致、delta 缺失或重叠都
  必须失败。

## 备选方案

- 拒绝按原始行自动分桶：多条件重复观测不能代表独立体系数。
- 拒绝 owner 冻结时删除任务：会同时剥夺 shared owner 所需的监督信号。
- 拒绝早停或验证最优拼接：与固定预算、报告-仅合同冲突。
- 拒绝同时修改采样、loss weighting、PCGrad 或专家数量：会混淆本轮实验变量。

## 关联

- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：身份与审计合同](0021-identity-audit-contract-v1.md)
- [ADR-0027：后期逐任务精调](0027-late-taskwise-refinement.md)
- [ADR-0046：按参数归属裁剪与原始样本采样](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0047：deterministic 四阶段训练（历史）](history.md#adr-0047)
