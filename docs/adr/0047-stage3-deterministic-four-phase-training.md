# ADR-0047：Stage 3 deterministic 四阶段训练

- 状态：Accepted
- 日期：2026-09-07
- 修订：ADR-0027 的现役 v2 Stage 3 refinement、selection、checkpoint 与 final artifact

## 背景

现役 Stage 3 使用固定 `80% joint + 20% PRIVATE-only refinement`，并按 validation
normalized MAE 为每个 task 选择不同 epoch 的 PRIVATE state。该策略不能分别控制 GLOBAL、
GROUP 与 PRIVATE 的学习率，也不能让六个 GROUP 从相同 shared checkpoint 独立专门化；最终
artifact 还混合了 validation-best state，而不是预注册预算的最后状态。

## 决定

1. 现役 v2 Base、全部 v2 split，以及 ADR-0034/0036 的两个 Stage 3 消融使用
   `schedule_mode=four_phase`：Phase A 为 5 epoch joint bootstrap，Phase B 为 10 epoch
   shared consolidation，Phase C 为六个 group 分支，Phase D 为每个 task 的独立分支。所有
   scope 均使用固定预算的最后 epoch，validation 只记录、不选择模型。
2. A/B 使用 hierarchical PCGrad。C 冻结 GLOBAL，只在当前 GROUP block 对组内 tasks 执行
   PCGrad，同时更新这些 task 的 PRIVATE；D 冻结 GLOBAL/GROUP，只更新一个 PRIVATE 且关闭
   PCGrad 和 task/group weight。四阶段继续使用 ADR-0046 的 raw sampling 与 ownership-aware
   clipping，并全程复用按全部 active tasks 解析的 `B_t`。
3. 每个 phase/branch 新建 AdamW 与 phase-local cosine scheduler。A 的 GLOBAL/GROUP/PRIVATE
   LR 均为 `3e-4`，5% warmup 后到 0.5；B 为 `3e-5/1e-4/1.5e-4×task_scale`，到 0.2；
   C 为 `7.5e-5/1e-4×task_scale`，到 0.2；D 为 `1e-4×task_scale`，到 0.1。
   冻结通过 `requires_grad=False`，不得使用 LR=0。
4. GROUP 的 expert 数、expert hidden ratio 与 C epoch 由各 group 配置。task 可覆盖
   `private_experts`、private/FiLM/tower hidden ratio 与 PRIVATE LR scale；TaskGate 候选数按
   本 task 的实际 GLOBAL、GROUP、PRIVATE expert 数构造。结构从 Phase A 起固定，不在分叉时
   改 shape。
5. 六个 C 分支绑定同一个 Phase-B full-state hash，结束后 stitch 互不重叠的 GROUP 与组内
   PRIVATE state。21 个 D 分支绑定同一个 Phase-C stitched hash，结束后只 stitch PRIVATE。
   branch RNG 由 seed、fold、phase、scope 派生，因此结果不依赖执行顺序。
6. A/B checkpoint 保存完整状态；C/D checkpoint 保存 anchor hash、owner delta、optimizer、
   scheduler、RNG 与 update count。只有 history 尾与 checkpoint 完全一致时才能从完整 epoch
   恢复；所有 branch final 完整且 hash 合法后才原子发布 stitched artifact。
7. 现役最终模型为 `four_phase_final.pt` 与 `four_phase_final.json`，记录 phase LR/epoch、
   capacity、anchor、per-owner/final-state hash 和最终完整 validation；不包含 `best_metric`、
   `selected_epoch` 或 `best_state`。evaluate 默认 selector 为 `four_phase_final`，现役配置拒绝
   `--checkpoint-epoch`。
8. legacy v1 与 Capacity v1 继续使用 ADR-0027 的 joint/refinement、validation-best stitching、
   format v2 checkpoint 与 `taskwise_refined` artifact。默认 schedule mode 在序列化中省略，
   旧 training identity 不变；four-phase 使用独立 identity/checkpoint/final-artifact contract。

## 后果

- 同一 GROUP 或 task 的训练预算、学习率、容量和最终 state 都可直接审计，validation 不再对
  正式 benchmark 模型产生反馈。
- 分支 checkpoint 通过 anchor+owner delta 避免重复保存冻结 tensor，但恢复和 stitch 必须额外
  校验 anchor 与 owner 集合。
- Stage 1、Stage 2 与 Stage 3 prepared artifact 可复用；现役 v2 与两个消融必须在新输出目录
  重新执行 Stage 3 train、validation 和 test evaluation，旧 Stage 3 checkpoint 不可续训。

## 备选方案

- 拒绝延续统一 optimizer 与 LR：无法实现 ownership-specific LR 和真实冻结。
- 拒绝顺序训练六组或 21 tasks：后训练 scope 会继承前一分支的非目标随机状态，破坏同源比较。
- 拒绝 validation-best stitching：与 deterministic fixed-budget 最后一轮合同冲突。
- 拒绝每个 branch 保存完整模型：会重复存储大量相同 GLOBAL/GROUP tensor。

## 关联

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：identity/audit contract](0021-identity-audit-contract-v1.md)
- [ADR-0027：late taskwise refinement](0027-late-taskwise-refinement.md)
- [ADR-0046：ownership clipping 与 raw sampling](0046-stage3-ownership-clipping-raw-sampling.md)
