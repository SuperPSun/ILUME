# ADR-0043：退役 Stage 2 评估与现役 v2 单任务微调

- 状态：Accepted
- 日期：2026-09-05
- 修订：取代 ADR-0023、ADR-0024、ADR-0025、ADR-0027 中的 Stage 2 reporting、最终评估 artifact 与现役 v2 refinement 决定；同时取代 ADR-0022、ADR-0028～0030、ADR-0032、ADR-0035、ADR-0037、ADR-0038、ADR-0040、ADR-0042 中的 Stage 2 baseline/reporting 部分

## 背景

Stage 2 的任务表现不再作为论文比较目标。继续训练四个 task head 的末期 refinement，并让主模型、消融和 baseline 生成 Stage 2 test 榜单，会增加运行成本和汇总复杂度，但不再服务当前结论。Stage 2 作为 Stage 3 的表示学习阶段仍然保留。

## 决定

1. 现役 `configs/v2/stage2/base.yaml` 只运行九任务的 10 个 joint epochs，配置为 `refinement_epochs: 0` 和空 `refinement_tasks`。训练完成后直接发布最终 joint checkpoint、`stage2_encoder.pt` 与仅含最终 joint validation 的 `final_metrics.json`，不生成 `taskwise_refined.pt`、`taskwise_refinement.json` 或 stitched validation。
2. Stage 2 配置与训练器仍接受正数 refinement epochs 和非空任务列表，以保持 legacy v1、Capacity v1 与 No-Stage1 消融的冻结训练定义；零 epochs 与空任务列表必须同时出现。
3. 删除 Stage 2 test evaluation 公共入口及其专用实现。主模型、消融和 baseline 均不再生成新的 Stage 2 test evaluation。
4. baseline 配置、数据解析、训练/evaluate CLI、adapter 与 sweep 只支持 Stage 3。七个 baseline 的正式 sweep 均为 21 tasks × 5 folds，即 105 个训练 job。
5. 统一汇总只发布 Stage 3 test/validation 的 leaderboard、metrics、wins、health、overview、radar 与 `summary.json`。旧 candidate 中存在的 Stage 2 section 仅被忽略，不作为错误或 health 字段传播。
6. 既有 `outputs/`、legacy 配置、Capacity 配置和历史 ADR 正文保持只读；不迁移或删除历史 Stage 2 产物。

## 兼容性与恢复

- `stage2_encoder.pt` 的 state-based semantic identity 和跨 Stage 绑定方式不变；是否可复用下游 Stage 3 由新旧 encoder semantic identity 是否一致决定。
- 旧式 v2 refinement run 不能在 joint-only 合同下 resume。若新旧 encoder identity 不一致，才需要重跑 Stage 3 prepare、train 与 evaluation。
- ADR-0027 对 Stage 3 PRIVATE-only refinement 以及 legacy/Capacity Stage 2 refinement 的定义继续有效；本 ADR 只取代现役 v2 Stage 2 的对应部分。

## 后果

- `scripts/stage2/evaluate.py` 不再存在；`scripts/benchmarks/{train,evaluate}.py --benchmark` 只接受 `stage3`。
- 新 baseline YAML 不再接受 `stage2_physics`，D-MPNN 不再包含 Stage 2 atom evaluation authority。
- 依赖旧 Stage 2 summary schema 的消费者必须停止读取相关键和文件。
- 现有 Stage 3 训练与预测合同未改变，已有结果只需重新运行 summarizer。

## 拒绝方案

- 只隐藏 Stage 2 榜单但继续训练/evaluate：仍保留无效运行成本和维护面。
- 删除所有 refinement 支持：会改写 legacy、Capacity 与 No-Stage1 的冻结训练定义。
- 强制迁移或清理历史输出：会破坏 provenance，且不是本次合同变更所需。
