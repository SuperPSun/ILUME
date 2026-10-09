# ADR-0043：退役 Stage 2 评估与现役 v2 单任务微调

- 状态：已接受
- 日期：2026-09-05
- 修订：取代 ADR-0023、ADR-0024、ADR-0025、ADR-0027 中的 Stage 2 报告、最终评估产物与现役 v2 精调决定；同时取代 ADR-0022、ADR-0028～0030、ADR-0032、ADR-0035、ADR-0037、ADR-0038、ADR-0040、ADR-0042 中的 Stage 2 基线/报告部分

> 后续边界：[ADR-0083](0083-stage2-home-full-artifact-evaluation.md) 曾恢复正式 HoME 独立评估，现由 [ADR-0085](0085-retire-stage2-home-evaluation.md) 再次退役。完整 Stage2 产物合同仍按 0083，正式训练配方按 0082；本文旧训练描述不恢复。

## 背景

Stage 2 的任务表现不再作为论文比较目标。继续训练四个任务预测头的末期精调，并让主模型、消融和基线生成 Stage 2 测试集榜单，会增加运行成本和汇总复杂度，但不再服务当前结论。Stage 2 作为 Stage 3 的表示学习阶段仍然保留。

## 决定

1. 现役 `configs/v2/stage2/base.yaml` 只运行九任务的 10 个联合训练轮，配置为 `refinement_epochs: 0` 和空 `refinement_tasks`。训练完成后直接发布最终联合训练检查点、`stage2_encoder.pt` 与仅含最终联合训练验证的 `final_metrics.json`，不生成 `taskwise_refined.pt`、`taskwise_refinement.json` 或拼接后的验证。
2. Stage 2 配置与训练器仍接受正数精调轮和非空任务列表，以保持历史实现 v1、Capacity v1 与 No-Stage1 消融的冻结训练定义；零轮与空任务列表必须同时出现。
3. 删除 Stage 2 测试集评估公共入口及其专用实现。主模型、消融和基线均不再生成新的 Stage 2 测试集评估。
4. 基线配置、数据解析、训练/evaluate CLI、adapter 与 sweep 只支持 Stage 3。七个基线的正式 sweep 均为 21 任务 × 5折，即 105 个训练作业。
5. 统一汇总只发布 Stage 3 测试集/验证的榜单、metrics、wins、health、overview、radar 与 `summary.json`。旧候选中存在的 Stage 2 section 仅被忽略，不作为错误或 health 字段传播。
6. 既有 `outputs/`、历史实现配置、Capacity 配置和历史 ADR 正文保持只读；不迁移或删除历史 Stage 2 产物。

## 兼容性与恢复

- `stage2_encoder.pt` 的状态-based 语义身份和跨 Stage 绑定方式不变；是否可复用下游 Stage 3 由新旧编码器语义身份是否一致决定。
- 旧式 v2 精调运行不能在联合训练-仅合同下恢复。若新旧编码器身份不一致，才需要重跑 Stage 3 prepare、训练集与评估。
- ADR-0027 对 Stage 3 仅PRIVATE更新精调以及历史实现/Capacity Stage 2 精调的定义继续有效；本 ADR 只取代现役 v2 Stage 2 的对应部分。

## 后果

- `scripts/stage2/evaluate.py` 不再存在；`scripts/benchmarks/{train,evaluate}.py --benchmark` 只接受 `stage3`。
- 新基线 YAML 不再接受 `stage2_physics`，D-MPNN 不再包含 Stage 2 原子评估合同依据。
- 依赖旧 Stage 2 汇总结构定义的消费者必须停止读取相关键和文件。
- 现有 Stage 3 训练与预测合同未改变，已有结果只需重新运行 summarizer。

## 拒绝方案

- 只隐藏 Stage 2 榜单但继续训练/evaluate：仍保留无效运行成本和维护面。
- 删除所有精调支持：会改写历史实现、Capacity 与 No-Stage1 的冻结训练定义。
- 强制迁移或清理历史输出：会破坏来源记录，且不是本次合同变更所需。
