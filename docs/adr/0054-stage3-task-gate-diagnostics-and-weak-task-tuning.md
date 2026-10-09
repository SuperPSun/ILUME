# ADR-0054：Stage 3 任务门控诊断

- 状态：已接受
- 日期：2026-09-14
- 数值修订：当时同时调整四项任务配方；最终设置统一见 [ADR-0055](0055-stage3-pec50-phase3-single-variable-rollback.md)

## 决定

为观察任务门控对 GLOBAL、GROUP 和 PRIVATE 候选的使用情况，three-phase 验证
与独立 evaluator 复用同一次前向已返回的 `task_gate`，不增加前向或参与训练决策。

- 按实际任务-specific 候选 count，将权重依序聚合为 GLOBAL、GROUP、PRIVATE 权重占比。
- 每个任务报告三类均值权重占比、PRIVATE 权重占比的 p10/p50/p90，以及逐样本
  `-sum(p*log(p))/log(candidate_count)` 后取均值的归一化熵。
- 验证按折/任务报告；测试集同时报每折与 aggregate，aggregate 合并全部
  折-样本后计算，不对不同折的专家索引作语义对齐。
- `gate_diagnostics` 进入每轮 three-phase 验证历史记录、拼接后的/final 验证
  清单与独立验证/测试集 `summary.json`；历史实现 v1/Capacity 不输出。

预测 CSV、原 metrics、`reporting` block、common 报告结构定义、验证报告-仅
语义均不变；不改变预测、loss、优化器、PCGrad、采样、裁剪、模型结构或检查点
选择。owner 与恢复合同见 [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 和
[ADR-0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md)。

当年的配方调整为 volume expansion Phase 2 PRIVATE `4→0`、pEC50/xCO2 Phase 3 `3→5`、
viscosity PRIVATE dropout `0.10→0.15`；pEC50 后由 0055 回调。数值变化进入训练身份，
诊断字段不改变报告结构定义；不要将当时要求新目录重跑的说明套用于纯文档整理。
