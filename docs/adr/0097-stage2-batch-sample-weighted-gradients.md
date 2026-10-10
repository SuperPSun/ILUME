# ADR-0097：Stage2 按当前 batch 样本数加权共享梯度

- 状态：已接受
- 日期：2026-10-10
- 修订范围：ADR-0095 的现役 Entity-HoME Stage2 任务补偿和逐任务更新；Stage1、Stage3及历史非 Entity-HoME 实现不变。

## 决定

Stage2 与 [ADR-0096](0096-stage3-batch-sample-weighted-gradients.md) 使用相同的共享梯度公式：`sum(B_t * g_t) / sum(B_t)`。`B_t` 是当前任务逻辑 batch 的真实记录数，`g_t` 是该任务平均 SmoothL1 Loss 梯度；GLOBAL直接跨任务聚合，GROUP在组内聚合，PRIVATE保留本任务原始平均梯度。共享分母不按参数是否有梯度重算，完全没有梯度的参数不赋值。

Stage2五项模拟都是共享预训练任务，全部参与GLOBAL及对应GROUP；不套用Stage3模拟辅助任务的PRIVATE隔离规则。移除正式Entity-HoME的任务等权补偿系数，不乘配置task/group权重、任务总行数或独立体系数。YAML保留原task权重字段以维持配置接口，但该路径不使用其数值聚合或缩放PRIVATE。

原来每个任务batch立即更新一次，改为每个联合step取各未耗尽任务的一个batch，在同一模型状态计算各任务梯度后聚合、裁剪并更新一次。复用原epoch_batch_schedule的轮内顺序和任务内随机排列，保留Raw Sampling、每轮完整行覆盖，不重复、不补齐。尾batch按实际长度计权，耗尽任务不进入后续分子或分母；单任务step共享梯度保持原值。microbatch以完整任务batch为分母累积，不改变平均Loss。

每轮联合优化器步数由各任务batch数之和改为最大值。AdamW、原整模梯度裁剪、LR1e-4、256逻辑/微批、10轮、cosine warmup函数及warmup比例不变；调度器每联合更新推进一次，依据新的联合步数覆盖完整10轮。Stage1仍永久冻结/eval，模型、owner、归一化、验证仅报告和末轮导出不变。

## 身份与产物

训练身份的math_contract加入：

```yaml
gradient_aggregation: batch_sample_weighted_owner_raw_v1
gradient_weighting:
  weight_source: actual_task_batch_size
  normalization: participating_simulation_samples
  global: cross_task_sample_mean
  group: within_group_sample_mean
  private: raw_task_mean_gradient
  simulation: shared_pretraining
  step: one_batch_per_active_task
```

合同字段进入原训练身份hash，不升级format4或kind。恢复严格匹配新身份、联合更新数、scheduler、RNG、owner和历史尾部；final加载要求新聚合合同，Stage3迁移也通过同一加载校验，拒绝旧配方final。历史检查点及产物只读，复现需对应历史Git版本。

采用新规则需要在新输出目录从头训练Stage2，再由新Stage2 final迁移训练Stage3。身份一致的Stage1和Stage2数据/冻结实体prepare可复用；Stage3数据准备在既有数据/Stage1身份一致时可复用，不改写历史产物。

## 验证

手算跨GROUP梯度1/3和batch长度3/1得到GLOBAL1.5，GROUP及PRIVATE保持1/3；microbatch切分结果一致。临时CPU端到端覆盖真实尾batch、任务耗尽、联合更新计数、完整行覆盖、冻结Stage1、final重载及Stage3迁移；中断恢复与连续训练张量hash一致，旧合同checkpoint/final拒载。执行完整pytest、相关入口help、compileall、diff及文档链接检查，不启动正式GPU训练。
