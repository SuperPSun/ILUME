# ADR-0098：Stage3 整轮训练诊断

- 状态：已接受
- 日期：2026-10-10
- 范围：现役 Entity-HoME Stage3 的只读训练诊断；训练算法、身份、checkpoint格式和五折评估不变。

## 决定

Phase1/2联合训练和Phase3单任务训练在现有 `diagnostics.jsonl` 写入整轮统计，替换此前仅代表最后一步的梯度字段。统计不进入loss、优化、调度、选模或科研身份；不增加前向、反向、梯度张量扫描或逐步日志。仅累计已有CPU范数标量和真实batch长度，内存与owner/任务数成正比，不随step数增长。门控诊断继续遵守 [ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)。

## 梯度统计口径

新行使用 `diagnostics_version: 2`，`epoch_gradient_stats` 包含：

- `steps`、`clip_enabled`、`max_grad_norm`：整轮实际step数及当前裁剪配置。
- `total.pre_norm/post_norm`：已有整体裁剪前后范数的整轮算术均值和最大值。
- `owners.<owner>`：`trainable_steps`、`gradient_steps`、`pre_norm/post_norm` 的 `mean/max`、`clip_count`、`clip_rate`。
- `tasks.<task>`：实际处理的 `samples`、参与的 `steps`，以及原始任务梯度范数的 `mean/max`；任务列表对应当前训练scope。

owner列表来自公开参数归属接口，包含冻结owner和冻结模拟backbone。冻结在轮边界执行，因此当前轮可训练owner的 `trainable_steps` 等于整轮step数；任务耗尽可能使其 `gradient_steps` 更小。只有实际有梯度的owner才进入范数与裁剪率分母：零梯度张量算观测，缺失梯度不算。冻结或整轮无梯度的owner，`gradient_steps/clip_count` 为0，范数及 `clip_rate` 为null；无梯度但仍可训练的owner保留非零 `trainable_steps`。

裁剪触发按现有PyTorch系数 `min(max_grad_norm / (pre_norm + 1e-6), 1)` 小于1判定。阈值非正时裁剪未启用，观测owner的触发数及比例为0。范数使用已有实际裁剪返回值，不能以聚合前的任务范数替代；Phase3归入当前任务PRIVATE，不改变任何裁剪调用。

## 共享样本贡献

[ADR-0096](0096-stage3-batch-sample-weighted-gradients.md) 的实际batch记录数是唯一权重来源。`shared_sample_contributions` 按GLOBAL和各GROUP分别记录：

- `steps/samples`：该共享owner可训练且有实验任务参与时的聚合step数和实际记录数。
- `tasks.<task>.samples/sample_fraction`：整轮参与记录数及其在该共享owner全部记录中的占比。
- `tasks.<task>.weight_sum/weight_mean`：逐step的 `B_t / sum(B_t)` 累计值，以及除以该共享owner聚合step数的均值。任务耗尽后对后续step贡献为零，不缩小这个均值分母。
- GLOBAL另有 `groups.<group>`，将同GROUP任务记录数及其逐步权重合并，观察真正跨GROUP样本加权的结果。

GLOBAL分母跨当前全部实验任务，GROUP分母仅含当前组内实验任务。参数缺失梯度不改变样本分母；贡献表示实际聚合的名义系数，不表示梯度向量大小、方向或因果贡献。共享owner冻结时不累计；模拟任务永远不进入共享统计，仅保留任务处理记录及PRIVATE梯度统计。只有模拟参与的step和Phase3不产生共享贡献。无有效分母时记录数/累计权重为0，占比/平均权重为null。

整轮记录占比与逐步权重均值不同。例如A/B两步分别处理2/1、1/0条，A记录占比为3/4，平均权重为 `(2/3 + 1)/2 = 5/6`。两项指标分别回答整轮覆盖和逐步归一化影响，不能互相替代。尾batch使用真实长度，microbatch不重复计数。

## 日志与恢复兼容

保留原phase、epoch、scope、owner更新计数及其他诊断。新版不再输出顶层 `task_gradient_norms`、`assembled_owner_norms`、`clip_pre_norm`、`clip_post_norm`、`clip_owner_pre_norms`、`clip_owner_post_norms` 和Phase3的 `gradient_norm`，分析代码应读取新块。历史非Entity-HoME保持其原字段。

历史日志与实验产物只读，不补写旧轮统计。身份一致的既有checkpoint继续按原严格规则恢复；恢复后新增行使用version2，允许同一文件保留未标版本的旧行及新行。累计器每epoch重新建立，不需要checkpoint状态，不修改连续历史、owner更新/冻结、RNG和checkpoint合法性检查。仅本次诊断修改不要求重训Stage1/2/3；此前算法变更的重训边界仍有效。

## 验证

手算均值/最大值、裁剪边界、禁用裁剪、零梯度与缺失梯度；覆盖跨GROUP、尾batch、任务耗尽、冻结、实验/模拟混合、仅模拟及Phase3，检查权重守恒及两种贡献占比的区别。临时CPU三阶段训练比较启用累计与测试中禁用累计的最终模型张量hash、训练身份、优化器、调度器及RNG完全一致；中断恢复保持结果一致，旧诊断行字节保留，追加新版行。完整pytest、入口help、compileall、diff及文档链接检查，不运行正式GPU训练或覆盖历史产物。
