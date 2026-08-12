# ADR-0006：覆盖型 epoch 与正式实验配置

- 状态：Accepted
- 日期：2026-07-26
- 部分取代：ADR-0008 已取代 Base micro-batch 与 learning rate profile；覆盖型 epoch 决定仍有效。

## 决定

正式单卡训练器使用覆盖型 epoch，不再接受 `training.max_steps`。每个 epoch 先按45/45/10计算覆盖全部 role 训练池所需的最小抽样量，再补齐到完整 effective batch。role 内完成一轮无放回覆盖后才开始下一轮，并以 `seed + epoch_index` 产生确定性顺序。

用户通过 `training.epochs` 控制训练程度。warmup、cosine scheduler、masking curriculum 和日志仍使用由 epoch 数推导出的内部 global step。validation 和 checkpoint 只在 epoch 边界执行；checkpoint 格式升级为v3，旧step配置和v2 checkpoint不迁移。

正式 Base/Large/XLarge 默认训练5个覆盖 epoch。Base 使用512 micro-batch；Large 使用256；XLarge 使用128 micro-batch 和2步梯度累积，与 Large 保持相同的 effective batch=256。三种容量的数据准备参数相同，因此共享 Base prepared artifact；Large/XLarge 只改变模型和训练参数。OOM 时通过等比例增大梯度累积保持 effective batch；核心消融以 Base 为参考，一次只改变一个因素并隔离训练输出。

## 理由

自然遍历一次训练池会得到接近数据自然分布的 role 比例，与既定45/45/10冲突。覆盖型 epoch 同时保留离子主导采样和“一个周期内所有实体至少出现一次”的直观语义。内部保留 global step 是 optimizer、scheduler 和 curriculum 连续推进所必需的实现细节，但不再要求用户手工计算训练预算。

旧checkpoint只记录全运行step预算和sampler偏移，不能无歧义地转换为完整epoch边界。显式拒绝旧格式比隐式迁移更可审计。

## 后果

不同训练池大小会自动产生不同的 `steps_per_epoch`。不同 effective batch 的配置可能因补齐产生不足一个 batch 的抽样量差异，但每个 epoch 都满足完整覆盖；Large 与 XLarge 因 effective batch 相同而具有相同的 epoch 抽样预算。checkpoint 只能从已完成 epoch 恢复；若训练在 epoch 中途退出，该 epoch 会重新执行。

本决定取代 ADR-0001 中“通过全运行 `max_steps` 校验覆盖”的训练预算部分，不改变其数据边界和45/45/10角色比例。
