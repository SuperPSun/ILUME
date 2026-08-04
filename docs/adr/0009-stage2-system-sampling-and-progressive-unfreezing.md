# ADR-0009：Stage 2 体系采样、PairEncoder 与渐进解冻

- 状态：Accepted
- 日期：2026-08-04
- 部分取代：ADR-0007 的任务内行级采样、共享 IL 回归 trunk、立即训练 backbone 和完整 QM 标签假设。

## 决定

density、heat capacity 和 thermal expansion 在任务内部按有序 cation/anion 体系均匀采样。每轮先无放回覆盖体系，每次访问体系只取一个条件点；体系内条件点也无放回轮换并在耗尽后确定性重排。QM 和 transfer 继续逐行采样，验证与选模继续采用逐行指标。

三项 IL 共享一个有序 PairEncoder，transfer 使用独立 PairEncoder。PairEncoder 联合左右 CLS、绝对差和逐元素乘积，经残差 MLP 产生 `d_model` 维 pair embedding。五项任务的回归 MLP 参数完全独立；Stage 1 reconstruction heads 永久冻结。

训练前10%完整任务块只更新 PairEncoder 和回归器，随后解冻全部 Stage 1 编码路径。新模块使用全程 warmup+cosine，backbone 在冻结期 learning-rate factor 为零，解冻后按剩余步数单独 warmup+cosine。总训练步数和五任务块比例不变。

QM 允许单行部分标签缺失。常见缺失标记写入布尔 target mask，scaler 只拟合 train 有效值，SmoothL1 先按标签聚合再对有效标签等权平均。全缺失行审计后排除；train 每列必须有值，valid 整列缺失时该列指标为 NaN 并从 QM 宏平均中跳过。

## 理由

按行采样会让拥有更多温度点的 IL 获得更大期望权重；体系分层采样把权重归还给化学体系，同时保留全部条件信息。显式 PairEncoder 能表达离子间交互而不把三项 IL 的回归器继续绑定。先稳定新模块再微调 backbone，可减少随机回归头对预训练表示的早期扰动。masked 多目标损失避免因少数缺失标签丢弃仍有监督信息的分子。

## 后果

Stage 2 data artifact 和 checkpoint 均升级到 v2，旧 checkpoint 不可恢复。正式数据和训练输出迁移到 `artifacts/stage_v2/`，保留 `artifacts/stage2/` 不变。checkpoint 新增体系 cursor、解冻 step 与阶段语义；恢复后必须重现任务、体系、源行和 loss 序列。
