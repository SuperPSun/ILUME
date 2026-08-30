# ADR-0030：MoLFormer 吞吐与训练预算合同

- 状态：Accepted
- 日期：2026-08-30

## 背景

ADR-0029 的首版实现会在每个 epoch 的每个 batch 重复 tokenize，并按 component 串行调用共享 backbone。正式 sweep 尚未运行，因此在不迁移既有 checkpoint 的前提下，可以先消除这些重复工作并冻结新的训练预算。

## 决定

1. 每个 run 在内存中为 train/valid 的 unique model-input SMILES 建立一次未 padding 的 `input_ids`/`attention_mask` cache；test 只在最佳 checkpoint 确定后的 evaluation 中加入。cache 不落盘、不跨 task/run 复用，超长 train 跳过和 valid/test 截断仍遵循 ADR-0029。
2. collate 按 registry slot 形成 component-major 的 `(C×B,L)` batch，所有 component 只调用一次共享 MoLFormer backbone；pooled output 恢复为 `(B,C,768)` 后继续使用既有 ordered concat、单层 fusion 和官方 head。
3. 训练固定 deterministic sortish bucketing：每轮以 `seed+epoch` 洗牌，在 `20×batch_size` 窗口内按 row 最大 component token length 排序成批，再确定性打乱 batch 顺序。每轮完整覆盖 retained rows且不 drop last。
4. 科研训练合同改为 batch 256、最多50 epochs、patience 8，并在 train、checkpoint validation 和独立 valid/test 中启用 TF32。AdamW、两组学习率、weight decay、normalized MSE、5% warmup、cosine decay和validation MAE选择保持不变。
5. DataLoader 固定4 workers、pin memory、persistent workers、prefetch factor 2和non-blocking H2D。这些运行参数进入公开provenance但不进入scientific identity；batch、bucketing、TF32和训练预算进入training identity。
6. 任一严格batch发生OOM、NaN或CUDA错误时立即失败；禁止自动缩批、改变精度、CPU fallback或修改bucketing。旧MoLFormer checkpoint与新training identity不兼容，不迁移或resume。

## 后果

- 本ADR仅取代ADR-0029第6条的逐component执行方式和第7条的训练预算；模型、数据、normalization、loss、evaluation与reporting合同不变。
- 合并forward和TF32会改变浮点与随机数轨迹，因此不与ADR-0029首版checkpoint做bitwise或续训兼容。
- 正式108-job sweep仍由用户显式运行，本决定只授权临时测试、forward/backward与吞吐smoke。
