# ADR-0030：MoLFormer 吞吐与训练预算合同

- 状态：已接受
- 日期：2026-08-30

> Patience、验证最优检查点与训练预算由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代；吞吐和调度器条款保持有效。

## 背景

ADR-0029 的首版实现会在每个轮的每个 batch 重复 tokenize，并按组分串行调用共享 backbone。正式 sweep 尚未运行，因此在不迁移既有检查点的前提下，可以先消除这些重复工作并冻结新的训练预算。

## 决定

1. 每个运行在内存中为训练集/验证集的 unique 模型-input SMILES 建立一次未 padding 的 `input_ids`/`attention_mask` 缓存；测试集只在最佳检查点确定后的评估中加入。缓存不落盘、不跨任务/运行复用，超长训练集跳过和验证集/测试集截断仍遵循 ADR-0029。
2. collate 按注册表槽位形成组分-major 的 `(C×B,L)` batch，所有组分只调用一次共享 MoLFormer backbone；pooled 输出恢复为 `(B,C,768)` 后继续使用既有 ordered concat、单层 fusion 和官方预测头。
3. 训练固定 deterministic 近似按长度排序 bucketing：每轮以 `seed+epoch` 洗牌，在 `20×batch_size` 窗口内按行最大组分 token length 排序成批，再确定性打乱 batch 顺序。每轮完整覆盖 retained 行且不 drop last。
4. 科研训练合同固定为 batch 128、编码器/预测头 learning rate `5e-6/5e-5`、最多10 轮，并在训练集、检查点验证和独立验证集/测试集中启用 TF32。两组learning rate由batch 256合同按linear scaling减半；AdamW、权重衰减、归一化 MSE、5% 预热和余弦 decay保持不变。
5. DataLoader 固定4 workers、pin memory、persistent workers、prefetch factor 2和non-blocking H2D。这些运行参数进入公开来源记录但不进入scientific 身份；batch、bucketing、TF32和训练预算进入训练身份。
6. 任一严格batch发生OOM、NaN或CUDA错误时立即失败；禁止自动缩批、改变精度、CPU fallback或修改bucketing。旧MoLFormer 检查点与新训练身份不兼容，不迁移或恢复。

## 后果

- 本ADR仅取代ADR-0029第6条的逐组分执行方式和第7条的训练预算；模型、数据、归一化、loss、评估与报告合同不变。
- 合并前向和TF32会改变浮点与随机数轨迹，因此不与ADR-0029首版检查点做bitwise或续训兼容。
- 正式108-作业 sweep仍由用户显式运行，本决定只授权临时测试、前向/反向与吞吐smoke。
