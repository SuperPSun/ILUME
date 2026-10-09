# ADR-0042：D-MPNN 组分编码器共享权重

- 状态：已接受
- 日期：2026-09-04
- 修订：仅取代 ADR-0028 第 4 条中的多组分独立 block 决定

## 背景

ADR-0028 的多组分 D-MPNN 为每个注册表槽位建立独立 `BondMessagePassing` block。这样会让
阳离子、阴离子、solute 与 solvent 的图表示学习依赖槽位，并让组分数不同的任务具有不同数量的
消息传递参数。当前实验希望所有组分服从同一个分子图编码函数，同时继续保留槽位顺序
和下游有序拼接。

## 决定

1. 所有多组分标量任务只实例化一个 Chemprop `BondMessagePassing` block，并通过
   `MulticomponentMessagePassing(shared=True, n_components=C)` 在全部注册表槽位间复用。
2. 各组分仍分别构图和编码，输出继续按注册表槽位顺序拼接；预测器输入宽度、numeric
   条件、目标 scaler、训练预算、检查点与评估/报告口径不变。
3. 不同任务与折仍是独立训练任务，不共享参数或优化器。单组分标量任务与 Partial Charge
   路径保持不变。
4. `multicomponent_shared: true` 属于正式模型合同并进入既有训练/报告身份。旧
   `outputs/benchmarks/v1/dmpnn` 保留；共享权重正式运行必须使用新的
   `outputs/benchmarks/v2/dmpnn` 根，不迁移或复用旧检查点。

## 后果

- 二组分和三组分任务的消息传递参数量相同，组分角色差异只通过有序拼接位置进入
  预测器。
- 新旧 D-MPNN 结果具有不同 scientific 身份，不能在同一报告 study 中混用。
- 训练任务数仍为 109，独立环境、失败重跑和不支持恢复的合同不变。

## 拒绝方案

- 保留每个槽位独立 block：不满足组分编码器共享的实验目标。
- 跨任务或折共享编码器：这会把独立单任务基线改成多任务学习，超出本次变更范围。
