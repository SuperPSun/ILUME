# ADR-0042：D-MPNN 组分编码器共享权重

- 状态：Accepted
- 日期：2026-09-04
- 修订：仅取代 ADR-0028 第 4 条中的多组分独立 block 决定

## 背景

ADR-0028 的多组分 D-MPNN 为每个 registry slot 建立独立 `BondMessagePassing` block。这样会让
cation、anion、solute 与 solvent 的图表示学习依赖槽位，并让组分数不同的任务具有不同数量的
message-passing 参数。当前实验希望所有组分服从同一个分子图编码函数，同时继续保留 slot 顺序
和下游有序拼接。

## 决定

1. 所有多组分标量任务只实例化一个 Chemprop `BondMessagePassing` block，并通过
   `MulticomponentMessagePassing(shared=True, n_components=C)` 在全部 registry slot 间复用。
2. 各组分仍分别构图和编码，输出继续按 registry slot 顺序拼接；predictor 输入宽度、numeric
   condition、target scaler、训练预算、checkpoint 与 evaluation/reporting 口径不变。
3. 不同 task 与 fold 仍是独立训练任务，不共享参数或优化器。单组分标量任务与 Partial Charge
   路径保持不变。
4. `multicomponent_shared: true` 属于正式模型合同并进入既有 training/reporting identity。旧
   `outputs/benchmarks/v1/dmpnn` 保留；共享权重正式运行必须使用新的
   `outputs/benchmarks/v2/dmpnn` 根，不迁移或复用旧 checkpoint。

## 后果

- 二组分和三组分任务的 message-passing 参数量相同，组分角色差异只通过有序拼接位置进入
  predictor。
- 新旧 D-MPNN 结果具有不同 scientific identity，不能在同一 reporting study 中混用。
- 训练任务数仍为 109，独立环境、失败重跑和不支持 resume 的合同不变。

## 拒绝方案

- 保留每个 slot 独立 block：不满足组分编码器共享的实验目标。
- 跨 task 或 fold 共享 encoder：这会把独立单任务 baseline 改成多任务学习，超出本次变更范围。
