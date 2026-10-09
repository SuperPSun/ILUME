# ADR-0065：Stage2→Stage3 等行数迁移矩阵

- 状态：已退役（由 [ADR-0071](0071-retire-balanced-stage2-stage3-transfer.md) 退役）
- 日期：2026-09-21
- 范围：`configs/ablations/stage2_stage3_transfer_balanced.yaml`
- 关联：[ADR-0062](0062-stage2-stage3-full-transfer-matrix.md)

> 2026-09-22：下游训练按
> [ADR-0068](0068-stage2-stage3-joint-downstream-adaptation.md) 同步更新ObjectEncoder与MLP；
> 本文的等行数抽样与Stage2更新预算合同不变。

> 2026-09-23：等行数实验的配置与执行能力已按 ADR-0071 移除。以下为历史合同，
> 旧输出只读，不可用当前代码恢复、prepare、训练集或 summarize。

## 决定

保留 ADR-0062 的全量数据 matrix，新增独立等行数 matrix。九个来源的训练子集
大小为全部过滤完成后的准备产物训练集行数的最小值 N。每个来源使用
`seed/source/balanced-v1` 派生的独立 CPU RNG，无放回抽取一次固定子集，并按原始行顺序
保存；十个轮复用该子集，各轮仍使用原原始随机排列，无 padding/drop-last。
`--source` 只控制执行范围，N 始终从全部九个来源计算。

所有来源的 batch=256、轮=10、调度器总步数 `10*ceil(N/256)` 相同。
第一个轮冻结 backbone，因此冻结/解冻更新预算也相同。仍使用全部九任务初始化、
仅物理监督 supervision、相同 Stage1 锚点、最终轮编码器和不变的 Stage3 MLP。
零更新基线、20 targets、五折验证、TG 定义和输出 CSV/SVG 结构不变；
任务集合由 [ADR-0067](0067-stage3-twenty-task-catalog.md) 修订。

## 抽样与身份

- Stage2 根目录的 `resolved_sampling_plan.json` 在训练前写入每来源的可用/选中行数、
  选中索引与来源行、选择 hash、种子、unique ordered 实体 systems、unique
  实体、每 system 行数直方图及共同更新/冻结 budget。这里 system 定义为有序
  准备产物实体-index tuple，不包括条件，不声称等同于每种任务目录拓扑的统计口径。
- Partial Charge 按分子行抽样，完整保留该行原子标签/mask，并重建 ragged offsets。
- 来源检查点训练身份绑定选择 hash；final 清单验证实际更新次数。
  编码器清单、表示、MLP 和汇总绑定等行数 experiment 身份，
  禁止 full/等行数交叉消费或恢复。完整来源跳过前还需核对重建的抽样计划。
- full 是缺省且序列化省略，既有全量数据配置/身份保持原值。
- 使用独立 `outputs/ablations/stage2_stage3_transfer_balanced/`；现有结果不覆盖。

## 解释边界

该实验控制训练行数、每行 exposure 和优化器次更新，并不统一分子/体系多样性、
原子标签数量或 FLOPs。使用原准备产物训练集归一化（含完整来源训练集的
scaler），不重新拟合；所以也不是仅访问 N 行信息的严格样本-efficiency 实验。
验证仅报告、不选模，不使用测试集。先用等行数与全量数据的差异判断规模效应，
不能仅由单次子集实验断言 physics 任务的因果价值；unique-system balancing 和多抽样种子
属于后续独立对照。本次不执行正式训练、表示 prepare 或汇总。
