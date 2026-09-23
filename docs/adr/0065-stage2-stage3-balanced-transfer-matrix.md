# ADR-0065：Stage2→Stage3 等行数迁移矩阵

- 状态：Retired（由 [ADR-0071](0071-retire-balanced-stage2-stage3-transfer.md) 退役）
- 日期：2026-09-21
- 范围：`configs/ablations/stage2_stage3_transfer_balanced.yaml`
- 关联：[ADR-0062](0062-stage2-stage3-full-transfer-matrix.md)

> 2026-09-22：下游训练按
> [ADR-0068](0068-stage2-stage3-joint-downstream-adaptation.md) 同步更新ObjectEncoder与MLP；
> 本文的等行数抽样与Stage2更新预算合同不变。

> 2026-09-23：等行数实验的配置与执行能力已按 ADR-0071 移除。以下为历史合同，
> 旧输出只读，不可用当前代码 resume、prepare、train 或 summarize。

## 决定

保留 ADR-0062 的 full-data matrix，新增独立 balanced matrix。九个 source 的训练子集
大小为全部过滤完成后的 prepared train rows 数的最小值 N。每个 source 使用
`seed/source/balanced-v1` 派生的独立 CPU RNG，无放回抽取一次固定子集，并按原始行顺序
保存；十个 epoch 复用该子集，各 epoch 仍使用原 raw permutation，无 padding/drop-last。
`--source` 只控制执行范围，N 始终从全部九个 source 计算。

所有 source 的 batch=256、epochs=10、scheduler 总步数 `10*ceil(N/256)` 相同。
第一个 epoch 冻结 backbone，因此冻结/解冻更新预算也相同。仍使用全部九任务初始化、
physics-only supervision、相同 Stage1 anchor、最终 epoch encoder 和不变的 Stage3 MLP。
零 update baseline、20 targets、五折 validation、TG 定义和输出 CSV/SVG 结构不变；
任务集合由 [ADR-0067](0067-stage3-twenty-task-catalog.md) 修订。

## 抽样与身份

- Stage2 root 的 `resolved_sampling_plan.json` 在训练前写入每 source 的可用/选中行数、
  选中索引与 source rows、selection hash、seed、unique ordered entity systems、unique
  entities、每 system 行数直方图及共同 update/freeze budget。这里 system 定义为有序
  prepared entity-index tuple，不包括条件，不声称等同于每种 catalog topology 的统计口径。
- Partial Charge 按分子行抽样，完整保留该行原子标签/mask，并重建 ragged offsets。
- source checkpoint training identity 绑定 selection hash；final manifest 验证实际更新次数。
  encoder manifest、representation、MLP 和 summary 绑定 balanced experiment identity，
  禁止 full/balanced 交叉消费或 resume。完整 source 跳过前还需核对重建的抽样计划。
- full 是缺省且序列化省略，既有 full-data config/identity 保持原值。
- 使用独立 `outputs/ablations/stage2_stage3_transfer_balanced/`；现有结果不覆盖。

## 解释边界

该实验控制训练行数、每行 exposure 和 optimizer updates，并不统一分子/体系多样性、
原子标签数量或 FLOPs。使用原 prepared train normalization（含完整 source train 的
scaler），不重新拟合；所以也不是仅访问 N 行信息的严格 sample-efficiency 实验。
validation 仅报告、不选模，不使用 test。先用 balanced 与 full-data 的差异判断规模效应，
不能仅由单次子集实验断言 physics task 的因果价值；unique-system balancing 和多抽样 seed
属于后续独立对照。本次不执行正式训练、representation prepare 或汇总。
