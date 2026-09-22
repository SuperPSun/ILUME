# ADR-0067：Stage 3 二十任务 catalog 合同

- 状态：Accepted
- 日期：2026-09-22
- 范围：现役 v2 Stage 3、Stage 3 消融、baseline 与 Stage2→Stage3 transfer target

## 背景

现役数据 catalog 已移除 `experiment/isobaric_coefficient_of_volume_expansion`，Stage 3
observation task 因此由 21 个变为 20 个。Stage 1 数据和 Stage 2 九个 physics task、源行、
registry 均未改变。本修订只同步消费者的任务集合，不改变余下任务的数据、分组 recipe、模型、
训练数学或评估口径。

## 决定

1. `configs/v2/stage3/` 下全部现役配置、RDKit-HoME、No-Stage1 以及 full/balanced
   Stage2→Stage3 transfer 配置删除该 task。知识图谱分组保留原六组；
   `thermophysical_interfacial_response` 由九个 task 变为八个。
2. Stage2→Stage3 transfer 不再硬编码 target 数量；target 必须非空、唯一，并与指定
   Stage 3 authority 的 enabled task 集合完全一致。当前 full/balanced 矩阵因此均为
   `9×20`，每套共有 `10×20×5=1000` 个 Stage 3 MLP job。
3. `configs/v1` 与 `configs/experiments_v1` 属于冻结合同，不修改其 21-task fallback、YAML
   或历史 identity。绑定旧 v1 512D prepared artifact 的 Single-task MLP同样保持冻结。
   它们不能与本修订后的 catalog/artifact 混用。
4. 余下 20 个 task 的 GLOBAL/GROUP/PRIVATE capacity、LR、epoch、dropout、raw sampling、
   PCGrad、optimizer、loss、clipping 和 fixed-final-state 规则保持不变。

## Artifact 边界

- Stage 1 artifact 与现役 Stage 2 prepared data/checkpoint/encoder 可复用；本次 catalog 变化
  不构成重跑 Stage 2 的理由。
- Stage 3 prepared identity包含 task catalog和源数据集合，必须为 20-task catalog重新
  prepare；旧 21-task Stage 3 prepared/train/evaluation artifact保持历史只读。
- 主模型和 Stage 3 消融必须从新的 prepared artifact重新执行 Stage 3 train/evaluation。
  baseline必须按20-task catalog重新执行。
- full-data transfer 的既有十个 Stage 2 encoder可复用，但 representation bank、1000个
  Stage 3 MLP job和summary必须按20-task authority重新生成。balanced实验仍服从其独立
  identity与完整性校验，不跨合同强行加载。

## 关联

- [ADR-0048：Stage 3 three-phase](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0062：Stage2→Stage3 full transfer](0062-stage2-stage3-full-transfer-matrix.md)
- [ADR-0063：知识图谱分组](0063-stage3-knowledge-graph-grouping-candidate.md)
- [ADR-0065：等行数 transfer](0065-stage2-stage3-balanced-transfer-matrix.md)
