# ADR-0067：Stage 3 二十任务目录合同

- 状态：已接受
- 日期：2026-09-22
- 范围：现役 v2 Stage 3、Stage 3 消融、基线与 Stage2→Stage3 transfer 目标

## 背景

现役数据任务目录已移除 `experiment/isobaric_coefficient_of_volume_expansion`，Stage 3
observation 任务因此由 21 个变为 20 个。Stage 1 数据和 Stage 2 九个 physics 任务、源行、
注册表均未改变。本修订只同步消费者的任务集合，不改变余下任务的数据、分组配方、模型、
训练数学或评估口径。

## 决定

1. `configs/v2/stage3/` 下全部现役配置、RDKit-HoME、No-Stage1 以及 full/等行数
   Stage2→Stage3 transfer 配置删除该任务。知识图谱分组保留原六组；
   `thermophysical_interfacial_response` 由九个任务变为八个。
2. Stage2→Stage3 transfer 不再硬编码目标数量；目标必须非空、唯一，并与指定
   Stage 3 合同依据的 enabled 任务集合完全一致。当前 full/等行数矩阵因此均为
   `9×20`，每套共有 `10×20×5=1000` 个 Stage 3 MLP 作业。
3. `configs/v1` 与 `configs/experiments_v1` 属于冻结合同，不修改其 21-任务 fallback、YAML
   或历史身份。绑定旧 v1 512D 准备产物的 Single-任务 MLP同样保持冻结。
   它们不能与本修订后的任务目录/产物混用。
4. 余下 20 个任务的 GLOBAL/GROUP/PRIVATE容量、LR、轮、dropout、原始样本采样、
   PCGrad、优化器、loss、裁剪和固定-末轮状态规则保持不变。

## 产物边界

- Stage 1 产物与现役 Stage 2 准备产物数据/检查点/编码器可复用；本次任务目录变化
  不构成重跑 Stage 2 的理由。
- Stage 3 准备产物身份包含任务目录和源数据集合，必须为二十任务目录重新
  prepare；旧 21-任务 Stage 3 准备产物/训练/评估产物保持历史只读。
- 主模型和 Stage 3 消融必须从新的准备产物重新执行 Stage 3 训练/评估。
  基线必须按二十任务目录重新执行。
- 全量数据 transfer 的既有十个 Stage 2 编码器可复用，但表示 bank、1000个
  Stage 3 MLP 作业和汇总必须按20-任务合同依据重新生成。等行数实验仍服从其独立
  身份与完整性校验，不跨合同强行加载。

## 关联

- [ADR-0048：Stage3三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0062：Stage2→Stage3全量迁移](0062-stage2-stage3-full-transfer-matrix.md)
- [ADR-0063：知识图谱分组](0063-stage3-knowledge-graph-grouping-candidate.md)
- [ADR-0065：等行数 transfer](0065-stage2-stage3-balanced-transfer-matrix.md)
