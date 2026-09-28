# ADR-0082：Stage2-HoME 升为正式 ILUME 主线

- 状态：Accepted
- 日期：2026-09-27
- 范围：`configs/v3/stage2/base.yaml`、`configs/v3/stage3/base.yaml`、三项核心消融
- 替代：ADR-0079 的实验身份；ADR-0019/0043/0044 的正式 Stage2 浅层 teacher 训练配方；ADR-0075 的旧 Base/No-Stage2 配对身份

> 完整九任务 `stage2_final.pt` 由 [ADR-0083](0083-stage2-home-full-artifact-evaluation.md) 扩展；独立评估现已由 [ADR-0085](0085-retire-stage2-home-evaluation.md) 退役。

> Stage3 Phase2/3 增加五项 simulation 训练与预测能力的后续决定见 [ADR-0084](0084-stage3-simulation-phase2-phase3.md)。本篇的 20-task Stage3 描述只适用于实验任务子集和旧 20-task 产物。

> 配置目录于 2026-09-28 对齐正式输出版本：Stage2/3 Base 使用 `configs/v3`，Stage1 继续使用 `configs/v2/stage1/base.yaml`。此次仅迁移配置路径，数值配方、artifact kind 与输出路径不变；历史 split 配置仍保留在 `configs/v2/stage3/splits`。

## 决定

正式链固定为 Stage1 通用分子表示预训练 → 九任务 simulation-guided Stage2-HoME → 迁移表示、完整 GLOBAL 与匹配的 thermophysical、solvation GROUP owner → 20 任务 experimental Stage3-HoME 三阶段训练。Stage2 physics-only、无 teacher cache，逻辑 batch 256、微批 256、10 轮末轮发布，每个逻辑 batch 恰好一次 optimizer/scheduler update。Stage2 的 electronic GROUP、simulation PRIVATE 与预测头不迁移。Stage3 维持既有 Phase1 ObjectEncoder 适配、冻结 Stage1 slots、`weighted_owner_raw_v1`、Phase2/3 冻结 ObjectEncoder、固定末轮 stitch 与只读 validation。具体数值以两份正式 YAML 为准。本 ADR 只改变代码身份与仓库结构，不改变已验证 HoME 科学配方。

正式入口为 `scripts/stage1/{prepare,train}.py`、`scripts/stage2/{prepare,train}.py`、`scripts/stage3/{prepare,train,evaluate}.py`。Stage1 仍取 v2 来源；Stage2、Stage3 新产物分别在 `outputs/v3/stage2/base`、`outputs/v3/stage3/base`。Stage2 最终输出 `stage2_final.pt`、同名 manifest、`stage2_encoder.pt`，Stage3 为 `three_phase_final.pt` 及 manifest。正式 checkpoint/final kind 与训练身份独立于历史实验，加载时校验 Stage1 来源、Stage2 artifact SHA、owner 集合、tensor 形状/dtype/state hash 与 GROUP 配方；历史产物不迁移、不覆盖。

## 核心消融

| 对照 | 唯一改变 | 隔离边界 |
|---|---|---|
| w/o Stage1 | 同结构 Stage1 编码器按固定 seed 随机初始化，仍训练完整九任务 Stage2-HoME 和正式 Stage3 配方 | 不读取 Stage1 预训练权重；配置、随机 seed、来源身份与产物独立 |
| w/o Stage2 | 相同 Stage1 checkpoint 和 Stage2 seed 导出零更新 ObjectEncoder；Stage3 Phase1 正常适配，不加载 Stage2 owner | manifest 绑定正式配对来源与零更新状态；独立 prepared、训练及评估身份 |
| w/o Stage3-HoME | 正式 Stage3 prepared 的冻结 1024D 表示输入独立 single-task MLP，`1024→512`、10 epochs、末轮发布 | 不做预算匹配；完整 Stage3 后端对照，不能单独归因于 routing |

No Stage1、No Stage2、No Stage3-HoME 分别对应 `configs/ablations/no_stage1_stage{2,3}.yaml`、`no_stage2_stage3.yaml`、`no_stage3_home.yaml`。不保留其它活跃消融。旧实验和结果仅由历史 ADR、Git history 追溯，不与新身份混用。
