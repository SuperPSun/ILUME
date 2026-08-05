# ADR-0010：Stage 3 HoME 与 late-solute 多任务训练

- 状态：Superseded by ADR-0011
- 日期：2026-08-05

## 背景

Stage 1/2 已分别提供四模态实体 CLS、IL PairEncoder 和 transfer PairEncoder。Stage 3 需要在不破坏这些表示的前提下同时训练直接 IL、IL–solute、neutral-pair 和 single-entity 任务。若让 solute 在第一层进入共享专家，它会改变所有 IL 任务依赖的公共表示，也无法判断收益来自 IL 表示还是特定 solute 交互。

## 决定

1. Stage 3 v1 绑定 `artifacts/stage_v2/training/comparisons/base_reference/best.pt` 的 SHA-256，并严格校验 Stage 2 checkpoint v2、kind、内嵌配置、Stage 2 data metadata 和 Stage 1 来源哈希。Stage 2 backbone 与两个 PairEncoder 只在准备阶段运行，训练时只读取冻结 FP32 缓存。
2. Phase 1 使用 19 个直接 IL 任务及 solvation/transfer，共 21 个任务；Phase 2 从同 fold 的 Phase 1 `best.pt` 扩展到 27 个任务。扩展加载器只容许显式登记的 neutral-pair/single-entity adapters、quantum group 和新增任务 private/gate/tower 参数缺失。
3. HoME 使用两层层级专家。每层有 2 个 global expert 和每组 2 个 group expert，第二层另有每任务 1 个 private expert。任务 gate 的候选仅包含 global、所属 meta group 和 private 输出，这一候选集合就是 hierarchy mask。每个 expert 固定为 `Linear(d,2d) → SiLU → Linear(2d,d) → BatchNorm1d → SiLU`。
4. solvation/transfer 的第一层只接收冻结 IL embedding 与条件。solute CLS 仅在第二层通过共享 `SoluteInteraction` 进入 solvation group、任务 private expert 和 gate；第二层 global expert 仍只接收 `z_shared`。直接 IL 任务不构造 null-solute。
5. 每个任务独立使用自己的五折定义。Phase 1 固定使用 `IL` fold；仅提供重复交叉验证子目录的任务固定选取 `cv1`。solvation/transfer 的验证边界是 IL pair，但训练采样体系为 `(cation, anion, solute)`。同任务 train/valid IL pair 重叠是错误；跨任务看到同一 IL pair 只写入曝光审计，因为 task-local validation 不等价于全局冷启动验证。
6. 一个 optimizer block 对每个活动任务执行一个 micro-batch，按任务数等权缩放 loss，完成 21/27 个 micro-batch 后才更新一次。Phase 2 前 10% 完整 block 只训练新增模块，之后训练全部 Stage 3 参数；冻结缓存永不进入 optimizer。

## 兼容性与解释限制

- artifact 位于 `artifacts/stage3_v1/data/phase{1,2}`，checkpoint format 为 v1；任何哈希、phase、fold 或配置不一致都拒绝恢复。
- `--resume-from` 仅用于同 phase/fold/config 精确恢复，`--init-from` 仅用于同 fold Phase 1 → Phase 2 扩展，两者互斥。
- fold scaler 只拟合另外四个训练分片；目标列一律 identity，名称中已有 `_log10` 的目标不会再次变换。
- 五折结果说明 task-local 泛化；`cross_task_exposure.csv` 中的曝光必须随结果保留，不能把它解释成所有任务联合意义上的 unseen-IL 泛化。

## 后果

Stage 3 准备会生成 entity CLS、IL pair、neutral pair 三类冻结缓存，以及条件/目标 scaler、实体/任务行排除审计和跨任务曝光审计。正式准备、五折训练、基线、扩展与固定 test ensemble 均由用户运行；代码验收只使用临时小数据。
