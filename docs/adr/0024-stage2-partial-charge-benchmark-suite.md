# ADR-0024：Stage 2 Partial Charge 基准比较与三榜汇总

- 状态：已接受
- 日期：2026-08-24

> 2026-08-25：Core/完整单元与 Stage 2 子合同版本由 [ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 取代；Partial Charge 评价、同运行约束与基线 unsupported 合同保持不变。

> 现役边界：Stage 2 评估/报告已由 [ADR-0043](0043-retire-stage2-evaluation-and-v2-refinement.md) 退役，下文 Stage 2 榜单与执行步骤仅作历史记录。Stage 3 与通用合同按 [ADR 索引](README.md) 的后续修订读取。

## 背景

Stage 2 报告 v1 只覆盖 3 个任务、5 个标量目标，未评估现役
`simulation/partial_atomic_charge` AtomPropertyHead。Atom 目标又依赖 MOL2 到 canonical
RDKit 原子 order 的确定性 mapping，不能套用标量 evaluator，也不能由模型自行挑选可评估分子。

## 决定

1. 保持报告与汇总结构定义 version 1，并增加具名子合同
   `stage2-benchmark-suite-v1`。缺少该标记的 Stage 2 报告-v1 结果是
   `legacy_stage2_reporting_contract`，只进入 health；Stage 3 结果不因此失效。
2. 发布三个独立比较：`Stage 2 CORE` 是原 5 个标量 unit 的等权 macro
   归一化MAE；`Partial Charge` 是 all-mapped 分子-macro 归一化MAE；
   `Stage 2 FULL` 是前 5 unit 加 Partial Charge 1 unit 的六 unit 等权平均。
3. Partial Charge evaluator 按测试集 `source_row` 顺序验证任务目录、清单、canonical
   SMILES、角色与正式 charge，并复用现役 deterministic MOL2 mapper。Mapper 继续执行
   typed 键 match、connectivity fallback、首个确定性 mapping 和现役隐式氢策略；模型不得
   进一步筛选分子，也不得按预测 error 选择随机排列。
4. Mapping 失败的分子从固定 evaluated set 排除，但预测 CSV 仍为每个测试集
   分子保留一行。Mapping 审计、evaluated 分子 set 和 canonical 目标 arrays 都进入
   Partial 比较身份。缺分子、额外分子、长度错误或非有限预测使
   Partial 与完整为 `incomplete`。
5. Partial 主指标按分子等权：先对每个分子求原子 MAE，再跨分子平均，并除以
   仅训练集 `scalers.json` 中 `weighting=molecule_equal` 的规模。另报原子-micro MAE、RMSE、R²
   及 `all_mapped`、`unique`、`ambiguous`、`typed`、`connectivity_only` 五个可重叠诊断 subset。
   空 subset 保留 null metrics 和 `no_samples`。不报告 charge-conservation 或 charge-sum error。
6. 完整身份只绑定 Core 身份 hash、Partial 身份 hash 和有序六 unit 定义。完整只能由
   同一个 completed 候选内的六个 unit 计算，禁止跨运行拼接；ILUME 三个 section 共同绑定
   检查点 SHA 和轮。
7. Capability/status 只有三种结果：`supported+complete` 可参榜；`unsupported` 不参 Partial/完整
   且不算错误；`supported+incomplete` 不参 Partial/完整并在 health 记录原因。MLP 与
   ECFP+XGBoost 本合同显式声明 Partial/完整为 `unsupported`，仍可参加 Core。
8. `summary/` 原子发布固定 13 文件：Stage 3 测试集/验证、Stage 2 Core/Partial/完整三榜，
   Stage 3 两份 metrics、Core metrics、Partial subset metrics、health、overview、雷达图 SVG 与 JSON。
   删除旧 `stage2_physics_{leaderboard,metrics}.csv` 名称。

## 后果

- 不改变数据 layout、准备产物、检查点、训练、验证或 Stage 1 特征合同。
- ILUME 必须在新的不可覆盖输出路径重新评估。基线 sweep 可复用既有训练
  检查点，但旧 Stage 2 child 评估视为 stale，并在新尝试重新执行 Core 测试集。
- 不需要重跑 Stage 3 评估或任何训练；旧 Stage 2 结果保留为历史 health 证据。
