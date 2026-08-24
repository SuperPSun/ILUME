# ADR-0024：Stage 2 Partial Charge benchmark 与三榜汇总

- 状态：Accepted
- 日期：2026-08-24

## 背景

Stage 2 reporting v1 只覆盖 3 个 task、5 个 scalar target，未评估现役
`simulation/partial_atomic_charge` AtomPropertyHead。Atom target 又依赖 MOL2 到 canonical
RDKit atom order 的确定性 mapping，不能套用 scalar evaluator，也不能由模型自行挑选可评估分子。

## 决定

1. 保持 reporting 与 summary schema version 1，并增加具名子合同
   `stage2-benchmark-suite-v1`。缺少该标记的 Stage 2 reporting-v1 结果是
   `legacy_stage2_reporting_contract`，只进入 health；Stage 3 结果不因此失效。
2. 发布三个独立 comparison：`Stage 2 CORE` 是原 5 个 scalar unit 的等权 macro
   normalized MAE；`Partial Charge` 是 all-mapped molecule-macro normalized MAE；
   `Stage 2 FULL` 是前 5 unit 加 Partial Charge 1 unit 的六 unit 等权平均。
3. Partial Charge evaluator 按 test `source_row` 顺序验证 catalog、manifest、canonical
   SMILES、role 与 formal charge，并复用现役 deterministic MOL2 mapper。Mapper 继续执行
   typed bond match、connectivity fallback、首个确定性 mapping 和现役隐式氢策略；模型不得
   进一步筛选 molecule，也不得按 prediction error 选择 permutation。
4. Mapping 失败的 molecule 从固定 evaluated set 排除，但 prediction CSV 仍为每个 test
   molecule 保留一行。Mapping audit、evaluated molecule set 和 canonical target arrays 都进入
   Partial comparison identity。缺 molecule、额外 molecule、长度错误或非有限 prediction 使
   Partial 与 Full 为 `incomplete`。
5. Partial 主指标按 molecule 等权：先对每个 molecule 求 atom MAE，再跨 molecule 平均，并除以
   train-only `scalers.json` 中 `weighting=molecule_equal` 的 scale。另报 atom-micro MAE、RMSE、R²
   及 `all_mapped`、`unique`、`ambiguous`、`typed`、`connectivity_only` 五个可重叠诊断 subset。
   空 subset 保留 null metrics 和 `no_samples`。不报告 charge-conservation 或 charge-sum error。
6. Full identity 只绑定 Core identity hash、Partial identity hash 和有序六 unit 定义。Full 只能由
   同一个 completed candidate 内的六个 unit 计算，禁止跨 run 拼接；ILUME 三个 section 共同绑定
   checkpoint SHA 和 epoch。
7. Capability/status 只有三种结果：`supported+complete` 可参榜；`unsupported` 不参 Partial/Full
   且不算错误；`supported+incomplete` 不参 Partial/Full并在 health 记录原因。MLP 与
   ECFP+XGBoost 本合同显式声明 Partial/Full 为 `unsupported`，仍可参加 Core。
8. `summary/` 原子发布固定 12 文件：Stage 3 test/validation、Stage 2 Core/Partial/Full 三榜，
   Stage 3 两份 metrics、Core metrics、Partial subset metrics、health、overview 与 JSON。
   删除旧 `stage2_physics_{leaderboard,metrics}.csv` 名称。

## 后果

- 不改变 data layout、prepared artifact、checkpoint、训练、validation 或 Stage 1 feature 合同。
- ILUME 必须在新的不可覆盖 output 路径重新 evaluation。Baseline sweep 可复用既有训练
  checkpoint，但旧 Stage 2 child evaluation 视为 stale，并在新 attempt 重新执行 Core test。
- 不需要重跑 Stage 3 evaluation 或任何训练；旧 Stage 2 结果保留为历史 health 证据。
