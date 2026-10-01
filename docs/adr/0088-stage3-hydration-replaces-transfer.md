# ADR-0088：Stage3 hydration 替换 transfer

- 状态：Accepted
- 日期：2026-10-01
- 修订：[ADR-0067](0067-stage3-twenty-task-catalog.md) 的现役任务集合；冻结 v1/Capacity 不变。

## 决定

现役 Stage3 的 `experiment/hydration` 替代 `experiment/transfer`，实验任务数仍为 20。
正式 Base、十组 base1 候选、核心消融以及 baseline authority 配置同步替换。
`transfer_organic` 与 Stage2 九任务、Stage3 五项模拟任务不变。

hydration 属于 solvation GROUP，151 个 unique systems，采用 small 默认 PRIVATE 配方，
不继承 transfer 的 large class 或 Phase3 10轮 override。模型输入是单个 solute，
`partner_mode: none`、无 partner slots；catalog 提供 temperature_K 和 hydration_kcal/mol。
所有活跃 split authority 显式选择 hydration 的 random split，沿用默认 cv_repeat=1，
五折交叉验证；不创建新划分，不生成不存在的 test 样本。

## 产物与验证

任务 roster、源文件与 catalog SHA 进入既有 prepared/training/evaluation identity。
旧 transfer 产物不转名、不跨合同加载。使用新输出重新 prepare、训练与评估 Stage3 和 baseline；
既有输出保持历史只读。catalog 改动仍须通过 Stage2 的既有数据完整性检查，不能手改 hash。

用临时五折单 solute 数据验证 train-only normalization、输入槽位、small 配方、
缺失 test 与旧 roster 拒载；同步检查 baseline 解析、报告任务顺序及冻结合同。
