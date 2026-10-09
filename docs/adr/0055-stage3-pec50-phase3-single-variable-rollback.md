# ADR-0055：Stage 3 现役任务配方（pEC50 单变量回调后）

- 状态：已接受
- 日期：2026-09-14
- 修订：ADR-0054 的 pEC50 Phase 3 PRIVATE 存续期；汇集 ADR-0051～0054 延续的任务例外

以当时最佳配方为基线，只将 pEC50 Phase 3 从 5 回调到 3，避免混入其他变量。
本页是现役任务配方的阅读入口；完整参数以各正式 YAML 为准，主线见
[Base YAML](../../configs/v3/stage3/base.yaml)。历史尝试见 [历史摘要](history.md#adr-0051)。

## 现役例外

下表只列 0051～0055 修订链涉及的最终设置；未列字段沿用 YAML 的显式值或
[ADR-0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md) 的 class 默认。
比例顺序为 PRIVATE / Tower / FiLM；轮是 nominal 存续期，Phase 2 实际存续期受 GROUP 预算上限约束。

| Task | 最终设置 |
|---|---|
| 静态介电常数 | 比例 `0.25/0.25/0.25`；Phase1/2/3轮数 `4/1/0` |
| 体积膨胀 | 比例 `0.25/0.25/0.25`；Phase2/3轮数 `0/0` |
| 热导率 | 比例 `0.5/0.5/0.5`；Phase2/3轮数 `3/2` |
| pEC50 | 比例 `0.75/0.75/0.75`；Phase3轮数 `3` |
| 声速 | Phase3轮数 `0` |
| 玻璃化转变 | 比例 `1.25/1.25/1.0`；Phase3轮数 `2` |
| 热分解 | 比例 `1.25/1.25/1.0`；PRIVATE dropout `0.10`；Phase3轮数 `6` |
| 电导率 | 比例 `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase3轮数 `4` |
| 黏度 | 比例 `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase3轮数 `6` |
| 表面张力 | 比例 `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase3轮数 `5` |
| 折射率 | 比例 `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase3轮数 `3` |
| 自扩散 / melting point / xCO2 | Phase 3 轮分别为 `4 / 5 / 5` |

六套现役 Stage 3 配置使用同一配方；GLOBAL/GROUP、LR、优化器、PCGrad、原始样本采样、
loss、按参数归属裁剪与固定末轮规则不变。任务门控诊断单独遵循
[ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)。

## 身份与历史结果

训练身份合同 v5、resolved-plan 格式 v3 不升级，完整 resolved 配方已进入身份。
当年的数值修改使身份发生变化，旧检查点不可恢复到不同配方；这不表示文档整理本身
要求重跑。当前身份一致的结果继续按原规则 summarize/evaluate/恢复，无需改写产物。
Stage 1/2 与 Stage 3 准备产物可复用，基线不受此次配方决策影响。
