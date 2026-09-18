# ADR-0055：Stage 3 现役 task recipe（pEC50 单变量回调后）

- 状态：Accepted
- 日期：2026-09-14
- 修订：ADR-0054 的 pEC50 Phase 3 PRIVATE lifetime；汇集 ADR-0051～0054 延续的 task 例外

以当时最佳 recipe 为基线，只将 pEC50 Phase 3 从 5 回调到 3，避免混入其他变量。
本页是现役 task recipe 的阅读入口；完整参数以各正式 YAML 为准，主线见
[Base YAML](../../configs/v2/stage3/base.yaml)。历史尝试见 [历史摘要](history.md#adr-0051)。

## 现役例外

下表只列 0051～0055 修订链涉及的最终设置；未列字段沿用 YAML 的显式值或
[ADR-0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md) 的 class 默认。
ratio 顺序为 PRIVATE / Tower / FiLM；epoch 是 nominal lifetime，Phase 2 实际 lifetime 受 GROUP 预算上限约束。

| Task | 最终设置 |
|---|---|
| static permittivity | ratio `0.25/0.25/0.25`；Phase 1/2/3 epoch `4/1/0` |
| volume expansion | ratio `0.25/0.25/0.25`；Phase 2/3 epoch `0/0` |
| thermal conductivity | ratio `0.5/0.5/0.5`；Phase 2/3 epoch `3/2` |
| pEC50 | ratio `0.75/0.75/0.75`；Phase 3 epoch `3` |
| speed of sound | Phase 3 epoch `0` |
| glass transition | ratio `1.25/1.25/1.0`；Phase 3 epoch `2` |
| thermal decomposition | ratio `1.25/1.25/1.0`；PRIVATE dropout `0.10`；Phase 3 epoch `6` |
| electrical conductivity | ratio `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase 3 epoch `4` |
| viscosity | ratio `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase 3 epoch `6` |
| surface tension | ratio `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase 3 epoch `5` |
| refractive index | ratio `0.75/0.75/0.75`；PRIVATE dropout `0.15`；Phase 3 epoch `3` |
| self diffusion / melting point / xCO2 | Phase 3 epoch 分别为 `4 / 5 / 5` |

六套现役 Stage 3 配置使用同一 recipe；GLOBAL/GROUP、LR、optimizer、PCGrad、raw sampling、
loss、ownership clipping 与固定末轮规则不变。task-gate diagnostics 单独遵循
[ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)。

## Identity 与历史结果

training identity contract v5、resolved-plan format v3 不升级，完整 resolved recipe 已进入 identity。
当年的数值修改使身份发生变化，旧 checkpoint 不可恢复到不同 recipe；这不表示文档整理本身
要求重跑。当前 identity 一致的结果继续按原规则 summarize/evaluate/resume，无需改写产物。
Stage 1/2 与 Stage 3 prepared artifact 可复用，baseline 不受此次 recipe 决策影响。
