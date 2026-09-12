# ADR-0051：Stage 3 弱任务 PRIVATE 定向正则化

- 状态：Accepted
- 日期：2026-09-12
- 修订：ADR-0050 的 task-specific PRIVATE capacity、dropout 与 epoch 数值

## 背景

最新五折 validation 与 test benchmark 表明，ADR-0050 的 medium/large Phase 1 LR 调整整体有效，
无需改变三阶段结构或扩大 GLOBAL/GROUP。剩余弱项主要分为 tiny validation 过拟合，以及五折强但
test 泛化较弱的 task。本轮只定向控制 PRIVATE specialization。

## 决定

1. ADR-0050 的 GLOBAL/GROUP、size-class 默认、medium/large Phase 1 LR 与 task-specific LR
   例外保持不变。六套现役 Stage 3 配置继续共享同一 recipe。
2. static permittivity 使用 `0.25/0.25/0.25` width，Phase 1/2/3 PRIVATE epoch 为
   `4/1/0`；volume expansion 同宽度，epoch 为 `6/0/0`。Phase 2 零预算 owner 从 branch
   开始即冻结且不进入 optimizer，但 task 继续参与 raw sampling 并向 GROUP 提供梯度。
3. thermal conductivity 撤回扩容，使用 `0.5/0.5/0.5` width 与 `6/3/4` epoch；pEC50
   使用 `0.75/0.75/0.75`；speed of sound Phase 3 为 0；glass transition 和 thermal
   decomposition 保留 `1.25/1.25/1.0`，Phase 3 分别为 2 和 7。
4. electrical conductivity、viscosity、surface tension 保持 `0.75/0.75/0.75`，Phase 3
   分别为 4、6、5；refractive index 改为 `1/1/1` 且 Phase 3 为 3。它们与 thermal
   decomposition 的 PRIVATE-owned Expert、Tower 和 FiLM 使用 task-specific dropout `0.15`；
   其他 task 及 GLOBAL/GROUP 继续使用 `0.10`。
5. `model_overrides.private_dropout` 限定在 `[0, 0.15]`，缺省回退 `model.dropout`。
   `phase2_private_epochs` 允许为 0。最终 resolved plan 展开 dropout、capacity、LR、
   nominal/effective epoch、freeze epoch 与 update budget。
6. training identity 和 resolved-plan 格式升级；旧 three-phase checkpoint 不得交叉 resume。
   prepared identity以及 legacy v1/Capacity 合同不变。

## 后果

- 不改变 raw sampling、loss weighting、PCGrad、AdamW、weight decay、ownership clipping、
  GLOBAL/GROUP capacity、专家数量、固定最后 epoch 或 validation reporting-only 语义。
- Phase 2 零预算 PRIVATE 仍进入 owner delta 与 stitch 完整性校验，其 state 必须逐 bit 继承
  Phase-1 anchor。
- 六套现役配置必须在新目录重新执行 Stage 3 五折 train、validation 和 test；Stage 1/2 及
  Stage 3 prepared artifact 可复用，历史输出保持只读。

## 关联

- [ADR-0046：ownership clipping 与 raw sampling](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0048：owner-lifetime 三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：task-specific owner budget 与 PRIVATE capacity](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
