# ADR-0031：Stage 3 汇总放松训练归一化一致性

- 状态：Accepted
- 日期：2026-08-31

## 背景

MoLFormer 按 ADR-0029 跳过超过 202 tokens 的训练行，并只用 retained training rows 拟合 target scaler。因此其部分 Stage 3 fold 与使用完整训练行的模型具有相同 valid/test 数据，但 comparison identity 中的 normalization 不同。ADR-0023 的全 identity 一致 gate 会阻止这些结果进入同一汇总。

## 决定

1. Stage 3 test 与 validation 汇总比较时忽略 comparison identity payload 中的 `normalization`，允许不同 train-only target scaler 的模型进入同一 normalized leaderboard。
2. 除 `normalization` 外，`benchmark`、`split`、`expected`、`sources`、`folds` 与 `ensemble` 必须完全一致；valid/test source 或协议不一致仍硬失败。
3. 原始完整 comparison identity 继续保存在各 evaluation summary 与全局 comparison catalog 中，不修改既有训练、evaluation artifact 或 reporting schema。
4. 本放松仅适用于 Stage 3；Stage 2 Core、Partial Charge 与 Full 继续要求各自现役 comparison identity 一致。

## 后果

- Stage 3 的 `macro_normalized_mae`、per-task wins 与排名可以混合不同 train-only scaler；这是本决定接受的比较口径。
- 本 ADR 仅取代 ADR-0023 第 7 条在 Stage 3 normalization 差异上的限制；损坏输入、不同 valid/test source 和其他 comparison identity 差异仍保持硬失败。
- 既有模型无需重训或重新 evaluation，只需重新运行全局 summarizer。
