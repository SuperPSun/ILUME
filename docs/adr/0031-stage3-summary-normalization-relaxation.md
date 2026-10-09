# ADR-0031：Stage 3 汇总放松训练归一化一致性

- 状态：已接受
- 日期：2026-08-31

## 背景

MoLFormer 按 ADR-0029 跳过超过 202 tokens 的训练行，并只用 retained 训练行拟合目标 scaler。因此其部分 Stage 3 折与使用完整训练行的模型具有相同验证集/测试集数据，但比较身份中的归一化不同。ADR-0023 的全身份一致门控会阻止这些结果进入同一汇总。

## 决定

1. Stage 3 测试集与验证汇总比较时忽略比较身份 payload 中的 `normalization`，允许不同仅训练集目标 scaler 的模型进入同一归一化榜单。
2. 除 `normalization` 外，`benchmark`、`split`、`expected`、`sources`、`folds` 与 `ensemble` 必须完全一致；验证集/测试集来源或协议不一致仍硬失败。
3. 原始完整比较身份继续保存在各评估汇总与全局比较任务目录中，不修改既有训练、评估产物或报告结构定义。
4. 本放松仅适用于 Stage 3；Stage 2 Core、Partial Charge 与完整继续要求各自现役比较身份一致。

## 后果

- Stage 3 的 `macro_normalized_mae`、逐任务 wins 与排名可以混合不同仅训练集 scaler；这是本决定接受的比较口径。
- 本 ADR 仅取代 ADR-0023 第 7 条在 Stage 3 归一化差异上的限制；损坏输入、不同验证集/测试集来源和其他比较身份差异仍保持硬失败。
- 既有模型无需重训或重新评估，只需重新运行全局 summarizer。
