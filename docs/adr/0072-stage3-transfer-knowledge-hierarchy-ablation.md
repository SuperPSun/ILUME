# ADR-0072：Stage 3 三级 Stage 2 transfer knowledge 消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-24
- 范围：`configs/ablations/stage3_transfer_knowledge.yaml`，不替代 v2 Base

## 决定

复用 Base 准备产物的联合训练 Stage 2 1024D embedding 作为每个 ObjectKey 的
`R_joint`。从全量数据 Stage2→Stage3 transfer 的零更新基线与九个单来源
final ObjectEncoder，按同一 ObjectKey 顺序在冻结、eval 状态生成 FP32 bank；来源 knowledge
定义为 `ΔR_s = R_s - R_0`。bank 同时绑定 Stage 3 准备产物身份、ObjectKey 顺序、
十份 transfer 产物身份/SHA、张量 shape/hash。它是新消融输入，不能写回准备产物。

对配置了来源的 GLOBAL、GROUP 和 PRIVATE scope，依次计算
`R_next = R_parent + γ × Σ softmax(logits)_s ΔR_s`。`γ` 和 logits 均初始化为零，
分别归属该 scope 现有 owner；未配置来源的 scope 原样继承上层表示。solvation
GROUP 的额外 delta 仅作用于 solvation、transfer、transfer_organic，不作用于同组 x_co2。
完整来源表以消融 YAML 为准。

L1/L2 GLOBAL 使用 GLOBAL 表示；GROUP local 使用 GROUP 表示。配置 PRIVATE 来源的任务
从 PRIVATE 表示额外经过同一套 GROUP L1、归一化、FiLM 和 PartnerInteraction，所得 local
只供 PRIVATE 专家使用；交互 partner 同样按各层表示转换。TaskGate 仍使用
`concat(z_global, group_local)`，GROUP 专家与最终残差仍使用组 local，
Flat 候选顺序、专家数量、loss、`weighted_owner_raw_v1`、owner 生命周期和固定末轮规则均不变。

新 bank、来源/任务映射、参数归属、公式进入独立训练身份；周期检查点、
owner delta、拼接后的和 final 产物 kind 及模型/owner 状态hash namespace 与 Base 隔离。
训练集、恢复、evaluate 均校验 bank 与产物的绑定。Base 未启用时不创建新参数，
原配置序列化、训练身份和预测路径保持不变。

## 解释边界

迁移 matrix 的下游实验同步更新 ObjectEncoder 与 MLP，并未直接证明冻结
`ΔR_s` 对 HoME 有利。本消融须以相同五折 protocol 独立检验。`γ=0` 的首个更新步
中 logits 没有梯度；这是既定公式的结果。PRIVATE 来源会增加一次局部前向；训练态
dropout 可重新采样，因此仅要求初始 eval 与 Base 逐 bit 对齐。

本 ADR 不授权在实现验证时启动正式 bank 生成、训练或评估；历史/Base 输出不得覆盖。
