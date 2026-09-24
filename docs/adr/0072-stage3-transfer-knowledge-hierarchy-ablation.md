# ADR-0072：Stage 3 三级 Stage 2 transfer knowledge 消融

- 状态：Experimental
- 日期：2026-09-24
- 范围：`configs/ablations/stage3_transfer_knowledge.yaml`，不替代 v2 Base

## 决定

复用 Base prepared artifact 的 joint Stage 2 1024D embedding 作为每个 ObjectKey 的
`R_joint`。从 full-data Stage2→Stage3 transfer 的零 update baseline 与九个单 source
final ObjectEncoder，按同一 ObjectKey 顺序在冻结、eval 状态生成 FP32 bank；source knowledge
定义为 `ΔR_s = R_s - R_0`。bank 同时绑定 Stage 3 prepared identity、ObjectKey 顺序、
十份 transfer artifact 身份/SHA、张量 shape/hash。它是新消融输入，不能写回 prepared artifact。

对配置了 source 的 GLOBAL、GROUP 和 PRIVATE scope，依次计算
`R_next = R_parent + γ × Σ softmax(logits)_s ΔR_s`。`γ` 和 logits 均初始化为零，
分别归属该 scope 现有 owner；未配置 source 的 scope 原样继承上层表示。solvation
GROUP 的额外 delta 仅作用于 solvation、transfer、transfer_organic，不作用于同组 x_co2。
完整 source 表以消融 YAML 为准。

L1/L2 GLOBAL 使用 GLOBAL 表示；GROUP local 使用 GROUP 表示。配置 PRIVATE source 的 task
从 PRIVATE 表示额外经过同一套 GROUP L1、归一化、FiLM 和 PartnerInteraction，所得 local
只供 PRIVATE expert 使用；交互 partner 同样按各层表示转换。TaskGate 仍使用
`concat(z_global, group_local)`，GROUP expert 与最终 residual 仍使用 group local，
Flat candidate 顺序、专家数量、loss、`weighted_owner_raw_v1`、owner 生命周期和固定末轮规则均不变。

新 bank、source/task 映射、ownership、公式进入独立 training identity；周期 checkpoint、
owner delta、stitched 和 final artifact kind 及模型/owner state hash namespace 与 Base 隔离。
train、resume、evaluate 均校验 bank 与 artifact 的绑定。Base 未启用时不创建新参数，
原配置序列化、训练身份和预测路径保持不变。

## 解释边界

Transfer matrix 的下游实验同步更新 ObjectEncoder 与 MLP，并未直接证明冻结
`ΔR_s` 对 HoME 有利。本消融须以相同五折 protocol 独立检验。`γ=0` 的首个更新步
中 logits 没有梯度；这是既定公式的结果。PRIVATE source 会增加一次局部前向；训练态
dropout 可重新采样，因此仅要求初始 eval 与 Base 逐 bit 对齐。

本 ADR 不授权在实现验证时启动正式 bank 生成、训练或评估；历史/Base 输出不得覆盖。
