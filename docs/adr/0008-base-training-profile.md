# ADR-0008：Base 正式训练 profile 收敛到 effective batch 256

- 状态：Superseded by ADR-0013
- 日期：2026-08-03

> global batch 256 与 learning rate `1e-4` 被 ADR-0013 延续；覆盖采样、多容量和 checkpoint v3 已被取代。

## 决定

Stage 1 的 Base 正式配置及其 Base 消融统一使用micro-batch 256、梯度累积1和learning rate `1e-4`，训练5个覆盖型epoch。Large 使用256 × 1，XLarge 使用128 × 2，因此三个容量档位的effective batch均为256，并在相同训练池上产生相同的epoch抽样预算。

正式 Base checkpoint 写入 `outputs/formal_v1/stage1/base/train`；checkpoint 内嵌配置是恢复语义的权威来源。旧 `artifacts/training/pretrain_base_bs512` 仅是历史路径，不对已生成 checkpoint 就地改名。

本决定取代ADR-0006中“Base使用512 micro-batch”的profile描述，不改变覆盖型epoch、45/45/10采样或checkpoint v3语义。

## 理由

2026-08-02起，正式YAML、Base消融和实际epoch 5 checkpoint均记录256 × 1与`1e-4`，而旧README、ADR和测试仍保留512 × 1与`4e-4`，导致运行入口、验证门禁和恢复认知分叉。以当前配置和checkpoint内嵌事实收敛文档及测试，可避免未来训练被旧断言误导，同时保持已有Stage 2 checkpoint路径稳定。

## 后果

在当前 `(24908, 27907, 56532)` 训练实体上，Base、Large和XLarge每个覆盖型epoch都执行2,209个optimizer step、抽样565,504次。Base若发生显存压力，可改为128 × 2；只要effective batch保持256，epoch预算和学习率调度步数不变。历史目录名仍可能造成误读，因此配置索引必须明确这一兼容约束。
