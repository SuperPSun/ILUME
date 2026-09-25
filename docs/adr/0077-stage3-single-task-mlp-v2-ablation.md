# ADR-0077：现役 v2 Stage 3 单任务 MLP 整体消融

- 状态：Accepted
- 日期：2026-09-26
- 范围：`ilume_stage3_single_task_mlp_v2`

## 背景

[ADR-0033](0033-stage3-single-task-mlp-ablation.md) 的 512D、21-task 实验已冻结，不能与现役 v2 的 1024D、20-task Stage 3 prepared artifact 混用。为了与现役 Stage 3 Base 比较，建立新的消融身份，不迁移或重解释旧 checkpoint。

## 决定

1. 方法 ID 为 `ilume_stage3_single_task_mlp_v2`，显示名为 `ILUME Stage3 Single-task MLP v2`；配置位于 `configs/ablations/ilume_stage3_single_task_mlp_v2.yaml`，输出根为 `outputs/ablations/stage3_single_task_mlp_v2`。实现复用 `ablations/stage3_single_task_mlp`，调度与汇总复用 benchmark 入口。
2. 仅使用 `configs/v2/stage3/base.yaml` 对应的 20-task、1024D prepared artifact。校验 prepared、Stage 2 encoder、registry 和 artifact identity；冻结 Object representation，不加载或更新 Stage 2 encoder。每个 task/fold 有独立模型、优化器、RNG 与 checkpoint。
3. 输入按 primary embedding、声明时的 partner embedding、prepared train-only normalized conditions 顺序直接拼接。MLP 为 `input → 1024 → 512 → 1`，隐藏层均采用 SiLU 和 dropout 0.1。
4. 每个 task 的自然训练集完整遍历定义一个 epoch。固定训练 10 epochs，batch 128，normalized SmoothL1(beta=1)，AdamW(lr 3e-4、weight decay 1e-2)，5% linear warmup、cosine 到 5% base LR、global grad clip 1.0、BF16。validation 每轮记录，不驱动选模；发布第 10 轮末轮模型，不支持 resume。
5. 五折 validation 覆盖全部 20 task；test 仅覆盖实际存在非空 test split 的 task，先逐样本平均五折 raw prediction 再计分。Reporting 使用 `model_selector=final_training_state`、`checkpoint_epoch=null`，100 个独立模型由一个 Stage3-only sweep 汇总。
6. v2 使用独立的 method ID、input contract、checkpoint kind 和 state hash namespace；v1/v2 checkpoint 必须互相拒载。旧 ADR、YAML、checkpoint 和输出路径保持原样。

## 解释边界

该结果比较冻结表示上的简单单任务 MLP recipe 与完整 Stage 3 HoME pipeline。共享、routing、sampling、ObjectEncoder Phase 1 适配及训练预算同时变化，不能将差异归因于单一组件，也不声称参数量或计算量匹配。
