# ADR-0077：现役 v2 Stage 3 单任务 MLP 整体消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-26
- 范围：`ilume_stage3_single_task_mlp_v2`

## 背景

[ADR-0033](0033-stage3-single-task-mlp-ablation.md) 的 512D、21-任务实验已冻结，不能与现役 v2 的 1024D、20-任务 Stage 3 准备产物混用。为了与现役 Stage 3 Base 比较，建立新的消融身份，不迁移或重解释旧检查点。

## 决定

1. 方法 ID 为 `ilume_stage3_single_task_mlp_v2`，显示名为 `ILUME Stage3 Single-task MLP v2`；配置位于 `configs/ablations/ilume_stage3_single_task_mlp_v2.yaml`，输出根为 `outputs/ablations/stage3_single_task_mlp_v2`。实现复用 `ablations/stage3_single_task_mlp`，调度与汇总复用基准比较入口。
2. 仅使用 `configs/v2/stage3/base.yaml` 对应的 20-任务、1024D 准备产物。校验准备产物、Stage 2 编码器、注册表和产物身份；冻结 Object 表示，不加载或更新 Stage 2 编码器。每个任务/折有独立模型、优化器、RNG 与检查点。
3. 输入按 primary embedding、声明时的 partner embedding、准备产物仅训练集归一化条件顺序直接拼接。MLP 为 `input → 1024 → 512 → 1`，隐藏层均采用 SiLU 和 dropout 0.1。
4. 每个任务的自然训练集完整遍历定义一个轮。固定训练 10 轮，batch 128，归一化 SmoothL1(beta=1)，AdamW(lr 3e-4、权重衰减 1e-2)，5% linear 预热、余弦到 5% base LR、global grad clip 1.0、BF16。验证每轮记录，不驱动选模；发布第 10 轮末轮模型，不支持恢复。
5. 五折验证覆盖全部 20 任务；测试集仅覆盖实际存在非空测试集 split 的任务，先逐样本平均五折原始预测再计分。结果报告使用 `model_selector=final_training_state`、`checkpoint_epoch=null`，100 个独立模型由一个 Stage3-仅 sweep 汇总。
6. v2 使用独立的 method ID、input 合同、检查点 kind 和状态hash namespace；v1/v2 检查点必须互相拒载。旧 ADR、YAML、检查点和输出路径保持原样。

## 解释边界

该结果比较冻结表示上的简单单任务 MLP 配方与完整 Stage 3 HoME pipeline。共享、路由、采样、ObjectEncoder Phase 1 适配及训练预算同时变化，不能将差异归因于单一组件，也不声称参数量或计算量匹配。
