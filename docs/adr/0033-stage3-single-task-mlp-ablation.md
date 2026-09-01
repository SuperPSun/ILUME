# ADR-0033：Stage3 frozen Object representation + single-task MLP 整体消融

- 状态：Accepted
- 日期：2026-08-31

## 背景

现役 Stage3 由 sparse-label HoME、跨任务共享、hierarchical PCGrad、composite
sampling 和末期 task-wise refinement 共同组成。为了回答完整 Stage3 体系相对最简单
的 frozen Stage2 representation + single-task predictor 是否带来价值，需要一个覆盖全部
21 个 observation task、但不声称进行单组件归因的整体架构消融。

已有 `benchmarks/mlp` 使用 RDKit 2D descriptors，不能代表本实验的 Stage2 Object
representation 输入。因此本实验使用独立 model/reporting identity，同时复用 benchmark
调度、evaluation 和 summary 合同。

## 决定

1. 方法 ID 固定为 `ilume_stage3_single_task_mlp`，显示名为
   `ILUME Stage3 Single-task MLP`。它是 Stage3-only 内部 ablation，不提供 Stage2
   Core、Partial Charge 或 Full capability。专用实现位于
   `ablations/stage3_single_task_mlp`，配置位于
   `configs/ablations/ilume_stage3_single_task_mlp.yaml`；运行与 reporting 继续复用
   `scripts/benchmarks` 和 `benchmarks/common`。
2. 训练直接读取 `configs/v1/stage3/base.yaml` 指向的现役 Stage3 prepared artifact。
   必须校验完整 prepared identity、Stage2 encoder identity、registry、source 和 artifact
   hashes；不得重新加载、微调或重算 Stage2 encoder。
3. 每个 task/fold 使用独立模型、优化器、RNG 和 checkpoint，不共享任何 Stage3 参数。
   输入严格为 primary embedding，随后按 registry 声明依次追加 partner embedding 和
   prepared train-only normalized conditions。embedding 保持原始 FP32 值；不增加
   LayerNorm、FiLM、PartnerInteraction、expert、gate、residual 或 group 表示。
4. 模型固定为 `input -> 512 -> 256 -> 1`，两个隐藏层均为 SiLU 后接 dropout 0.1。
   Base prepared embedding 必须为 512 维；无 task override、容量匹配或 HPO。
5. 每个 task 的自然训练集完整遍历定义一个 epoch，batch 128，固定训练 100 epochs。
   使用 normalized SmoothL1(beta=1)、AdamW(`3e-4`, weight decay `1e-2`)、5% linear
   warmup、cosine 到 5% base LR、global grad clip 1.0 和 BF16。不存在 composite
   allocation、virtual oversampling、PCGrad、joint phase、refinement 或 early stopping。
6. 每 epoch validation，以 normalized MAE 严格下降选择 best；平局保留较早 epoch。
   训练仍必须完成全部 100 epochs，只发布 validation-best model、完整 history、state
   hash 和 identity manifest。失败由 sweep 在新 attempt 中完整重跑，不支持 resume。
7. Validation 覆盖 21 个 task 的五折统计。Test 仅覆盖 catalog 中实际存在非空 test
   split 的任务，并按现役合同先逐样本平均五折 raw prediction 再计分；不得制造缺失 test。
8. Reporting 使用 `model_selector=validation_best`、`checkpoint_epoch=null`。105 个独立
   task/fold 模型由一个 sweep study 汇总为一个 leaderboard 方法。Stage3-only reporting
   不伪造 Stage2 suite，但必须继续满足 schema v1、comparison identity 和 prediction
   manifest 合同。

## 后果

- 结果只能解释为“固定简单 single-task MLP recipe 与完整 Stage3 HoME pipeline 的比较”。
  因为同时移除了共享、routing、PCGrad 和 sampling，它不是纯 HoME 或纯 PCGrad 消融。
- 不要求与 HoME 参数量或计算量匹配，也不代表 MLP 的任务特定调参上限。
- 现役 Stage1/2/3 配置、HoME checkpoint、prepared artifact 和正式输出均不改变。
- 既有 `outputs/benchmarks/v1/ilume_stage3_single_task_mlp` 保持原位且兼容；迁移不提供
  旧 Python import 或旧配置路径的 alias、symlink 或 fallback。
