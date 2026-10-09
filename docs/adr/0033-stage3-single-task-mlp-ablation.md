# ADR-0033：Stage3 冻结Object表示加单任务MLP 整体消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-08-31

> 2026-09-22：本消融冻结为旧 v1 512D、21-任务历史合同。现役 Stage 3 任务集合已由
> [ADR-0067](0067-stage3-twenty-task-catalog.md) 修订为20个；不得仅过滤一个任务后把该
> 产物解释为现役20-任务消融，也不得与v2 1024D 准备产物表示混用。

## 背景

现役 Stage3 由稀疏标签 HoME、跨任务共享、分层 PCGrad、复合
采样和末期任务-wise 精调共同组成。为了回答完整 Stage3 体系相对最简单
的冻结 Stage2 表示 + single-任务预测器是否带来价值，需要一个覆盖全部
21 个 observation 任务、但不声称进行单组件归因的整体架构消融。

已有 `benchmarks/mlp` 使用基础分子统计特征，不能代表本实验的 Stage2 Object 表示
输入。因此本实验使用独立模型/报告身份，同时复用基准比较
调度、评估和汇总合同。

## 决定

1. 方法 ID 固定为 `ilume_stage3_single_task_mlp`，显示名为
   `ILUME Stage3 Single-task MLP`。它是 Stage3-仅内部 ablation，不提供 Stage2
   Core、Partial Charge 或完整 capability。专用实现位于
   `ablations/stage3_single_task_mlp`，配置位于
   `configs/ablations/ilume_stage3_single_task_mlp.yaml`；运行与报告继续复用
   `scripts/benchmarks` 和 `benchmarks/common`。
2. 训练直接读取 `configs/v1/stage3/base.yaml` 指向的现役 Stage3 准备产物。
   必须校验完整准备产物身份、Stage2 编码器身份、注册表、来源和产物
   hashes；不得重新加载、微调或重算 Stage2 编码器。
3. 每个任务/折使用独立模型、优化器、RNG 和检查点，不共享任何 Stage3 参数。
   输入严格为 primary embedding，随后按注册表声明依次追加 partner embedding 和
   准备产物仅训练集归一化条件。embedding 保持原始 FP32 值；不增加
   LayerNorm、FiLM、PartnerInteraction、专家、门控、残差或组表示。
4. 模型固定为 `input -> 512 -> 256 -> 1`，两个隐藏层均为 SiLU 后接 dropout 0.1。
   Base 准备产物 embedding 必须为 512 维；无任务覆盖项、容量匹配或 HPO。
5. 每个任务的自然训练集完整遍历定义一个轮，batch 128，固定训练 100 轮。
   使用归一化 SmoothL1(beta=1)、AdamW(`3e-4`, 权重衰减 `1e-2`)、5% linear
   预热、余弦到 5% base LR、global grad clip 1.0 和 BF16。不存在复合
   allocation、虚拟过采样、PCGrad、联合训练 phase、精调或早停。
6. 每轮验证，以归一化MAE 严格下降选择最优；平局保留较早轮。
   训练仍必须完成全部 100 轮，只发布验证最优模型、完整历史记录、状态
   hash 和身份清单。失败由 sweep 在新尝试中完整重跑，不支持恢复。
7. 验证覆盖 21 个任务的五折统计。测试仅覆盖任务目录中实际存在非空测试集
   split 的任务，并按现役合同先逐样本平均五折原始预测再计分；不得制造缺失测试集。
8. 结果报告使用 `model_selector=validation_best`、`checkpoint_epoch=null`。105 个独立
   任务/折模型由一个 sweep study 汇总为一个榜单方法。Stage3-仅报告
   不伪造 Stage2 suite，但必须继续满足结构定义 v1、比较身份和预测
   清单合同。

## 后果

- 结果只能解释为“固定简单 single-任务 MLP 配方与完整 Stage3 HoME pipeline 的比较”。
  因为同时移除了共享、路由、PCGrad 和采样，它不是纯 HoME 或纯 PCGrad 消融。
- 不要求与 HoME 参数量或计算量匹配，也不代表 MLP 的任务特定调参上限。
- 现役 Stage1/2/3 配置、HoME 检查点、准备产物和正式输出均不改变。
- 既有 `outputs/benchmarks/v1/ilume_stage3_single_task_mlp` 保持原位且兼容；迁移不提供
  旧 Python import 或旧配置路径的 alias、symlink 或 fallback。
