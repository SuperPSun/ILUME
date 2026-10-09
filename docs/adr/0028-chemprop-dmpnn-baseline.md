# ADR-0028：Chemprop D-MPNN 强图基线

- 状态：已接受
- 日期：2026-08-26
- 后续修订：ADR-0042 将多组分消息传递改为跨组分槽共享权重；其余决定保持有效。

> 训练预算、早停与验证最优检查点已由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代；现役 D-MPNN 使用固定 10轮末轮状态合同。

## 背景

ILUME 需要一个强图神经网络基线，与 ADR-0022 的 MLP 和 ECFP4-XGBoost 使用相同注册表、split、条件、目标与评估/报告口径。该基线不得进入或改变 Stage 1/2/3 的训练合同，也不得复制 Chemprop 的模型实现。

## 决定

1. 新增 `configs/benchmarks/dmpnn.yaml`，继续复用 `scripts/benchmarks/{train,evaluate,sweep}.py`。模型 adapter 独立位于 `benchmarks/dmpnn/`；不新增正式入口、特征缓存、专属 split、mapper、evaluator 或报告结构定义。
2. D-MPNN 使用独立 `ilume-dmpnn` 环境。仓库提交最小 Conda 定义与 Linux x86_64/CUDA 12.8 hash lock，固定 Python 3.12.12、Chemprop 2.3.1、PyTorch 2.9.0+cu128 和 RDKit 2026.03.5。原拟采用的 RDKit 2025.09.5 与 Chemprop 的 `cuik-molmaker-pin` 在 Python 3.12 上不可安装，因此采用支持 Python 3.12 的最早兼容 pin；不得绕过该官方依赖。
3. D-MPNN 入口在创建运行输出前通过 `conda run --no-capture-output -n ilume-dmpnn` 重启自身并严格核对完整 lock、直接依赖、CUDA 和 GPU。缺失或不一致均硬失败；不自动创建、安装、升级或回退 CPU。每个运行记录不含用户名、hostname 和私有绝对路径的环境快照与 lock SHA。
4. 标量任务使用 Chemprop 2.3.1 的 `BondMessagePassing`；多组分任务按注册表槽位顺序建立独立 block，以 `MulticomponentMessagePassing(shared=False)` 和 `MulticomponentMPNN` 组合。仅训练集归一化 numeric 条件仅作为 `x_d` 在聚合后进入预测器。HOMO/LUMO 各自使用 pooled 阳离子/阴离子行、一个训练集 scaler、一个单组分标量模型，`ion_role` 只用于诊断。
5. Partial Charge 直接读取现役 Stage 2 prepare 产物的 retained 行、canonical 原子-order targets 与分子-equal scaler，使用 `MABBondMessagePassing`、`MolAtomBondMPNN` 和原子 `RegressionFFN`。训练身份绑定合同依据配置、准备产物元数据、scaler、相关张量、原始训练集/验证集 split、mapping 合同、模型合同、graph 合同与环境 lock。
6. 所有任务只拟合仅训练集归一化目标，以归一化验证 MAE 选择检查点；单标量的正定规模保证它与原始 MAE 的最优轮相同。训练固定种子 42、FP32、Adam/NoamLR、batch 64、最多 50 轮、patience 10，无预训练、HPO、multitask 或恢复。正式产物只保留 Chemprop 官方保存的 `model.pt`、ILUME `checkpoint.json` 和训练历史。
7. 标量继续使用公共原始-unit evaluator，Stage 3 测试集继续五折逐样本预测平均。Partial 测试集由公共 mapper 构造 evaluated set并由公共 scorer/writer 评分；缺失、额外、长度错误或非有限预测保持 `supported+incomplete`，不增加 fallback。
8. D-MPNN 的 Stage 2 Core 是 HoV、HOMO、LUMO 三任务等权；Partial Charge 单独报告；仅同一 sweep 中 Core 与 Partial 均完整时生成四单元等权完整。MLP/XGBoost 的 Partial/完整 capability 不变。正式规模为 21×5 + 4 = 109 个单种子训练任务。

## 后果

- 高级基线可以一模型一环境，主 `pyproject.toml` 与主运行环境保持不变。
- 环境解析或 CUDA 不一致会在任何运行输出创建前暴露，不会产生看似正式的失败目录。
- `configs/v1`、Stage 1/2/3 实现、正式数据、现役 prepare 产物与已有 outputs 均不变；正式 sweep 仍由用户显式执行。
