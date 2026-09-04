# ADR-0028：Chemprop D-MPNN 强图基线

- 状态：Accepted
- 日期：2026-08-26
- 后续修订：ADR-0042 将多组分 message-passing 改为跨组分槽共享权重；其余决定保持有效。

## 背景

ILUME 需要一个强图神经网络 baseline，与 ADR-0022 的 MLP 和 ECFP4-XGBoost 使用相同 registry、split、条件、target 与 evaluation/reporting 口径。该 baseline 不得进入或改变 Stage 1/2/3 的训练合同，也不得复制 Chemprop 的模型实现。

## 决定

1. 新增 `configs/benchmarks/dmpnn.yaml`，继续复用 `scripts/benchmarks/{train,evaluate,sweep}.py`。模型 adapter 独立位于 `benchmarks/dmpnn/`；不新增正式入口、feature cache、专属 split、mapper、evaluator 或 reporting schema。
2. D-MPNN 使用独立 `ilume-dmpnn` 环境。仓库提交最小 Conda 定义与 Linux x86_64/CUDA 12.8 hash lock，固定 Python 3.12.12、Chemprop 2.3.1、PyTorch 2.9.0+cu128 和 RDKit 2026.03.5。原拟采用的 RDKit 2025.09.5 与 Chemprop 的 `cuik-molmaker-pin` 在 Python 3.12 上不可安装，因此采用支持 Python 3.12 的最早兼容 pin；不得绕过该官方依赖。
3. D-MPNN 入口在创建 run output 前通过 `conda run --no-capture-output -n ilume-dmpnn` 重启自身并严格核对完整 lock、直接依赖、CUDA 和 GPU。缺失或不一致均硬失败；不自动创建、安装、升级或回退 CPU。每个 run 记录不含用户名、hostname 和私有绝对路径的环境快照与 lock SHA。
4. 标量任务使用 Chemprop 2.3.1 的 `BondMessagePassing`；多组分任务按 registry slot 顺序建立独立 block，以 `MulticomponentMessagePassing(shared=False)` 和 `MulticomponentMPNN` 组合。train-only normalized numeric conditions 仅作为 `x_d` 在聚合后进入 predictor。HOMO/LUMO 各自使用 pooled cation/anion rows、一个 train scaler、一个单组分 scalar model，`ion_role` 只用于诊断。
5. Partial Charge 直接读取现役 Stage 2 prepare artifact 的 retained rows、canonical atom-order targets 与 molecule-equal scaler，使用 `MABBondMessagePassing`、`MolAtomBondMPNN` 和 atom `RegressionFFN`。训练身份绑定 authority config、prepared metadata、scaler、相关 tensor、原始 train/valid split、mapping contract、模型合同、graph contract 与环境 lock。
6. 所有任务只拟合 train-only normalized target，以 normalized validation MAE 选择 checkpoint；单 scalar 的正定 scale 保证它与 raw MAE 的最优 epoch 相同。训练固定 seed 42、FP32、Adam/NoamLR、batch 64、最多 50 epochs、patience 10，无 pretrained、HPO、multitask 或 resume。正式产物只保留 Chemprop 官方保存的 `model.pt`、ILUME `checkpoint.json` 和训练历史。
7. 标量继续使用公共 raw-unit evaluator，Stage 3 test 继续五折逐样本预测平均。Partial test 由公共 mapper 构造 evaluated set并由公共 scorer/writer 评分；缺失、额外、长度错误或非有限预测保持 `supported+incomplete`，不增加 fallback。
8. D-MPNN 的 Stage 2 Core 是 HoV、HOMO、LUMO 三任务等权；Partial Charge 单独报告；仅同一 sweep 中 Core 与 Partial 均完整时生成四单元等权 Full。MLP/XGBoost 的 Partial/Full capability 不变。正式规模为 21×5 + 4 = 109 个单 seed 训练任务。

## 后果

- 高级 baseline 可以一模型一环境，主 `pyproject.toml` 与主运行环境保持不变。
- 环境解析或 CUDA 不一致会在任何 run output 创建前暴露，不会产生看似正式的失败目录。
- `configs/v1`、Stage 1/2/3 实现、正式数据、现役 prepare artifact 与已有 outputs 均不变；正式 sweep 仍由用户显式执行。
