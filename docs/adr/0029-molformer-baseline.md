# ADR-0029：MoLFormer 分子语言模型基线

- 状态：已接受
- 日期：2026-08-28

> Early stopping 与验证最优检查点由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代。

## 背景

ILUME 需要一个预训练分子语言模型基线，与 ADR-0022/0028 的既有基线使用相同注册表、split、canonical 身份、仅训练集归一化与评估/报告口径。MoLFormer 没有原生离子液体多组分结构，也没有满足 Partial Charge 严格原子 mapping 的可靠输出合同。

## 决定

1. 新增 `configs/benchmarks/molformer.yaml` 和 `benchmarks/molformer/adapter.py`，继续复用四个公共基准比较入口；不新增特征/embedding 缓存、split、evaluator 或报告结构定义。
2. 固定 `ibm-research/MoLFormer-XL-both-10pct@361063d0ad524ef77cf39b08469f6be770dc550f`、Transformers 5.12.1、`trust_remote_code=True` 与 `deterministic_eval=True`。模型快照由用户显式下载，正式运行仅离线读取并校验；不 clone IBM 仓库，不自动下载或切换修订版本。
3. 使用独立 `ilume-molformer` hash-lock 环境，固定 Python 3.12.12、PyTorch 2.9.0+cu128、CUDA 12.8 与 RDKit 2026.03.5。环境、快照或 CUDA 不匹配时在创建运行输出前硬失败，不自动安装、升级或回退 CPU。
4. ILUME isomeric canonical SMILES 继续作为身份。模型输入由其派生为 RDKit canonical `isomericSmiles=False`；stereo collapse 只审计，不修补。token 长度包含特殊 tokens，最大 202。
5. 超长训练集行在任一组分超限时整行跳过，条件/目标 scaler只拟合 retained 训练集行。验证集/测试集保留全部行并显式截断到202；这些结果保持完整和可参榜，但必须记录原始长度、affected 槽位与来源行。训练与独立验证集评估使用同一截断规则，测试集只在检查点确定后读取。
6. 所有组分共享一个全量微调后的预训练 backbone。纯单组分无条件时直接使用官方池化与`MolformerClassificationHead`；其他任务按注册表槽位有序拼接pooled vectors和训练集-归一化条件，只增加一个`Linear(input_dim, 768)`后进入官方预测头。
7. 训练固定FP32、batch 32、归一化 MSE、AdamW、编码器/预测头 LR `1e-5/1e-4`、权重衰减 `1e-2`、5% linear 预热与余弦 decay、最多100 轮、验证归一化MAE选择、patience 15和种子 42；不使用AMP/TF32、LoRA、HPO、multitask或恢复。
8. Stage 3为21任务×5折，Stage 2 Core为HoV/HOMO/LUMO，共108个训练任务。HOMO/LUMO各自pooled 阳离子/阴离子使用一个scaler/模型/预测头，角色仅用于诊断。Partial Charge与完整为unsupported。

## 后果

- MoLFormer特有代码局限在薄adapter与独立环境；Stage 1/2/3数值合同和既有基线行为不变。
- 超长训练行改变该基线的有效训练覆盖，因此retained/skipped 行与scaler均进入训练身份；评估截断进入评估身份和公开审计。
- 正式权重、sweep和评估仍由用户显式运行，本ADR不授权修改正式数据或已有outputs。
