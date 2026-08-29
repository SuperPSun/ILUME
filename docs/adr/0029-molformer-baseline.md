# ADR-0029：MoLFormer 分子语言模型基线

- 状态：Accepted
- 日期：2026-08-28

## 背景

ILUME 需要一个 pretrained 分子语言模型 baseline，与 ADR-0022/0028 的既有 baseline 使用相同 registry、split、canonical identity、train-only normalization 与 evaluation/reporting 口径。MoLFormer 没有原生离子液体多组分结构，也没有满足 Partial Charge 严格 atom mapping 的可靠输出合同。

## 决定

1. 新增 `configs/benchmarks/molformer.yaml` 和 `benchmarks/molformer/adapter.py`，继续复用四个公共 benchmark 入口；不新增 feature/embedding cache、split、evaluator 或 reporting schema。
2. 固定 `ibm-research/MoLFormer-XL-both-10pct@361063d0ad524ef77cf39b08469f6be770dc550f`、Transformers 5.12.1、`trust_remote_code=True` 与 `deterministic_eval=True`。模型 snapshot 由用户显式下载，正式运行仅离线读取并校验；不 clone IBM 仓库，不自动下载或切换 revision。
3. 使用独立 `ilume-molformer` hash-lock 环境，固定 Python 3.12.12、PyTorch 2.9.0+cu128、CUDA 12.8 与 RDKit 2026.03.5。环境、snapshot 或 CUDA 不匹配时在创建 run output 前硬失败，不自动安装、升级或回退 CPU。
4. ILUME isomeric canonical SMILES 继续作为 identity。模型输入由其派生为 RDKit canonical `isomericSmiles=False`；stereo collapse 只审计，不修补。token 长度包含 special tokens，最大 202。
5. 超长 train row 在任一 component 超限时整行跳过，condition/target scaler只拟合 retained train rows。valid/test 保留全部 row并显式截断到202；这些结果保持完整和可参榜，但必须记录原始长度、affected slot与source row。训练与独立valid evaluation使用同一截断规则，test只在checkpoint确定后读取。
6. 所有component共享一个full-finetuned pretrained backbone。纯单组分无condition时直接使用官方pooling与`MolformerClassificationHead`；其他任务按registry slot有序拼接pooled vectors和train-normalized conditions，只增加一个`Linear(input_dim, 768)`后进入官方head。
7. 训练固定FP32、batch 32、normalized MSE、AdamW、encoder/head LR `1e-5/1e-4`、weight decay `1e-2`、5% linear warmup与cosine decay、最多100 epochs、validation normalized MAE选择、patience 15和seed 42；不使用AMP/TF32、LoRA、HPO、multitask或resume。
8. Stage 3为21任务×5 folds，Stage 2 Core为HoV/HOMO/LUMO，共108个训练任务。HOMO/LUMO各自pooled cation/anion使用一个scaler/model/head，role仅用于诊断。Partial Charge与Full为unsupported。

## 后果

- MoLFormer特有代码局限在薄adapter与独立环境；Stage 1/2/3数值合同和既有baseline行为不变。
- 超长训练row改变该baseline的有效训练覆盖，因此retained/skipped rows与scaler均进入训练identity；evaluation截断进入evaluation identity和公开audit。
- 正式权重、sweep和evaluation仍由用户显式运行，本ADR不授权修改正式数据或已有outputs。
