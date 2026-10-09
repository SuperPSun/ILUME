# ADR-0032：ILBERT 离子液体语言模型基线

- 状态：已接受
- 日期：2026-08-31

> 验证-driven 调度器、早停、验证最优检查点，以及现役固定训练预算与 learning rate 由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代。

## 背景

ILUME 需要一个保留 ILBERT 原生 AIS 分词器、RoBERTa、TextCNN 与下游预测器的语言模型基线。上游不是可安装的模型包，且截至本决定没有显式许可证；其通用预训练检查点独立发布于 Zenodo。ILBERT 原生处理完整离子液体序列，但没有满足 Partial Charge 严格原子 mapping 的官方输出合同。

## 决定

1. 固定外部上游 `Yu-Xin-Qiu/ILBERT@f9dc6f1b23a40b6988480735f3724a6332f68c12` 和 Zenodo `pretrained_model.pth`。ILUME 只动态加载用户本地 checkout，并校验 commit、`model.py`、`ILtokenizer.py`、`merged_vocab.txt` 与检查点 SHA；不复制、提交或再分发无许可证的源码和权重。
2. 使用独立 `ilume-ilbert` hash-lock 环境，固定 Python 3.11.9、PyTorch 2.9.0+cu128、CUDA 12.8、Transformers 4.39.1、tokenizers 0.15.2、atomInSmiles 1.0.2、RDKit 2023.9.5 与 NumPy 1.26.4。缺少环境、CUDA 或资产不匹配时在创建运行输出前硬失败，不自动安装、下载或回退。
3. 固定官方 AIS 分词器、vocab 2000、512 隐藏宽度、6层、4 注意力头、FFN 1024、dropout 0、TextCNN kernels `1–10,15` 及官方 filters。通用检查点只初始化实际使用的RoBERTa 编码器；仅允许LM 预测头为unexpected，未使用pooler和随机初始化TextCNN/预测器为missing，并保存完整加载审计。
4. 普通IL严格编码一个`cation.anion` sequence。solvation/transfer编码`[ionic_liquid, solute]`，organic transfer编码`[solute, solvent]`；所有视图合成一次`V×B` 前向并共享唯一全量微调后的 RoBERTa+TextCNN。适配只扩展官方预测器第一层输入，保持`Linear(input_dim,256) → Softplus → Linear(256,1)`。
5. HOMO/LUMO直接编码single-ion sequence；阳离子与阴离子共同使用一个任务池、scaler、模型和标量预测头，`ion_role`只用于审计与诊断，不构造dummy counterion或角色特征。
6. 全部输入固定AIS tokenization、`max_length=100`、官方式truncation和max-length padding；截断前长度包含特殊 tokens。训练集+验证集在运行内缓存unique sequence，测试集只在检查点确定后的评估读取；所有truncation记录任务、split、视图和来源行。
7. numeric 条件按注册表列顺序使用原始物理单位，不拟合条件 scaler。目标仍由ILUME仅用训练行拟合；归一化 MSE训练，原始验证 RMSE驱动ReduceLROnPlateau，原始验证 MAE驱动检查点与patience。
8. 训练固定Adam、LR `1e-4`、权重衰减 0、调度器 patience 7/factor 0.3/min LR `3e-5`、batch 16、最多100 轮、early-stopping patience 15、种子 42、FP32+TF32。OOM、NaN或CUDA错误不触发缩批、精度或设备fallback；不支持恢复、HPO或multitask。
9. Stage 3为21 任务×5折，Stage 2 Core为HoV/HOMO/LUMO，共108个训练任务。Partial Charge与Stage 2 完整为unsupported；不增加原子 mapper、原子预测头、split、evaluator或报告结构定义。

## 后果

- ILBERT特有逻辑局限在薄adapter、严格配置和独立环境；Stage 1/2/3与既有基线数值合同不变。
- whole-IL预训练表示应用到single-ion orbital任务构成明确的域 shift；它作为基线 limitation记录，不通过伪化学上下文修补。
- 上游无显式许可证意味着仓库只能保存引用、hash与准备说明。用户必须自行确认其使用场景符合上游和Zenodo条款。
- 正式权重、108-作业 sweep和评估仍由用户显式执行；本决定不修改现有outputs或正式数据。

## 来源

- [固定 GitHub commit](https://github.com/Yu-Xin-Qiu/ILBERT/tree/f9dc6f1b23a40b6988480735f3724a6332f68c12)
- [官方 Zenodo record](https://zenodo.org/records/14601320)
