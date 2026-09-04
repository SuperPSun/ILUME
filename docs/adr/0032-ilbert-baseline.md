# ADR-0032：ILBERT 离子液体语言模型基线

- 状态：Accepted
- 日期：2026-08-31

## 背景

ILUME 需要一个保留 ILBERT 原生 AIS tokenizer、RoBERTa、TextCNN 与 downstream predictor 的语言模型 baseline。上游不是可安装的模型包，且截至本决定没有显式许可证；其 generic pretrained checkpoint 独立发布于 Zenodo。ILBERT 原生处理完整离子液体序列，但没有满足 Partial Charge 严格 atom mapping 的官方输出合同。

## 决定

1. 固定外部上游 `Yu-Xin-Qiu/ILBERT@f9dc6f1b23a40b6988480735f3724a6332f68c12` 和 Zenodo `pretrained_model.pth`。ILUME 只动态加载用户本地 checkout，并校验 commit、`model.py`、`ILtokenizer.py`、`merged_vocab.txt` 与 checkpoint SHA；不复制、提交或再分发无许可证的源码和权重。
2. 使用独立 `ilume-ilbert` hash-lock 环境，固定 Python 3.11.9、PyTorch 2.9.0+cu128、CUDA 12.8、Transformers 4.39.1、tokenizers 0.15.2、atomInSmiles 1.0.2、RDKit 2023.9.5 与 NumPy 1.26.4。缺少环境、CUDA 或资产不匹配时在创建 run output 前硬失败，不自动安装、下载或回退。
3. 固定官方 AIS tokenizer、vocab 2000、512 hidden、6层、4 heads、FFN 1024、dropout 0、TextCNN kernels `1–10,15` 及官方 filters。generic checkpoint只初始化实际使用的RoBERTa encoder；仅允许LM head为unexpected，未使用pooler和随机初始化TextCNN/predictor为missing，并保存完整load audit。
4. 普通IL严格编码一个`cation.anion` sequence。solvation/transfer编码`[ionic_liquid, solute]`，organic transfer编码`[solute, solvent]`；所有view合成一次`V×B` forward并共享唯一full-finetuned RoBERTa+TextCNN。适配只扩展官方predictor第一层输入，保持`Linear(input_dim,256) → Softplus → Linear(256,1)`。
5. HOMO/LUMO直接编码single-ion sequence；cation与anion共同使用一个task pool、scaler、model和scalar head，`ion_role`只用于审计与diagnostics，不构造dummy counterion或role feature。
6. 全部输入固定AIS tokenization、`max_length=100`、官方式truncation和max-length padding；截断前长度包含special tokens。train+valid在run内缓存unique sequence，test只在checkpoint确定后的evaluation读取；所有truncation记录task、split、view和source row。
7. numeric conditions按registry列顺序使用原始物理单位，不拟合condition scaler。target仍由ILUME仅用training rows拟合；normalized MSE训练，raw validation RMSE驱动ReduceLROnPlateau，raw validation MAE驱动checkpoint与patience。
8. 训练固定Adam、LR `1e-4`、weight decay 0、scheduler patience 7/factor 0.3/min LR `3e-5`、batch 16、最多100 epochs、early-stopping patience 15、seed 42、FP32+TF32。OOM、NaN或CUDA错误不触发缩批、精度或设备fallback；不支持resume、HPO或multitask。
9. Stage 3为21 tasks×5 folds，Stage 2 Core为HoV/HOMO/LUMO，共108个训练任务。Partial Charge与Stage 2 Full为unsupported；不增加atom mapper、atom head、split、evaluator或reporting schema。

## 后果

- ILBERT特有逻辑局限在薄adapter、严格配置和独立环境；Stage 1/2/3与既有baseline数值合同不变。
- whole-IL预训练表示应用到single-ion orbital任务构成明确的domain shift；它作为baseline limitation记录，不通过伪化学上下文修补。
- 上游无显式许可证意味着仓库只能保存引用、hash与准备说明。用户必须自行确认其使用场景符合上游和Zenodo条款。
- 正式权重、108-job sweep和evaluation仍由用户显式执行；本决定不修改现有outputs或正式数据。

## 来源

- [固定 GitHub commit](https://github.com/Yu-Xin-Qiu/ILBERT/tree/f9dc6f1b23a40b6988480735f3724a6332f68c12)
- [官方 Zenodo record](https://zenodo.org/records/14601320)
