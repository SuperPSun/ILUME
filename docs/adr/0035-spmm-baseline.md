# ADR-0035：SPMM 仅SMILES 语言模型基线

- 状态：已接受
- 日期：2026-09-01

> 训练预算、早停与验证最优检查点已由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代；现役 SPMM 使用固定 10轮末轮状态合同。

## 背景

ILUME需要加入官方SPMM 预训练模型作为advanced 基线。SPMM预训练同时包含SMILES text与53维property vector，但官方MoleculeNet regression只使用SMILES text 分支的`[CLS]`表示。其原生下游没有ILUME多组分拓扑，也没有满足Partial Charge 原子 mapping合同的输出路径。

## 决定

1. 固定外部`jinhojsk515/SPMM@046976484f31b3cbc862b8f2094e38df72fcfce7`及官方`checkpoint_SPMM.ckpt`；运行前校验checkout、相关源码、vocab、BERT 配置、检查点 SHA与2358591924字节大小。上游为Apache-2.0，但ILUME仍不复制或提交源码和权重。
2. 使用独立`ilume-spmm` hash-lock环境，固定Python 3.10.14、PyTorch 2.9.0+cu128、CUDA 12.8、Transformers 4.30.1、tokenizers 0.13.3、RDKit 2023.3.1和NumPy 1.24.3。该环境不安装要求Python ≥3.11的ILUME editable package，统一launcher通过现役脚本的仓库路径引导运行；缺少环境、CUDA或资产不匹配时在创建运行输出前硬失败。
3. 基线只实例化官方`xbert.py`的SMILES text 编码器。加载主`text_encoder.bert` embeddings与层 0–5的102个状态 entries；层 6–11、LM 预测头、PV、momentum 编码器和queues均不进入下游模型。官方Lightning 检查点包含Python pickle对象，只有固定SHA与大小通过后才允许反序列化，并保存加载审计。
4. ILUME isomeric canonical SMILES保持基准比较身份；模型输入重新canonicalize为`isomericSmiles=False`。使用固定BERT WordPiece vocab，严格执行官方`"[CLS]" + SMILES`、分词器自动特殊 tokens、`max_length=100`截断及删除最外层首token，因此编码器最大长度为99。去立体collision与全部split截断均公开审计。
5. 阳离子、阴离子、solute和solvent始终按注册表槽位分别编码，但共享唯一全量微调后的编码器；一个batch以组分-major顺序合并成一次`C×B` 前向。表示按槽位顺序concat，归一化 numeric 条件随后拼接，预测器仅为`Linear(input_dim,1536) → GELU → Linear(1536,1)`。
6. 目标和条件 scaler只由当前折/任务训练集行拟合。HOMO/LUMO将阳离子与阴离子组成一个池、一个scaler、一个模型和一个标量预测头，`ion_role`只用于审计及诊断。
7. 训练固定归一化 MSE、AdamW、LR `5e-5`、权重衰减 `0.02`、batch 8、50 轮和种子 42。首个完整轮从`5e-6`线性预热至`5e-5`，随后按优化器步余弦 decay至`3e-6`；原始验证 MAE选择检查点并以patience 10早停。FP32且不使用AMP；OOM、NaN或CUDA错误不触发fallback。
8. Stage 3为21 任务×5折，Stage 2 Core为HoV/HOMO/LUMO，共108个训练任务。Partial Charge与Stage 2 完整为unsupported；不增加原子 mapper、原子预测头、split、evaluator或报告结构定义。

## 后果

- SPMM专属行为局限在薄adapter、严格配置和独立环境；Stage 1/2/3与既有基线合同不变。
- 组分-wise编码是对ILUME 拓扑的最薄扩展，不把阳离子与阴离子拼成pseudo sequence，也不引入新的交互 architecture。
- 去除立体信息、99-token 编码器上限及single-ion orbital 域 shift均作为基线能力限制记录，不进行人工修补。
- 正式108-作业 sweep和评估仍由用户显式执行；实现与验证不修改现有outputs。

## 来源

- [固定GitHub commit](https://github.com/jinhojsk515/SPMM/tree/046976484f31b3cbc862b8f2094e38df72fcfce7)
- [官方检查点目录](https://drive.google.com/drive/folders/1ARrSg9kXdXAL5VGgDBwizpSgcJwauPua)
