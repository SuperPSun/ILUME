# ADR-0035：SPMM SMILES-only 语言模型基线

- 状态：Accepted
- 日期：2026-09-01

## 背景

ILUME需要加入官方SPMM pretrained model作为advanced baseline。SPMM预训练同时包含SMILES text与53维property vector，但官方MoleculeNet regression只使用SMILES text branch的`[CLS]`表示。其原生downstream没有ILUME多组分topology，也没有满足Partial Charge atom mapping合同的输出路径。

## 决定

1. 固定外部`jinhojsk515/SPMM@046976484f31b3cbc862b8f2094e38df72fcfce7`及官方`checkpoint_SPMM.ckpt`；运行前校验checkout、相关源码、vocab、BERT config、checkpoint SHA与2358591924字节大小。上游为Apache-2.0，但ILUME仍不复制或提交源码和权重。
2. 使用独立`ilume-spmm` hash-lock环境，固定Python 3.10.14、PyTorch 1.13.1+cu117、CUDA 11.7、Transformers 4.30.1、tokenizers 0.13.3、RDKit 2023.3.1和NumPy 1.24.3。该环境不安装要求Python ≥3.11的ILUME editable package，统一launcher通过现役脚本的仓库路径引导运行；缺少环境、CUDA或资产不匹配时在创建run output前硬失败。
3. baseline只实例化官方`xbert.py`的SMILES text encoder。加载主`text_encoder.bert` embeddings与layers 0–5的102个state entries；layers 6–11、LM head、PV、momentum encoder和queues均不进入downstream模型。官方Lightning checkpoint包含Python pickle对象，只有固定SHA与大小通过后才允许反序列化，并保存load audit。
4. ILUME isomeric canonical SMILES保持benchmark identity；模型输入重新canonicalize为`isomericSmiles=False`。使用固定BERT WordPiece vocab，严格执行官方`"[CLS]" + SMILES`、tokenizer自动special tokens、`max_length=100`截断及删除最外层首token，因此encoder最大长度为99。去立体collision与全部split截断均公开审计。
5. cation、anion、solute和solvent始终按registry slot分别编码，但共享唯一full-finetuned encoder；一个batch以component-major顺序合并成一次`C×B` forward。表示按slot顺序concat，normalized numeric conditions随后拼接，predictor仅为`Linear(input_dim,1536) → GELU → Linear(1536,1)`。
6. target和condition scaler只由当前fold/task train rows拟合。HOMO/LUMO将cation与anion组成一个pool、一个scaler、一个model和一个scalar head，`ion_role`只用于审计及diagnostics。
7. 训练固定normalized MSE、AdamW、LR `5e-5`、weight decay `0.02`、batch 8、50 epochs和seed 42。首个完整epoch从`5e-6`线性warmup至`5e-5`，随后按optimizer step cosine decay至`3e-6`；raw validation MAE选择checkpoint并以patience 10早停。FP32且不使用AMP；OOM、NaN或CUDA错误不触发fallback。
8. Stage 3为21 tasks×5 folds，Stage 2 Core为HoV/HOMO/LUMO，共108个训练任务。Partial Charge与Stage 2 Full为unsupported；不增加atom mapper、atom head、split、evaluator或reporting schema。

## 后果

- SPMM专属行为局限在薄adapter、严格配置和独立环境；Stage 1/2/3与既有baseline合同不变。
- component-wise编码是对ILUME topology的最薄扩展，不把cation与anion拼成pseudo sequence，也不引入新的interaction architecture。
- 去除立体信息、99-token encoder上限及single-ion orbital domain shift均作为baseline能力限制记录，不进行人工修补。
- 正式108-job sweep和evaluation仍由用户显式执行；实现与验证不修改现有outputs。

## 来源

- [固定GitHub commit](https://github.com/jinhojsk515/SPMM/tree/046976484f31b3cbc862b8f2094e38df72fcfce7)
- [官方checkpoint目录](https://drive.google.com/drive/folders/1ARrSg9kXdXAL5VGgDBwizpSgcJwauPua)
