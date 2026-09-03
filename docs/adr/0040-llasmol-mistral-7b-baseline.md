# ADR-0040：LlaSMol-Mistral-7B QLoRA 基线

- 状态：Accepted
- 日期：2026-09-03

## 背景

ILUME需要加入chemical LLM baseline，以统一regression协议评估大语言模型式chemical representation。官方`osunlp/LlaSMol-Mistral-7B`只发布了基于`mistralai/Mistral-7B-v0.1`的LoRA adapter，而不是完整7B checkpoint；其官方adapter同时覆盖attention与MLP projection。

## 决定

1. 固定`mistralai/Mistral-7B-v0.1@27d67f1b5f57dc0953326b2601d68371d40ea8da`和`osunlp/LlaSMol-Mistral-7B@044d6124448733615c5a3d6ab14b947f71fc6728`。两份snapshot由用户显式下载至`artifacts/benchmarks/llasmol/`，运行只读本地文件并校验全部使用文件的SHA和大小。
2. 使用独立`ilume-llasmol` hash-lock环境，固定Python 3.12.12、PyTorch 2.9.0+cu128、Transformers 4.57.6、PEFT 0.18.1、bitsandbytes 0.49.2、Accelerate 1.12.0、RDKit 2026.3.5和NumPy 2.5.2。缺少环境、CUDA BF16/4-bit backend或资产不匹配时，在创建run output前硬失败。
3. 基座以NF4 double-quant 4-bit冻结加载，BF16计算并开启gradient checkpointing。继续训练官方adapter的全部q/k/v/o与gate/up/down LoRA参数；不创建第二套adapter。官方adapter固定rank 16、alpha 16、dropout 0.05，共448个BF16 tensor。
4. 输入使用普通文本`<{task leaf uppercase}>\nSMILES`，不扩展tokenizer词表。普通IL把canonical isomeric cation和anion组成单条`cation.anion`；solvation/transfer分别编码whole-IL和solute，transfer organic分别编码solute和solvent。所有view共享唯一backbone，并合并成一次forward。
5. 基座tokenizer固定左padding、右截断、BOS且无EOS，最大长度512；所有split的截断均公开审计。最后一层hidden states使用attention-mask mean pooling。多view representation按registry顺序concat，train-only normalized conditions随后拼入`Linear(input_dim,256) → SiLU → Linear(256,1)`。
6. target和condition scaler只由当前fold/task train rows拟合。HOMO/LUMO将cation与anion组成一个pool、一个scaler、一个model和一个head；`ion_role`只用于provenance和diagnostics。
7. 训练固定normalized MSE、AdamW、LoRA/head LR `2e-5/1e-4`、weight decay `0.01`、row batch 8、gradient accumulation 4、30 epochs、5% warmup后cosine decay至0、raw validation MAE selection、patience 8和seed 42。训练采用确定性sortish长度分桶；OOM、NaN或CUDA错误不触发fallback。
8. Stage 3为21 tasks×5 folds，Stage 2 Core为HoV/HOMO/LUMO，共108个训练任务。Partial Charge与Stage 2 Full为unsupported；不增加生成式数值解析、atom mapper、split、evaluator或reporting schema。
9. 正式checkpoint只保存fine-tuned LoRA和regression head。4-bit基座从固定snapshot重建；test只在best validation checkpoint确定后评估一次。

## 后果

- 该baseline比较LlaSMol hidden representation的可迁移性，而不是prompt engineering或生成能力。
- 官方adapter中的MLP LoRA继续训练是本baseline的settled decision，取代早期仅更新q/k/v/o的设想。
- 多view concat和condition拼接是ILUME topology所需的最薄扩展；不增加cross-attention或自定义interaction network。
- LlaSMol adapter是CC-BY-4.0，Mistral基座是Apache-2.0。官方adapter为pickle；只有固定SHA和大小校验通过后才使用`weights_only=True`反序列化。
- 实现验证不运行正式108-job sweep，也不修改现有outputs。

## 来源

- [固定LlaSMol adapter快照](https://huggingface.co/osunlp/LlaSMol-Mistral-7B/tree/044d6124448733615c5a3d6ab14b947f71fc6728)
- [固定Mistral基座快照](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/27d67f1b5f57dc0953326b2601d68371d40ea8da)
- [官方adapter配置](https://huggingface.co/osunlp/LlaSMol-Mistral-7B/blob/044d6124448733615c5a3d6ab14b947f71fc6728/adapter_config.json)
- [官方fine-tune实现](https://github.com/OSU-NLP-Group/LLM4Chem/blob/43ab5fccd14514ddf756534d18a6917e7e11d0ae/finetune.py)
