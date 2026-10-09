# ADR-0040：LlaSMol-Mistral-7B QLoRA 基线

- 状态：已接受
- 日期：2026-09-03
- 修订：2026-09-10 将固定预算调整为 10 轮、batch 16、梯度 accumulation 2

> Early stopping 与验证最优检查点由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代。

## 背景

ILUME需要加入chemical LLM 基线，以统一regression协议评估大语言模型式chemical 表示。官方`osunlp/LlaSMol-Mistral-7B`只发布了基于`mistralai/Mistral-7B-v0.1`的LoRA adapter，而不是完整7B 检查点；其官方adapter同时覆盖attention与MLP 投影。

## 决定

1. 固定`mistralai/Mistral-7B-v0.1@27d67f1b5f57dc0953326b2601d68371d40ea8da`和`osunlp/LlaSMol-Mistral-7B@044d6124448733615c5a3d6ab14b947f71fc6728`。两份快照由用户显式下载至`artifacts/benchmarks/llasmol/`，运行只读本地文件并校验全部使用文件的SHA和大小。
2. 使用独立`ilume-llasmol` hash-lock环境，固定Python 3.12.12、PyTorch 2.9.0+cu128、Transformers 4.57.6、PEFT 0.18.1、bitsandbytes 0.49.2、Accelerate 1.12.0、RDKit 2026.3.5和NumPy 2.5.2。缺少环境、CUDA BF16/4-bit 后端或资产不匹配时，在创建运行输出前硬失败。
3. 基座以NF4 double-quant 4-bit冻结加载，BF16计算并关闭梯度检查点。继续训练官方adapter的全部q/k/v/o与门控/up/down LoRA参数；不创建第二套adapter。官方adapter固定rank 16、alpha 16、dropout 0.05，共448个BF16 张量。
4. 输入使用普通文本`<{task leaf uppercase}>\nSMILES`，不扩展分词器词表。普通IL把canonical isomeric 阳离子和阴离子组成单条`cation.anion`；solvation/transfer分别编码whole-IL和solute，transfer organic分别编码solute和solvent。所有视图共享唯一backbone，并合并成一次前向。
5. 基座分词器固定左padding、右截断、BOS且无EOS，最大长度512；所有split的截断均公开审计。最后一层隐藏宽度状态使用attention-mask 均值池化。多视图表示按注册表顺序concat，仅训练集归一化条件随后拼入`Linear(input_dim,256) → SiLU → Linear(256,1)`。
6. 目标和条件 scaler只由当前折/任务训练集行拟合。HOMO/LUMO将阳离子与阴离子组成一个池、一个scaler、一个模型和一个预测头；`ion_role`只用于来源记录和诊断。
7. 训练固定归一化 MSE、AdamW、LoRA/预测头 LR `2e-5/1e-4`、权重衰减 `0.01`、行 batch 16、梯度 accumulation 2、10 轮、5% 预热后余弦 decay至0和种子 42。每轮验证只写入历史记录，不参与训练决策；正式发布第10轮状态。训练采用确定性近似按长度排序长度分桶；OOM、NaN或CUDA错误不触发fallback。
8. Stage 3为21 任务×5折，共105个训练任务。Stage 2 报告已经退役；不增加生成式数值解析、原子 mapper、split、evaluator或报告结构定义。
9. 正式检查点只保存fine-tuned LoRA和regression 预测头。4-bit基座从固定快照重建；测试集只使用固定训练预算结束后的末轮状态评估一次。

## 后果

- 该基线比较LlaSMol 隐藏宽度表示的可迁移性，而不是prompt engineering或生成能力。
- 官方adapter中的MLP LoRA继续训练是本基线的settled 决策，取代早期仅更新q/k/v/o的设想。
- 多视图 concat和条件拼接是ILUME 拓扑所需的最薄扩展；不增加cross-attention或自定义交互 network。
- LlaSMol adapter是CC-BY-4.0，Mistral基座是Apache-2.0。官方adapter为pickle；只有固定SHA和大小校验通过后才使用`weights_only=True`反序列化。
- 实现验证不运行正式105-作业 sweep，也不修改现有outputs。

## 来源

- [固定LlaSMol adapter快照](https://huggingface.co/osunlp/LlaSMol-Mistral-7B/tree/044d6124448733615c5a3d6ab14b947f71fc6728)
- [固定Mistral基座快照](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/27d67f1b5f57dc0953326b2601d68371d40ea8da)
- [官方adapter配置](https://huggingface.co/osunlp/LlaSMol-Mistral-7B/blob/044d6124448733615c5a3d6ab14b947f71fc6728/adapter_config.json)
- [官方fine-tune实现](https://github.com/OSU-NLP-Group/LLM4Chem/blob/43ab5fccd14514ddf756534d18a6917e7e11d0ae/finetune.py)
