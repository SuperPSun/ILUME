# ADR-0057：ILTransR Stage 3 基线

- 状态：已接受
- 日期：2026-09-14
- 范围：ILTransR 基线；独立于 ADR-0045 的七个旧固定-budget 基线
- 修订：2026-09-19 所有任务配方统一为 10 轮

## 目标与资产边界

ILTransR 用于评估 ionic-liquid-specific 通用预训练 Transformer 在 ILUME 自己的
Stage 3 splits、targets 和原始-unit metrics 下完整 fine-tune 后的表现。唯一上游实现为
`GuzhongChen/ILTransR` commit `ff5e55cfb8162b0706fc2d88ca5cd384705686b1`，唯一初始化
检查点为 `pretraining/valid_best.params`，SHA-256 为
`36a3401175dd372725bd2f5e4e041642a754eeaf3f46845b6ff949065871735a`。

不重新 pretrain，不读取作者 property datasets、`density_best.params`、`viscosity_best.params`、
`co2_best.params` 或其他监督训练 property 检查点。重资产不进入 Git；运行时必须校验
原始检查点、两个 vocab、转换产物与转换清单的哈希和结构。

正式训练使用 PyTorch `2.9.0+cu128` 的 FP32 等价实现。MXNet 1.9.1/GluonNLP 0.10.0 只在
Python 3.8.20 CPU 环境执行一次性转换：只导出来源 embedding 与三层 Transformer 编码器；
decoder、one-步 decoder、目标 embedding 和目标投影明确列入 ignored tensors。
固定 token batch 的 MXNet reference 作为独立 safetensors 资产锁定；正式 validator 用 PyTorch
CPU 重放并要求 max absolute error 不超过 `1e-5`、均值 absolute error 不超过 `1e-6`。

## 输入、拓扑与模型合同

Stage 3 注册表/任务目录是任务身份、组分 slots、条件顺序、split 路径和
目标的唯一合同依据。所有输入先用 RDKit 2022.3.2.1 做 `canonical=True,
isomericSmiles=False` 的模型-specific 视图；ILUME 的原始分子身份不变。

- `(cation, anion)` 作为一条整体 canonicalized `cation.anion` IL 视图。
- `(cation, anion, solute)` 按 `ionic_liquid, solute` 顺序产生两个视图。
- `(solute, solvent)` 按 `solute, solvent` 顺序产生两个视图。

所有视图共用唯一、可训练的 embedding、Transformer 和 TextCNN，并合并为一次
`views x batch` backbone 前向；融合严格按注册表角色顺序 concat，不把 partner 任务
拼成新的三组分 dot-separated SMILES。

Tokenizer 固定上游 72-token character vocab：不加 BOS、末尾加 EOS、未知字符映射 `<unk>`、
batch padding value 固定为上游代码实际使用的 0。应用 `ClipSequence(100)` 的语义，保留并
截断超过 100 tokens 的样本；逐角色记录截断前字符/token 长度、截断数、丢失 EOS 数和
unknown-token 数。

Transformer 固定为 128D、三层、四注意力头、FFN 1024、ReLU、post-norm、输入 LayerNorm、
sinusoidal position encoding、embedding 乘 `sqrt(128)`、dropout 0、context 100。TextCNN 保留
上游三种拓扑及 filters/highway：无条件 kernels `1..10`；T-仅 kernels
`1..10,15,20`；含 P kernels `1..10,15`。无条件预测头是无激活/dropout 的
`Dense(512) -> Dense(1)`；conditioned 预测头保留上游 MLP。Partner 任务只扩展预测头第一层
输入宽度。Pretrained embedding 和每层编码器与新建 TextCNN/预测头全部参与训练。

## 归一化与训练合同

Temperature、pressure、frequency 和 wavelength 均按注册表中的原始条件列顺序，
使用任务-global 总体z-score：`z=(x-mean_all_stage3)/std_all_stage3`，`ddof=0`。
总体是该任务的五个折文件与测试集，每行恰好一次。这是明确的传导式特征 scaling：
允许读取验证集/测试集协变量，不读取其目标；统计值、条件-仅 hash 与源路径身份进入
每个折检查点。恒定条件保存原均值，使用规模 1 并映射为 0。Frequency 和
wavelength 是 ILUME-specific extension。

Target 身份保持 ILUME 原始定义，只用当前折训练集 targets 做总体z-score；在
归一化目标上统一使用 L1，评估 inverse-transform 到原单位后调用现役 metrics。
这也明确覆盖上游 temperature-仅源码中的 L2 不一致。

同一 property 只在有公开 notebook 时采用官方配方：density、viscosity、heat capacity、
melting point、热分解 temperature、x_CO2 和 pEC50；其 batch大小和 dropout 由
正式 YAML 冻结，但所有任务统一训练 10 轮。所有任务使用 Adam (`lr=1e-3`、默认 betas/epsilon、无权重衰减)，每 10 轮
学习率乘 0.5，每 batch 一次更新，full fine-tuning，FP32 且不启用 TF32。无条件拓扑使用上游 two-bucket
shuffled 采样；conditioned 拓扑保持来源 order。

Seed 为 `42 + fold - 1`。每个任务跑满预算，验证每轮只记录，不进入调度器、停止或
选择；发布 final 轮状态，不 early stop、不按最低训练集 loss 或验证选择。基线
不支持恢复，失败由 sweep 在新尝试目录完整重跑。10轮正式结果根为
`outputs/benchmarks/model-native-10e-v1/iltransr/`，不得复用旧预算或其他基线的结果。

## 后果

- 公开检查点的模型语义由固定转换和跨框架 parity 保证，不要求在新 GPU 上运行 MXNet。
- Condition scaling 是主动登记的传导式 covariate 使用，不能描述成仅训练集 scaling。
- 全部任务使用同一 10轮预算；上游 notebook 仅保留 batch大小与 dropout 来源记录。
- 超过 100 tokens 的样本不会被删除，但其尾部和可能的 EOS 会被截断，必须在输入审计中报告。

实施时使用 RDKit 2022.03.2 对当前 system-split Stage 3 的 21 个任务做只读审计：245,614 条
任务局部物理行产生的视图中，1,697 条超过 100 tokens，分布在 13 个任务，最大截断前长度
为 287，official vocab unknown-token 总数为 0。该数字是当前数据快照的审计结果而非冻结常量；
正式运行仍按其实际来源 hash 重新记录逐 split/角色分位数和截断统计。
