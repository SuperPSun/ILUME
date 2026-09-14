# ADR-0057：ILTransR Stage 3 Baseline

- 状态：Accepted
- 日期：2026-09-14
- 范围：ILTransR baseline；独立于 ADR-0045 的七个旧 fixed-budget baseline

## 目标与资产边界

ILTransR 用于评估 ionic-liquid-specific generic pretrained Transformer 在 ILUME 自己的
Stage 3 splits、targets 和 raw-unit metrics 下完整 fine-tune 后的表现。唯一上游实现为
`GuzhongChen/ILTransR` commit `ff5e55cfb8162b0706fc2d88ca5cd384705686b1`，唯一初始化
checkpoint 为 `pretraining/valid_best.params`，SHA-256 为
`36a3401175dd372725bd2f5e4e041642a754eeaf3f46845b6ff949065871735a`。

不重新 pretrain，不读取作者 property datasets、`density_best.params`、`viscosity_best.params`、
`co2_best.params` 或其他 supervised property checkpoint。重资产不进入 Git；运行时必须校验
原始 checkpoint、两个 vocab、转换产物与转换 manifest 的哈希和结构。

正式训练使用 PyTorch `2.9.0+cu128` 的 FP32 等价实现。MXNet 1.9.1/GluonNLP 0.10.0 只在
Python 3.8.20 CPU 环境执行一次性转换：只导出 source embedding 与三层 Transformer encoder；
decoder、one-step decoder、target embedding 和 target projection 明确列入 ignored tensors。
固定 token batch 的 MXNet reference 作为独立 safetensors 资产锁定；正式 validator 用 PyTorch
CPU 重放并要求 max absolute error 不超过 `1e-5`、mean absolute error 不超过 `1e-6`。

## 输入、拓扑与模型合同

Stage 3 registry/task catalog 是 task identity、component slots、condition 顺序、split path 和
target 的唯一 authority。所有输入先用 RDKit 2022.3.2.1 做 `canonical=True,
isomericSmiles=False` 的 model-specific view；ILUME 的原始 molecule identity 不变。

- `(cation, anion)` 作为一条整体 canonicalized `cation.anion` IL view。
- `(cation, anion, solute)` 按 `ionic_liquid, solute` 顺序产生两个 view。
- `(solute, solvent)` 按 `solute, solvent` 顺序产生两个 view。

所有 view 共用唯一、可训练的 embedding、Transformer 和 TextCNN，并合并为一次
`views x batch` backbone forward；融合严格按 registry role 顺序 concat，不把 partner task
拼成新的三组分 dot-separated SMILES。

Tokenizer 固定上游 72-token character vocab：不加 BOS、末尾加 EOS、未知字符映射 `<unk>`、
batch padding value 固定为上游代码实际使用的 0。应用 `ClipSequence(100)` 的语义，保留并
截断超过 100 tokens 的样本；逐 role 记录截断前字符/token 长度、截断数、丢失 EOS 数和
unknown-token 数。

Transformer 固定为 128D、三层、四 heads、FFN 1024、ReLU、post-norm、输入 LayerNorm、
sinusoidal position encoding、embedding 乘 `sqrt(128)`、dropout 0、context 100。TextCNN 保留
上游三种 topology 及 filters/highway：无条件 kernels `1..10`；T-only kernels
`1..10,15,20`；含 P kernels `1..10,15`。无条件 head 是无 activation/dropout 的
`Dense(512) -> Dense(1)`；conditioned head 保留上游 MLP。Partner task 只扩展 head 第一层
输入宽度。Pretrained embedding 和每层 encoder 与新建 TextCNN/head 全部参与训练。

## Normalization 与训练合同

Temperature、pressure、frequency 和 wavelength 均按 registry 中的原始 condition 列顺序，
使用 task-global population z-score：`z=(x-mean_all_stage3)/std_all_stage3`，`ddof=0`。
总体是该 task 的五个 fold 文件与 test，每行恰好一次。这是明确的 transductive feature scaling：
允许读取 valid/test covariates，不读取其 target；统计值、condition-only hash 与源路径身份进入
每个 fold checkpoint。恒定 condition 保存原 mean，使用 scale 1 并映射为 0。Frequency 和
wavelength 是 ILUME-specific extension。

Target identity 保持 ILUME 原始定义，只用当前 fold train targets 做 population z-score；在
normalized target 上统一使用 L1，evaluation inverse-transform 到 raw units 后调用现役 metrics。
这也明确覆盖上游 temperature-only 源码中的 L2 不一致。

同一 property 只在有公开 notebook 时采用官方 recipe：density、viscosity、heat capacity、
melting point、thermal decomposition temperature、x_CO2 和 pEC50；准确 epochs、batch size 和
dropout 由正式 YAML 冻结。其余 14 个任务统一使用 150 epochs、batch 64、dropout 0.1 的
fallback。所有任务使用 Adam (`lr=1e-3`、默认 betas/epsilon、无 weight decay)，每 10 epochs
学习率乘 0.5，每 batch 一次更新，full fine-tuning，FP32 且不启用 TF32。无条件 topology 使用上游 two-bucket
shuffled sampling；conditioned topology 保持 source order。

Seed 为 `42 + fold - 1`。每个任务跑满预算，validation 每轮只记录，不进入 scheduler、停止或
选择；发布 final epoch state，不 early stop、不按最低 train loss 或 validation 选择。Baseline
不支持 resume，失败由 sweep 在新 attempt 目录完整重跑。正式结果根为
`outputs/benchmarks/model-native-v1/iltransr/`，不得复用其他 baseline 的结果。

## 后果

- 公开 checkpoint 的模型语义由固定转换和跨框架 parity 保证，不要求在新 GPU 上运行 MXNet。
- Condition scaling 是主动登记的 transductive covariate 使用，不能描述成 train-only scaling。
- 不同 property 的训练预算不同，优先保留 model-native recipe，不能解释成统一算力比较。
- 超过 100 tokens 的样本不会被删除，但其尾部和可能的 EOS 会被截断，必须在输入审计中报告。

实施时使用 RDKit 2022.03.2 对当前 system-split Stage 3 的 21 个 task 做只读审计：245,614 条
task-local 物理行产生的 view 中，1,697 条超过 100 tokens，分布在 13 个 task，最大截断前长度
为 287，official vocab unknown-token 总数为 0。该数字是当前数据快照的审计结果而非冻结常量；
正式 run 仍按其实际 source hash 重新记录逐 split/role 分位数和截断统计。
