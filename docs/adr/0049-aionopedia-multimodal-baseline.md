# ADR-0049：AIonopedia Stage 3 多模态 Baseline

- 状态：Accepted
- 日期：2026-09-10
- 范围：AIonopedia baseline；在重叠范围内取代 ADR-0045 的统一 50-epoch 规则

## 目标与来源边界

AIonopedia 用于回答：公开的 ionic-liquid-specific multimodal foundation model 在 ILUME
自己的 Stage 3 数据、fold 和指标下重新 fine-tune 后能达到什么水平。它只使用
`AIonopedia/AIonopedia` revision
`448236ef3532efd67b7956472df4e8f539b22629` 与 `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`。模型实现固定到公开仓库 commit
`17e2f550f91eadcdec39f467c0443f5446d9713c`；重资产不进入 Git，运行前必须校验每个实际
读取文件的 SHA-256、字节数和 state-dict 结构。正式 pretrained snapshot 固定为本地 generic
pretraining export `qwen0.6b-pretrain_simple2.8m(itg_loss)`；不得读取同级的 density、melt、
solvation、tension、transfer 或 viscosity property-specific 目录。

该 snapshot 的 11 个 tensor/checkpoint 文件逐一匹配上述 released revision；本地
`adapter_config.json` 是旧 PEFT 0.14 导出元数据，并不与 Hugging Face 当前文件逐字节相同。
它以自己的 SHA-256/字节数进入身份，并精确校验 LoRA rank、alpha、dropout、target modules、
Qwen3 auto-mapping 等语义。`task_type: null` 与作者机器的 private base path 只存在于原始输入
文件，不写入公开 run metadata；运行时始终把 adapter 加载到本地、另行锁定的 Qwen base，
并把内存中的 PEFT base provenance 重绑定到公开 repository 标识。

不重新预训练，也不使用作者未公开的原始 molecule pools、2.8M instantiated corpus、graph
dictionary、property-specific 数据、checkpoint 或 normalization statistics。Stage 2 不在该
baseline 范围内。可复现性边界是锁定 released artifact 与公开 downstream code path，而不是
从私有语料重建 pretraining。

## 输入与模型合同

1. Stage 3 registry/task catalog 是 component slots、conditions 与 task identity 的唯一 authority。
   允许的映射是 IL、IL+conditions、solute+IL+conditions 与
   solute+solvent+conditions，对应 AIonopedia 的四个原生 topology。
2. Prompt 只写 system composition 和原始单位的 conditions，不写 target/task/property 名称。
3. 分子图保留官方公开 preprocessing：35D atom features、11D edge features、无 explicit-H
   expansion，并保留 released checkpoint 所依赖的历史行为。
4. Temperature 使用 `temperature_K / 1000`。Pressure 使用当前 fold train rows 的 sample
   z-score（`ddof=1`）；常量 pressure 的 scale 固定为 1 并审计。Frequency 与 wavelength
   分别使用 `frequency_MHz / 1000` 和 `wavelength_nm / 1000`。
5. Pressure、frequency、wavelength 都同时进入 text prompt 和 graph-side fusion；各自新增独立
   projector 与 segment token。新增模块随机初始化，不复制 temperature 权重，也不宣称具有
   pretrained condition representation。
6. 加载 released LoRA、GNN、LLM/GNN/temperature projectors、topology embedding、graph merge
   Transformer、两个 cross-modal decoder 和五个 segment embedding。官方 71-output pretraining
   head 仅做 artifact 结构审计，不加载到 downstream；每个 task/fold 新建
   `Linear(512,1024) -> ReLU -> Linear(1024,1)` scalar head。
7. Qwen 非 LoRA 参数冻结；released LoRA、上述完整多模态模块、scalar head 和新增 condition
   modules 均可训练。不得退化为 pure-Qwen encoder baseline。

Target 保持 ILUME Stage 3 原始定义，只用当前 fold train rows 做 sample z-score，MSE 在 normalized
space 训练；validation/test 预测反归一化后以 ILUME 现役 raw-unit 指标评估。

## 训练、选择与输出

每个 task/fold 独立训练 10 epochs，seed 为 `42 + fold - 1`，单 GPU batch size 16，BF16，
AdamW，weight decay 0.01，max grad norm 1。scalar head learning rate 为 `4e-5`，两个 decoder
为 `3e-5`，其余 trainable 参数为 `3e-5`；scheduler 为 50-step linear warmup 后 cosine decay。

Validation 每轮计算且只用于 history/reporting。当前发布 epoch 10 final training state，不 early
stop、不按 validation 选 checkpoint，scheduler 也不读取 validation。每轮 trainable-state snapshot
只用于不可恢复的审计；baseline 不支持 resume，失败任务必须在新 attempt 目录从头运行。正式
结果使用独立根 `outputs/benchmarks/model-native-v1/aionopedia/`，不得与 ADR-0045 的
`fixed-budget-v1` 或任何旧 candidate 混用。

独立环境使用 PyTorch `2.9.0+cu128`，与其他 CUDA 12.8 advanced baseline 对齐。这是为
Blackwell/RTX PRO 6000 服务器采用的执行兼容扩展；公开 fine-tuning snapshot 原始记录的
PyTorch 2.6/CUDA 12.4 仍保留为 provenance，模型、优化器和数据合同不因运行时升级而改变。
环境版本和 lock hash 必须进入每次 run metadata。

## 后果与风险

- 训练预算保留官方 downstream recipe 的 10 epochs，不能解释为与其他 baseline 的统一算力比较，
  也不是对 ILUME 数据规模的最优预算声明。
- 新 condition path 只能从 fold-local train data 学习；近常量 pressure 或小数据 task 中可能贡献很弱。
- 深层 multimodal fusion 中的新 token 稳定性必须由逐 epoch loss、validation history 与 non-finite
  hard failure 审计。
- 后续 baseline 的 epoch、validation、selection、early stopping、scheduler 和 reporting policy
  逐模型冻结；ADR-0045 只继续约束其列出的七个旧 baseline。
