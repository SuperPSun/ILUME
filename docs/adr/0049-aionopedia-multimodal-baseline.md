# ADR-0049：AIonopedia Stage 3 多模态基线

- 状态：已接受
- 日期：2026-09-10
- 范围：AIonopedia 基线；使用独立的 10轮模型-native 合同
- 修订：2026-09-19 固定训练预算为 10 轮

## 目标与来源边界

AIonopedia 用于回答：公开的 ionic-liquid-specific 多模态 foundation 模型在 ILUME
自己的 Stage 3 数据、折和指标下重新 fine-tune 后能达到什么水平。它只使用
`AIonopedia/AIonopedia` 修订版本
`448236ef3532efd67b7956472df4e8f539b22629` 与 `Qwen/Qwen3-0.6B` 修订版本
`c1899de289a04d12100db370d81485cdf75e47ca`。模型实现固定到公开仓库 commit
`17e2f550f91eadcdec39f467c0443f5446d9713c`；重资产不进入 Git，运行前必须校验每个实际
读取文件的 SHA-256、字节数和状态-dict 结构。正式预训练快照固定为本地通用
预训练导出 `qwen0.6b-pretrain_simple2.8m(itg_loss)`；不得读取同级的 density、melt、
solvation、tension、transfer 或 viscosity property-specific 目录。

该快照的 11 个张量/检查点文件逐一匹配上述已发布修订版本；本地
`adapter_config.json` 是旧 PEFT 0.14 导出元数据，并不与 Hugging Face 当前文件逐字节相同。
它以自己的 SHA-256/字节数进入身份，并精确校验 LoRA rank、alpha、dropout、目标 modules、
Qwen3 auto-mapping 等语义。`task_type: null` 与作者机器的 PRIVATE base 路径只存在于原始输入
文件，不写入公开运行元数据；运行时始终把 adapter 加载到本地、另行锁定的 Qwen base，
并把内存中的 PEFT base 来源记录重绑定到本地 base 快照，避免 PEFT 触发在线查询。

不重新预训练，也不使用作者未公开的原始分子 pools、2.8M instantiated 语料、graph
dictionary、property-specific 数据、检查点或归一化 statistics。Stage 2 不在该
基线范围内。可复现性边界是锁定已发布产物与公开下游 code 路径，而不是
从私有语料重建预训练。

## 输入与模型合同

1. Stage 3 注册表/任务目录是组分 slots、条件与任务身份的唯一合同依据。
   允许的映射是 IL、IL+条件、solute+IL+条件与
   solute+solvent+条件，对应 AIonopedia 的四个原生拓扑。
2. Prompt 只写 system composition 和原始单位的条件，不写目标/任务/property 名称。
3. 分子图保留官方公开预处理：35D 原子特征、11D edge 特征、无 explicit-H
   expansion，并保留已发布检查点所依赖的历史行为。
4. Temperature 使用 `temperature_K / 1000`。Pressure 使用当前折训练集行的样本
   z-score（`ddof=1`）；常量 pressure 的规模固定为 1 并审计。Frequency 与 wavelength
   分别使用 `frequency_MHz / 1000` 和 `wavelength_nm / 1000`。
5. Pressure、frequency、wavelength 都同时进入 text prompt 和 graph-side fusion；各自新增独立
   投影层与 segment token。新增模块随机初始化，不复制 temperature 权重，也不宣称具有
   预训练条件表示。
6. 加载已发布 LoRA、GNN、LLM/GNN/temperature 投影层、拓扑 embedding、graph merge
   Transformer、两个 cross-modal decoder 和五个 segment embedding。官方 71-输出预训练
   预测头仅做产物结构审计，不加载到下游；每个任务/折新建
   `Linear(512,1024) -> ReLU -> Linear(1024,1)` 标量预测头。
7. Qwen 非 LoRA 参数冻结；已发布 LoRA、上述完整多模态模块、标量预测头和新增条件
   modules 均可训练。不得退化为 pure-Qwen 编码器基线。

Target 保持 ILUME Stage 3 原始定义，只用当前折训练集行做样本 z-score，MSE 在归一化
space 训练；验证/测试集预测反归一化后以 ILUME 现役原始-unit 指标评估。

## 训练、选择与输出

每个任务/折独立训练 10 轮，种子为 `42 + fold - 1`，单 GPU batch大小 16，BF16，
AdamW，权重衰减 0.01，max grad norm 1。标量预测头 learning rate 为 `4e-5`，两个 decoder
为 `3e-5`，其余可训练参数为 `3e-5`；调度器为 50-步 linear 预热后余弦 decay。

验证每轮计算且只用于历史记录/报告。当前发布轮 10 final 训练状态，不 early
stop、不按验证选检查点，调度器也不读取验证。每轮可训练-状态快照
只用于不可恢复的审计；基线不支持恢复，失败任务必须在新尝试目录从头运行。正式
结果使用独立根 `outputs/benchmarks/model-native-v1/aionopedia/`，不得与 ADR-0045 的
`fixed-budget-v1` 或任何旧候选混用。

独立环境使用 PyTorch `2.9.0+cu128`，与其他 CUDA 12.8 advanced 基线对齐。这是为
Blackwell/RTX PRO 6000 服务器采用的执行兼容扩展；公开 fine-tuning 快照原始记录的
PyTorch 2.6/CUDA 12.4 仍保留为来源记录，模型、优化器和数据合同不因运行时升级而改变。
环境版本和 lock hash 必须进入每次运行元数据。

## 后果与风险

- 训练预算保留官方下游配方的 10 轮，不能解释为与其他基线的统一算力比较，
  也不是对 ILUME 数据规模的最优预算声明。
- 新条件路径只能从折内训练集数据学习；近常量 pressure 或小数据任务中可能贡献很弱。
- 深层多模态 fusion 中的新 token 稳定性必须由逐轮 loss、验证历史记录与非有限
  hard failure 审计。
- 后续基线的轮、验证、选择、早停、调度器和报告策略
  逐模型冻结；ADR-0045 只继续约束其列出的七个旧基线。

## 历史容量对照

正式配置保持 `Linear(512,1024) → ReLU → Linear(1024,1)`。128维宽度对照已退役，历史决定与
退役原因见 [ADR-0076](0076-aionopedia-scalar-head-capacity-comparison.md)。
