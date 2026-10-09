# ADR-0060：AIFC Stage 3 基线

- 状态：已接受
- 日期：2026-09-15
- 范围：AIFC 基线；独立于现役 Stage 1/2/3 数值合同
- 修订：2026-09-19 将固定训练预算统一为 10 轮

## 目标与上游边界

AIFC（Attentive Ionic Fragment Contribution）代表 ionic-liquid-specific fragment prior、
GNN 与 attention 路线。实现以 `2022kaikaili/ILs-AIFC` commit
`569bc338a1dbe8aad9770a5562e27e23f9ab3cdb` 的公开模型为上游来源身份，但不照搬
作者的 dataset split、Bayesian optimization、验证-driven 调度器、早停或
100模型集成。

Fragment dictionary 必须使用作者历史 commit
`23973da709212d1e13fd4b8e0968c8109f78fa75` 的 `My_fragments.csv`：Git blob
`90d8bafc2eb07835854e183df48a637d97b8c09b`，SHA-256
`ccc5cfefe87ebf3e1d98762eb1c58170bc00c9f7cfd12b6c9ce23fa957ccdf95`。文件包含 100 个
唯一 `First-Order Group / SMARTs / Priority` 条目；加载器校验完整文件 hash、列、条目数和
SMARTS 可解析性，不允许替换或扩展 dictionary。未被 SMARTS 覆盖的原子保持作者语义：每个
原子形成一个 motif，fragment type 是全零 100D 向量。

## 模型与拓扑合同

正式实现用 PyTorch `2.9.0+cu128` FP32 等价复刻作者 DGL 模型：46D 原子特征、12D
有向键特征、片段级 AttentiveFP、motif/junction AttentiveFP、多头
attention 聚合与两层 regression MLP。固定图和确定性参数在 DGL 1.1.2 CPU 与正式
PyTorch 实现间的预测、system 表示和 motif attention 最大绝对误差必须不超过
`1e-5`。作者 DGL alpha 在整个 batch 没有 fragment/motif edge 时无法产生 message context；
现代实现只对此情况补零 context，从而保留节点自身 GRU 更新并避免合法单原子体系失败。

Stage 3 注册表/任务目录是任务、槽位顺序、条件、split 和目标的唯一合同依据：

- 普通 `(cation, anion)` 生成一个 canonical `cation.anion` AIFC graph。
- `(cation, anion, solute)` 分别编码阳离子、阴离子、solute。
- `(solute, solvent)` 分别编码 solute、solvent。

所有组分在同一个 batch 中只调用一次共享 AIFC 编码器；每个组分的 system
表示严格按注册表槽位 order concat。Partner 组分之间不增加 cross-attention、
message passing 或其他交互模块。Registry 条件在 ordered 组分表示后按
原顺序 concat，整体经过作者原有 ReLU 后进入 regression MLP。Temperature/pressure 恢复作者源码中被注释的 concat 设计；
frequency/wavelength 采用相同标量 concat，明确登记为 ILUME-specific extension。只有
authoritative `condition_columns` 实际存在的条件才进入模型。

公开、可信的同性质 architecture 配置仅映射到 viscosity（fallback architecture 本身）和
热分解 temperature（隐藏宽度 208、dropout 0.598）；其余任务不按“相近性质”
借用配置，统一使用隐藏宽度 128、1 预测头、dropout 0、depth 3、层 3，且残差、batch norm、
layer norm 均关闭。所有任务/折从随机初始化训练，不存在预训练或 property-specific
检查点。

## 缩放、训练与选择合同

每个折的目标与所有注册表条件都只在该折训练集行上拟合 population
z-score（`ddof=0`）；验证集/测试集复用这些 statistics。恒定条件保存原均值、尺度设为 1，
因此归一化 value 为 0。模型以归一化目标的 MSE 训练；评估 inverse-transform
预测后调用现役 ILUME 原始-unit metrics。

所有任务/折固定使用一个种子 `1000` 的模型：Adam、learning rate `1e-3`、默认
betas/epsilon、权重衰减 0、恒定调度器、batch大小 64、FP32、10 轮。Forward
路径中的 fragment 编码器、attention 与 regression 预测头参数均参与训练；作者定义但未在前向
调用的 `motif_attend` placeholder 原样保留，不宣称其得到更新。不 early stop；验证每轮只记录原始-unit MAE，不驱动优化器、
调度器或检查点选择；测试集在训练阶段完全不读取。正式模型只能是轮 10 final
状态，不保存或加载验证最优状态。基线不支持恢复，失败由 sweep 在新尝试
中完整重跑，也不做 multi-种子选择或集成。10轮正式输出根为
`outputs/benchmarks/model-native-10e-v1/aifc/`。

## 审计结果与后果

对当前 system-split Stage 3 数据做了只读全量审计：21 个任务的五折和测试集共 245,614 条
任务局部行，形成 11,282 个唯一 canonical 模型视图；所有视图均可解析并完成分片，
failure 为 0。唯一视图共含 251,297 个原子，其中 5,618 个使用 official unknown motif，原子级
unknown 比例为 2.236%；1,548 个唯一视图至少含一个 unknown 原子。数字只描述当前数据
快照，不是冻结常量；每次正式训练/评估仍按实际来源身份在检查点和
评估元数据中记录逐角色分片审计。

该基线是对公开 alpha architecture 的审计化恢复，不宣称复现作者论文的集成或
性质数据集结果。10轮末轮状态策略、partner ordered concat 和非 T/P 条件是
ILUME adaptation，必须与作者原生实验设置区分描述。
