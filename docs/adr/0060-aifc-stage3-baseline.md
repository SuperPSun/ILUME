# ADR-0060：AIFC Stage 3 Baseline

- 状态：Accepted
- 日期：2026-09-15
- 范围：AIFC baseline；独立于现役 Stage 1/2/3 数值合同
- 修订：2026-09-19 将固定训练预算统一为 10 epochs

## 目标与上游边界

AIFC（Attentive Ionic Fragment Contribution）代表 ionic-liquid-specific fragment prior、
GNN 与 attention 路线。实现以 `2022kaikaili/ILs-AIFC` commit
`569bc338a1dbe8aad9770a5562e27e23f9ab3cdb` 的公开模型为上游 source identity，但不照搬
作者的 dataset split、Bayesian optimization、validation-driven scheduler、early stopping 或
100-model ensemble。

Fragment dictionary 必须使用作者历史 commit
`23973da709212d1e13fd4b8e0968c8109f78fa75` 的 `My_fragments.csv`：Git blob
`90d8bafc2eb07835854e183df48a637d97b8c09b`，SHA-256
`ccc5cfefe87ebf3e1d98762eb1c58170bc00c9f7cfd12b6c9ce23fa957ccdf95`。文件包含 100 个
唯一 `First-Order Group / SMARTs / Priority` 条目；加载器校验完整文件 hash、列、条目数和
SMARTS 可解析性，不允许替换或扩展 dictionary。未被 SMARTS 覆盖的原子保持作者语义：每个
原子形成一个 motif，fragment type 是全零 100D 向量。

## 模型与拓扑合同

正式实现用 PyTorch `2.9.0+cu128` FP32 等价复刻作者 DGL 模型：46D atom features、12D
directed bond features、fragment-level AttentiveFP、motif/junction AttentiveFP、multi-head
attention aggregation 与两层 regression MLP。固定图和确定性参数在 DGL 1.1.2 CPU 与正式
PyTorch 实现间的 prediction、system representation 和 motif attention 最大绝对误差必须不超过
`1e-5`。作者 DGL alpha 在整个 batch 没有 fragment/motif edge 时无法产生 message context；
现代实现只对此情况补零 context，从而保留节点自身 GRU 更新并避免合法单原子体系失败。

Stage 3 registry/task catalog 是 task、slot 顺序、conditions、split 和 target 的唯一 authority：

- 普通 `(cation, anion)` 生成一个 canonical `cation.anion` AIFC graph。
- `(cation, anion, solute)` 分别编码 cation、anion、solute。
- `(solute, solvent)` 分别编码 solute、solvent。

所有 component 在同一个 batch 中只调用一次共享 AIFC encoder；每个 component 的 system
representation 严格按 registry slot order concat。Partner components 之间不增加 cross-attention、
message passing 或其他交互模块。Registry conditions 在 ordered component representation 后按
原顺序 concat，整体经过作者原有 ReLU 后进入 regression MLP。Temperature/pressure 恢复作者源码中被注释的 concat 设计；
frequency/wavelength 采用相同 scalar concat，明确登记为 ILUME-specific extension。只有
authoritative `condition_columns` 实际存在的条件才进入模型。

公开、可信的同性质 architecture config 仅映射到 viscosity（fallback architecture 本身）和
thermal decomposition temperature（hidden 208、dropout 0.598）；其余任务不按“相近性质”
借用配置，统一使用 hidden 128、1 head、dropout 0、depth 3、layers 3，且 residual、batch norm、
layer norm 均关闭。所有 task/fold 从随机初始化训练，不存在 pretrained 或 property-specific
checkpoint。

## Scaling、训练与选择合同

每个 fold 的 target 与所有 registry conditions 都只在该 fold train rows 上拟合 population
z-score（`ddof=0`）；valid/test 复用这些 statistics。恒定 condition 保存原 mean、scale 设为 1，
因此 normalized value 为 0。模型以 normalized target 的 MSE 训练；evaluation inverse-transform
prediction 后调用现役 ILUME raw-unit metrics。

所有 task/fold 固定使用一个 seed `1000` 的模型：Adam、learning rate `1e-3`、默认
betas/epsilon、weight decay 0、constant scheduler、batch size 64、FP32、10 epochs。Forward
路径中的 fragment encoder、attention 与 regression head 参数均参与训练；作者定义但未在 forward
调用的 `motif_attend` placeholder 原样保留，不宣称其得到更新。不 early stop；validation 每 epoch 只记录 raw-unit MAE，不驱动 optimizer、
scheduler 或 checkpoint selection；test 在训练阶段完全不读取。正式模型只能是 epoch 10 final
state，不保存或加载 validation-best state。Baseline 不支持 resume，失败由 sweep 在新 attempt
中完整重跑，也不做 multi-seed selection 或 ensemble。10-epoch 正式输出根为
`outputs/benchmarks/model-native-10e-v1/aifc/`。

## 审计结果与后果

对当前 system-split Stage 3 数据做了只读全量审计：21 个 task 的五折和 test 共 245,614 条
task-local 行，形成 11,282 个唯一 canonical model view；所有 view 均可解析并完成 fragmentation，
failure 为 0。唯一 view 共含 251,297 个原子，其中 5,618 个使用 official unknown motif，原子级
unknown ratio 为 2.236%；1,548 个唯一 view 至少含一个 unknown atom。数字只描述当前数据
snapshot，不是冻结常量；每次正式 train/evaluate 仍按实际 source identity 在 checkpoint 和
evaluation metadata 中记录逐 role fragmentation audit。

该 baseline 是对公开 alpha architecture 的审计化恢复，不宣称复现作者论文的 ensemble 或
property dataset 结果。10-epoch final-state policy、partner ordered concat 和非 T/P conditions 是
ILUME adaptation，必须与作者原生实验设置区分描述。
