# ADR-0034：RDKit 2D → HoME Stage1+2 表示消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-08-31

> 2026-09-07：Stage 3 采样与联合训练裁剪已随现役 Base 由 [ADR-0046](0046-stage3-ownership-clipping-raw-sampling.md) 修订；本消融继续只替换表示。
>
> 2026-09-07：Stage 3 四阶段训练与 final 产物随现役 Base 由
> [ADR-0047](history.md#adr-0047) 修订；本消融仍不改变优化合同。
>
> 2026-09-10：现役 Stage 3 优化合同进一步由
> [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 取代为 owner-存续期三阶段训练；
> 本消融仍只改变表示。

## 背景

现役 Stage3 使用冻结的 Stage2 Object v3 512D 表示，再进入动态 HoME、
条件 FiLM、PartnerInteraction、分层 PCGrad、复合采样和末期
任务-wise 精调。为了隔离 Stage1+2 预训练表示的贡献，需要只替换该
512D 接口，不同时消融 Stage3 的共享、路由、优化或评估合同。

该实验比较：

`Stage1/2 pretrained Object representation → HoME`

与：

`RDKit 2D descriptors → minimal trainable adapter → HoME`。

它不是零 learned-表示实验，也不是完整 pipeline 基线。

## 决定

1. 配置固定为 `configs/ablations/stage1_stage2_rdkit_home.yaml`，输出根固定为
   `outputs/ablations/stage1_stage2_rdkit_home`。它不修改 `configs/v1`、Capacity 或现役
   Stage1/2/3 产物。
2. 描述符 family、名称顺序和计算复用现役 RDKit 2D pipeline。每个 held-out 折
   使用其余四折中全部 21 任务的 observation occurrence 拟合预处理；重复行保留
   权重，验证/测试集不参与拟合。
3. IL 预处理以固定 `cation || anion` 列拟合。所有单对象 primary/partner occurrence
   共同拟合另一套预处理。只删除训练中整列无有限值的列；其余非有限值按训练
   median 填充，使用 population 均值/std z-score，近常量规模设为 1，最后 clip 到
   `[-10, 10]`。
4. IL adapter 固定为 `Linear(retained_il_width, 512) → LayerNorm(512)`；single-object
   adapter 固定为 `Linear(retained_single_width, 512) → LayerNorm(512)`。不存在激活、
   dropout、额外层或任务/组/角色-specific adapter。
5. 条件不进入 adapter，继续走原 FiLM；partner 使用同一个 single-object adapter 后
   进入原 PartnerInteraction。两个 adapter 都属于 GLOBAL，联合训练 phase 沿用原分层
   PCGrad 聚合，精调与其他 GLOBAL/GROUP 一起冻结。
6. 不加载 Stage1 检查点、Stage2 编码器/Object embedding 或 Stage3 插件。RDKit
   准备产物数据、普通检查点与逐任务精调后产物使用独立 kind，身份绑定
   来源、注册表、RDKit version、描述符结构定义、五折预处理、输入宽度和模型
   状态；不得与 Object 后端交叉恢复或评估。
7. HoME、21 任务、5折、100轮 80/20 联合训练/精调、loss、优化器、调度器、
   复合 allocation、虚拟过采样、microbatch、PCGrad、检查点与评估
   合同全部沿用 Stage3 Base，不进行独立 HPO或自动 fallback。
8. 报告身份固定为 `rdkit_2d_home` / `RDKit 2D + HoME`。验证要求
   21 任务五折完整；测试集只评估任务目录中实际存在非空测试集的任务，并先逐样本平均五折
   原始预测。结果进入现有 `scripts/benchmarks/summarize.py`，不定义胜负阈值。

## 后果

- adapter 会通过 Stage3 supervision 学到最低限度的共享投影，因此结果应解释为
  “大型预训练表示”与“handcrafted descriptors + minimal 监督训练投影”
  的比较。
- 每个折的仅训练集 invalid-column mask 可以不同，因此 adapter 输入宽度和检查点
  shape 也可以不同；这些差异由折-specific 身份严格绑定。
- 两个 GLOBAL adapter 对不同拓扑的任务可能没有梯度；沿用现有 PCGrad 对缺失梯度的
  处理和既有聚合除数，不做实验特有重加权。
- 现役 Object 后端未声明 `representation` 时保持原序列化、产物、检查点和
  语义身份，不迁移已有正式结果。

## 备选方案

- 拒绝逐任务预处理：同一 object 会因任务不同得到不同输入，破坏共享
  表示语义。
- 拒绝 unique-object 预处理：它会改变现有 MLP 按训练行拟合的计权合同。
- 拒绝固定随机/PCA 投影：会人为限制 RDKit 表示能力。
- 拒绝多层 MLP/attention adapter：会重新引入一个新的复杂 learned 编码器。
- 拒绝独立基准比较 trainer：会复制 HoME、PCGrad、恢复和精调合同。

## 参考

- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0022：MLP 与 ECFP-XGBoost 基线](0022-mlp-ecfp-xgboost-baselines.md)
- [ADR-0023：统一报告](0023-unified-evaluation-reporting.md)
- [ADR-0027：后期逐任务精调](0027-late-taskwise-refinement.md)
- [ADR-0031：Stage3 汇总归一化](0031-stage3-summary-normalization-relaxation.md)
