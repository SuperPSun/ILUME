# ADR-0034：RDKit 2D → HoME Stage1+2 representation 消融

- 状态：Accepted
- 日期：2026-08-31

## 背景

现役 Stage3 使用冻结的 Stage2 Object v3 512D representation，再进入动态 HoME、
condition FiLM、PartnerInteraction、hierarchical PCGrad、composite sampling 和末期
task-wise refinement。为了隔离 Stage1+2 预训练 representation 的贡献，需要只替换该
512D 接口，不同时消融 Stage3 的共享、routing、优化或评估合同。

该实验比较：

`Stage1/2 pretrained Object representation → HoME`

与：

`RDKit 2D descriptors → minimal trainable adapter → HoME`。

它不是零 learned-representation 实验，也不是完整 pipeline baseline。

## 决定

1. 配置固定为 `configs/ablations/stage1_stage2_rdkit_home.yaml`，输出根固定为
   `outputs/ablations/stage1_stage2_rdkit_home`。它不修改 `configs/v1`、Capacity 或现役
   Stage1/2/3 artifact。
2. descriptor family、名称顺序和计算复用现役 RDKit 2D pipeline。每个 held-out fold
   使用其余四折中全部 21 tasks 的 observation occurrence 拟合 preprocessing；重复行保留
   权重，validation/test 不参与拟合。
3. IL preprocessing 以固定 `cation || anion` 列拟合。所有单对象 primary/partner occurrence
   共同拟合另一套 preprocessing。只删除训练中整列无有限值的列；其余非有限值按训练
   median 填充，使用 population mean/std z-score，近常量 scale 设为 1，最后 clip 到
   `[-10, 10]`。
4. IL adapter 固定为 `Linear(retained_il_width, 512) → LayerNorm(512)`；single-object
   adapter 固定为 `Linear(retained_single_width, 512) → LayerNorm(512)`。不存在 activation、
   dropout、额外层或 task/group/role-specific adapter。
5. condition 不进入 adapter，继续走原 FiLM；partner 使用同一个 single-object adapter 后
   进入原 PartnerInteraction。两个 adapter 都属于 GLOBAL，joint phase 沿用原 hierarchical
   PCGrad 聚合，refinement 与其他 GLOBAL/GROUP 一起冻结。
6. 不加载 Stage1 checkpoint、Stage2 encoder/Object embedding 或 Stage3 plugin。RDKit
   prepared data、普通 checkpoint 与 taskwise-refined artifact 使用独立 kind，identity 绑定
   source、registry、RDKit version、descriptor schema、五折 preprocessing、输入宽度和模型
   state；不得与 Object backend 交叉恢复或评估。
7. HoME、21 tasks、5 folds、100-epoch 80/20 joint/refinement、loss、optimizer、scheduler、
   composite allocation、virtual oversampling、microbatch、PCGrad、checkpoint 与 evaluation
   合同全部沿用 Stage3 Base，不进行独立 HPO或自动 fallback。
8. reporting identity 固定为 `rdkit_2d_home` / `RDKit 2D + HoME`。Validation 要求
   21 tasks 五折完整；test 只评估 catalog 中实际存在非空 test 的任务，并先逐样本平均五折
   raw prediction。结果进入现有 `scripts/benchmarks/summarize.py`，不定义胜负阈值。

## 后果

- adapter 会通过 Stage3 supervision 学到最低限度的共享 projection，因此结果应解释为
  “大型预训练 representation”与“handcrafted descriptors + minimal supervised projection”
  的比较。
- 每个 fold 的 train-only invalid-column mask 可以不同，因此 adapter 输入宽度和 checkpoint
  shape 也可以不同；这些差异由 fold-specific identity 严格绑定。
- 两个 GLOBAL adapter 对不同 topology 的任务可能没有梯度；沿用现有 PCGrad 对缺失梯度的
  处理和既有聚合除数，不做实验特有重加权。
- 现役 Object backend 未声明 `representation` 时保持原序列化、artifact、checkpoint 和
  semantic identity，不迁移已有正式结果。

## 备选方案

- 拒绝逐 task preprocessing：同一 object 会因 task 不同得到不同输入，破坏共享
  representation 语义。
- 拒绝 unique-object preprocessing：它会改变现有 MLP 按训练行拟合的计权合同。
- 拒绝固定随机/PCA projection：会人为限制 RDKit 表示能力。
- 拒绝多层 MLP/attention adapter：会重新引入一个新的复杂 learned encoder。
- 拒绝独立 benchmark trainer：会复制 HoME、PCGrad、resume 和 refinement 合同。

## 参考

- [ADR-0020：Stage3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0022：RDKit MLP baseline](0022-mlp-ecfp-xgboost-baselines.md)
- [ADR-0023：统一 reporting](0023-unified-evaluation-reporting.md)
- [ADR-0027：late task-wise refinement](0027-late-taskwise-refinement.md)
- [ADR-0031：Stage3 summary normalization](0031-stage3-summary-normalization-relaxation.md)
