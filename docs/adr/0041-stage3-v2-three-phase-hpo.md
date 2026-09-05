# ADR-0041：v2 Stage 3 三阶段固定预算搜索

- 状态：Accepted
- 日期：2026-09-03
- 修订：ADR-0020 的 expert 数下界与 Stage 3 prepared identity 边界

## 背景

Global-RDKit v2 已固定 Stage 1 Base 与 Stage 2 Base。新的搜索只优化 Stage 3 对困难性质的
task organization、expert specialization 和 task-aware loss weighting。搜索必须可恢复、可审计，
且不能因中间结果改变总 trial 数。旧 Capacity v1 HPO 继续冻结在
`configs/experiments_v1`，不得与本研究共享 study、checkpoint 或输出。

## 决定

1. 搜索入口为 `scripts/stage3/search.py`，配置为
   `configs/v2/stage3/search.yaml`。A/B/C 使用三个独立 Optuna study；每个 trial 固定训练
   fold1/2、20 epochs，即16 joint + 4 PRIVATE-only refinement。A/B/C 分别为50、30、20个
   唯一配置，总计100 trial和200个正常fold run，不补跑fold3/4/5。
2. Search A 固定5个group-count anchor、20个人工层级分组和25个seed-42组合分组。
   `G∈{2,3,6,9,12}` 各10个。每个分组恰好覆盖21任务且组非空；允许singleton、规模不平衡
   和随G增长的参数量。不使用自动clustering、残差相关性或表示相似度。
3. Search B 完整覆盖
   `global∈{0,1,2} × group∈{1,2,3,4,6} × private∈{0,1}` 的30个tuple。
   local、global/private ablation、higher-capacity各10个，并确定性轮转到Search A Top-3，
   使每个grouping恰好评估10次。排名后只传递三个不同的expert tuple。
4. `global_experts=0` 表示L1 global采用无参数identity，不创建global gate、L1/L2 global
   expert或global candidate；`private_experts=0` 只删除private expert candidate，继续保留
   TaskGate、FiLM、task normalization和tower。`group_experts>=1`，因此TaskGate候选永不为空。
   ownership、hierarchical PCGrad、checkpoint和diagnostics必须接受空GLOBAL expert block。
5. Search C 使用Top-3 grouping与Top-3 expert tuple的九个交叉组合。前九个trial逐一运行
   Base优化参数和全1训练权重；其余11个由seeded TPE搜索pair、三项Tier共享训练权重、LR、
   dropout和weight decay。范围固定为Tier weight `[1,5]`、LR log
   `[1e-4,5e-4]`、dropout `[0.05,0.20]`、weight decay log `[1e-3,3e-2]`。
   其他Stage 3 Base训练参数不搜索。
6. task weight仍只在joint phase的PCGrad投影后聚合中生效。ADR-0027定义的refinement继续
   不使用task/group weight，不修改PRIVATE-only选择语义。
7. 排名指标为每个fold的
   `sum(task normalized MAE * fixed evaluation weight) / 33`，再对fold1/2算术平均。
   Tier1权重3、Tier2权重2、Tier3权重1.5、其他任务权重1；该评价权重与训练task weight
   分离。精确并列依次用原macro-task normalized MAE、合计GPU seconds和trial编号裁决。
8. 每阶段至少需要三个成功trial。失败配置最多原样重试一次；第二次失败后记为failed，
   不生成替补trial。Search C发布完整排名、Top-3、唯一winner和冻结Base YAML，不自动推广
   S/L/XL，也不运行test。

## Prepared identity 修订

Object-backed v2主线的Stage 3 prepared contract升级为v2。prepared registry与identity只包含会改变物化数据的catalog
事实、split/repeat、topology/slots、partner mode、normalization、object集合和Stage 2 encoder
identity；不再包含`meta_group`、`enabled`、`task_weight`、group weight或expert topology。
这些模型/训练事实继续完整进入training identity。因此本研究所有trial复用同一份v2 Base
prepared artifact，但任一grouping、weight或expert变化仍禁止checkpoint交叉resume。

旧prepared artifact不静默兼容；启用本合同后必须重新执行一次Stage 3 v2 Base prepare。
Stage 1/2 artifact、Stage 3源数据、normalization与数值tensor格式不改变。

## 后果

- A/B是固定候选枚举，Optuna负责持久化、状态与恢复；只有C的后11个trial使用TPE。
- 每个fold summary额外记录wall/GPU seconds、峰值allocated显存、总参数量和可训练参数量。
- 搜索结果只是在两折20-epoch筛选口径下的Base recipe决定，不能解释为五折正式结果或
  scale-independent结论。
- `scripts/stage3/search_report.py`只读汇总A/B/C保存的fold1/2 taskwise-refined stitched
  validation并生成逐trial/逐性质表格与SVG；它不重新evaluation、不读取test，也不能把
  性质对比图解释为五折确认。
- 正式prepare、100-trial搜索及最终test必须由用户显式执行；代码验收只运行临时小数据测试。

## 参考

- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0027：Late Task-wise Refinement](0027-late-taskwise-refinement.md)
- [ADR-0039：Global-RDKit v2](0039-global-rdkit-v2-mainline.md)
