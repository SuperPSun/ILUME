# ADR-0041：v2 Stage 3 三阶段固定预算搜索

- 状态：已退役
- 日期：2026-09-03
- 退役日期：2026-09-06
- 修订：ADR-0020 的专家数下界与 Stage 3 准备产物身份边界

> 2026-09-06：本 ADR 定义的搜索入口、报告入口、配置、实现与 Optuna 依赖均已移除。
> 下文只保留历史设计与结果解释边界，不再构成可运行合同；现役 v2 Stage 3 直接使用
> `configs/v3/stage3/base.yaml`。

## 背景

Global-RDKit v2 已固定 Stage 1 Base 与 Stage 2 Base。新的搜索只优化 Stage 3 对困难性质的
任务 organization、专家 specialization 和任务-aware loss weighting。搜索必须可恢复、可审计，
且不能因中间结果改变总试验数。旧 Capacity v1 HPO 继续冻结在
`configs/experiments_v1`，不得与本研究共享 study、检查点或输出。

## 决定

1. 搜索入口为 `scripts/stage3/search.py`，配置为
   `configs/v2/stage3/search.yaml`。A/B/C 使用三个独立 Optuna study；每个试验固定训练
   fold1/2、20 轮，即16 联合训练 + 4 仅PRIVATE更新精调。A/B/C 分别为50、30、20个
   唯一配置，总计100 试验和200个正常折运行，不补跑fold3/4/5。
2. Search A 固定5个组-count 锚点、20个人工层级分组和25个种子-42组合分组。
   `G∈{2,3,6,9,12}` 各10个。每个分组恰好覆盖21任务且组非空；允许singleton、规模不平衡
   和随G增长的参数量。不使用自动clustering、残差相关性或表示相似度。
3. Search B 完整覆盖
   `global∈{0,1,2} × group∈{1,2,3,4,6} × private∈{0,1}` 的30个tuple。
   local、global/PRIVATE ablation、higher-capacity各10个，并确定性轮转到Search A Top-3，
   使每个grouping恰好评估10次。排名后只传递三个不同的专家 tuple。
4. `global_experts=0` 表示L1 global采用无参数身份，不创建global 门控、L1/L2 global
   专家或global 候选；`private_experts=0` 只删除PRIVATE 专家候选，继续保留
   TaskGate、FiLM、任务归一化和tower。`group_experts>=1`，因此TaskGate候选永不为空。
   参数归属、分层 PCGrad、检查点和诊断必须接受空GLOBAL 专家 block。
5. Search C 使用Top-3 grouping与Top-3 专家 tuple的九个交叉组合。前九个试验逐一运行
   Base优化参数和全1训练权重；其余11个由seeded TPE搜索pair、三项Tier共享训练权重、LR、
   dropout和权重衰减。范围固定为Tier 权重 `[1,5]`、LR log
   `[1e-4,5e-4]`、dropout `[0.05,0.20]`、权重衰减 log `[1e-3,3e-2]`。
   其他Stage 3 Base训练参数不搜索。
6. 任务权重仍只在联合训练 phase的PCGrad投影后聚合中生效。ADR-0027定义的精调继续
   不使用任务/组权重，不修改仅PRIVATE更新选择语义。
7. 排名指标为每个折的
   `sum(task normalized MAE * fixed evaluation weight) / 33`，再对fold1/2算术平均。
   Tier1权重3、Tier2权重2、Tier3权重1.5、其他任务权重1；该评价权重与训练任务权重
   分离。精确并列依次用原macro-任务归一化MAE、合计GPU seconds和试验编号裁决。
8. 每阶段至少需要三个成功试验。失败配置最多原样重试一次；第二次失败后记为failed，
   不生成替补试验。Search C发布完整排名、Top-3、唯一胜出方案和冻结Base YAML，不自动推广
   S/L/XL，也不运行测试集。

## 准备数据身份修订

Object-backed v2主线的Stage 3 准备产物合同升级为v2。准备产物注册表与身份只包含会改变物化数据的任务目录
事实、split/repeat、拓扑/slots、partner mode、归一化、object集合和Stage 2 编码器
身份；不再包含`meta_group`、`enabled`、`task_weight`、组权重或专家拓扑。
这些模型/训练事实继续完整进入训练身份。因此本研究所有试验复用同一份v2 Base
准备产物，但任一grouping、权重或专家变化仍禁止检查点交叉恢复。

旧准备产物不静默兼容；启用本合同后必须重新执行一次Stage 3 v2 Base prepare。
Stage 1/2 产物、Stage 3源数据、归一化与数值张量格式不改变。

## 后果

- A/B是固定候选枚举，Optuna负责持久化、状态与恢复；只有C的后11个试验使用TPE。
- 每个折汇总额外记录wall/GPU seconds、峰值allocated显存、总参数量和可训练参数量。
- 搜索结果只是在两折20-轮筛选口径下的Base 配方决定，不能解释为五折正式结果或
  规模-independent结论。
- `scripts/stage3/search_report.py`只读汇总A/B/C保存的fold1/2 逐任务精调后拼接后的
  验证并生成逐试验/逐性质表格与SVG；它不重新评估、不读取测试集，也不能把
  性质对比图解释为五折确认。
- 正式prepare、100-试验搜索及最终测试集必须由用户显式执行；代码验收只运行临时小数据测试。

## 参考

- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0027：后期逐任务精调](0027-late-taskwise-refinement.md)
- [ADR-0039：Global-RDKit v2](0039-global-rdkit-v2-mainline.md)
