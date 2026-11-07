# ADR-0114：以 G2_C01 作为主模型

## 决定与范围

已接受（2026-11-04）。研究者指定较小的 HoME `G2_C01` 为后续主模型，并明确沿用 `optimized_3407` 的 Stage2 `007` / Stage3 `032` 训练超参。正式入口继续为 `configs/v4/stage{1,2,3}/base.yaml`，不增加另一套正式 Base。

Stage2/3 Base 使用 `data/3407` 及同目录 catalog，数据划分 seed 为3407，训练 seed 为42（Stage3按42+fold派生）。Stage1维持[ADR-0111](0111-rdkit-2026-09-1.md)、[ADR-0112](0112-v4-explicit-rdkit-211-selection.md)、[ADR-0113](0113-v4-descriptor-validity-ring-log-review.md)的现役配置及来源：RDKit2026.09.1、211列、Phi适用性mask与环计数log1p、learned1024/atom512。新配方需要匹配的新Stage1产物，不读取旧表示替代。

## 分组与容量

G2保持原任务分组，但将两项电位任务归入 `phase_stability`，不建立无活跃任务的 `electrochemical` GROUP。C01使用一个GLOBAL专家，GROUP专家数相对C05各减少一个、最低为一个；L1/L2同步采用既有专家数配置。

| owner | 专家数量 | 隐藏比例 |
|---|---:|---:|
| GLOBAL | 1 | 2.0 |
| GROUP:transport | 1 | 1.5 |
| GROUP:thermophysical | 1 | 1.5 |
| GROUP:phase_stability | 2 | 1.5 |
| GROUP:dielectric_optical | 1 | 0.75 |
| GROUP:solvation | 2 | 1.5 |
| GROUP:biological | 1 | 0.5 |

Stage2使用其中thermophysical/solvation两组，与Stage3严格匹配。每任务一个PRIVATE专家，PRIVATE/Tower/FiLM宽度与dropout、条件通路、flat gate算法及其他结构均沿用原优化配方；gate输出宽度由活跃GLOBAL/GROUP/PRIVATE候选数量自动决定。

## 训练和身份

- Stage2沿用007：10轮、LR1e-4、weight decay0.01、逻辑batch/微批256；Stage1永久冻结。
- Stage3完整沿用032的GLOBAL/GROUP预算、PRIVATE学习率及轮数、任务覆盖、优化器、采样、loss和末轮发布。没有混入WeakOpt三个后续搜索的参数。
- 保持[ADR-0096](0096-stage3-batch-sample-weighted-gradients.md)、[ADR-0097](0097-stage2-batch-sample-weighted-gradients.md)的样本加权、owner归属与模拟隔离，不修改Stage科研算法或放宽加载。
- Stage2/3准备、训练及评估仍配对到 `outputs/v4/stage{2,3}/base/`；准备审计明确写在各自 `prepare/data_metadata.json`。已有目录不能覆盖；如需复用标准路径，先按精确清单另行确认归档。
- 新旧专家数量/分组及Stage1特征来源均进入既有训练身份。G2_C05的Stage2 final、Stage3锚点、检查点与预测不得作为G2_C01产物加载或改名复用。必须重新训练配对Stage2及Stage3五折。

## 历史结果与证据边界

历史split42两折容量搜索中，G2_C01记录可训练参数164,117,502、Q=0.298914155；G2_C05为209,810,641、Q=0.296411853，参数减少21.78%、Q增加0.84%。该参数口径不含冻结Stage1。它们不是新211维表示、3407五折或032配方下的已完成结果；主模型新参数量以新产物实际记录为准，不把历史指标移植为新模型性能。

保留 `configs/HPO/selected.yaml` 的原G2_C05选择及各轮静态配置、generation绑定和全部历史输出；`configs/v4/stage{2,3}/optimized_3407.yaml`继续表示旧G2_C05优化配方。旧HPO生成入口改读冻结的 `configs/HPO/HoME/stage{2,3}/G0_C05.yaml` 模板，确保同一个候选编号仍生成相同配方。

既有 `configs/ablations/v4_3407/` 仍是旧G2_C05的配对消融，不能解释为新G2_C01主模型的配对证据。基线仍使用原注册表authority与自身训练合同，不随主模型容量切换重写。论文汇总默认Stage3配置改为Base，历史结果仍需显式传原配置和原运行根；不改写旧报告。

## 验证与运行

后续[ADR-0117](0117-base-22-tasks-updated-solvation.md)已根据用户确认，将现役Base对齐为22实验＋两模拟辅助，并按更新数据将solvation声明改为3684体系。下面的24任务缺失说明保留为本决策实施时的历史核查；新Base不再请求已排除的两个任务，严格来源与体系数检查仍保留。

检查新Base相对旧优化配方仅改变专家数、移除空GROUP及Base路径；任务、训练数值及split保持一致。验证Stage2/3配对、旧容量迁移拒绝及旧HPO全部200份HoME YAML再生成一致；执行pytest、入口help、编译、Markdown链接及diff检查，只使用临时数据。

实现时本地3407 catalog缺少 `experiment/equilibrium_pressure` 与 `experiment/water_activity_coefficient`，完整24任务注册表解析仍拒绝。另外，solvation的catalog体系数为3684，原优化配方声明3611，现有一致性校验也会拒绝。需要先确认配套数据版本，不能静默删任务、改计数绕过或用历史预测补齐；正式训练前须恢复匹配的数据与catalog，或另行确认数据协议变更。运行命令见[v4手册](../v4-runbook.md#g2_c01-主模型)，本次不启动正式prepare、训练或评估。
