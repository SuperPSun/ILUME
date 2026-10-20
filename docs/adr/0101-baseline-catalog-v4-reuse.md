# ADR-0101：当前任务目录的基线补训与完整产物复用

- 状态：已接受
- 日期：2026-10-15
- 范围：十个正式基线的新 `catalog_v4` 协议；历史20实验/四模拟配置和产物保持原合同。

2026-10-20：现役运行改为[ADR-0102](0102-baseline-split3407-full-retraining.md)的3407十模型全量重训。本篇保留split42配置与复用审计的历史合同，825项补训安排不再执行；旧产物清理以0102及用户确认的精确清单为准。

## 数据与训练

新配置在 `configs/benchmarks/catalog_v4/`，使用 `data/42` 和正式Stage3 Base的任务注册表，24实验、五折；hydration为单solute加温度、random/cv1，其余按system/cv1。模拟仅thermal_expansion、homo、lumo，各一次独立训练，沿用原始训练/验证/测试集。模拟汽化热退出新协议，实验汽化焓不改名、不与模拟混合。新协议不改变Stage1/2/3主线。

各模型的结构、资产、训练预算、模型特定种子和末轮发布规则沿用各自ADR；不新增搜索或验证选模。新任务使用原注册默认配方。AIonopedia新增按任务启用的水摩尔分数projector/token及文本，数值不缩放、随机初始化；hydration使用现有单分子拓扑id4、solute graph、其余empty graph及官方温度通路。其余模型扩展单solute视图并按registry顺序接收全部数值条件，不伪造离子。

显式 `protocol: catalog_v4` 进入配置及任务训练身份。历史配置省略该字段并保留原身份。原有严格checkpoint加载校验不放宽。

## 复用与复制

九个候选任务：self_diffusion_coefficient、thermal_conductivity、heat_capacity、speed_of_sound、equilibrium_pressure、dynamic_relative_permittivity、static_relative_permittivity、transfer_organic、pec50。其他15实验从头训练。AIonopedia原权重确认丢失，24实验的120折及三模拟全部从头训练，不复用历史预测。

其他九模型逐任务核验五折来源SHA、registry输入定义、顺序/重复行、条件与目标、配方/种子、预处理/scaler、独立环境锁、资产、权重及预测完整性。ILTransR还核验全量条件统计总体。仅允许路径迁移以及不参与基线计算的新增registry字段，不修改原训练身份。数据/配方不符以及源产物缺失/损坏均记录原因并转为整个任务补训；目标文件冲突或迁移清单本身的完整性异常必须停止。

审计通过的完整任务目录复制到 `outputs/benchmarks/v4/<model>/imported/`，v3原件只读保留。复制前报告文件数、字节数、目标和冲突策略；目标相同SHA可复用，差异立即失败，不覆盖。复制后逐文件核验。每次审计生成独立 `migrations/attempt-NNN/manifest.json`，绑定当前配置、完整数据来源和复制/派生产物SHA。

新评估记录使用已验证完整行集合的原单位预测重新计算指标，在普通任务评估目录生成新比较身份及原训练身份/迁移来源。原目录内的checkpoint、配置、日志和报告不改写。完成标记不能替代身份和文件校验。

## 入口与报告

`sweep.py --prepare-reuse-from`只审计/复制，不启动训练；`--reuse-manifest`重新验证后跳过复用任务并训练其余任务。AIonopedia直接运行新配置，不使用复用参数。新运行只写v4目录。

24实验形成完整五折验证，现有10项测试集使用五模型集成，其余不造测试集。三模拟独立发布逐任务及三任务等权榜单。与现役Stage3只对显式共同任务thermal_expansion生成独立比较表，核验完整来源、行集合及尺度；不将两任务和三任务macro混排。历史四模拟和20实验报告不自动拼入新榜单。

候选全部通过时，405个旧实验折次可复用，新训练795个实验折次及30模拟，共825个作业。实现阶段仅审计、复制和小数据验证；正式训练/评估由用户启动。运行命令见[基线手册](../baseline-catalog-v4-runbook.md)。
