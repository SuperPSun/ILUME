# ADR-0086：四项标量模拟基线与 Stage3 独立榜单

- 状态：已接受
- 日期：2026-09-28
- 扩展：各模型基线合同及 [ADR-0084](0084-stage3-simulation-phase2-phase3.md) 的最终模拟预测能力；[ADR-0085](0085-retire-stage2-home-evaluation.md) 的 Stage2 evaluator 退役保持有效。

## 决定

十个正式基线新增汽化热、热膨胀、HOMO、LUMO，各任务独立训练一次并使用任务目录原有训练集/验证集/测试集。partial charge 排除，内部单任务 MLP 消融及历史 split 配置仍只训练实验任务。模型不跨任务共享更新；实验任务配方不变。神经模型保持各自现役 10轮配方，XGBoost 保持 1000 trees，固定末轮发布；验证不影响训练或选模。

基线训练/评估增加 `--benchmark simulation`，拒绝折/集成；sweep 默认执行全部启用任务，`--benchmark simulation|stage3|all` 可选择域。每模型共有 100 个实验折训练作业和 4 个模拟训练作业，模拟训练后分别运行验证集/测试集。模拟任务独立输出到 `simulation/<task>/`，新根目录使用 `outputs/benchmarks/v4/<model>`；旧输出不迁移、不覆盖。任务、注册表、来源 hash、仅训练集 scaler、种子与原模型/优化配方进入训练身份。

两项 thermophysical 使用阳离子/阴离子和 temperature，HOMO/LUMO 使用单个 SMILES、无条件。ILTransR/AIFC 新增单分子视图，沿用现有共享编码器/预测头。AIonopedia 沿用完整图文模型，将单分子放入 solute graph 槽位，其余图为现有 empty graph，temperature 输入为 0，使用现有 7-行拓扑 embedding 的 id 4 和 `molecule <SMILES>` prompt；该输入合同进入身份，不新增参数或伪造另一种离子。既有实验拓扑路径不变，不改为 pure-Qwen。

Stage3 evaluate 现支持 `--domain all|experimental|simulation`，默认all：同一次调用评估同一split的实验与模拟，保留单域选择。联合模式实验验证仍逐折、模拟验证仍五模型集成；实验输出位置保持原约定，模拟输出位于同根simulation子目录。各域独立身份、预测与报告结构不变，任务集按ADR-0095的现役两任务或显式历史协议校验。模拟仅加载完整或 no-Stage1 的五个 `three_phase_final.pt`，验证集/测试集均要求集成：同一 split 上每折先用来源 scaler 还原到原单位，再平均预测，最后计算指标。单独模拟评估拒绝折/轮/任务-subset selector 和无模拟 owner 的产物；联合模式的折/任务selector仅作用于实验，模拟始终完整五模型集成，轮selector被拒绝。验证集复用 Stage2 准备产物特征，测试集按任务目录原始数据与完整产物特征快照构建输入；QC 失败或缺失行硬失败，不静默筛选或取交集。

## 比较与产物

报告原单位 MAE/RMSE/R² 和归一化MAE/RMSE。统一报告尺度为对应原始训练集 split 的总体标准差，零方差用 1；常量评估目标的 R² 为 null。模型训练内部原有 scaler 公式保持不变（例如 AIonopedia 样本 std、模型特定 retained 训练集行）；只在原单位预测上统一报告尺度。headline 为四任务等权 macro 归一化MAE。

比较身份使用 `simulation_property`，绑定四项任务、原始训练/评估文件 SHA、完整来源-行集合和尺度，不绑定模型特定编码器、训练 scaler、折或集成以便 ILUME 与独立基线比较。这些模型来源及五模型/单模型预测协议另进入各自评估/报告身份。独立 `simulation_{validation,test}` 榜单、metrics、任务 MAE/rank 表与 JSON sections 发布；汇总快照结构定义升为 v4。模拟结果不参与实验榜单/wins/radar/scatter，不恢复 Stage2 报告。

## 验证

仅用临时小数据检查四任务数据、HOMO/LUMO 审计、无条件单分子适配、单次作业、固定训练预算与禁止折/集成，验证完整重载与迁移、五 final 原单位集成、行集合与尺度、模拟/实验榜单隔离、缺失/篡改产物拒载。执行 pytest、入口 help、compileall、文档链接和 diff 检查；高级模型以 adapter 合同与环境校验为边界，不下载模型、安装环境或运行正式数据准备/训练/评估。
