# ADR-0086：四项 scalar simulation baseline 与 Stage3 独立榜单

- 状态：Accepted
- 日期：2026-09-28
- 扩展：各模型 baseline 合同及 [ADR-0084](0084-stage3-simulation-phase2-phase3.md) 的最终模拟预测能力；[ADR-0085](0085-retire-stage2-home-evaluation.md) 的 Stage2 evaluator 退役保持有效。

## 决定

十个正式 baseline 新增 heat of vaporization、thermal expansion、HOMO、LUMO，各任务独立训练一次并使用 catalog 原有 train/valid/test。partial charge 排除，内部单任务 MLP 消融及历史 split 配置仍只训练实验任务。模型不跨任务共享更新；实验任务配方不变。神经模型保持各自现役 10-epoch recipe，XGBoost 保持 1000 trees，固定末轮发布；validation 不影响训练或选模。

baseline train/evaluate 增加 `--benchmark simulation`，拒绝 fold/ensemble；sweep 默认执行全部启用任务，`--benchmark simulation|stage3|all` 可选择域。每模型共有 100 个实验 fold training job 和 4 个模拟 training job，模拟训练后分别运行 valid/test。模拟 task 独立输出到 `simulation/<task>/`，新 root 使用 `outputs/benchmarks/v4/<model>`；旧输出不迁移、不覆盖。任务、registry、来源 hash、train-only scaler、seed 与原模型/优化 recipe 进入训练身份。

两项 thermophysical 使用 cation/anion 和 temperature，HOMO/LUMO 使用单个 SMILES、无 condition。ILTransR/AIFC 新增单分子 view，沿用现有共享 encoder/head。AIonopedia 沿用完整图文模型，将单分子放入 solute graph 槽位，其余图为现有 empty graph，temperature 输入为 0，使用现有 7-row topology embedding 的 id 4 和 `molecule <SMILES>` prompt；该输入合同进入身份，不新增参数或伪造另一种离子。既有实验拓扑路径不变，不改为 pure-Qwen。

Stage3 evaluate 增加 `--domain experimental|simulation`，默认 experimental。simulation 仅加载 Full 或 no-Stage1 的五个 `three_phase_final.pt`，valid/test 均要求 ensemble：同一 split 上每折先用来源 scaler 还原到原单位，再平均预测，最后计算指标。拒绝 fold/epoch/task-subset selector 和无 simulation owner 的产物。valid 复用 Stage2 prepared features，test 按 catalog 原始数据与完整产物特征快照构建输入；QC 失败或缺失行硬失败，不静默筛选或取交集。

## 比较与产物

报告原单位 MAE/RMSE/R² 和 normalized MAE/RMSE。统一报告尺度为对应原始 train split 的 population std，零方差用 1；常量 evaluation target 的 R² 为 null。模型训练内部原有 scaler 公式保持不变（例如 AIonopedia sample std、模型特定 retained train rows）；只在原单位预测上统一报告尺度。headline 为四任务等权 macro normalized MAE。

比较 identity 使用 `simulation_property`，绑定四项任务、原始 train/evaluation 文件 SHA、完整 source-row 集合和尺度，不绑定模型特定 encoder、训练 scaler、fold 或 ensemble 以便 ILUME 与独立 baseline 比较。这些模型来源及五模型/单模型预测协议另进入各自 evaluation/reporting identity。独立 `simulation_{validation,test}` leaderboard、metrics、task MAE/rank 表与 JSON sections 发布；summary snapshot schema 升为 v4。模拟结果不参与实验 leaderboard/wins/radar/scatter，不恢复 Stage2 reporting。

## 验证

仅用临时小数据检查四任务数据、HOMO/LUMO audit、无 condition 单分子适配、单次作业、固定训练预算与禁止 fold/ensemble，验证完整重载与迁移、五 final 原单位 ensemble、行集合与尺度、simulation/experimental 榜单隔离、缺失/篡改产物拒载。执行 pytest、入口 help、compileall、文档链接和 diff 检查；高级模型以 adapter 合同与环境校验为边界，不下载模型、安装环境或运行正式 prepare/train/evaluate。
