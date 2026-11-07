# ADR-0117：Base 对齐22项实验任务与更新的 solvation 数据

## 状态与范围

已接受（2026-11-06，用户明确确认）。仅修订现役 `configs/v4/stage3/base.yaml` 的任务集合及solvation数据规模声明。Stage1按ADR-0116使用DGT；Stage2五模拟、Stage3两模拟辅助、G2_C01容量、007/032训练超参及3407划分均不变。历史optimized_3407、HPO、消融、基线配置及结果保留原身份，不批量改写。

## 决策

- 从Base显式任务清单移除 `experiment/equilibrium_pressure` 与 `experiment/water_activity_coefficient`，不加载、训练或评估这两个任务，不新增占位数据。内置legacy/default任务表不修改。
- Base由24项实验改为**22项实验＋两项模拟辅助**。六个GROUP均保留，专家数、剩余任务归属及完整训练配方不变。
- solvation沿用任务ID、target、GROUP、交互路径、large类别、capacity、dropout和LR/epoch；`unique_systems`由3611改为当前catalog声明的3684。正式prepare继续读取更新的真实行、split及目标，不修改原始数据或catalog，不手改SHA。
- 现有严格任务与体系数检查保持；配置必须与当前数据匹配，后续不匹配仍失败。此变更是用户明确选择的新数据协议，不是为了通过校验而临时删任务或改计数。

## 身份、运行与比较

新任务集合、数据来源及解析计划自然改变prepared/training/evaluation身份；保留既有kind/format，禁止用历史24任务或旧solvation的Stage3 checkpoint恢复到新Base。Stage2配方不因本次实验任务调整而改变；当前整条链路另因Stage1 DGT和Gasteiger v2需要新表示与训练，不能混用旧来源。

Stage1旧prepared和电荷sidecar按单独确认的精确清单归档，保留全部内容；其余历史输出不删除、不覆盖。正式准备、teacher、训练和评估仍由用户启动。

新22任务macro不能直接与历史24任务macro解释为性能增益；比较须对齐任务集合、数据、fold及归一化来源。历史基线和消融若要配对新协议，需另行建立配置与运行身份，不在本次改动中迁移。

## 验证

配置精确diff仅允许删除两个任务和solvation体系数变化；验证22任务与catalog一致、GROUP及剩余recipe不变、错误体系数仍拒绝。使用临时小型DGT来源覆盖Stage2训练/迁移及Stage3三阶段、恢复、模拟PRIVATE和评估链路；执行定向/完整pytest、CLI帮助、配置round-trip、compileall、链接及diff检查。不运行正式prepare或训练。
