# ADR-0085：退役 Stage2-HoME 独立评估与榜单

- 状态：已接受
- 日期：2026-09-28
- 替代：[ADR-0083](0083-stage2-home-full-artifact-evaluation.md) 的独立 evaluator/报告部分；完整产物合同保持有效。

> 后续边界：[ADR-0086](0086-scalar-simulation-baselines-and-reporting.md) 新增 Stage3 final 的四项标量模拟比较，不恢复 Stage2 evaluator。

## 决定

五项模拟任务已在正式完整 ILUME 与 no-Stage1 的 Stage3 Phase2/3 继续训练并进入最终模型（[ADR-0084](0084-stage3-simulation-phase2-phase3.md)）。Stage2 不再提供独立 evaluate 入口或验证/测试集榜单，删除 `scripts/stage2/evaluate.py`、对应实现与专属报告测试。统一 summarizer 忽略历史 Stage2 evaluate 输出，只读取 Stage3 与独立基线报告，不生成 Stage2 CSV、比较或 JSON section；汇总快照结构定义升为 v3。

Stage2 仍发布完整九任务 `stage2_final.pt`、清单和 `stage2_encoder.pt`。保留 Stage1 backbone、ObjectEncoder、全部 GLOBAL/GROUP/PRIVATE、路由/towers、原子 adapter、特征/scaler 快照与 `SimulationHoME.predict`；完整重载、来源 SHA、owner 集合及状态hash 校验保持原样。Stage3 初始化范围、Phase2 共享 GROUP 模拟任务权重 0.1、Phase3 训练和独立模拟验证均不改变。Stage3 evaluator/榜单仍只汇总实验任务；本次不新增 Stage3 模拟任务测试集报告入口。

此次仅删除评估维护面，不改变模型数值、训练配方或检查点/final kind；已有 Stage2/3 模型产物仍遵循原兼容合同。旧输出不移动、不删除、不覆盖；README 重训链仅保留 Stage2 数据准备/训练，随后 Stage3 数据准备/训练/评估。

## 验证边界

临时数据检查完整 Stage2 重载与 Stage3 迁移、模拟分支训练和恢复；报告测试确认历史 Stage2 输入不进入榜单且不生成 Stage2 文件。运行完整测试、入口 help、compileall、文档链接和 diff 检查，不启动正式训练或评估。
