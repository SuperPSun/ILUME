# 完整22任务 baseline 报告

本入口按[ADR-0123](adr/0123-complete-baseline-22-task-report.md)组装十个baseline的22项实验与三项模拟结果，不包含ILUME。读取已有原单位预测重算指标，不调用模型，不训练、不改写历史身份或结果。

2026-11-11起，XGBoost按[ADR-0126](adr/0126-xgboost-lightweight-variants.md)只保留500棵/深度4；原1000棵产物已按确认清单删除。2026-11-10生成报告是包含原版的历史快照，指标不改写。新报告入口只选择`lightweight/ecfp_xgboost_500_d4/`，须先完成113次新训练、评估与来源审计，并在报告YAML的`source_audits`登记新审计文件及SHA；当前尚未训练，不能重新发布完整榜单，不回退旧原版。

## 生成

在4090仓库根使用主环境执行；报告目录必须不存在。`summary_22.yaml`显式固定来源审计及路径映射的SHA，相关文件与训练产物均须在本地。

```bash
cd /data/pengs/ILUME
conda activate ilume
PYTHONPATH=src:. python scripts/benchmarks/summarize.py \
  --baseline-config configs/benchmarks/FLftv4_3407/reports/summary_22.yaml \
  --output outputs/benchmarks/reports/FLftv4_3407_22_20261110
```

该日期目录已用于本次交付。以后重新生成须换一个不存在的输出目录；不能覆盖已有报告。输入来源变化时先重新审计，禁止为了消除错误直接修改旧身份或绕过SHA。

## 查看

- `overview.md`：实验验证、实际有测试行的实验测试、三模拟验证、三模拟测试，四张独立榜单及归一化说明。
- `stage3_validation/{overall,tasks,folds}.csv`：完整22任务，逐折指标、五折均值和标准差、任务排名及宏平均。
- `stage3_test/{overall,tasks,folds}.csv`：有测试数据的任务，五模型原单位集成结果；无测试集任务列在summary的 `coverage.experimental_no_test`。
- `simulation_validation/`、`simulation_test/`：热膨胀、HOMO、LUMO逐任务指标及三任务等权汇总，与实验完全分开。
- `source_manifest.json`：250组任务的原始训练身份、当前与原路径、评估身份和所有已核验文件的大小/SHA。迁移仅改变定位，不改原身份。
- `summary.json`、`manifest.json`：完整指标、覆盖、新比较身份及报告文件完整性清单。

验证尺度为每折其余四折完整训练目标总体标准差，测试尺度为五个训练尺度的平均；模拟使用原训练集总体标准差。均保留原始重复行，零方差取1。宏平均先按任务内折均值计算，再对任务等权；不同性质的原单位MAE不直接相加。

## 来源边界

固定来源为10项已完成定向补训＋其余12项实验＋三项模拟。其中MLP、XGBoost、ILBERT后15项使用本轮重训，其余七模型使用已经审计通过的历史来源。当前模型根可能同时存在不同协议的任务与旧根summary；入口只使用明确批准的任务/折来源，不扫描根summary拼表，不按结果好坏选择attempt。

归档上移后，原 `completion_22/` 已移除，任务位于 `FLftv4_3407/<model>/`；本轮sweep记录位于 `FLftv4_3407/sweeps/completion_22/`，缓存位于 `FLftv4_3407/cache/completion_22/`。训练时冻结的原路径继续保留，报告通过SHA绑定的映射定位。

本次重新核验来源身份、数据/配方、权重和完整有序预测行集合。原生环境、scaler和预处理的深度核验承接SHA固定的既有来源审计，不重跑模型特征计算、原生分词或模型前向。现役ILUME的训练和评估完成后，仍需单独适配比较入口，不能把本报告当作包含ILUME的论文比较报告。
