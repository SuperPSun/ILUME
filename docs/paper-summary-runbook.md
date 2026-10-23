# 论文结果汇总

本入口按[ADR-0103](adr/0103-paper-summary-common-scale.md)生成独立论文报告。
训练和评估由用户完成；汇总只读取现有完整预测，不执行模型前向，也不修改训练产物。

## 前提

- 在4090仓库根执行，主环境为`ilume`。若尚无绘图依赖，先自行执行
  `python -m pip install -e '.[reports]'`；脚本不自动安装依赖。
- 十模型完整回传至`outputs/benchmarks/v4_3407/`，保留原相对路径。
- 明确选定ILUME运行根，例如`outputs/v4/stage3/optimized_3407/`。
  该目录必须已有`valid/fold1`至`valid/fold5`的完整评估和预测。训练完成本身不足以汇总。
- `--with-test`还要求同一运行根的`test/`已完成五模型原单位预测集成。
  Stage3评估命令见[v4手册](v4-runbook.md#最终优化配方3407)。

## 命令

仅汇总全部24项实验的五折验证：

```bash
python scripts/benchmarks/summarize_paper.py \
  --config configs/v4/stage3/optimized_3407.yaml \
  --ilume outputs/v4/stage3/optimized_3407 \
  --baselines outputs/benchmarks/v4_3407 \
  --output outputs/reports/paper-3407-valid-001
```

验证与现有10项测试任务一起发布：

```bash
python scripts/benchmarks/summarize_paper.py \
  --config configs/v4/stage3/optimized_3407.yaml \
  --ilume outputs/v4/stage3/optimized_3407 \
  --baselines outputs/benchmarks/v4_3407 \
  --with-test \
  --output outputs/reports/paper-3407-valid-test-001
```

输出目录已存在即拒绝。再次运行使用新的编号，不删除或覆盖历史报告。
不把`outputs/HPO`整体作为输入，也不把不同split、候选或study的折凑成五折。

## 输出

| 文件 | 内容 |
|---|---|
| `overview.md` | 中文口径说明、解释边界、全部模型与逐任务增益表 |
| `summary.json` | 共同尺度结果、新比较身份、源评估身份和输入指纹 |
| `manifest.json` | 本报告各文件的大小和SHA256 |
| `valid/overall.csv` | 十个基线及ILUME的总体共同尺度nMAE、平均名次、第一/前三数 |
| `valid/folds.csv` | 每模型每任务每折的重算指标、完整训练数据参考行数/体系数、共同尺度和评估来源 |
| `valid/tasks.csv` | 全部任务的五折平均MAE/RMSE/R²、共同尺度nMAE及折间标准差 |
| `valid/gains.csv` | 每任务相对最强基线的原单位MAE降低率，包含负值 |
| `valid/performance_profile.csv` | 原单位MAE相对全部十一模型最优值的完整覆盖曲线 |
| `valid/{overall,task_gains,performance_profile}.{svg,pdf}` | 总体误差、完整逐任务增益及性能覆盖图 |
| `valid/ilume_scatter/*.{svg,pdf}` | 所有实验任务的ILUME预测散点，坐标共享范围、显示y=x |

启用`--with-test`时，`test/`发布相同类别输出，仅覆盖实际有非空测试集的任务。
验证散点合并五个留出折；大样本散点图的点层栅格化，坐标和文本仍为矢量。
图内使用英文标签，中文说明保留在overview。

## 指标与使用边界

总体归一化MAE使用完整训练数据的共同`ddof=0`标准差，先五折等权再24任务等权。
测试的共同尺度使用五折训练标准差的平均。原来的统一summarizer仍按
[ADR-0031](adr/0031-stage3-summary-normalization-relaxation.md)允许模型各自scaler，两个
报告的总体数字可能不同，应明确标注采用哪种口径。

任务排名、第一/前三数和性能覆盖使用原单位MAE；总体nMAE与平均名次可能给出不同
排序。相对“最强基线”表示每任务十模型的最优结果集合。最优误差为0时，不伪造有限
误差比或百分比增益。所有负增益均保留，不自动挑选有利任务。

3407来自跨划分验证选择；五折离散度不是独立随机种子重复或显著性。hydration仍为
random/cv1、无测试集；其他任务按正式配置。消融应与同split的配套Base比较，条件
曲线和计算成本需要独立测量。共同尺度不表示预训练数据或HPO预算相同，应披露各模型
训练信息。`reference_train_rows/systems`表示完整训练数据参考规模，不是MoLFormer等
模型经过自身筛选后实际保留的训练规模。

三项模拟及ILUME共同热膨胀比较继续按[基线手册](baseline-catalog-v4-runbook.md#汇总和回传)
使用原入口生成独立榜单，不加入实验总体指标。权重完整性核验继续按
[回传规则](v4-runbook.md#跨设备训练产物回传)执行；本入口核验报告、数据、预测和声明的
检查点清单，不重新加载权重或重算全部权重SHA。
