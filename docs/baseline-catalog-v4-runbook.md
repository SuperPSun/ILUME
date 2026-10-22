# 当前数据版本基线运行手册

现役合同见[ADR-0102](adr/0102-baseline-split3407-full-retraining.md)。全部命令在仓库根执行；主环境为`ilume`，高级模型由入口进入已有hash-lock环境，环境和资产按[README](../README.md#baselines-and-ablations)验证，不自动下载或安装。

## 3407配置、任务与结果

使用`configs/benchmarks/catalog_v4_3407/<model>.yaml`，显式`protocol: catalog_v4_3407`，数据根和catalog配对为`data/3407`。注册表authority为`configs/v4/stage3/optimized_3407.yaml`，只采用其任务与划分；各基线自己的模型、资产和训练配方保持原样。

24实验、cv1五折，hydration使用random/cv1且无测试集，其余按当前system规则；10项实验有测试集。模拟为thermal_expansion、homo、lumo，各独立一次训练。热膨胀来自data/3407/stage2；HOMO/LUMO按Stage1共用合同读取data/stage1/properties的固定原划分，不伪造3407电子数据，模型同样重训。AIFC保留seed1000，其他模型保留各自原种子；3407不替换训练种子。神经模型10轮、XGBoost 1000 trees，末轮发布，验证只报告。

所有十模型全部重训，每模型`24×5+3=123`个作业，总计**1,230个**。不传复用参数，不复制旧预测；新协议拒绝旧迁移清单。新输出为`outputs/benchmarks/v4_3407/<model>/`，MLP/XGBoost特征缓存独立使用该根的cache。

## 旧结果清理

2026-10-20，经用户确认后已完成：4090旧`outputs/benchmarks/`整体移至`trash/outputs/benchmarks-split42-20261020/`，25,370个文件、98,291,585,453字节，归档前后清单一致；szx旧`/home/sunp/ILUME/outputs/benchmarks/`已删除，32,602个文件、247,456,697,332字节，包含用户明确确认删除的历史v2结果。执行记录和逐文件清单位于`outputs/audits/baseline_3407/attempt-001/`的`cleanup_result.json`及`cleanup_{4090,szx}.tsv`。

此类清理须先报告逐文件清单、精确数量和字节数，按工作区规则确认后执行；trash目标冲突则停止，不能覆盖或合并。历史v2在4090当前目录中没有副本，此次也已纳入明确确认的删除范围。

预训练`artifacts/benchmarks/`、模型环境、原始数据、Stage/HPO/消融产物和Slurm脚本/日志继续保留。清理完成后不再向szx同步旧`outputs/benchmarks/v4/`或split42迁移清单。

## 数据与资产迁移

代码先通过Git更新；本次新增配置和入口必须在目标机就位。以下命令从4090执行，仅同步3407原始数据和预训练资产，不同步旧训练结果：

```bash
cd /data/pengs/ILUME
rsync -aHcR --partial --info=progress2 \
  ./data/3407 ./data/stage1/properties/homo ./data/stage1/properties/lumo \
  ./artifacts/benchmarks \
  szx:/home/sunp/ILUME/
rsync -acniR ./data/3407 ./data/stage1/properties/homo \
  ./data/stage1/properties/lumo ./artifacts/benchmarks szx:/home/sunp/ILUME/
```

MoLFormer还要求固定HF快照`ibm-research/MoLFormer-XL-both-10pct@361063d0ad524ef77cf39b08469f6be770dc550f`位于目标用户HF缓存，完整目录包括blobs和snapshots；szx已有缓存时仍由入口核验。其余资产使用YAML固定的仓库相对路径和SHA。不要修改hash以迁就不匹配的副本。

## szx：gpu05提交

两份sbatch位于`/home/sunp/slurm/baseline/`：

| 文件 | 资源与范围 |
|---|---|
| `baseline-catalog-v4-gpu.sbatch` | gpu_6000D、gpu05；9项数组、每项1卡/8 CPU/64G、最多4项并发、7天 |
| `baseline-catalog-v4-xgboost.sbatch` | 同节点8 CPU/32G，不占GPU；单独运行XGBoost |

GPU数组索引依次为`0 llasmol / 1 aionopedia / 2 spmm / 3 molformer / 4 ilbert / 5 iltransr / 6 aifc / 7 dmpnn / 8 mlp`。每个模型内部max-workers1；脚本不传`--devices`，保留Slurm设置的CUDA_VISIBLE_DEVICES。资源空闲数量会变化，Slurm负责排队。配置和数据同步、旧产物清理完成后在szx执行：

```bash
cd /home/sunp/slurm/baseline
sbatch baseline-catalog-v4-gpu.sbatch
sbatch baseline-catalog-v4-xgboost.sbatch
squeue -u sunp
```

日志在`/home/sunp/slurm/logs/baseline-v4-3407-*`，作业状态在`/home/sunp/ILUME/outputs/benchmarks/v4_3407/<model>/status.tsv`。单独重交某个GPU模型可用`sbatch --array=索引 baseline-catalog-v4-gpu.sbatch`，需先确认此前同模型作业已结束；不要并发运行同一模型的两个sweep。没有训练恢复，中断作业在新attempt完整重跑，已完成作业经严格核验后跳过。

直接运行示例（在已分配GPU的节点）：

```bash
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/catalog_v4_3407/aionopedia.yaml \
  --output outputs/benchmarks/v4_3407/aionopedia \
  --max-workers 1
```

其他模型只替换模型名，XGBoost同样使用max-workers1且YAML固定n_jobs8。不对同一完整sweep根混用`--benchmark all/stage3/simulation`；单任务调试使用独立输出。数据审计位置从data_root派生。

### ILTransR三项模拟的SP2D导入错误

ILTransR按合同固定RDKit 2022.03.2。旧版Stage2包入口在读取模拟任务注册表时提前导入Stage1图特征，因该环境没有`HybridizationType.SP2D`而使三项模拟在训练初始化前失败。现已将Stage2的冻结编码器公开导出改为按需加载，注册表可独立导入；环境锁、图特征映射和科研身份保持原合同。

更新仓库并确认此前ILTransR作业已退出后，可重新提交`sbatch --array=5 /home/sunp/slurm/baseline/baseline-catalog-v4-gpu.sbatch`。沿用原完整sweep及输出根，严格校验并跳过已完成的120个实验训练作业，在新attempt中补训三项模拟及其评估。

## 汇总和回传

模型24实验及三模拟完整后再发布完整榜单。只扫描新的3407输出及相同split的Stage3评估，不读取trash或旧split42报告：

```bash
python scripts/benchmarks/summarize.py \
  --input outputs/benchmarks/v4_3407 \
    outputs/v4/stage3/optimized_3407/valid \
    outputs/v4/stage3/optimized_3407/test \
  --output outputs/reports/baseline-catalog-v4-3407
```

Stage3路径替换为实际已完成目录；只汇总基线时仅传`outputs/benchmarks/v4_3407`。实验为24任务五折验证、10任务五模型测试集成；模拟为三任务等权汇总，Stage3仅参加独立thermal_expansion比较，来源、完整行集合和尺度必须一致。HOMO/LUMO独立报告。

结束写入后将完整`outputs/benchmarks/v4_3407/`回传4090相同相对目录，包含全部权重、配置、清单、日志与预测，并核验大小和SHA；冲突不得覆盖不同身份产物。遵循[v4回传规则](v4-runbook.md#跨设备训练产物回传)。

## 历史split42安排

[ADR-0101](adr/0101-baseline-catalog-v4-reuse.md)及`configs/benchmarks/catalog_v4/`保留历史身份。2026-10-15曾审计81个任务、405个候选折次，计划补训825个作业；此计划已被本次3407全量重训替代。旧迁移清单不再作为现役入口，历史产物按上述已确认的清理范围归档/删除，历史报告不改写。
