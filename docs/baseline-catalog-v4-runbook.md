# 当前数据版本基线运行手册

合同见[ADR-0101](adr/0101-baseline-catalog-v4-reuse.md)。全部命令在仓库根执行；主环境为`ilume`，高级模型训练仍由入口切换至各自hash-lock环境，环境和资产按[README](../README.md#baselines-and-ablations)验证，不自动下载或安装。

## 配置和结果

使用`configs/benchmarks/catalog_v4/<model>.yaml`，默认data/42、24实验五折以及thermal_expansion/homo/lumo三模拟。AIFC保留模型合同seed1000，其他模型保留各自原种子；data/42不替换模型种子。

v3结果保留。新结果根为`outputs/benchmarks/v4/<model>/`；`imported/`保存完整旧任务，普通stage3目录保存新训练/派生评估，simulation目录保存新模拟作业。不要将旧根summary复制成新根summary。

## 审计与复制（不训练）

九个可复用模型逐一执行以下命令；AIonopedia不执行此步骤。此模式不加载大模型或启动GPU训练，会核验来源、预训练资产、权重和预测SHA，可能持续数分钟。

```bash
conda activate ilume
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/catalog_v4/mlp.yaml \
  --output outputs/benchmarks/v4/mlp \
  --prepare-reuse-from outputs/benchmarks/v3/mlp
```

其余模型为ecfp_xgboost、dmpnn、molformer、ilbert、spmm、llasmol、iltransr、aifc，各自替换模型名。命令返回`migrations/attempt-NNN/manifest.json`。查看同目录summary.json和manifest中的任务action：reuse可复用、train需要补训、integrity_error需要先解决完整性异常。重复审计生成新attempt；目标已存在文件只有SHA相同才复用。

## 用户正式启动补训

先确认审计清单没有integrity_error，并使用命令实际返回的清单路径。初次从一个GPU任务开始，确认主机内存和显存后再增加并发。

```bash
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/catalog_v4/mlp.yaml \
  --output outputs/benchmarks/v4/mlp \
  --reuse-manifest outputs/benchmarks/v4/mlp/migrations/attempt-001/manifest.json \
  --max-workers 1 --devices cuda:0
```

XGBoost不传`--devices`；其他高级模型使用对应模型配置/清单。包含imported目录时必须提供清单，不能只根据summary存在跳过训练。基线没有训练恢复，失败训练在新attempt从头重跑，已有成功作业在严格验证后跳过。

AIonopedia旧权重已确认丢失，直接训练完整24实验和三模拟：

```bash
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/catalog_v4/aionopedia.yaml \
  --output outputs/benchmarks/v4/aionopedia \
  --max-workers 1 --devices cuda:0
```

不对完整sweep根混用`--benchmark all/stage3/simulation`；单任务调试使用独立输出。所有新训练和评估的数据审计路径从data_root派生，不使用旧的data/stage3元数据。

## 汇总

仅当模型的24实验及三模拟完成后发布完整榜单。下面的输出目录单独保存本协议快照，不把v3根作为输入。

```bash
python scripts/benchmarks/summarize.py \
  --input outputs/benchmarks/v4 outputs/v4/stage3/base/evaluate_valid outputs/v4/stage3/base/evaluate_test \
  --output outputs/reports/baseline-catalog-v4
```

Stage3输入替换为实际完成的现役评估目录；只汇总基线时仅传`outputs/benchmarks/v4`。实验榜单要求完整相同的24任务/五折或对应10测试任务。simulation榜单比较三任务基线均值；`thermal_expansion_{validation,test}_*.csv`只比较热膨胀，包括完整现役Stage3两任务报告中的对应任务。不同来源或归一尺度拒绝混排。HOMO/LUMO保留逐任务表，不伪造Stage3电子预测。

任意设备运行后的完整模型、配置、日志、清单和预测均须回传4090同相对目录，遵循[v4产物归档规则](v4-runbook.md#跨设备训练产物回传)。
