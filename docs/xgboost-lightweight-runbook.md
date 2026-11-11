# ECFP+XGBoost：500棵树、最大深度4

按[ADR-0126](adr/0126-xgboost-lightweight-variants.md)，现役只保留500棵/深度4。原版1000棵与500棵/深度6退役。22实验×五折＋三模拟＝113次训练；每作业CPU八线程，无需GPU或DDP。输入特征、物理条件、seed42及其他训练参数保持原定义，训练固定跑满500棵末轮发布，不早停。

现役配置按用户要求恢复原命名 `configs/benchmarks/FLftv4_3407/completion/ecfp_xgboost.yaml`，显示名为 `ECFP+XGBoost`，实际仍为500棵/深度4。此次仅统一配置命名，已有输出与缓存不移动；不能将新身份写进3090仍存在的旧1000棵输出根。若已启动旧长文件名的500棵/深度4进程，先让该进程结束，再更新仓库或执行后续评估。

## 3090启动

4090先提交并推送本次代码和配置，在3090拉取后执行。已有主环境、3407数据及共享电子模拟数据沿用原安装，不安装新依赖。

```bash
cd /mnt/backup8t/pengs/ILUME
git pull --ff-only
conda activate ilume
mkdir -p outputs/benchmarks/FLftv4_3407/lightweight/logs
```

先只训练并评估验证集：

```bash
set -o pipefail
PYTHONPATH=src:. python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/completion/ecfp_xgboost.yaml \
  --output outputs/benchmarks/FLftv4_3407/lightweight/ecfp_xgboost_500_d4 \
  --max-workers 1 --validation-only \
  2>&1 | tee -a outputs/benchmarks/FLftv4_3407/lightweight/logs/ecfp_xgboost_500_d4.log
```

可放在已有tmux中运行；相同输出根不要启动多个进程。中断后先确认原进程结束，再执行原命令；已完整且严格核验通过的作业跳过，失败作业在新attempt从头重训，不恢复部分boosting预算。

## 验证、测试与报告

查看输出根 `summary.json` 的 `stage3_property_benchmark.validation_five_fold` 和 `simulation_property_benchmark.simulation_validation`。测试section在 `reporting.benchmarks` 中为 `not_requested`，不代表失败，也不能作为测试榜单使用。没有测试集的实验任务不填造测试结果。

验证与资源检查完成后，重新执行上述sweep命令并去掉 `--validation-only`，保持配置与输出根一致，补齐测试评估。训练/验证仍通过完整性核验后跳过；不通过时明确失败，禁止手改身份或SHA。每次训练都固定500棵末轮发布，不按验证选树或调学习率。

评估轻量化开销时，使用相同任务、五折与行集合，记录MAE、共同完整训练集尺度的NMAE、模型JSON总字节、训练耗时及相同CPU/线程/batch下的推理耗时。树数或深度比例不代表实际加速；本次尚无新正式结果。

原版三个产物目录已按用户确认的精确清单清理，其他模型、共享缓存和历史只读配置保留。2026-11-10的报告包含1000棵原版，仅作为历史快照，不代表当前500棵模型。后续[完整22任务报告](baseline-22-summary-runbook.md)只接受500棵/深度4的新产物；必须先完成113作业、所需评估与新来源审计，并在报告配置的 `source_audits` 登记新审计SHA。未完成时报告组装明确失败，不拿旧原版或未审核预测补位。

所有模型、运行配置、日志、身份和完整检查点最终回传4090项目的同相对路径；不能只传summary或排除权重。回传按[v4手册](v4-runbook.md#跨设备训练产物回传)执行，不覆盖或删除其他基线。此次未启动正式训练，3090旧产物也未自动清理。
