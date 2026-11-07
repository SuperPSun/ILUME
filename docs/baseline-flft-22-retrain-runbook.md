# FLft baseline：3090优先的22任务协议定向补训

本轮按[ADR-0118](adr/0118-flft-22-task-baseline-retraining.md)，仅重新训练10项数据变化的实验任务：电导率、黏度、密度、表面张力、玻璃化转变温度、熔点、热分解温度、折射率、solvation、hydration。十模型各50个折次，共500个训练作业；包含训练后的逐折验证，以及有测试行任务的五模型测试集成。模拟和其他任务不运行。

## 执行前

在各服务器的ILUME仓库根目录执行。先同步本次代码与 `configs/benchmarks/FLftv4_3407/`、`configs/v4/stage3/base.yaml`、当前 `data/3407/` 及各模型的预训练资产。未提交的本地修改不会通过 `git pull` 传输；使用既有Git发布或显式同步流程，不以旧配置代替。两台服务器应使用相同代码版本和同一份数据。

本次交付另提供 `outputs/benchmarks/maintenance/20261107-FLft-retrain-checks/flft-22-retrain.patch` 和 `flft-22-data.sha256`。补丁已在基于 `eedbcfb301f22ecd14519f0c6152fc8080636514` 的干净临时仓库重放验证。若未通过Git发布本次修改，将这两个文件复制到目标服务器 `/tmp/`，同步新数据后，在仓库根执行以下命令；任何一步失败都先处理差异，不开始训练。补丁仅应用一次。

```bash
git pull --ff-only
git apply --check /tmp/flft-22-retrain.patch && \
git apply /tmp/flft-22-retrain.patch && \
sha256sum --quiet -c /tmp/flft-22-data.sha256
```

主环境使用 `ilume`，八个高级模型由入口自动切换到各自已安装的独立环境，不自动安装依赖。MoLFormer还需要其固定HF快照缓存。MLP/XGBoost使用当前主环境RDKit2026.09.1及全新特征缓存；不能复用旧版本缓存。

先确认设备编号及新文件已就位：

```bash
nvidia-smi
python scripts/benchmarks/sweep.py --help
test -f configs/benchmarks/FLftv4_3407/llasmol.yaml
```

下面3090按0、1、2、3四卡均空闲书写；H100的编号应按实际两张空闲卡填写。独占节点内可直接传物理编号；如Slurm已设置CUDA_VISIBLE_DEVICES，不传会覆盖分配的物理编号，仅使用调度系统分配的卡。

## 3090：四卡、每卡一个独立折次

使用bash。依次完成八个GPU模型，每个模型的50个折次由现有sweep分配到四张卡；每张卡最多一个独立train/evaluate子进程，没有DDP。保留单独日志，某模型失败后继续下一个模型，失败模型根据日志再处理。

```bash
conda activate ilume
mkdir -p outputs/benchmarks/FLftv4_3407/retrain_22/logs
for model in aionopedia molformer spmm iltransr ilbert dmpnn aifc mlp; do
  python scripts/benchmarks/sweep.py \
    --config "configs/benchmarks/FLftv4_3407/${model}.yaml" \
    --output "outputs/benchmarks/FLftv4_3407/retrain_22/${model}" \
    --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-workers 4 \
    > "outputs/benchmarks/FLftv4_3407/retrain_22/logs/${model}.log" 2>&1 || \
    echo "FAILED: ${model}; inspect its log before retrying."
done
```

在另一终端运行XGBoost，不占GPU、每次8个CPU线程：

```bash
conda activate ilume
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/ecfp_xgboost.yaml \
  --output outputs/benchmarks/FLftv4_3407/retrain_22/ecfp_xgboost \
  --max-workers 1
```

## H100：LlaSMol，两卡各一个独立折次

用户已确认H100空闲卡为0、1，以下使用这两个编号。保持当前QLoRA、batch16、梯度累积2、关闭梯度检查点的配方。

```bash
conda activate ilume
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/llasmol.yaml \
  --output outputs/benchmarks/FLftv4_3407/retrain_22/llasmol \
  --devices cuda:0,cuda:1 --max-workers 2
```

## 失败、转移与回传

- 3090出现OOM时保留失败attempt，不修改batch、精度、长度或梯度累积。不要在原模型仍运行时另开同模型sweep。
- 确认原sweep结束后，把该模型完整输出同步至H100相同相对路径，使用同一份配置及数据，再以同一入口运行、只替换模型名与设备列表。经严格验证的完成折次跳过，失败折在新attempt重跑。不要删除或仅挑选复制预测文件。
- 日志中的 `FAILED` 以及各模型 `status.tsv` / `metadata.json` 是状态依据；shell循环继续不代表前一个模型已成功。重跑命令不会恢复失败训练的中间epoch。
- 保持本轮新输出的相对路径不变，将3090、H100的完整 `retrain_22/<model>/` 分别回传4090同路径；遵循[v4回传规则](v4-runbook.md#跨设备训练产物回传)，核验全部权重、清单、配置、日志和预测，冲突不覆盖。
- szx本轮暂不启动训练。保留的历史12任务及三模拟尚未获得新协议复用资格；本轮完成后也不能直接运行旧论文汇总脚本生成完整22任务比较报告。
