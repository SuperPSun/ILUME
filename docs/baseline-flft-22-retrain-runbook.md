# FLft baseline：22任务协议定向补训

本轮按[ADR-0118](adr/0118-flft-22-task-baseline-retraining.md)，仅重新训练10项数据变化的实验任务：电导率、黏度、密度、表面张力、玻璃化转变温度、熔点、热分解温度、折射率、solvation、hydration。十模型各50个折次，共500个训练作业；包含训练后的逐折验证，以及有测试行任务的五模型测试集成。模拟和其他任务不运行。

## 当前安排：szx运行9个模型，H100保留LlaSMol

2026-11-07用户改为在szx运行其余9个模型，H100已经启动的LlaSMol保留跑完，3090不再启动本轮baseline。szx共450个训练作业：8个GPU模型各50折次、XGBoost50折次；H100的LlaSMol另50折次，总预算仍为500。原3090/H100整批命令仅作备用，不与当前安排同时执行。

代码、新配置、60个本轮数据文件、独立环境锁及预训练文件已在szx只读核对。GPU后端及权重完整SHA由原入口在分配的节点继续检查，不在登录节点训练。

两份新脚本已部署到 `/home/sunp/slurm/baseline/`，旧脚本保留。它们不属于Git运行入口，而是该服务器的Slurm资源提交文件；4090留存副本及检查记录于 `outputs/benchmarks/maintenance/20261107-szx-flft-sbatch/`。

| 脚本 | 范围与资源 |
|---|---|
| `baseline-flft-22-gpu.sbatch` | gpu_6000D/gpu05；数组0–7、最多并发5项；每项1卡、4 CPU、64GB内存、7天；每模型内部max-workers1 |
| `baseline-flft-22-xgboost.sbatch` | gpu05；不申请GPU，8 CPU、32GB内存、7天；max-workers1 |

GPU数组映射为 `0 aionopedia / 1 spmm / 2 molformer / 3 ilbert / 4 iltransr / 5 aifc / 6 dmpnn / 7 mlp`，不包含LlaSMol。每项只执行对应模型的10项实验任务五折训练、验证及有测试行任务的集成评估，不用DDP，不运行模拟。脚本保留Slurm的CUDA_VISIBLE_DEVICES，不传 `--devices`。

在szx登录节点执行以下命令提交；脚本准备阶段仅做过 `sbatch --test-only`，未提交实际训练：

```bash
cd /home/sunp/slurm/baseline
sbatch baseline-flft-22-gpu.sbatch
sbatch baseline-flft-22-xgboost.sbatch
squeue -u sunp
```

GPU日志为 `/home/sunp/slurm/logs/baseline-flft22-gpu_<array_job_id>_<index>.out/.err`，CPU日志为 `baseline-flft22-xgboost_<job_id>.out/.err`。产物为 `/home/sunp/ILUME/outputs/benchmarks/FLftv4_3407/retrain_22/<model>/`，状态在各模型的 `status.tsv`。同模型互斥锁保存在 `retrain_22/.locks/`，重复提交正在运行的模型会失败，不删除锁文件绕过检查。

并发5是上限，不是立即占满5卡的承诺。核对时gpu05剩余11个CPU，按每项4CPU预计最多先运行2个GPU数组项；XGBoost需要8CPU，也可能排队。实际并发由Slurm按CPU、GPU和其他队列限制决定，不修改训练batch或配方以迁就资源。

如只重交失败的GPU模型，确认原作业已结束后使用数组索引，例如 `sbatch --array=2 baseline-flft-22-gpu.sbatch` 仅重交MoLFormer。已完成折次经过原有严格校验后跳过，失败attempt不会中途恢复。

## ILBERT 独立条件标准化配方

2026-11-08按[ADR-0119](adr/0119-ilbert-train-only-condition-standardization.md)增加独立配方。仅温度、压力等注册表数值条件改为训练行z-score；保留模型结构、预训练资产、seed42及10轮末轮发布。每折分别拟合统计量并随训练身份保存于checkpoint，验证/测试复用；常量scale1，无条件使用空统计量。预测CSV仍保存原始物理条件。

原配置和 `retrain_22/ilbert/` 保留，不把标准化训练写成旧配方的下一次attempt，不按新旧验证分数择优拼接。下列新配置仍覆盖本轮10项变化任务五折，共50个训练作业；其余12任务和模拟不运行，也不自动发布完整22任务榜单。

在3090拉取包含本次修改的Git提交后执行；当前环境和预训练资产无需重新安装或下载。先确保所列GPU可用。完整sweep每张卡运行一个独立任务，不使用DDP：

```bash
cd /mnt/backup8t/pengs/ILUME
git pull --ff-only
conda activate ilume
PYTHONPATH=src:. python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/ilbert_condition_zscore.yaml \
  --output outputs/benchmarks/FLftv4_3407/condition_zscore/ilbert \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-workers 4
```

若仅先运行黏度fold4，可直接使用原单折入口，使用独立 `condition_zscore_check/` 预检根。以下训练与验证输出均须尚不存在；它只产生该折结果，不构成五折比较或测试集成，也不据此选择训练参数或检查点：

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 python scripts/benchmarks/train.py \
  --config configs/benchmarks/FLftv4_3407/ilbert_condition_zscore.yaml \
  --benchmark stage3 --task experiment/viscosity --fold 4 \
  --output outputs/benchmarks/FLftv4_3407/condition_zscore_check/ilbert/stage3/experiment__viscosity/fold4/attempt-001
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 python scripts/benchmarks/evaluate.py \
  --config configs/benchmarks/FLftv4_3407/ilbert_condition_zscore.yaml \
  --benchmark stage3 --task experiment/viscosity --fold 4 --split valid \
  --checkpoint outputs/benchmarks/FLftv4_3407/condition_zscore_check/ilbert/stage3/experiment__viscosity/fold4/attempt-001 \
  --output outputs/benchmarks/FLftv4_3407/condition_zscore_check/ilbert/stage3/experiment__viscosity/evaluate_valid_fold4/attempt-001
```

完整sweep在上方独立 `condition_zscore/` 根从头运行50折，不复制单折预检目录拼装sweep根；正常重复执行同一sweep时，已完成折次通过严格校验后跳过。新配方的完整权重、配置、日志、身份及预测最终仍回传4090同相对路径。

## 审计后剩余189作业补训

2026-11-09用户按[ADR-0121](adr/0121-baseline-22-task-completion-retraining.md)选择将缺口全部重训：MLP、XGBoost、标准化ILBERT各12项其余实验任务五折＋热膨胀/HOMO/LUMO三模拟，每模型63次，共189次。包含无条件ILBERT的pEC50/HOMO/LUMO，不采用旧身份的等价派生。其余七个模型和已经完成的10项变化实验不重复训练。

先将4090本次代码、配置提交推送，再在3090拉取。两环境及资产沿用原安装；下列GPU分配要求0–3号卡空闲，XGBoost保持配置的8线程。新目录 `completion_22/` 与既有所有结果隔离，不向旧目录写入。

```bash
cd /mnt/backup8t/pengs/ILUME
git pull --ff-only
conda activate ilume
mkdir -p outputs/benchmarks/FLftv4_3407/completion_22/logs
nvidia-smi
```

在三个独立终端分别激活 `ilume` 并从仓库根执行。ILBERT入口自动切换到 `ilume-ilbert`；每卡一个独立训练任务，无DDP。以下命令以前台运行并用tee保存完整日志，可放在现有tmux会话中：

```bash
set -o pipefail
PYTHONPATH=src:. python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/completion/ilbert.yaml \
  --output outputs/benchmarks/FLftv4_3407/completion_22/ilbert \
  --devices cuda:0,cuda:1 --max-workers 2 \
  2>&1 | tee -a outputs/benchmarks/FLftv4_3407/completion_22/logs/ilbert.log
```

```bash
set -o pipefail
PYTHONPATH=src:. python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/completion/mlp.yaml \
  --output outputs/benchmarks/FLftv4_3407/completion_22/mlp \
  --devices cuda:2,cuda:3 --max-workers 2 \
  2>&1 | tee -a outputs/benchmarks/FLftv4_3407/completion_22/logs/mlp.log
```

```bash
set -o pipefail
PYTHONPATH=src:. python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/completion/ecfp_xgboost.yaml \
  --output outputs/benchmarks/FLftv4_3407/completion_22/ecfp_xgboost \
  --max-workers 1 \
  2>&1 | tee -a outputs/benchmarks/FLftv4_3407/completion_22/logs/ecfp_xgboost.log
```

无需分开手动启动验证与测试；完整sweep自动执行实验逐折验证、可用测试集五折集成和模拟逐任务验证/测试。无测试行任务如实记录no_samples。中断或失败后，在原进程已结束的前提下重新执行同一命令；严格核验通过的已完成任务跳过，失败任务在新attempt完整重训。不要同时启动两个相同模型的相同输出根。

训练结束后在4090终端回传整个新目录；只新增、不覆盖任何已有文件、不删除源产物：

```bash
cd /data/pengs/ILUME
rsync -aH --ignore-existing --partial-dir=.rsync-partial --info=progress2 \
  3090:/mnt/backup8t/pengs/ILUME/outputs/benchmarks/FLftv4_3407/completion_22/ \
  outputs/benchmarks/FLftv4_3407/completion_22/
rsync -rHcni \
  3090:/mnt/backup8t/pengs/ILUME/outputs/benchmarks/FLftv4_3407/completion_22/ \
  outputs/benchmarks/FLftv4_3407/completion_22/
```

校验应无文件缺失/内容差异输出。保留3090原件直到最终归档确认。本次只补训练缺口，完整22任务论文汇总入口仍需另外适配，不能直接使用旧24任务汇总命令。

## 其他设备备用：同步资源

从4090直接使用rsync同步当前代码、配置、3407数据与预训练资产，不再使用补丁或独立SHA清单。2026-11-07已只读确认3090仓库为 `/mnt/backup8t/pengs/ILUME/`，H100为 `/workspace/pengs/ILUME/`。以下命令在4090执行；不传输主仓库 `.git`、训练输出、trash或Conda环境，不删除目标额外文件。同步期间不要在目标仓库修改同一批源文件。

```bash
cd /data/pengs/ILUME
rsync -aHcR --partial --info=progress2 \
  ./src ./benchmarks ./scripts ./configs ./docs \
  ./AGENTS.md ./README.md ./pyproject.toml \
  ./data/3407 ./artifacts/benchmarks \
  3090:/mnt/backup8t/pengs/ILUME/

rsync -aHcR --partial --info=progress2 \
  ./src ./benchmarks ./scripts ./configs ./docs \
  ./AGENTS.md ./README.md ./pyproject.toml \
  ./data/3407 ./artifacts/benchmarks \
  h100:/workspace/pengs/ILUME/
```

3090运行MoLFormer还需要其固定HF快照缓存，必须连同blobs和snapshots一起同步。若H100后续承接MoLFormer，对H100也执行同样缓存同步，仅替换目标主机：

```bash
rsync -aHc --mkpath --partial --info=progress2 \
  /home/pengs/.cache/huggingface/hub/models--ibm-research--MoLFormer-XL-both-10pct \
  3090:~/.cache/huggingface/hub/
```

同步完成后再到各服务器ILUME仓库根目录运行下面的训练命令。主环境使用 `ilume`，八个高级模型由入口自动切换到各自已安装的独立环境，不自动安装依赖。MLP/XGBoost使用当前主环境RDKit2026.09.1及全新特征缓存；不能复用旧版本缓存。rsync只同步文件，不创建Git提交或同步Conda环境。

先确认设备编号及新文件已就位：

```bash
nvidia-smi
python scripts/benchmarks/sweep.py --help
test -f configs/benchmarks/FLftv4_3407/llasmol.yaml
```

下面3090按0、1、2、3四卡均空闲书写；H100的编号应按实际两张空闲卡填写。独占节点内可直接传物理编号；如Slurm已设置CUDA_VISIBLE_DEVICES，不传会覆盖分配的物理编号，仅使用调度系统分配的卡。

## 备用：3090四卡、每卡一个独立折次

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

## H100：已有LlaSMol继续运行

用户已确认H100使用0、1号卡，原LlaSMol已经启动，保留该运行，不重复执行以下命令。仅在原进程结束且需要重试时使用相同输出和配置。保持当前QLoRA、batch16、梯度累积2、关闭梯度检查点的配方。

```bash
cd /workspace/pengs/ILUME
ILUME_BENCHMARK_ENVIRONMENT=ilume-llasmol \
/workspace/pengs/.conda/envs/ilume-llasmol/bin/python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/FLftv4_3407/llasmol.yaml \
  --output outputs/benchmarks/FLftv4_3407/retrain_22/llasmol \
  --devices cuda:0,cuda:1 --max-workers 2
```

## 失败、转移与回传

- 3090出现OOM时保留失败attempt，不修改batch、精度、长度或梯度累积。不要在原模型仍运行时另开同模型sweep。
- 确认原sweep结束后，把该模型完整输出同步至H100相同相对路径，使用同一份配置及数据，再以同一入口运行、只替换模型名与设备列表。经严格验证的完成折次跳过，失败折在新attempt重跑。不要删除或仅挑选复制预测文件。
- 日志中的 `FAILED` 以及各模型 `status.tsv` / `metadata.json` 是状态依据；shell循环继续不代表前一个模型已成功。重跑命令不会恢复失败训练的中间epoch。
- 保持本轮新输出的相对路径不变，将szx、H100的完整 `retrain_22/<model>/` 分别回传4090同路径，同时保留szx对应Slurm日志和脚本；遵循[v4回传规则](v4-runbook.md#跨设备训练产物回传)，核验全部权重、清单、配置、日志和预测，冲突不覆盖。
- 保留的历史12任务及三模拟尚未获得新协议复用资格；本轮完成后也不能直接运行旧论文汇总脚本生成完整22任务比较报告。
