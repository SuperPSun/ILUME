# 配置索引

## 运行配置

- `smoke.yaml`：两步最小 forward/backward，仅用于确认模型链路；smoke 的 `steps` 不代表正式训练周期。
- `train_test.yaml`：两个小型覆盖 epoch，用于 validation、checkpoint 和 resume 验收。
- `pretrain_base.yaml`：Base 模型，micro-batch 256，5个覆盖 epoch。
- `pretrain_large.yaml`：Large 模型，micro-batch 256，5个覆盖 epoch，并启用 gradient checkpointing。
- `pretrain_xlarge.yaml`：XLarge 模型，micro-batch 128、梯度累积2，5个覆盖 epoch，并启用 gradient checkpointing；直接复用 Base artifact。
- `ablations/`：Base reference 与九个单因素消融配置。
- `archive/`：切换 epoch trainer 前的历史 step 配置；当前代码不能执行。
- `stage2_base.yaml`：Base reference，使用默认35/20/15/15/15任务采样。
- `stage2_base_sampling_balanced.yaml`：Base 五任务20/20/20/20/20均衡采样。
- `stage2_base_sampling_il_heavy.yaml`：Base 10/30/25/25/10 IL-heavy采样。
- `stage2_large.yaml`：复用同一 Stage 2 data artifact、改用 Large checkpoint。
- `stage2_xlarge.yaml`：复用同一 Stage 2 data artifact、改用 XLarge checkpoint。

| 档位 | `d_model` | heads | SMILES/Fusion 层数 | graph depth | descriptor hidden | 当前 artifact 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 384 | 6 | 6/6 | 5 | 768 | 约65.37M |
| Large | 512 | 8 | 8/8 | 6 | 1024 | 约127.95M |
| XLarge | 640 | 10 | 10/10 | 7 | 1280 | 约218.79M |

## 覆盖型 epoch

正式 trainer 根据当前 artifact 的三类训练实体数自动计算每个 epoch 的抽样预算：

```text
required_draws = max(ceil(N_role / role_probability))
effective_batch = batch_size * gradient_accumulation_steps
steps_per_epoch = ceil(required_draws / effective_batch)
draws_per_epoch = steps_per_epoch * effective_batch
```

因此一个 epoch 会保持45/45/10，并让每个入选实体至少出现一次。离子实体为了维持90%配额会在同一 epoch 中进入后续无放回循环；epoch 不是按训练池自然比例简单遍历一次。

以当前正式 artifact 的 `(24908, 27907, 56532)` 三类训练实体为例：Base、Large 和 XLarge 的 effective batch 都是256，因此每个 epoch 均派生2,209步和565,504次抽样。数据池变化后这些数值会自动重算。

若正式配置 OOM，只改变下面两项即可保持有效 batch 和 epoch 预算不变：

```text
Base:  batch_size 256, accumulation 1 → 128, 2
Large: batch_size 256, accumulation 1 → 128, 2
XLarge: batch_size 128, accumulation 2 → 64, 4
```

若 XLarge 的显存占用仍偏低，可改为 `batch_size: 256`、`gradient_accumulation_steps: 1`。两种调整都保持 effective batch=256，因此不改变每个 epoch 的 optimizer step、抽样预算或学习率调度。

所有新实验必须使用独立 `training.output_dir`。只有 data、tokenizer、descriptor 或 fingerprint 配置变化时才需要新的 prepared artifact。

Base 的现役训练参数为micro-batch 256和learning rate `1e-4`。已有 checkpoint 目录仍沿用历史名称 `pretrain_base_bs512`；恢复和 Stage 2 初始化必须使用该实际路径，并以 checkpoint 内嵌配置为准。

## Stage 2 配置

Stage 2 配置与 `PretrainConfig` 分离，模型结构从 checkpoint v3 恢复，不在 YAML 中重复声明。Base、Large 与 XLarge 的 Stage 1 artifact 合同哈希相同，共享 `artifacts/stage_v2/data` 下的实体和任务工件；冻结教师 CLS 位于该目录的 `teachers/<checkpoint-sha256>/`。旧 `artifacts/stage2/` 和 Stage 2 v1 checkpoint 只保留审计，不能恢复到 v2 trainer。

`sampling.probabilities` 必须包含五个任务且总和为1；概率乘以 `sampling.block_size` 必须为整数。默认20-step块对应7/4/3/3/3。三个 IL 任务内部按有序 cation/anion 体系无放回采样，并在每次体系访问时无放回轮换一个条件点；QM 和 transfer 保持逐行采样。验证仍按行统计。

三项 IL 共享一个交互 PairEncoder，transfer 使用独立 PairEncoder，五个回归 MLP 参数互不共享。`training.backbone_freeze_fraction: 0.10` 先训练新模块，再在最近的完整20-step块解冻编码 backbone；新模块和 backbone 使用各自的 warmup+cosine。QM 支持部分缺失标签，并按有效标签列等权计算 masked SmoothL1。`--lambda-alignment`、`--output-dir` 和 `--resume-from` 会形成有效配置并参与 checkpoint 一致性校验。

旧 v1 Stage 2 data artifact 含10个实体 shard，v2 正式配置继续使用 `shard_cache_size: 10`。首次完整生成 `artifacts/stage_v2/data` 后必须根据 metadata 复核实际 shard 数；较小缓存可能反复反序列化约60 MiB的 shard，使GPU长时间等待。

### Stage 2 对比矩阵

下表中的比例顺序为QM/density/heat capacity/thermal expansion/transfer。三种Base采样策略均让transfer执行3,516个optimizer step，即抽样900,096行，覆盖当前899,992行transfer训练集一次。模型容量对比固定默认采样、有效batch 256和23,440个optimizer step。

| 配置 | 采样比例 | 20-step配额 | micro-batch × accumulation | 解冻step | max steps |
|---|---|---|---:|---:|---:|
| `stage2_base.yaml` | 35/20/15/15/15 | 7/4/3/3/3 | 256 × 1 | 2,340 | 23,440 |
| `stage2_base_sampling_balanced.yaml` | 20/20/20/20/20 | 4/4/4/4/4 | 256 × 1 | 1,760 | 17,580 |
| `stage2_base_sampling_il_heavy.yaml` | 10/30/25/25/10 | 2/6/5/5/2 | 256 × 1 | 3,520 | 35,160 |
| `stage2_large.yaml` | 35/20/15/15/15 | 7/4/3/3/3 | 128 × 2 | 2,340 | 23,440 |
| `stage2_xlarge.yaml` | 35/20/15/15/15 | 7/4/3/3/3 | 64 × 4 | 2,340 | 23,440 |

### 单卡串行运行

下面的Bash命令依次准备教师缓存并运行五个独立实验。util-linux的 `script` 会为每项命令分配伪终端，因此训练器仍显示原生动态进度条，同时将完整终端输出写入独立日志；某一项失败时记录退出码并继续后续实验。命令不会删除或覆盖已有目录，因此重跑前应为已有实验选择新的 `--output-dir`，或使用匹配的checkpoint恢复。

```bash
bash <<'BASH'
set -u

log_dir=artifacts/stage_v2/training/comparisons/logs
mkdir -p "$log_dir"

run_and_continue() {
  local name="$1"
  shift
  local command_string
  local command_status=0
  printf -v command_string '%q ' "$@"
  script --quiet --flush --return \
    --command "$command_string" "$log_dir/${name}.log" </dev/null \
    || command_status="$?"
  if [ "$command_status" -eq 0 ]; then
    echo "[OK] $name"
  else
    echo "[FAILED:$command_status] $name" >&2
  fi
  return 0
}

run_and_continue base_prepare \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-prepare --config configs/stage2_base.yaml

run_and_continue base_reference \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_base.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage_v2/training/comparisons/base_reference

run_and_continue base_sampling_balanced \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_base_sampling_balanced.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage_v2/training/comparisons/base_sampling_balanced

run_and_continue base_sampling_il_heavy \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_base_sampling_il_heavy.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage_v2/training/comparisons/base_sampling_il_heavy

run_and_continue large_prepare \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-prepare --config configs/stage2_large.yaml

run_and_continue large_reference \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_large.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage_v2/training/comparisons/large_reference

run_and_continue xlarge_prepare \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-prepare --config configs/stage2_xlarge.yaml

run_and_continue xlarge_reference \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_xlarge.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage_v2/training/comparisons/xlarge_reference
BASH
```

## Stage 3 单阶段双域训练

Stage 3 v2 固定从 `artifacts/stage_v2/training/comparisons/base_reference/best.pt` 生成冻结表示。训练进程不加载 Stage 2 backbone、IL PairEncoder 或 transfer PairEncoder。Stage 3 v1 的 artifact、checkpoint 和日志只保留审计；v2 不读取、不恢复，也不静默迁移它们。

主线一次训练覆盖 27 项任务，但内部是两个数值状态完全隔离的训练域：

- `il21` 包含 19 个直接 IL 任务及 solvation/transfer，使用 HoME 和 late-solute。
- `aux6` 包含 transfer organic、四个离子 HOMO/LUMO 和 charge，每项使用参数完全独立的 `IndependentTaskHead`。

两个域只依赖同一个冻结 Stage 2 来源，不共享任何可训练参数、optimizer、scheduler、AMP scaler、梯度裁剪、BatchNorm、抽样 cursor、Torch/CUDA dropout RNG、早停或最佳指标。每个外层 cycle 先执行一个 21-task IL block，再执行一个 6-task aux block；任一域早停后，另一域仍独立继续。训练输出分别保存 `best_il21.pt` 与 `best_aux6.pt`，结束时组装仅用于评估的 `best.pt`。恢复只能使用 v2 的 `checkpoint_cycle_*.pt`，不能使用这三个 best 文件。

正式高吞吐配置让当前 fold 的冻结表示、条件、目标和预计算 row-to-embedding index 常驻 GPU，并将 PyTorch CPU intra-op/inter-op 线程限制为 4/1。每个域完成全部任务 forward 后只执行一次 backward，loss 与验证统计也只在域边界同步到 CPU。IL21 使用 `128 × 5000 blocks`、每 50 blocks 验证；Aux6 使用 `256 × 2500 blocks`、每 25 blocks 验证。两域均保持每任务最多 640,000 个样本和每 6,400 个样本一次验证，LR 仍为 `3e-4`。

late-solute HoME 的第一层只接收冻结 IL embedding 和条件。solute CLS 仅在第二层进入共享 `SoluteInteraction`、solvation group expert、任务 private expert 和 gate；第二层 global expert 不接收 solute。直接 IL 任务完全绕过这一路径。条件固定编码 temperature/pressure/frequency/wavelength 的标准化值与 presence mask，以及带 `<missing>/<unk>` 的 phase；fold scaler 只拟合另外四折，已有 `_log10` 目标保持 identity。

artifact 按域分隔在 `artifacts/stage3_v2/data/{il21,aux6}`。IL pair 泄漏和 `cross_task_exposure.csv` 只审计 `il21`。task-local fold 只保证同一任务 train/valid 的 IL pair 不重叠，不保证一个任务的验证 IL 未在另一任务训练集中出现，因此五折指标只能解释为 task-local 泛化。

正式配置包括：

- `stage3_home.yaml`：同时训练相互隔离的 `il21` 和 `aux6`。
- `stage3_home_legacy.yaml`：`64 × 10000 + per_task backward`，仅用于审计或恢复优化前的 Stage 3 v2 training checkpoint。
- `stage3_shared_bottom.yaml`、`stage3_mmoe.yaml`：只训练 `il21` 的共享结构基线。
- `stage3_early_solute.yaml`：只训练 `il21` 的 early-solute 对照。
- `stage3_without_feature_gate.yaml`、`stage3_without_self_gate.yaml`：只训练 `il21` 的单因素消融。

下面的 Bash 块无需替换 fold 或输出路径，完整命令保存在 [`scripts/run_stage3_matrix.sh`](../scripts/run_stage3_matrix.sh)。它会安装环境，执行一次幂等 v2 prepare，串行运行主线与全部 IL-only 基线五折，分别汇总验证指标，并用主线五个组合 `best.pt` 做固定 test ensemble。优化训练和日志统一写入 `artifacts/stage3_v2/training_optimized/`；优化前的 `artifacts/stage3_v2/training/` 不会被读取或覆盖。脚本固定使用 GPU 1；`script` 保留原生 tqdm，每项写独立日志与状态表，某项失败后继续后续项目。

2026年8月5日的正式优化矩阵已经完成；[`status.tsv`](../artifacts/stage3_v2/training_optimized/logs/20260805_152606/status.tsv) 中主线、五组 IL-only 对照、各五折汇总和主线 test ensemble 均为 `OK`。可跟踪结果位于各实验的 `five_fold_summary.json` 和 `home/test_ensemble_metrics.json`；忽略的模型 checkpoint 与终端日志仍保留在同一 artifact 树中。

```bash
bash <<'BASH'
set -u

test -f scripts/run_stage3_matrix.sh
bash -n scripts/run_stage3_matrix.sh
bash scripts/run_stage3_matrix.sh
BASH
```

单 fold 的精确恢复接口如下。checkpoint 必须来自同一 fold、配置、active domains 和 v2 artifact；CLI 不再提供 `--init-from`：

```bash
ilume-stage3-train \
  --config configs/stage3_home.yaml \
  --fold 1 \
  --resume-from artifacts/stage3_v2/training_optimized/home/fold1/checkpoint_cycle_00000100.pt \
  --output-dir artifacts/stage3_v2/training_optimized/home/fold1
```

优化前的 fold1 只能使用 legacy 配置恢复，不能传给 `stage3_home.yaml`：

```bash
ilume-stage3-train \
  --config configs/stage3_home_legacy.yaml \
  --fold 1 \
  --resume-from artifacts/stage3_v2/training/home/fold1/checkpoint_cycle_00003300.pt \
  --output-dir artifacts/stage3_v2/training/home/fold1
```
