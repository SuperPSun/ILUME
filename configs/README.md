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

Stage 2 配置与 `PretrainConfig` 分离，模型结构从 checkpoint v3 恢复，不在 YAML 中重复声明。Base、Large 与 XLarge 的 Stage 1 artifact 合同哈希相同，共享 `artifacts/stage2/data` 下的实体和任务工件；冻结教师 CLS 位于该目录的 `teachers/<checkpoint-sha256>/`。

`sampling.probabilities` 必须包含五个任务且总和为1；概率乘以 `sampling.block_size` 必须为整数。默认20-step块对应7/4/3/3/3。`--lambda-alignment`、`--output-dir` 和 `--resume-from` 会形成有效配置并参与 checkpoint 一致性校验。

当前 Stage 2 data artifact 含10个实体 shard，正式配置使用 `shard_cache_size: 10` 将它们全部保留在主机内存。较小缓存会在随机实体批次间反复反序列化约60 MiB的 shard，使GPU长时间等待；数据重建后若 shard 数变化，应同步复核该值。

### Stage 2 对比矩阵

下表中的比例顺序为QM/density/heat capacity/thermal expansion/transfer。三种Base采样策略均让transfer执行3,516个optimizer step，即抽样900,096行，覆盖当前899,992行transfer训练集一次。模型容量对比固定默认采样、有效batch 256和23,440个optimizer step。

| 配置 | 采样比例 | 20-step配额 | micro-batch × accumulation | max steps |
|---|---|---|---:|---:|
| `stage2_base.yaml` | 35/20/15/15/15 | 7/4/3/3/3 | 256 × 1 | 23,440 |
| `stage2_base_sampling_balanced.yaml` | 20/20/20/20/20 | 4/4/4/4/4 | 256 × 1 | 17,580 |
| `stage2_base_sampling_il_heavy.yaml` | 10/30/25/25/10 | 2/6/5/5/2 | 256 × 1 | 35,160 |
| `stage2_large.yaml` | 35/20/15/15/15 | 7/4/3/3/3 | 128 × 2 | 23,440 |
| `stage2_xlarge.yaml` | 35/20/15/15/15 | 7/4/3/3/3 | 64 × 4 | 23,440 |

### 单卡串行运行

下面的Bash命令依次准备教师缓存并运行五个独立实验。util-linux的 `script` 会为每项命令分配伪终端，因此训练器仍显示原生动态进度条，同时将完整终端输出写入独立日志；某一项失败时记录退出码并继续后续实验。命令不会删除或覆盖已有目录，因此重跑前应为已有实验选择新的 `--output-dir`，或使用匹配的checkpoint恢复。

```bash
bash <<'BASH'
set -u

log_dir=artifacts/stage2/training/comparisons/logs
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
  --output-dir artifacts/stage2/training/comparisons/base_reference

run_and_continue base_sampling_balanced \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_base_sampling_balanced.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage2/training/comparisons/base_sampling_balanced

run_and_continue base_sampling_il_heavy \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_base_sampling_il_heavy.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage2/training/comparisons/base_sampling_il_heavy

run_and_continue large_prepare \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-prepare --config configs/stage2_large.yaml

run_and_continue large_reference \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_large.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage2/training/comparisons/large_reference

run_and_continue xlarge_prepare \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-prepare --config configs/stage2_xlarge.yaml

run_and_continue xlarge_reference \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ilume-stage2-train --config configs/stage2_xlarge.yaml \
  --lambda-alignment 0.1 \
  --output-dir artifacts/stage2/training/comparisons/xlarge_reference
BASH
```
