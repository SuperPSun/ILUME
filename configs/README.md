# 配置索引

## 运行配置

- `smoke.yaml`：两步最小 forward/backward，仅用于确认模型链路；smoke 的 `steps` 不代表正式训练周期。
- `train_test.yaml`：两个小型覆盖 epoch，用于 validation、checkpoint 和 resume 验收。
- `pretrain_base.yaml`：Base 模型，micro-batch 512，5个覆盖 epoch。
- `pretrain_large.yaml`：Large 模型，micro-batch 256，5个覆盖 epoch，并启用 gradient checkpointing。
- `pretrain_xlarge.yaml`：XLarge 模型，micro-batch 128、梯度累积2，5个覆盖 epoch，并启用 gradient checkpointing；直接复用 Base artifact。
- `legacy.yaml`：Full/1-token/AIS/无指纹/无 role embedding/linear graph head 的旧架构消融。
- `ablations/`：Base reference 与九个单因素消融配置。
- `archive/`：切换 epoch trainer 前的历史 step 配置；当前代码不能执行。

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

以当前正式 artifact 的 `(24908, 27907, 56532)` 三类训练实体为例：Base 每个 epoch 派生1,105个 optimizer step 和565,760次抽样；Large 和 XLarge 的有效 batch 都是256，因此每个 epoch 均派生2,209步和565,504次抽样。数据池变化后这些数值会自动重算。

若正式配置 OOM，只改变下面两项即可保持有效 batch 和 epoch 预算不变：

```text
Base:  batch_size 512, accumulation 1 → 256, 2
Large: batch_size 256, accumulation 1 → 128, 2
XLarge: batch_size 128, accumulation 2 → 64, 4
```

若 XLarge 的显存占用仍偏低，可改为 `batch_size: 256`、`gradient_accumulation_steps: 1`。两种调整都保持 effective batch=256，因此不改变每个 epoch 的 optimizer step、抽样预算或学习率调度。

所有新实验必须使用独立 `training.output_dir`。只有 data、tokenizer、descriptor 或 fingerprint 配置变化时才需要新的 prepared artifact。
