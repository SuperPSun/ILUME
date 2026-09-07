# ILUME Capacity v1 操作手册

本文只给出正式运行命令；实现验收不会执行这些 prepare/train/evaluate。所有命令从仓库
根目录运行。开始前必须确认 Git clean、没有仍在写入的现役 Stage job，并保留全部
`outputs/v1` 与 `summary/`。

## 1. Stage 1 Base selection

只用 Base 配置 prepare 一次实验 corpus；R1–R8 selection 只使用 Base epoch-10 checkpoint：

```bash
python scripts/stage1/prepare.py \
  --config configs/experiments_v1/stage1/base.yaml \
  --output outputs/experiments_v1/stage1/prepare
```

prepare 完整成功后，先训练 Base：

四份 Capacity Stage 1 YAML 已共同冻结为 global batch 512、LR `4e-4`。这是从原
batch 128 约 20GB 显存占用线性估算得到的 84GB 单卡配置，目标峰值约 80GB；它不是
自动调参，也不适用于 48GB 卡。正式运行前确认目标 GPU 空闲且为同类 84GB 硬件，并在
训练日志中记录实际 peak VRAM、吞吐和稳定性。若出现 OOM/NaN/divergence，停止研究，
不得改 batch、LR、gradient checkpointing 或 horizon 后续跑。

现有共享 corpus 仍可直接复用：它在此前的 prepare run 中记录了 batch 128/LR `1e-4`，
但这两个训练字段不进入 corpus identity。不得仅因本次 batch/LR 变化重新 prepare；反之，
任何不同 global batch 的 Stage 1 checkpoint 都不能 resume。本仓库当前没有 Capacity Stage 1
checkpoint，因此 Base 训练从 epoch 0 开始。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage1/train.py --config configs/experiments_v1/stage1/base.yaml --output outputs/experiments_v1/stage1/base/train
```

确认 Base epoch-10 full validation 与 checkpoint 完整。OOM/NaN 时停止研究，不改 batch 或
gradient checkpointing 后续跑。S/L/XL 只在 Base winner 后按另行冻结的 promotion 配置训练。

## 2. Stage 2 Base selection prepare 与 8 runs

先以 R4 调用一次 reusable prepare root，物化共享 data 与 Stage 1 Base encoder 的 teacher cache：

```bash
python scripts/stage2/prepare.py --config configs/experiments_v1/stage2/base-e09-r4.yaml --output outputs/experiments_v1/stage2/prepare
```

随后对 R1–R8 的八个 YAML 分别运行：

```bash
python scripts/stage2/train.py \
  --config configs/experiments_v1/stage2/base-e09-rN.yaml \
  --output outputs/experiments_v1/stage2/base/rN/train
```

每次先完成 10 个 joint epochs，再完成 YAML 指定的 10 个四任务 head-only refinement
epochs；连续发布 epoch 1–20 历史 checkpoint、joint epoch-10 boundary 的
`stage2_encoder.pt` 和 Stage 2 自身的 `taskwise_refined.pt`。Stage 3 只消费 encoder，
Stage 2 validation 不淘汰 candidate。

## 3. Stage 3 prepare、probe 与自动选 Base recipe

对八个 `base-rN` 配置逐一准备：

```bash
python scripts/stage3/prepare.py \
  --config configs/experiments_v1/stage3/probe/base-rN.yaml \
  --output outputs/experiments_v1/stage3/prepare/base-rN
```

每个 candidate 跑 folds 1/2：

```bash
python scripts/stage3/train.py \
  --config configs/experiments_v1/stage3/probe/base-rN.yaml \
  --fold 1 2 \
  --output outputs/experiments_v1/stage3/probe/base/rN \
  --max-parallel 2 \
  --devices cuda:0,cuda:1
```

全部完成后生成只读 probe 报告：

```bash
python scripts/stage3/capacity.py \
  --manifest configs/experiments_v1/stage3/probe-report.yaml \
  --output outputs/experiments_v1/reports/probe
```

报告的 `scale_winners` 含唯一的 Base winner，按各 fold taskwise-refined stitched validation
和 R4→R3→R5→R2→R6→R1→R7→R8 tie-break 选出。
该命令自动汇总主指标、task/group 指标、fold sample-SD 和原始 run 路径；参数量、峰值
显存、吞吐、wall time 与 Stage 1/2 稳定性不会由报告器推断，必须从同类硬件上的训练
日志和监控记录中另行取证，并随人工 decision 一起保留。
超参数搜索、五折 confirmation、seed 配置物化和基于 trial 输出生成 final recipe 的能力已于
2026-09-06 退役。不要创建新的 HPO study，也不要尝试用当前代码恢复旧 SQLite study。
既有 HPO 输出只可作为历史证据读取，不再是运行输入。

## 4. 冻结的四规模 formal 配置

Stage 3 正式配方已经冻结在 `configs/experiments_v1/stage3/formal/{s,base,l,xl}.yaml`。
这些 YAML 是唯一现役 Capacity Stage 3 训练输入；不得从旧 HPO artifact 重新生成或修改其
model、optimizer、seed、epoch 与数据身份。对每个 `<scale>` 先准备数据：

```bash
python scripts/stage3/prepare.py \\
  --config configs/experiments_v1/stage3/formal/<scale>.yaml \\
  --output outputs/experiments_v1/stage3/formal/<scale>/prepare
```

然后运行五折训练：

```bash
python scripts/stage3/train.py \\
  --config configs/experiments_v1/stage3/formal/<scale>.yaml \\
  --fold 1 2 3 4 5 \\
  --output outputs/experiments_v1/stage3/formal/<scale>/train \\
  --max-parallel 4 \\
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

四个 scale 全部完成后，使用提交到仓库的固定 manifest 汇总 validation：

```bash
python scripts/stage3/capacity.py \\
  --manifest configs/experiments_v1/stage3/formal-report.yaml \\
  --output outputs/experiments_v1/reports/formal-validation
```

再把同类硬件上的参数量、峰值显存、吞吐和 wall time 证据附入决策记录。在查看 test
前写入 `outputs/experiments_v1/decisions/main-scale.yaml`，记录所选 scale、validation/resource
Pareto 理由和 formal report。然后才允许对四个 scale 各执行一次 test ensemble：

```bash
python scripts/stage3/evaluate.py \\
  --config configs/experiments_v1/stage3/formal/<scale>.yaml \\
  --checkpoint-dir outputs/experiments_v1/stage3/formal/<scale>/train \\
  --split test \\
  --ensemble-folds \\
  --study-id capacity-v1-<scale> \\
  --output outputs/experiments_v1/stage3/test/<scale>
```

Test 只发布四点 capacity trend，不得修改 main scale、冻结配方或 refined artifact。任何正式
命令失败时先保存原日志和 metadata；同配置最多原样重跑一次，不做动态 rescue。
