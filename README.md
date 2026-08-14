# ILUME

ILUME 以四模态掩码预训练（Stage 1）、五任务物性 Object 建模（Stage 2）和双域 27 任务训练（Stage 3）组成科研 pipeline。仓库按 Stage 组织代码；现役科研合同由正式 YAML 与 [ADR](docs/adr/README.md) 共同定义。

## Installation

```bash
python -m pip install -e ".[dev,tokenizers]"
```

## Data

数据本体由独立的 ILUME-Data 仓库生成，并放在 `data/stage1`、`data/stage2`、`data/stage3`。CSV 不进入 Git。每次 prepare 会自动更新对应的 `data/stage*/metadata.json`，记录实际消费文件的相对路径、SHA256、字节数、数据行数及 ILUME-Data Git 状态。

## Stage 1

Stage 1 只保留一个正式 Base。prepare 会消费三类 original；当 `include_augmentation: true` 时还会要求并全量处理三份 augmentation CSV。正式 Base 默认使用 16 个 prepare worker；可用 `--workers` 临时覆盖。corpus v2 不兼容旧 artifact，必须重新 prepare：

```bash
python scripts/stage1/prepare.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/prepare

# 临时覆盖 YAML 中的 preparation.workers
python scripts/stage1/prepare.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/prepare \
  --workers 24

python scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train
```

正式 Base 默认 `training.compile: false`，直接使用 eager 执行。如需开启编译，必须在冻结的 YAML 中显式设为 `true`；编译失败仍会明确终止，不会静默回退。

prepare 期间可查看同一输出目录下的 `performance.json`，其中记录本次 invocation 各 phase 的处理量、耗时、吞吐和复用状态；该文件不参与 corpus 或 checkpoint 身份。

`training.batch_size: 128` 是跨所有 rank 的 global batch。四卡训练使用每卡 32：

```bash
torchrun --nproc-per-node=4 scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train
```

Stage 1 只从已完成 epoch 的 v2 checkpoint 恢复；中断的 epoch 会从头重跑。恢复仍需显式指定同一操作目录中的 checkpoint：

```bash
python scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train \
  --resume outputs/v1/stage1/base/train/last.pt
```

可以在 epoch 边界改变 GPU 数量；这会生成新的 execution attempt 并记录新的 `world_size`，恢复后的随机轨迹不保证与原运行一致。保持相同 GPU 数、compile 设置和软件/硬件环境时，新版训练器保证自身的 epoch-boundary 可复现性。

## Stage 2

Stage 2 只保留一个正式 Base，以共享 ObjectEncoder 建模 molecule 与 IL，并从 Stage 1 Base 的 v2 checkpoint 准备内容寻址的 FP32 entity teacher cache。Object v2 prepare 单次扫描 CSV，并行完成确定性的 entity feature/QC，直接发布 train-ready normalized tensor；train 启动时严格校验并 preload 全部 entity shard：

```bash
python scripts/stage2/prepare.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/prepare

python scripts/stage2/train.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/train
```

训练每个 epoch 完整覆盖五个任务的全部有效行。第一 epoch 直接复用 GPU teacher embedding，不运行 packer 或 Stage 1 backbone；后四个 epoch 在 accumulation window 内跨任务去重，只执行一次 pack/backbone encode。CUDA 固定使用 TF32、pinned/non-blocking transfer 与 fused AdamW。每个 epoch 在 full validation 后保存 `checkpoint_epoch_00001.pt` 至 `checkpoint_epoch_00005.pt`。只允许从完整 Object v2 epoch 恢复，例如增加 `--resume outputs/v1/stage2/base/train/checkpoint_epoch_00003.pt`；不生成 `best.pt` 或 `last.pt`。旧 Object v1 artifact/cache/checkpoint 不迁移，必须重新 prepare。

## Stage 3

Stage 3 到新 Stage 2 object v2 的表示迁移尚未完成。当前 `scripts/stage3/prepare.py` 会在写入 artifact 前明确拒绝运行；以下历史命令暂不可用于新 Stage 2 checkpoint，待 neutral-pair 与 single topology 合同另行确定后恢复：

```bash
python scripts/stage3/prepare.py \
  --config configs/v1/stage3/reference.yaml \
  --output outputs/v1/stage3/reference/prepare

python scripts/stage3/train.py \
  --config configs/v1/stage3/reference.yaml \
  --fold 1 \
  --output outputs/v1/stage3/reference/checkpoints/fold1
```

验证集汇总和 test ensemble：

```bash
python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/reference.yaml \
  --checkpoint-dir outputs/v1/stage3/reference/checkpoints \
  --split valid \
  --output outputs/v1/stage3/reference/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/reference.yaml \
  --checkpoint-dir outputs/v1/stage3/reference/checkpoints \
  --split test --ensemble-folds \
  --output outputs/v1/stage3/reference/evaluate_test
```

## Outputs

`--output` 是一次操作的独立目录。新训练和 evaluate 拒绝覆盖已有目录；只有显式 `--resume` 可继续训练。Prepare 保留可校验的幂等复用。

每个操作目录包含 Git 可跟踪的 `run_config.yaml`、`metadata.json` 和成功后生成的 `summary.json`。Prepare payload 位于 `artifacts/` 子目录；checkpoint、metrics JSONL、日志和 tensor 默认不进入 Git。所有阶段保留全部周期 checkpoint；Stage 1/3 用 `last.pt` 表示最新完整恢复状态，Stage 2 直接通过最新的完整 epoch checkpoint 恢复。

## Tests

正式运行前必须通过：

```bash
pytest -q
```

测试只使用临时小数据，不执行正式 prepare、教师缓存、训练或五折矩阵。科研与架构决定见 [docs/adr/README.md](docs/adr/README.md)。
