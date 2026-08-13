# ILUME

ILUME 以四模态掩码预训练（Stage 1）、五任务物性监督对齐（Stage 2）和双域 27 任务训练（Stage 3）组成科研 pipeline。仓库按 Stage 组织代码；现役科研合同由正式 YAML 与 [ADR](docs/adr/README.md) 共同定义。

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

`training.batch_size: 256` 是跨所有 rank 的 global batch。四卡训练使用每卡 64：

```bash
torchrun --nproc-per-node=4 scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train
```

恢复训练需显式指定同一操作目录中的 checkpoint，并保持 checkpoint 保存时的 `world_size`：

```bash
python scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train \
  --resume outputs/v1/stage1/base/train/last.pt
```

四卡 checkpoint 恢复时仍使用 `torchrun --nproc-per-node=4`，并在同一条命令末尾添加上述 `--resume` 参数。

## Stage 2

Stage 2 同样只保留一个正式 Base，并从 Stage 1 Base 的 v1 checkpoint 准备离线教师缓存：

```bash
python scripts/stage2/prepare.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/prepare

python scripts/stage2/train.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/train
```

恢复时增加 `--resume outputs/v1/stage2/base/train/last.pt`。

## Stage 3

Stage 3 reference 使用 Stage 2 Base 的 `best.pt`。五折需要分别执行 train，不提供 matrix runner：

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

每个操作目录包含 Git 可跟踪的 `run_config.yaml`、`metadata.json` 和成功后生成的 `summary.json`。Prepare payload 位于 `artifacts/` 子目录；checkpoint、metrics JSONL、日志和 tensor 默认不进入 Git。所有阶段保留全部周期 checkpoint，并用 `last.pt` 表示最新完整可恢复状态。

## Tests

正式运行前必须通过：

```bash
pytest -q
```

测试只使用临时小数据，不执行正式 prepare、教师缓存、训练或五折矩阵。科研与架构决定见 [docs/adr/README.md](docs/adr/README.md)。
