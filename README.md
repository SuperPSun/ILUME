# ILUME

ILUME 以四模态掩码预训练（Stage 1）、五任务物性监督对齐（Stage 2）和双域 27 任务训练（Stage 3）组成科研 pipeline。仓库按 Stage 组织代码；重构不改变数据、模型、loss、sampling、训练顺序或 evaluation protocol。

## Installation

```bash
python -m pip install -e ".[dev,tokenizers]"
```

## Data

数据本体由独立的 ILUME-Data 仓库生成，并放在 `data/stage1`、`data/stage2`、`data/stage3`。CSV 不进入 Git。每次 prepare 会自动更新对应的 `data/stage*/metadata.json`，记录实际消费文件的相对路径、SHA256、字节数、数据行数及 ILUME-Data Git 状态。

## Stage 1

三种模型容量共享 Base prepared data：

```bash
python scripts/stage1/prepare.py \
  --config configs/formal/stage1/base.yaml \
  --output outputs/formal_v1/stage1/base/prepare

python scripts/stage1/train.py \
  --config configs/formal/stage1/base.yaml \
  --output outputs/formal_v1/stage1/base/train
```

Base prepare 只需执行一次。训练 Large 或 XLarge 时复用该 prepared data，只替换训练命令中的配置名和训练输出容量，例如 `configs/formal/stage1/large.yaml` 与 `outputs/formal_v1/stage1/large/train`。恢复训练需显式指定同一操作目录中的 checkpoint：

```bash
python scripts/stage1/train.py \
  --config configs/formal/stage1/base.yaml \
  --output outputs/formal_v1/stage1/base/train \
  --resume outputs/formal_v1/stage1/base/train/last.pt
```

## Stage 2

Stage 2 三种容量分别准备与对应 Stage 1 checkpoint 匹配的离线教师缓存：

```bash
python scripts/stage2/prepare.py \
  --config configs/formal/stage2/base.yaml \
  --output outputs/formal_v1/stage2/base/prepare

python scripts/stage2/train.py \
  --config configs/formal/stage2/base.yaml \
  --output outputs/formal_v1/stage2/base/train
```

恢复时增加 `--resume outputs/formal_v1/stage2/base/train/last.pt`。

## Stage 3

Stage 3 reference 使用 Stage 2 Base 的 `best.pt`。五折需要分别执行 train，不提供 matrix runner：

```bash
python scripts/stage3/prepare.py \
  --config configs/formal/stage3/reference.yaml \
  --output outputs/formal_v1/stage3/reference/prepare

python scripts/stage3/train.py \
  --config configs/formal/stage3/reference.yaml \
  --fold 1 \
  --output outputs/formal_v1/stage3/reference/checkpoints/fold1
```

验证集汇总和 test ensemble：

```bash
python scripts/stage3/evaluate.py \
  --config configs/formal/stage3/reference.yaml \
  --checkpoint-dir outputs/formal_v1/stage3/reference/checkpoints \
  --split valid \
  --output outputs/formal_v1/stage3/reference/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/formal/stage3/reference.yaml \
  --checkpoint-dir outputs/formal_v1/stage3/reference/checkpoints \
  --split test --ensemble-folds \
  --output outputs/formal_v1/stage3/reference/evaluate_test
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
