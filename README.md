# ILUME

ILUME 以四模态掩码预训练（Stage 1）、catalog 驱动的九任务 physics representation 训练（Stage 2）和双域 27 任务训练（Stage 3）组成科研 pipeline。仓库按 Stage 组织代码；现役科研合同由正式 YAML 与 [ADR](docs/adr/README.md) 共同定义。

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

Stage 2 只保留一个正式 Base。Object v3 从 `task_catalog.csv` 动态加载九个 simulation task，以共享 ObjectEncoder 建模 single entity、ionic liquid 与 interaction，并从 Stage 1 Base 的 v2 checkpoint 准备内容寻址的 FP32 entity teacher cache。Prepared data只绑定数据、Stage 1 feature、registry与tensor contract；teacher cache只绑定Stage 1 encoder与entity artifact。ObjectEncoder layers、FFN或dropout变化不要求重新prepare或提取teacher，但不同模型配置仍不可互相resume。Prepare 发布 task-local normalized object/atom tensor；partial charge 通过可审计的 typed/fallback MOL2 graph mapping 对齐到 Stage 1 atom ordering：

```bash
python scripts/stage2/prepare.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/prepare

python scripts/stage2/train.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/train
```

训练每个 epoch 完整覆盖所有任务的全部有效行，并以 deterministic randomized round-robin 交替 task batch；Object v3 强制一个 batch 对应一个 optimizer step，不支持 gradient accumulation。第一 epoch 的 object/interaction task 直接复用 teacher CLS，不运行 Stage 1；atom task仍运行冻结 Stage 1 取得 fusion atom states。后四个 epoch联合微调。Partial charge 只去重 Stage 1 entity forward，ObjectEncoder 与 AtomHead 按 molecule sample 向量化执行；atom target 全量保持 CPU resident。Task weight 归一化后只补偿 physics loss，teacher loss保持独立。Base 使用 4 个 ordered packing worker、包含 H2D 在内的 4 个逻辑预取名额，以及单 batch CUDA lookahead；transfer stream 通过 event 交接，不做逐 batch synchronize。CUDA 固定使用 TF32、pinned/non-blocking transfer 与 fused AdamW。每个 epoch full validation 后保存 `checkpoint_epoch_00001.pt` 至 `checkpoint_epoch_00005.pt`，epoch 5 后额外导出不含 physics heads 的 `stage2_encoder.pt`。只允许从完整 Object v3 epoch 恢复；不生成 `best.pt` 或 `last.pt`。旧 Object v2 以及缺少当前 preparation/extraction contract 的开发期 v3 artifact/cache/checkpoint 不迁移，必须重新 prepare。

## Stage 3

Stage 3 到 Stage 2 Object v3 encoder artifact 的表示迁移尚未完成。当前 `scripts/stage3/prepare.py` 会在写入 artifact 前明确拒绝 Object v3 checkpoint 与 `stage2_encoder.pt`；以下历史命令暂不可用，待独立迁移完成后恢复：

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
