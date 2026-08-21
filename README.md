# ILUME

ILUME 以四模态掩码预训练（Stage 1）、catalog 驱动的九任务 physics representation 训练（Stage 2）和 sparse-label 21 任务训练（Stage 3）组成科研 pipeline。仓库按 Stage 组织代码；现役科研合同由正式 YAML 与 [ADR](docs/adr/README.md) 共同定义。

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

截至 2026-08-19 的正式 Base 产物缺少 ADR-0021 identity contract v1 block，升级后不再兼容，必须按 Stage 1 → Stage 2 → Stage 3 顺序重新生成。归档旧正式输出前先做精确清单并等待单独确认。

## Stage 3

Stage 3 v1 从 catalog 解析 21 个 scalar observation task 和 6 个 meta-group。prepare 通过自包含 `stage2_encoder.pt` 建立按 encoder semantic identity 内容寻址的 FP32 object cache；train/evaluate 只读完整 Stage 3 artifact，不反向定位 Stage 1 或加载完整 Stage 2 checkpoint。每个任务保留独立 dataset，按 task-local fold 拟合 normalization，并使用固定 composite allocation 与 ownership-aware hierarchical PCGrad：GLOBAL、GROUP block 始终独立投影，PRIVATE 不参与投影。

正式 Base 仅支持单进程单 CUDA GPU，默认 BF16；设备不支持时直接失败。测试可显式使用 CPU 与 `amp_dtype: none`。截至 2026-08-20，当前五折 CSV 中有 6 个任务同时包含有值和缺失的 `pressure_kPa`，缺失合计 50,300 行，并非这些任务整列都没有压力。正式 prepare 会按合同拒绝；需先由 ILUME-Data 逐行补齐，或在确认压力不是该任务合法条件后同步修改 catalog `condition_columns`，Stage 3 不负责填充或删行。

## Baselines

MLP（RDKit 2D descriptors）与 ECFP4-XGBoost 对比基线位于 `benchmarks/`，由 [ADR-0022](docs/adr/0022-mlp-ecfp-xgboost-baselines.md) 定义。安装与单任务运行示例：

```bash
pip install -e ".[benchmarks]"
python scripts/benchmarks/train.py \
  --config configs/benchmarks/mlp.yaml \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output outputs/benchmarks/mlp/stage3/experiment__density/fold1/attempt-001
```

批量入口为 `python scripts/benchmarks/sweep.py --config configs/benchmarks/mlp.yaml --output outputs/benchmarks/mlp`。它顺序执行并保留逐 job 状态；直接运行与 sweep 在交互终端显示 feature、epoch/boosting round 和训练 job 的 tqdm 进度，重定向到非 TTY 时保持静默。当前 Stage 3 缺失压力修复前，完整 21-task sweep 会按现役 condition 合同失败。正式训练与评估仍由用户显式运行。

```bash
python scripts/stage3/prepare.py \
  --config configs/v1/stage3/base.yaml \
  --output outputs/v1/stage3/base/prepare

python scripts/stage3/train.py \
  --config configs/v1/stage3/base.yaml \
  --fold 1 \
  --output outputs/v1/stage3/base/train/fold1
```

验证集汇总和 test ensemble：

```bash
python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train/fold1 \
  --split valid --fold 1 --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_valid_fold1

python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train \
  --split test --ensemble-folds --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_test
```

## Outputs

`--output` 是一次操作的独立目录。新训练和 evaluate 拒绝覆盖已有目录；只有显式 `--resume` 可继续训练。Stage 1/2 prepare 按各自 artifact 身份合同支持受控复用；Stage 3 prepare 拒绝已有输出目录，只能发布到新的空输出。

每个操作目录包含 Git 可跟踪的 `run_config.yaml`、`metadata.json`、逐 attempt 追加的 `attempts.jsonl` 和成功后生成的 `summary.json`。Prepare payload 位于 `artifacts/` 子目录；checkpoint、训练 metrics JSONL、日志和 tensor 默认不进入 Git。Stage 1 用 `last.pt` 表示最新完整恢复状态；Stage 2 直接通过最新的完整 epoch checkpoint 恢复；Stage 3 Base 默认只保存 epoch 10、20、…、100 的完整 checkpoint，不生成 best/last。

## Tests

正式运行前必须通过：

```bash
pytest -q
```

测试只使用临时小数据，不执行正式 prepare、教师缓存、训练或五折矩阵。科研与架构决定见 [docs/adr/README.md](docs/adr/README.md)。
