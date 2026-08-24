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

Stage 2 test evaluator 同时发布三个互不混淆的口径：Core 固定评估 heat of vaporization 和 cation/anion PBE/TZVP HOMO/LUMO 的 5 个 scalar unit；Partial Charge 使用确定性 MOL2 mapping 后的 all-mapped molecule-macro normalized MAE；Full 对 Core 5 unit 与 Partial 1 unit 等权平均。Test 实体只在 evaluation 进程内构建，不进入 prepared artifact 或训练：

```bash
python scripts/stage2/evaluate.py \
  --config configs/v1/stage2/base.yaml \
  --checkpoint-dir outputs/v1/stage2/base/train \
  --output outputs/v1/stage2/base/evaluate/test_benchmark_suite_v1
```

训练每个 epoch 完整覆盖所有任务的全部有效行，并以 deterministic randomized round-robin 交替 task batch；Object v3 强制一个 batch 对应一个 optimizer step，不支持 gradient accumulation。第一 epoch 的 object/interaction task 直接复用 teacher CLS，不运行 Stage 1；atom task仍运行冻结 Stage 1 取得 fusion atom states。后四个 epoch联合微调。Partial charge 只去重 Stage 1 entity forward，ObjectEncoder 与 AtomHead 按 molecule sample 向量化执行；atom target 全量保持 CPU resident。Task weight 归一化后只补偿 physics loss，teacher loss保持独立。Base 使用 4 个 ordered packing worker、包含 H2D 在内的 4 个逻辑预取名额，以及单 batch CUDA lookahead；transfer stream 通过 event 交接，不做逐 batch synchronize。CUDA 固定使用 TF32、pinned/non-blocking transfer 与 fused AdamW。每个 epoch full validation 后保存 `checkpoint_epoch_00001.pt` 至 `checkpoint_epoch_00005.pt`，epoch 5 后额外导出不含 physics heads 的 `stage2_encoder.pt`。只允许从完整 Object v3 epoch 恢复；不生成 `best.pt` 或 `last.pt`。旧 Object v2 以及缺少当前 preparation/extraction contract 的开发期 v3 artifact/cache/checkpoint 不迁移，必须重新 prepare。

截至 2026-08-19 的正式 Base 产物缺少 ADR-0021 identity contract v1 block，升级后不再兼容，必须按 Stage 1 → Stage 2 → Stage 3 顺序重新生成。归档旧正式输出前先做精确清单并等待单独确认。

## Stage 3

Stage 3 v1 从 catalog 解析 21 个 scalar observation task 和 6 个 meta-group。prepare 通过自包含 `stage2_encoder.pt` 建立按 encoder semantic identity 内容寻址的 FP32 object cache；train/evaluate 只读完整 Stage 3 artifact，不反向定位 Stage 1 或加载完整 Stage 2 checkpoint。每个任务保留独立 dataset，按 task-local fold 拟合 normalization，并使用固定 composite allocation 与 ownership-aware hierarchical PCGrad：GLOBAL、GROUP block 始终独立投影，PRIVATE 不参与投影。

每个正式 fold worker 只支持单进程单 CUDA GPU，默认 BF16；设备不支持时直接失败。唯一 train 入口可以用 spawn worker 并发调度多个独立 fold，测试可显式使用 CPU 与 `amp_dtype: none`。Base 的 training/validation `microbatch_size` 为 1024，改变它会产生新的 training identity。2026-08-23 对当前五折物化源的复核覆盖 21 个任务、232,889 行和 423,907 个 condition 值，未发现缺失或非有限 condition；未来任何声明 condition 的缺失仍然硬失败，Stage 3 不填充或删行。

## Baselines

MLP（RDKit 2D descriptors）与 ECFP4-XGBoost 对比基线位于 `benchmarks/`，由 [ADR-0022](docs/adr/0022-mlp-ecfp-xgboost-baselines.md) 定义。安装与单任务运行示例：

```bash
pip install -e ".[benchmarks]"
python scripts/benchmarks/train.py \
  --config configs/benchmarks/mlp.yaml \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output outputs/benchmarks/mlp/stage3/experiment__density/fold1/attempt-001
```

批量入口为 `python scripts/benchmarks/sweep.py --config configs/benchmarks/mlp.yaml --output outputs/benchmarks/mlp --max-workers 1`。`--max-workers` 控制同时运行的 train/evaluate 子进程数，默认值 1 保持串行行为；Stage 3 按 task × fold、Stage 2 Physics 按 task 并行，并保留逐 job 状态和依赖关系。MLP 与 ECFP+XGBoost 在 Stage 2 suite 中显式发布 Partial/Full `unsupported`，只参加 Core；旧 Stage 2 child evaluation 会视为 stale，在新 attempt 中复用已有训练 checkpoint 重新评估。MLP 可通过 `--devices cuda:0,cuda:1,...` 将逻辑 job 链 round-robin 分配到多张 GPU；不指定时多个 worker 共享 YAML 中现有的 `device: cuda`。XGBoost 继续使用 YAML 的 `training.n_jobs`，应按 `max_workers × training.n_jobs` 估算总 CPU 并行度并结合机器核心数设置，避免 oversubscription。直接运行在交互终端显示自身进度；sweep 只显示全局进度并关闭子进程动态进度，重定向到非 TTY 时保持静默。正式训练与评估仍由用户显式运行。

```bash
python scripts/stage3/prepare.py \
  --config configs/v1/stage3/base.yaml \
  --output outputs/v1/stage3/base/prepare

python scripts/stage3/train.py \
  --config configs/v1/stage3/base.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/v1/stage3/base/train \
  --max-parallel 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Stage 3 train 的 `--output` 始终表示五折共同 root，单 fold 也写入 `<output>/foldN`。默认 `--max-parallel 1`；并发大于 1 时必须用 `--devices` 显式给出 GPU，设备槽按列表 round-robin 绑定。一个设备配多个槽可显式同卡并发，例如 `--max-parallel 2 --devices cuda:0`。每个 fold 是独立的 spawn 进程；任一 fold 失败不阻断其余 fold，也不会自动降低并发或 microbatch。布尔 `--resume` 对已完成且 identity/历史完整的 fold 做 skip，对其余 fold 只从 checkpoint、metrics、diagnostics 尾部完全一致的位置恢复。

任何 `microbatch_size=128` 的旧训练都属于旧 training identity，不可恢复为新 Base。`--output` 路径不属于 training identity；截至 2026-08-24，本机 `outputs/v1/stage3/base/train/fold1` 至 `fold5` 均为 completed microbatch-1024 epoch-100 run，现役 evaluator 已只读解析五折 test ensemble identity，确认全部 checkpoint 与当前配置兼容。同卡并发的新运行可使用新的 output root：

```bash
python scripts/stage3/train.py \
  --config configs/v1/stage3/base.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/v1/stage3/base/train_same_gpu_new \
  --max-parallel 2 \
  --devices cuda:0
```

Validation 支持一次请求一个、部分或全部 fold，并始终按给定顺序单进程执行。`--output` 是 validation 的共同 root，每个 fold 都写入独立的 `<output>/foldN/` run directory：

```bash
python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train \
  --split valid --fold 1 2 3 4 5 --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train \
  --split valid --fold 2 4 --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_valid_subset

python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train \
  --split valid --fold 3 --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train \
  --split test --ensemble-folds --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_test
```

## Outputs

`--output` 是一次操作的独立目录；Stage 3 train 与 validation evaluate 使用共同 root，其下每个 `foldN/` 才是独立 run directory。Stage 3 test ensemble 仍直接使用自身的 `--output`。新训练和 evaluate 拒绝覆盖已有正式 run directory；只有显式 `--resume` 可继续训练。Stage 1/2 prepare 按各自 artifact 身份合同支持受控复用；Stage 3 prepare 拒绝已有输出目录，只能发布到新的空输出。

每个操作目录包含 Git 可跟踪的 `run_config.yaml`、`metadata.json`、逐 attempt 追加的 `attempts.jsonl` 和成功后生成的 `summary.json`。Prepare payload 位于 `artifacts/` 子目录；checkpoint、训练 metrics JSONL、日志和 tensor 默认不进入 Git。Stage 1 用 `last.pt` 表示最新完整恢复状态；Stage 2 直接通过最新的完整 epoch checkpoint 恢复；Stage 3 Base 默认只保存 epoch 10、20、…、100 的完整 checkpoint，不生成 best/last。

## 结果汇总

[ADR-0023](docs/adr/0023-unified-evaluation-reporting.md) 与 [ADR-0024](docs/adr/0024-stage2-partial-charge-benchmark-suite.md) 将完整运行与论文结果分层：`outputs/` 保留 prediction/checkpoint/audit，`summary/` 原子发布 Stage 3 TEST、Stage 3 VALIDATION、Stage 2 CORE、Partial Charge、Stage 2 FULL 三类 Stage 2/两类 Stage 3 榜单及相应明细、health、overview 和机器可读 summary。生成或刷新汇总：

```bash
python scripts/benchmarks/summarize.py --input outputs --output summary
```

只有 reporting schema 完整且 comparison identity 一致的 completed evaluation/sweep 进入对应排名；缺少 `stage2-benchmark-suite-v1` 的旧 Stage 2 结果、失败、运行中或不完整实验只进入 health。明确 `unsupported` 的模型仍可进入 Core，但不进入 Partial/Full。发现声称为当前合同的损坏正式结果时命令失败，已有 `summary/` 保持不变。

Stage 2 suite 的正式刷新命令如下；baseline sweep 会跳过已完成的 Stage 3 与训练，只重跑 stale 的 Stage 2 child evaluation。ILUME evaluation 必须使用新的 output 路径：

```bash
python scripts/stage2/evaluate.py \
  --config configs/v1/stage2/base.yaml \
  --checkpoint-dir outputs/v1/stage2/base/train \
  --output outputs/v1/stage2/base/evaluate/test_benchmark_suite_v1

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/mlp.yaml \
  --output outputs/benchmarks/mlp --max-workers 1

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/ecfp_xgboost.yaml \
  --output outputs/benchmarks/ecfp_xgboost --max-workers 1

python scripts/benchmarks/summarize.py --input outputs --output summary
```

Partial Charge 合同本身不要求重跑 Stage 3；但缺少 reporting schema v1 的旧 ILUME Stage 3 evaluation 仍只进入 health，不能进入 Stage 3 榜。需要让 ILUME 参加 Stage 3 TEST/VALIDATION 榜时，复用既有 epoch-100 checkpoint 执行上文的 Stage 3 evaluation 命令即可，不需要重跑训练。

## Tests

正式运行前必须通过：

```bash
pytest -q
```

测试只使用临时小数据，不执行正式 prepare、教师缓存、训练或五折矩阵。科研与架构决定见 [docs/adr/README.md](docs/adr/README.md)。
