# ILUME

ILUME 是按 Stage 组织的分子科研 pipeline：Stage 1 进行四模态掩码预训练，Stage 2 训练 catalog 驱动的九任务 physics representation，Stage 3 训练 21 个 sparse-label observation task。正式 YAML 与 [ADR 索引](docs/adr/README.md) 共同定义现役科研合同。

## 安装与数据

```bash
python -m pip install -e ".[dev,tokenizers]"
```

ILUME-Data 生成的数据放在 `data/stage1`、`data/stage2`、`data/stage3`；CSV 不进入 Git。prepare 会更新相应的 `data/stage*/metadata.json`，记录实际输入及其完整性信息。

## Stage 1

Stage 1 只有一个正式 Base。现役 corpus、训练、恢复和 runtime 合同见 [ADR-0013/0014/0015/0017](docs/adr/README.md)。

```bash
python scripts/stage1/prepare.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/prepare

python scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train
```

多卡训练使用原生 DDP；`training.batch_size` 是 global batch：

```bash
torchrun --nproc-per-node=4 scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train
```

只支持完整 epoch checkpoint 恢复。默认 eager；如在 YAML 中显式开启 compile，编译失败会直接终止，不会静默回退。

```bash
python scripts/stage1/train.py \
  --config configs/v1/stage1/base.yaml \
  --output outputs/v1/stage1/base/train \
  --resume outputs/v1/stage1/base/train/last.pt
```

## Stage 2

Stage 2 Object v3 从 catalog 加载九个 simulation task，共享 ObjectEncoder，并从 Stage 1 encoder 准备 entity teacher cache。模型、数据身份、恢复和 HOMO/LUMO 合同见 [ADR-0019/0021/0025](docs/adr/README.md)。

```bash
python scripts/stage2/prepare.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/prepare

python scripts/stage2/train.py \
  --config configs/v1/stage2/base.yaml \
  --output outputs/v1/stage2/base/train

python scripts/stage2/evaluate.py \
  --config configs/v1/stage2/base.yaml \
  --checkpoint-dir outputs/v1/stage2/base/train \
  --output outputs/v1/stage2/base/evaluate/test_benchmark_suite_v2
```

Evaluator 分别发布 Core、Partial Charge 和 Full 三榜；评分及 eligibility 合同见 [ADR-0023/0024/0025](docs/adr/README.md)。Stage 2 只从完整 Object v3 epoch 恢复，旧 Object v2 和缺少现役合同的开发期 v3 输出不迁移。

## Stage 3

Stage 3 使用冻结的 Stage 2 Object v3 表示、动态 HoME 和 ownership-aware hierarchical PCGrad。数据、模型、五折调度和恢复合同见 [ADR-0020/0021](docs/adr/README.md)。

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

`--output` 是所有 fold 的共同 root，实际 run 位于 `<output>/foldN`。默认串行；并发训练必须显式提供设备槽。`--resume` 会跳过身份一致且完整的 fold，其余 fold 只从相互一致的完整 epoch checkpoint 与历史尾部恢复。

```bash
python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train \
  --split valid --fold 1 2 3 4 5 --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/v1/stage3/base.yaml \
  --checkpoint-dir outputs/v1/stage3/base/train \
  --split test --ensemble-folds --checkpoint-epoch 100 \
  --output outputs/v1/stage3/base/evaluate_test
```

独立的 Capacity v1 研究不替换正式 v1 合同；设计见 [ADR-0026](docs/adr/0026-capacity-v1-pipeline-study.md)，正式命令集中在 [Capacity v1 操作手册](docs/capacity-v1-runbook.md)。

## Baselines

MLP 与 ECFP4-XGBoost 位于 `benchmarks/`，与 Stage 代码隔离；合同见 [ADR-0022](docs/adr/0022-mlp-ecfp-xgboost-baselines.md)。

```bash
python -m pip install -e ".[benchmarks]"

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/mlp.yaml \
  --output outputs/benchmarks/v1/mlp \
  --max-workers 1

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/ecfp_xgboost.yaml \
  --output outputs/benchmarks/v1/ecfp_xgboost \
  --max-workers 1
```

`--max-workers 1` 保持串行行为。MLP 多 GPU sweep 可通过 `--devices cuda:0,cuda:1,...` 分配逻辑 job；XGBoost 的 CPU 并行度由 YAML 中的 `training.n_jobs` 控制。

## 输出与结果汇总

新 train/evaluate 不覆盖既有输出，恢复必须显式请求。每个操作目录冻结 `run_config.yaml`，写入公开安全的 `metadata.json`，成功后生成 `summary.json`；checkpoint、训练日志和 tensor 默认不进入 Git。完整身份与 checkpoint 规则见 [ADR-0021](docs/adr/0021-identity-audit-contract-v1.md)。

全局 summarizer 只收录显式选中的目录。`--input` 提供一个或多个扫描根；可选 `--include` 是精确目录前缀白名单，省略时扫描全部 input。include 必须存在、位于某个 input 内并至少匹配一个 reporting candidate；重叠路径会去重，不支持 glob。

```bash
python scripts/benchmarks/summarize.py \
  --input outputs/v1 outputs/benchmarks \
  --include \
    outputs/v1/stage3/base \
    outputs/v1/stage2/base/evaluate/test_benchmark_suite_v2 \
    outputs/benchmarks/v1/mlp \
    outputs/benchmarks/v1/ecfp_xgboost \
  --output summary
```

只有 schema 完整且 comparison identity 一致的 completed run 进入榜单；其他 run 进入 health。损坏的选中正式结果会使发布失败，已有 `summary/` 保持不变。详细 reporting 合同见 [ADR-0023/0024/0025](docs/adr/README.md)。

## 验证

```bash
pytest -q
```

测试只使用临时小数据，不执行正式 prepare、teacher cache、训练或五折 evaluation。
