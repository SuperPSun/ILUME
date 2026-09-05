# ILUME

ILUME 是按 Stage 组织的分子科研 pipeline：Global-RDKit v2 主线在 Stage 1 进行 SMILES、Graph、RDKit 三模态四目标掩码预训练，Stage 2 训练 catalog 驱动的九任务 physics representation，Stage 3 训练 21 个 sparse-label observation task。正式 YAML 与 [ADR 索引](docs/adr/README.md) 共同定义现役科研合同。

## 安装与数据

```bash
python -m pip install -e ".[dev,tokenizers]"
```

ILUME-Data 生成的数据放在 `data/stage1`、`data/stage2`、`data/stage3`；CSV 不进入 Git。prepare 会更新相应的 `data/stage*/metadata.json`，记录实际输入及其完整性信息。

## Stage 1

Stage 1 只有一个 v2 正式 Base。架构与跨 Stage 表示见 [ADR-0039](docs/adr/0039-global-rdkit-v2-mainline.md)，corpus、训练、恢复和 runtime 合同见 [ADR-0013/0014/0015/0017](docs/adr/README.md)。

```bash
python scripts/stage1/prepare.py \
  --config configs/v2/stage1/base.yaml \
  --output outputs/v2/stage1/base/prepare

python scripts/stage1/train.py \
  --config configs/v2/stage1/base.yaml \
  --output outputs/v2/stage1/base/train
```

多卡训练使用原生 DDP；`training.batch_size` 是 global batch：

```bash
torchrun --nproc-per-node=4 scripts/stage1/train.py \
  --config configs/v2/stage1/base.yaml \
  --output outputs/v2/stage1/base/train
```

只支持完整 epoch checkpoint 恢复。默认 eager；如在 YAML 中显式开启 compile，编译失败会直接终止，不会静默回退。

```bash
python scripts/stage1/train.py \
  --config configs/v2/stage1/base.yaml \
  --output outputs/v2/stage1/base/train \
  --resume outputs/v2/stage1/base/train/last.pt
```

## Stage 2

Stage 2 Object v3 从 catalog 加载九个 simulation task，共享 ObjectEncoder，并从 Stage 1 encoder 准备 entity teacher cache；最后 20% epoch 冻结共享表示并独立优化各 task head。模型、数据身份、恢复和选择合同见 [ADR-0019/0021/0025/0027](docs/adr/README.md)。

```bash
python scripts/stage2/prepare.py \
  --config configs/v2/stage2/base.yaml \
  --output outputs/v2/stage2/base/prepare

python scripts/stage2/train.py \
  --config configs/v2/stage2/base.yaml \
  --output outputs/v2/stage2/base/train

python scripts/stage2/evaluate.py \
  --config configs/v2/stage2/base.yaml \
  --checkpoint-dir outputs/v2/stage2/base/train \
  --output outputs/v2/stage2/base/evaluate/test_benchmark_suite_v2
```

Evaluator 默认加载 `taskwise_refined.pt`；只有显式传入 `--checkpoint-epoch N` 时才加载普通历史 checkpoint。它分别发布 Core、Partial Charge 和 Full 三榜；评分及 eligibility 合同见 [ADR-0023/0024/0025](docs/adr/README.md)。Stage 2 只从完整 Object v3 epoch 恢复，旧 Object v2 和缺少现役合同的开发期 v3 输出不迁移。

## Stage 3

Stage 3 使用冻结的 Stage 2 Object v3 表示、动态 HoME 和 ownership-aware hierarchical PCGrad；最后 20% epoch 冻结 GLOBAL/GROUP，仅优化各 task PRIVATE。数据、模型、五折调度和恢复合同见 [ADR-0020/0021/0027](docs/adr/README.md)。

```bash
python scripts/stage3/prepare.py \
  --config configs/v2/stage3/base.yaml \
  --output outputs/v2/stage3/base/prepare

python scripts/stage3/train.py \
  --config configs/v2/stage3/base.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base/train \
  --max-parallel 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

`--output` 是所有 fold 的共同 root，实际 run 位于 `<output>/foldN`。默认串行；并发训练必须显式提供设备槽。`--resume` 会跳过身份一致且完整的 fold，其余 fold 只从相互一致的完整 epoch checkpoint 与历史尾部恢复。

```bash
python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base.yaml \
  --checkpoint-dir outputs/v2/stage3/base/train \
  --split valid --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base.yaml \
  --checkpoint-dir outputs/v2/stage3/base/train \
  --split test --ensemble-folds \
  --output outputs/v2/stage3/base/evaluate_test
```

Stage 3 evaluator 同样默认加载每个 fold 的 `taskwise_refined.pt`；显式
`--checkpoint-epoch N` 才选择对应普通 epoch checkpoint。

### v2 Stage 3 三阶段超参搜索

[ADR-0041](docs/adr/0041-stage3-v2-three-phase-hpo.md) 固定只搜索 Base Stage 3：
Search A 50个grouping、Search B 30个expert结构、Search C 20个loss/optimizer recipe。
每个trial只运行fold1/2和20 epochs，总预算为100个配置、200个正常fold run。搜索需要
Optuna，但不会改动Stage 1/2 Base参数：

```bash
python -m pip install -e ".[hpo]"

python scripts/stage3/prepare.py \
  --config configs/v2/stage3/base.yaml \
  --output outputs/v2/stage3/search/prepare

python scripts/stage3/search.py \
  --study-config configs/v2/stage3/search.yaml \
  --phase a \
  --output outputs/v2/stage3/search \
  --devices cuda:0,cuda:1 \
  --max-parallel 2

python scripts/stage3/search.py \
  --study-config configs/v2/stage3/search.yaml \
  --phase b \
  --output outputs/v2/stage3/search \
  --devices cuda:0,cuda:1 \
  --max-parallel 2

python scripts/stage3/search.py \
  --study-config configs/v2/stage3/search.yaml \
  --phase c \
  --output outputs/v2/stage3/search \
  --devices cuda:0,cuda:1 \
  --max-parallel 2
```

`--max-parallel`表示并行 fold worker 数，不要求手工重复设备名。程序会按 round-robin
自动扩展`--devices`：例如`--max-parallel 16 --devices cuda:0,cuda:1`会平均分配为每卡
8个槽，`--max-parallel 2 --devices cuda:0`则让两个fold竞争同一张卡。后者需要自行确认
显存足够。中断后原命令追加`--resume`。B严格绑定A的`result.json`，C严格绑定A/B结果；
不得修改已启动study的配置或候选manifest。最终Base recipe写入
`outputs/v2/stage3/search/search_c/winner_base.yaml`，不会自动运行test或推广到其他scale。

A/B/C完成后可生成只读的两折搜索报告：

```bash
python scripts/stage3/search_report.py \
  --search-output outputs/v2/stage3/search
```

报告默认写入`outputs/v2/stage3/search/search_report/`，包括完整trial与逐性质CSV、
三阶段性质矩阵和Base/A/B/C winner雷达图。报告只读取搜索保存的fold1/2
taskwise-refined stitched validation，不重新训练、不调用evaluation，也不读取test。

独立的 Capacity v1 研究继续冻结在 legacy v1 五模态合同，不替换或修改 Global-RDKit v2 主线；设计见 [ADR-0026](docs/adr/0026-capacity-v1-pipeline-study.md)，正式命令集中在 [Capacity v1 操作手册](docs/capacity-v1-runbook.md)。

### RDKit 2D → HoME representation ablation

[ADR-0034](docs/adr/0034-rdkit-2d-home-representation-ablation.md) 只用
RDKit 2D descriptors 与两个可训练的 `Linear → LayerNorm` adapter 替换 frozen
Stage2 Object representation；HoME、PCGrad、sampling、refinement 和 evaluation 保持
Stage3 Base 合同。该实验不读取 Stage1/2 checkpoint，也不单独 HPO。

```bash
python scripts/stage3/prepare.py \
  --config configs/ablations/stage1_stage2_rdkit_home.yaml \
  --output outputs/ablations/stage1_stage2_rdkit_home/prepare

python scripts/stage3/train.py \
  --config configs/ablations/stage1_stage2_rdkit_home.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/ablations/stage1_stage2_rdkit_home/train

python scripts/stage3/evaluate.py \
  --config configs/ablations/stage1_stage2_rdkit_home.yaml \
  --checkpoint-dir outputs/ablations/stage1_stage2_rdkit_home/train \
  --split valid --fold 1 2 3 4 5 \
  --output outputs/ablations/stage1_stage2_rdkit_home/evaluate/valid

python scripts/stage3/evaluate.py \
  --config configs/ablations/stage1_stage2_rdkit_home.yaml \
  --checkpoint-dir outputs/ablations/stage1_stage2_rdkit_home/train \
  --split test --ensemble-folds \
  --output outputs/ablations/stage1_stage2_rdkit_home/evaluate/test

python scripts/benchmarks/summarize.py \
  --input outputs/v1 outputs/benchmarks outputs/ablations \
  --output summary
```

五折训练命令默认串行；如需显式多 GPU 调度，只增加 `--max-parallel` 与 `--devices`，
不改变 scientific identity。正式执行前应保证对应输出目录不存在；恢复必须显式添加
`--resume`。

### No-Stage1：RDKit 2D → Stage2 → Stage3 HoME

[ADR-0036](docs/adr/0036-no-stage1-rdkit-stage2-stage3-ablation.md) 用共享的
`217D → 1024D → 512D` RDKit MLP 替换 Stage1 backbone，保留 Stage2 ObjectEncoder 与
Stage3 Base。该路径不读取 Stage1 artifact/checkpoint 或 teacher cache；Stage2 训练八个
object/interaction task，Partial Charge 与 Full 为 unsupported。

```bash
python scripts/stage2/prepare.py \
  --config configs/ablations/no_stage1_rdkit_stage2.yaml \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/prepare

python scripts/stage2/train.py \
  --config configs/ablations/no_stage1_rdkit_stage2.yaml \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/train

python scripts/stage2/evaluate.py \
  --config configs/ablations/no_stage1_rdkit_stage2.yaml \
  --checkpoint-dir outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/train \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/evaluate/test

python scripts/stage3/prepare.py \
  --config configs/ablations/no_stage1_rdkit_stage3.yaml \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/prepare

python scripts/stage3/train.py \
  --config configs/ablations/no_stage1_rdkit_stage3.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/train

python scripts/stage3/evaluate.py \
  --config configs/ablations/no_stage1_rdkit_stage3.yaml \
  --checkpoint-dir outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/train \
  --split valid --fold 1 2 3 4 5 \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/valid

python scripts/stage3/evaluate.py \
  --config configs/ablations/no_stage1_rdkit_stage3.yaml \
  --checkpoint-dir outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/train \
  --split test --ensemble-folds \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage3/evaluate/test

python scripts/benchmarks/summarize.py \
  --input outputs/v1 outputs/benchmarks outputs/ablations \
  --output summary
```

Stage2/Stage3 resume 分别在上述 train 命令追加 `--resume <checkpoint>` 与 `--resume`；新
输出不得覆盖既有目录。Stage3 五折默认串行，多 GPU 调度只增加 `--max-parallel` 和
`--devices`，不改变实验 identity。

## Baselines and Ablations

MLP、ECFP4-XGBoost、Chemprop D-MPNN、MoLFormer、ILBERT、SPMM 与 LlaSMol 位于 `benchmarks/`；
Stage3 Single-task MLP 内部消融位于 `ablations/`。二者均与 Stage 代码隔离，并继续
复用 benchmark 运行与 reporting 入口；合同见 [ADR-0022](docs/adr/0022-mlp-ecfp-xgboost-baselines.md)、[ADR-0028](docs/adr/0028-chemprop-dmpnn-baseline.md)、[ADR-0029](docs/adr/0029-molformer-baseline.md)、[ADR-0030](docs/adr/0030-molformer-throughput-contract.md)、[ADR-0032](docs/adr/0032-ilbert-baseline.md)、[ADR-0033](docs/adr/0033-stage3-single-task-mlp-ablation.md)、[ADR-0035](docs/adr/0035-spmm-baseline.md)、[ADR-0037](docs/adr/0037-spmm-wordpiece-character-limit.md)、[ADR-0038](docs/adr/0038-spmm-throughput-contract.md)、[ADR-0040](docs/adr/0040-llasmol-mistral-7b-baseline.md) 和 [ADR-0042](docs/adr/0042-dmpnn-shared-component-encoder.md)。

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

python scripts/benchmarks/sweep.py \
  --config configs/ablations/ilume_stage3_single_task_mlp.yaml \
  --output outputs/benchmarks/v1/ilume_stage3_single_task_mlp \
  --max-workers 1
```

Stage3 Single-task MLP 直接读取现役 Base prepared artifact，以冻结的 primary/partner
Object embedding 和 normalized conditions 做有序 concat。21 个 task × 5 folds 各自训练
完全独立的 `input -> 512 -> 256 -> 1` MLP，并由一个 Stage3-only sweep/reporting identity
汇总。它同时移除 HoME routing、跨任务共享、PCGrad 与 composite sampling，因此只能解释为
整体架构消融，不能解释成某个单组件的贡献。

D-MPNN 使用独立 hash-lock 环境，不修改主环境。多组分标量任务按 registry slot 分别构图并有序拼接表示，但所有组分共享唯一 message-passing encoder；不同 task/fold 仍独立训练。环境只由以下显式命令创建和安装；普通运行会通过 `conda run` 自动进入已有环境，不要求 `conda activate`，也不会自动安装或更新依赖。

```bash
conda env create -f benchmarks/dmpnn/environment.yml
conda run --no-capture-output -n ilume-dmpnn \
  python -m pip install --require-hashes \
  -r benchmarks/dmpnn/requirements-linux-x86_64-cu128.lock
conda run --no-capture-output -n ilume-dmpnn \
  python -m pip install --no-deps --no-build-isolation -e .

ILUME_BENCHMARK_ENVIRONMENT=ilume-dmpnn \
conda run --no-capture-output -n ilume-dmpnn \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.common.environment import validate_dmpnn_environment; validate_dmpnn_environment(load_benchmark_config("configs/benchmarks/dmpnn.yaml"))'
```

单任务与完整 sweep 仍复用公共入口：

```bash
python scripts/benchmarks/train.py \
  --config configs/benchmarks/dmpnn.yaml \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output outputs/benchmarks/v2/dmpnn/stage3/experiment__density/fold1/attempt-001

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/dmpnn.yaml \
  --output outputs/benchmarks/v2/dmpnn \
  --max-workers 1
```

`--max-workers 1` 保持串行行为。MLP、D-MPNN、MoLFormer、ILBERT、SPMM 与 LlaSMol 多 GPU sweep 可通过 `--devices cuda:0,cuda:1,...` 分配逻辑 job；XGBoost 的 CPU 并行度由 YAML 中的 `training.n_jobs` 控制。D-MPNN 正式 sweep 共 109 个单 seed 训练任务；上述命令不会 resume，失败任务由 sweep 在新 attempt 中完整重跑。

MoLFormer同样使用独立hash-lock环境。先显式安装环境并下载固定HF snapshot；正式launcher只使用本地cache，不会自动联网或切换revision。

```bash
conda env create -f benchmarks/molformer/environment.yml
conda run --no-capture-output -n ilume-molformer \
  python -m pip install --require-hashes \
  -r benchmarks/molformer/requirements-linux-x86_64-cu128.lock
conda run --no-capture-output -n ilume-molformer \
  python -m pip install --no-deps --no-build-isolation -e .
conda run --no-capture-output -n ilume-molformer \
  hf download ibm-research/MoLFormer-XL-both-10pct \
  --revision 361063d0ad524ef77cf39b08469f6be770dc550f

ILUME_BENCHMARK_ENVIRONMENT=ilume-molformer \
conda run --no-capture-output -n ilume-molformer \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.common.environment import validate_molformer_environment; validate_molformer_environment(load_benchmark_config("configs/benchmarks/molformer.yaml"))'
```

单任务与完整108-job sweep：

```bash
python scripts/benchmarks/train.py \
  --config configs/benchmarks/molformer.yaml \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output outputs/benchmarks/v1/molformer/stage3/experiment__density/fold1/attempt-001

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/molformer.yaml \
  --output outputs/benchmarks/v1/molformer \
  --max-workers 1
```

MoLFormer的超长train row整行跳过且不进入scaler；valid/test显式截断到202 tokens并在结果中审计。训练前为unique SMILES建立run-local内存token cache，多组分合并为一次共享backbone forward；正式合同固定batch 128、encoder/head learning rate `5e-6/5e-5`、50 epochs、patience 8和TF32，OOM/NaN不自动缩批或回退。多GPU sweep可使用`--devices cuda:0,cuda:1,...`。Partial Charge与Stage 2 Full保持unsupported。

ILBERT使用独立hash-lock环境和用户本地准备的固定上游资产。上游目前没有显式LICENSE，因此仓库不复制或再分发其源码与权重。

```bash
conda env create -f benchmarks/ilbert/environment.yml
conda run --no-capture-output -n ilume-ilbert \
  python -m pip install --require-hashes \
  -r benchmarks/ilbert/requirements-linux-x86_64-cu128.lock

mkdir -p artifacts/benchmarks/ilbert
git clone https://github.com/Yu-Xin-Qiu/ILBERT.git \
  artifacts/benchmarks/ilbert/upstream
git -C artifacts/benchmarks/ilbert/upstream \
  checkout --detach f9dc6f1b23a40b6988480735f3724a6332f68c12
curl --fail --location \
  https://zenodo.org/api/records/14601320/files/pretrained_model.pth/content \
  --output artifacts/benchmarks/ilbert/pretrained_model.pth

git -C artifacts/benchmarks/ilbert/upstream rev-parse HEAD
sha256sum \
  artifacts/benchmarks/ilbert/upstream/ILBERT/model.py \
  artifacts/benchmarks/ilbert/upstream/ILBERT/ILtokenizer.py \
  artifacts/benchmarks/ilbert/upstream/ILBERT/merged_vocab.txt \
  artifacts/benchmarks/ilbert/pretrained_model.pth

PYTHONPATH=src:. ILUME_BENCHMARK_ENVIRONMENT=ilume-ilbert \
conda run --no-capture-output -n ilume-ilbert \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.common.environment import validate_ilbert_environment; validate_ilbert_environment(load_benchmark_config("configs/benchmarks/ilbert.yaml"))'
```

单任务与完整108-job sweep：

```bash
python scripts/benchmarks/train.py \
  --config configs/benchmarks/ilbert.yaml \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output outputs/benchmarks/v1/ilbert/stage3/experiment__density/fold1/attempt-001

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/ilbert.yaml \
  --output outputs/benchmarks/v1/ilbert \
  --max-workers 1
```

ILBERT普通离子液体输入为单条`cation.anion` AIS sequence；solvation/transfer只增加共享backbone的有序双view。所有输入固定padding/truncation到100 tokens并公开审计，数值条件保持registry顺序和原始物理单位。Partial Charge与Stage 2 Full为unsupported。

SPMM使用独立hash-lock环境和固定的官方Apache-2.0上游checkout。仓库不复制或提交约2.20 GiB的官方Lightning checkpoint；运行前会校验commit、源码、vocab、config、checkpoint SHA和字节数。
该环境固定Python 3.10，因此不执行要求Python ≥3.11的ILUME editable install；四个benchmark脚本会从仓库根显式引导`src`和`benchmarks`导入。

```bash
conda env create -f benchmarks/spmm/environment.yml
conda run --no-capture-output -n ilume-spmm \
  python -m pip install --require-hashes \
  -r benchmarks/spmm/requirements-linux-x86_64-cu128.lock

mkdir -p artifacts/benchmarks/spmm
git clone https://github.com/jinhojsk515/SPMM.git \
  artifacts/benchmarks/spmm/upstream
git -C artifacts/benchmarks/spmm/upstream \
  checkout --detach 046976484f31b3cbc862b8f2094e38df72fcfce7
curl --fail --location --retry 5 --continue-at - \
  'https://drive.usercontent.google.com/download?id=1jNVII-ktlV17p6bCdzw0TiPsb6lGNgWY&export=download&confirm=t' \
  --output artifacts/benchmarks/spmm/checkpoint_SPMM.ckpt

sha256sum artifacts/benchmarks/spmm/checkpoint_SPMM.ckpt
stat --format='%s bytes' artifacts/benchmarks/spmm/checkpoint_SPMM.ckpt

PYTHONPATH=src:. ILUME_BENCHMARK_ENVIRONMENT=ilume-spmm \
conda run --no-capture-output -n ilume-spmm \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.common.environment import validate_spmm_environment; validate_spmm_environment(load_benchmark_config("configs/benchmarks/spmm.yaml"))'
```

单任务与完整108-job sweep：

```bash
python scripts/benchmarks/train.py \
  --config configs/benchmarks/spmm.yaml \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output outputs/benchmarks/v1/spmm/stage3/experiment__density/fold1/attempt-001

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/spmm.yaml \
  --output outputs/benchmarks/v1/spmm-wp350-bs128 \
  --max-workers 2 \
  --devices cuda:0,cuda:1
```

SPMM只使用官方text-mode前6层和768维`[CLS]`表示；各分子component由同一encoder分别编码并合并为一次forward，随后只扩宽官方regression head第一层。模型输入去除立体信息，保留官方100-token tokenizer加首token切片路径，实际encoder上限为99；WordPiece单词字符上限固定为350，collision和truncation均公开审计。训练固定batch 128、FP32+TF32及确定性sortish长度分桶；多GPU运行保持一张GPU一个job。旧`outputs/benchmarks/v1/spmm`不得与新合同混用，汇总时通过`--include outputs/benchmarks/v1/spmm-wp350-bs128`只选择新结果。Partial Charge与Stage 2 Full为unsupported。

LlaSMol使用固定Mistral-7B基座和官方LoRA adapter。仓库不复制或提交约13.5 GiB基座与84 MB adapter；必须先显式安装独立环境并将固定snapshot下载到已忽略目录。

```bash
conda env create -f benchmarks/llasmol/environment.yml
conda run --no-capture-output -n ilume-llasmol \
  python -m pip install --require-hashes \
  -r benchmarks/llasmol/requirements-linux-x86_64-cu128.lock

mkdir -p artifacts/benchmarks/llasmol/base artifacts/benchmarks/llasmol/adapter
conda run --no-capture-output -n ilume-llasmol \
  hf download mistralai/Mistral-7B-v0.1 \
  config.json model.safetensors.index.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json \
  --revision 27d67f1b5f57dc0953326b2601d68371d40ea8da \
  --local-dir artifacts/benchmarks/llasmol/base
conda run --no-capture-output -n ilume-llasmol \
  hf download osunlp/LlaSMol-Mistral-7B \
  adapter_config.json adapter_model.bin \
  --revision 044d6124448733615c5a3d6ab14b947f71fc6728 \
  --local-dir artifacts/benchmarks/llasmol/adapter

PYTHONPATH=src:. ILUME_BENCHMARK_ENVIRONMENT=ilume-llasmol \
conda run --no-capture-output -n ilume-llasmol \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.common.environment import validate_llasmol_environment; validate_llasmol_environment(load_benchmark_config("configs/benchmarks/llasmol.yaml"))'
```

单任务与完整108-job sweep：

```bash
python scripts/benchmarks/train.py \
  --config configs/benchmarks/llasmol.yaml \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output outputs/benchmarks/v1/llasmol/stage3/experiment__density/fold1/attempt-001

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/llasmol.yaml \
  --output outputs/benchmarks/v1/llasmol \
  --max-workers 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

普通IL使用带task marker的单条`cation.anion`sequence；solvation/transfer只增加whole-IL与partner的共享backbone双view。基座以NF4 double-quant冻结加载，并继续训练官方attention与MLP LoRA及`4096/8192 + conditions → 256 → 1`回归head。输入上限512 tokens并公开截断审计；target和numeric conditions只从train rows拟合scaler。建议每张GPU仅运行一个job；OOM/NaN不自动缩批或回退。Partial Charge与Stage 2 Full为unsupported。

## 输出与结果汇总

新 train/evaluate 不覆盖既有输出，恢复必须显式请求。每个操作目录冻结 `run_config.yaml`，写入公开安全的 `metadata.json`，成功后生成 `summary.json`；checkpoint、训练日志和 tensor 默认不进入 Git。完整身份与 checkpoint 规则见 [ADR-0021](docs/adr/0021-identity-audit-contract-v1.md)。

全局 summarizer 只收录显式选中的目录。`--input` 提供一个或多个扫描根；可选 `--include` 是精确目录前缀白名单，省略时扫描全部 input。include 必须存在、位于某个 input 内并至少匹配一个 reporting candidate；重叠路径会去重，不支持 glob。

```bash
python scripts/benchmarks/summarize.py \
  --input outputs/v2 outputs/benchmarks \
  --include \
    outputs/v2/stage3/base \
    outputs/v2/stage2/base/evaluate/test_benchmark_suite_v2 \
    outputs/benchmarks/v1/mlp \
    outputs/benchmarks/v1/ecfp_xgboost \
    outputs/benchmarks/v1/ilume_stage3_single_task_mlp \
  --output summary
```

只有 schema 完整且 comparison identity 兼容的 completed run 进入榜单；Stage 3 按 ADR-0031 允许 train-only normalization 不同，但 valid/test source 与其余协议必须一致。其他 run 进入 health。损坏的选中正式结果会使发布失败，已有 `summary/` 保持不变。详细 reporting 合同见 [ADR-0023/0024/0025/0031](docs/adr/README.md)。

## 验证

```bash
pytest -q
```

测试只使用临时小数据，不执行正式 prepare、teacher cache、训练或五折 evaluation。
