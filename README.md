# ILUME

ILUME 是按 Stage 组织的分子科研 pipeline：Global-RDKit v2 主线在 Stage 1 进行 SMILES、Graph、RDKit 三模态四目标掩码预训练，Stage 2 训练 catalog 驱动的九任务 physics representation，Stage 3 训练 21 个 sparse-label observation task。正式 YAML 与 [ADR 索引](docs/adr/README.md) 共同定义现役科研合同。

## 按任务阅读

| 目的 | 入口 |
|---|---|
| 跑现役主线 | [安装](#安装与数据) → [Stage 1](#stage-1) → [Stage 2](#stage-2) → [Stage 3](#stage-3) |
| 跑对比模型 | [Baseline 通用运行](#baselines-and-ablations)，再展开对应模型的环境准备 |
| 跑内部消融 | [RDKit-HoME](#rdkit-2d--home-representation-ablation)、[No-Stage1](#no-stage1rdkit-2d--stage2--stage3-home)；Single-task MLP 与 [Stage2→Stage3 transfer matrix](#stage2stage3-transfer-matrix) 见 baseline 部分 |
| 汇总结果 | [输出与结果汇总](#输出与结果汇总) |
| 查科学约束/历史 | [ADR 索引](docs/adr/README.md) / [已取代设计摘要](docs/adr/history.md) |
| 跑冻结的 legacy 研究 | [Capacity v1 手册](docs/capacity-v1-runbook.md) |

命令均从仓库根目录运行。按依赖顺序准备数据与模型，使用尚不存在的新 train/evaluate 输出目录；以下正式命令不属于自动验收步骤。

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

Stage 2 Object v3 从 catalog 加载九个 simulation task，共享 ObjectEncoder，并从 Stage 1 encoder 准备 entity teacher cache。现役 v2 固定训练 10 个 joint epochs，随后直接发布最终 checkpoint、`stage2_encoder.pt` 和 joint validation `final_metrics.json`；不再执行 taskwise refinement，也不提供 Stage 2 test evaluation。模型、数据身份和恢复合同见 [ADR-0019/0021/0025/0043](docs/adr/README.md)。

```bash
python scripts/stage2/prepare.py \
  --config configs/v2/stage2/base.yaml \
  --output outputs/v2/stage2/base/prepare

python scripts/stage2/train.py \
  --config configs/v2/stage2/base.yaml \
  --output outputs/v2/stage2/base/train
```

Stage 2 只从完整 Object v3 joint epoch 恢复，旧 Object v2、旧式 v2 refinement run 和缺少现役合同的开发期 v3 输出不迁移。legacy v1、Capacity v1 与 No-Stage1 的既有 refinement 训练定义和历史产物保持不变。

## Stage 3

Stage 3 使用冻结的 Stage 2 Object v3 表示、动态 HoME、raw sampling 与 ownership-aware clipping。现役 v2 按 owner-specific LR、固定训练寿命和 task-specific PRIVATE capacity/dropout 执行三阶段训练：15 个全任务 epoch、六个同源 GROUP 分支，以及从同一 anchor 独立解析的 21 个 PRIVATE task scope；零预算 scope 直接继承 anchor，其余 scope 固定训练并 stitch。owner 提前冻结不删除 task，validation 只记录，最终发布固定预算状态拼接的 `three_phase_final.pt`。three-phase validation 与独立 evaluation 额外按 task 报告 GLOBAL/GROUP/PRIVATE gate mass、归一化 gate entropy 和 PRIVATE mass 分位数，不改变预测或 prediction CSV。数据、模型、五折调度和恢复合同见 [ADR 索引](docs/adr/README.md)。

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

Stage 3 evaluator 对现役 v2 默认加载每个 fold 的 `three_phase_final.pt`；legacy v1 与
Capacity v1 仍默认加载 `taskwise_refined.pt`，且只有 legacy 配置支持显式
`--checkpoint-epoch N`。

### 知识图谱性质分组候选

`configs/v2/stage3/base1.yaml`仅改变六个GROUP的任务归属，并按
[ADR-0063](docs/adr/0063-stage3-knowledge-graph-grouping-candidate.md)继承对应旧组的capacity与训练预算。
它复用Base prepared artifact，但training identity和checkpoint不兼容；必须写入独立输出目录：

```bash
python scripts/stage3/train.py \
  --config configs/v2/stage3/base1.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base1/train \
  --max-parallel 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3

python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base1.yaml \
  --checkpoint-dir outputs/v2/stage3/base1/train \
  --split valid --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base1/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base1.yaml \
  --checkpoint-dir outputs/v2/stage3/base1/train \
  --split test --ensemble-folds \
  --output outputs/v2/stage3/base1/evaluate_test
```

该候选尚未取代现役Base；不要覆盖`outputs/v2/stage3/base`下的既有结果。

在相同知识图谱分组下，新增五个独立容量/预算候选，详见
[ADR-0064](docs/adr/0064-stage3-knowledge-graph-budget-candidates.md)：

| 配置名 | 相对base1的改动 |
|---|---|
| base1_1 | thermophysical/interfacial GROUP experts 2→3 |
| base1_2 | 该GROUP Phase 1 epochs 10→15 |
| base1_3 | 该GROUP Phase 1/2 LR改为2e-4/1e-4 |
| base1_4 | static GROUP Phase 1/2 LR改为5e-5/2.5e-5、Phase 2 epochs=1；static PRIVATE Phase 1 LR=2e-5 |
| base1_5 | 合并以上四项 |

所有候选的大组Phase 2仍为4 epochs，GLOBAL与PRIVATE capacity不变。
每个候选从头训练，下面以base1_1为例；运行其他候选时，将命令中的所有`base1_1`
一致替换为`base1_2`、`base1_3`、`base1_4`或`base1_5`，prepared artifact无需重建：

```bash
python scripts/stage3/train.py \
  --config configs/v2/stage3/base1_1.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base1_1/train \
  --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3

python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base1_1.yaml \
  --checkpoint-dir outputs/v2/stage3/base1_1/train \
  --split valid --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base1_1/evaluate_valid
```

先完成全部候选的五折validation比较，再确定一个候选运行test；test不得用于候选间调参。
例如仅当base1_1被选定时执行：

```bash
python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base1_1.yaml \
  --checkpoint-dir outputs/v2/stage3/base1_1/train \
  --split test --ensemble-folds \
  --output outputs/v2/stage3/base1_1/evaluate_test
```

以 `base1_5` 为共同锚点的五个定向小实验见
[ADR-0066](docs/adr/0066-stage3-knowledge-graph-targeted-small-experiments.md)：

| 配置名 | 相对base1_5的改动 |
|---|---|
| base2_1 | static GROUP expert hidden ratio 0.75→0.25 |
| base2_2 | speed of sound Phase 3 PRIVATE epochs 0→2 |
| base2_3 | self diffusion Phase 3 PRIVATE epochs 4→2 |
| base2_4 | thermophysical/interfacial GROUP experts 3→2 |
| base2_5 | 合并以上四项 |

下面以base2_1为例；运行其他候选时，将命令中的所有`base2_1`一致替换为
`base2_2`、`base2_3`、`base2_4`或`base2_5`：

```bash
python scripts/stage3/train.py \
  --config configs/v2/stage3/base2_1.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base2_1/train \
  --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3

python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base2_1.yaml \
  --checkpoint-dir outputs/v2/stage3/base2_1/train \
  --split valid --fold 1 2 3 4 5 \
  --output outputs/v2/stage3/base2_1/evaluate_valid

python scripts/stage3/evaluate.py \
  --config configs/v2/stage3/base2_1.yaml \
  --checkpoint-dir outputs/v2/stage3/base2_1/train \
  --split test --ensemble-folds \
  --output outputs/v2/stage3/base2_1/evaluate_test
```

五个候选都可以运行test ensemble，但test只作探索性报告；候选选择仍以完整五折validation
的task-equal macro NMAE为准。prepared artifact可复用，Base/base1系列checkpoint不可交叉加载。

### 超参数搜索退役

ILUME 的 v2 Stage 3 A/B/C 搜索与 Capacity v1 HPO 已于 2026-09-06 退役；仓库不再提供
搜索入口、搜索配置或 Optuna 依赖。现役 v2 直接使用自包含的
`configs/v2/stage3/base.yaml`，既有搜索输出只作为历史 artifact，不可由当前代码续跑。
退役背景见 [ADR-0041](docs/adr/0041-stage3-v2-three-phase-hpo.md)。

Capacity v1 继续冻结在 legacy v1 五模态合同，并直接使用已提交的四份
`configs/experiments_v1/stage3/formal/*.yaml`。只读 probe/robustness/comparison 报告仍由
`scripts/stage3/capacity.py --manifest ... --output ...` 生成；当前命令见
[Capacity v1 操作手册](docs/capacity-v1-runbook.md)。

### RDKit 2D → HoME representation ablation

[ADR-0034](docs/adr/0034-rdkit-2d-home-representation-ablation.md) 只用
RDKit 2D descriptors 与两个可训练的 `Linear → LayerNorm` adapter 替换 frozen
Stage2 Object representation；HoME、PCGrad、sampling、三阶段训练和 evaluation 保持
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
  --input outputs/v2 outputs/benchmarks outputs/ablations \
  --output summary
```

五折训练命令默认串行；如需显式多 GPU 调度，只增加 `--max-parallel` 与 `--devices`，
不改变 scientific identity。正式执行前应保证对应输出目录不存在；恢复必须显式添加
`--resume`。

### No-Stage1：RDKit 2D → Stage2 → Stage3 HoME

[ADR-0036](docs/adr/0036-no-stage1-rdkit-stage2-stage3-ablation.md) 用共享的
`217D → 1024D → 512D` RDKit MLP 替换 Stage1 backbone，保留 Stage2 ObjectEncoder 与
Stage3 Base。该路径不读取 Stage1 artifact/checkpoint 或 teacher cache；Stage2 仍按冻结配置
训练八个 object/interaction task，但不再执行或汇总 Stage 2 test evaluation。

```bash
python scripts/stage2/prepare.py \
  --config configs/ablations/no_stage1_rdkit_stage2.yaml \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/prepare

python scripts/stage2/train.py \
  --config configs/ablations/no_stage1_rdkit_stage2.yaml \
  --output outputs/ablations/no_stage1_rdkit_stage2_stage3/stage2/train

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
  --input outputs/v2 outputs/benchmarks outputs/ablations \
  --output summary
```

Stage2/Stage3 resume 分别在上述 train 命令追加 `--resume <checkpoint>` 与 `--resume`；新
输出不得覆盖既有目录。Stage3 五折默认串行，多 GPU 调度只增加 `--max-parallel` 和
`--devices`，不改变实验 identity。

## Baselines and Ablations

MLP、ECFP4-XGBoost、Chemprop D-MPNN、MoLFormer、ILBERT、SPMM、LlaSMol、AIonopedia、ILTransR 与 AIFC 位于 `benchmarks/`；
Stage3 Single-task MLP 内部消融位于 `ablations/`。二者均与 Stage 代码隔离，并继续
复用 benchmark 运行与 reporting 入口；旧七模型合同见 [ADR-0045](docs/adr/0045-fixed-budget-baseline-training.md) 及其引用，AIonopedia 合同见 [ADR-0049](docs/adr/0049-aionopedia-multimodal-baseline.md)，ILTransR 合同见 [ADR-0057](docs/adr/0057-iltransr-stage3-baseline.md)，AIFC 合同见 [ADR-0060](docs/adr/0060-aifc-stage3-baseline.md)。

所有模型发布固定预算的最终状态，validation 只用于 history/报告，不驱动训练决策；具体合同见对应 ADR。旧七模型 checkpoint format v2 不兼容旧 validation-selected artifact，旧结果不得混入同一汇总。

其中 `MLP` 固定表示每个 registry component 的 21 项 basic molecular statistics，按
`identity_columns` 顺序拼接后追加 authoritative conditions，再输入 `128 → 64 → 1`
浅层网络；它不再使用完整 RDKit 2D descriptor representation。

| 模型名（配置 basename） | 预算 | 输出根 |
|---|---|---|
| `mlp` | 10 epochs | `outputs/benchmarks/v3/mlp` |
| `dmpnn`、`molformer`、`ilbert`、`spmm` | 10 epochs | `outputs/benchmarks/fixed-budget-10e-v1/<model>` |
| `ecfp_xgboost` | 1000 trees | `outputs/benchmarks/fixed-budget-v1/ecfp_xgboost` |
| `llasmol` | 10 epochs | `outputs/benchmarks/fixed-budget-v1/llasmol-10e-bs16-ga2` |
| `aionopedia` | 10 epochs | `outputs/benchmarks/model-native-v1/aionopedia` |
| `iltransr` | 10 epochs | `outputs/benchmarks/model-native-10e-v1/iltransr` |
| `aifc` | 10 epochs | `outputs/benchmarks/model-native-10e-v1/aifc` |

先完成下方对应模型的环境、资产与 validator 步骤，再使用通用命令。以 D-MPNN 为例，替换下列两个变量即可选择其他模型；LlaSMol 输出后缀保持上表约定。

```bash
model=dmpnn
run_root=outputs/benchmarks/fixed-budget-10e-v1/dmpnn
python scripts/benchmarks/train.py \
  --config "configs/benchmarks/${model}.yaml" \
  --benchmark stage3 --task experiment/density --fold 1 \
  --output "${run_root}/stage3/experiment__density/fold1/attempt-001"

python scripts/benchmarks/sweep.py \
  --config "configs/benchmarks/${model}.yaml" \
  --output "${run_root}" \
  --max-workers 1
```

MLP、XGBoost 和 Single-task MLP 的基础环境及 sweep：

```bash
python -m pip install -e ".[benchmarks]"

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/mlp.yaml \
  --output outputs/benchmarks/v3/mlp \
  --max-workers 1

python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/ecfp_xgboost.yaml \
  --output outputs/benchmarks/fixed-budget-v1/ecfp_xgboost \
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

### Stage2→Stage3 transfer matrix

该隔离消融从同一个 Stage1 初始化构造零 update baseline 与九个 physics-only Stage2
single-source encoder，再对十份冻结的1024D表示运行21 tasks × 5 folds固定10轮MLP。
正式矩阵只使用 system-split validation raw MAE，不运行test。完整合同见
[ADR-0062](docs/adr/0062-stage2-stage3-full-transfer-matrix.md)。使用新的输出根依次运行：

```bash
root=outputs/ablations/stage2_stage3_transfer

python scripts/stage2/transfer.py \
  --config configs/ablations/stage2_stage3_transfer.yaml \
  --output "${root}/stage2" \
  --max-parallel 1

python scripts/stage3/transfer.py prepare \
  --config configs/ablations/stage2_stage3_transfer.yaml \
  --stage2-dir "${root}/stage2" \
  --output "${root}/representations"

python scripts/stage3/transfer.py train \
  --config configs/ablations/stage2_stage3_transfer.yaml \
  --representations "${root}/representations" \
  --output "${root}/stage3" \
  --max-parallel 1

python scripts/stage3/transfer.py summarize \
  --config configs/ablations/stage2_stage3_transfer.yaml \
  --stage3-dir "${root}/stage3" \
  --output "${root}/summary"
```

等行数对照使用 [balanced 配置](configs/ablations/stage2_stage3_transfer_balanced.yaml) 和
[ADR-0065](docs/adr/0065-stage2-stage3-balanced-transfer-matrix.md)：从九个 prepared train
数据集中各无放回抽取 N 行（N 为最小数据集行数），固定子集运行 10 epochs。
Stage2 根目录发布 `resolved_sampling_plan.json`，记录行索引/hash、体系覆盖和共同更新预算。
该实验保留 prepared normalization；控制行数/updates，不控制体系数或原子标签数。
依次运行（使用独立输出目录）：

```bash
python scripts/stage2/transfer.py \
  --config configs/ablations/stage2_stage3_transfer_balanced.yaml \
  --output outputs/ablations/stage2_stage3_transfer_balanced/stage2 \
  --max-parallel 1

python scripts/stage3/transfer.py prepare \
  --config configs/ablations/stage2_stage3_transfer_balanced.yaml \
  --stage2-dir outputs/ablations/stage2_stage3_transfer_balanced/stage2 \
  --output outputs/ablations/stage2_stage3_transfer_balanced/representations

python scripts/stage3/transfer.py train \
  --config configs/ablations/stage2_stage3_transfer_balanced.yaml \
  --representations outputs/ablations/stage2_stage3_transfer_balanced/representations \
  --output outputs/ablations/stage2_stage3_transfer_balanced/stage3 \
  --max-parallel 1

python scripts/stage3/transfer.py summarize \
  --config configs/ablations/stage2_stage3_transfer_balanced.yaml \
  --stage3-dir outputs/ablations/stage2_stage3_transfer_balanced/stage3 \
  --output outputs/ablations/stage2_stage3_transfer_balanced/summary
```

两套配置的两个训练命令都支持单卡多进程，例如
`--max-parallel 4 --devices cuda:0`；多GPU例如
`--max-parallel 8 --devices cuda:0,cuda:1,cuda:2,cuda:3`，即每张卡2个并发job。
`max-parallel`必须能被设备数整除；调度参数不进入科研identity。
`--source`、`--target`和`--fold`可用于分批执行，完整汇总仍严格要求全部1050个job。
交互终端会显示Stage2 encoder variant、representation variant和Stage3 MLP job总进度；
串行Stage3训练还会显示当前job的epoch与train/validation指标。并行时只保留主进程总进度，
避免多个worker进度条互相覆盖；非TTY日志保持安静，也可用`ILUME_DISABLE_PROGRESS=1`显式关闭。

`--max-workers 1` 保持串行行为。MLP、D-MPNN、MoLFormer、ILBERT、SPMM、LlaSMol、AIonopedia、ILTransR 与 AIFC 多 GPU sweep 可通过 `--devices cuda:0,cuda:1,...` 分配逻辑 job；XGBoost 的 CPU 并行度由 YAML 中的 `training.n_jobs` 控制。每个 baseline 正式 sweep 均为 21 tasks × 5 folds，即 105 个单 seed 训练任务；上述命令不会 resume，失败任务由 sweep 在新 attempt 中完整重跑。

<details>
<summary>D-MPNN：环境、资产与模型边界</summary>

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
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.dmpnn.environment import validate_dmpnn_environment; validate_dmpnn_environment(load_benchmark_config("configs/benchmarks/dmpnn.yaml"))'
```

</details>

<details>
<summary>MoLFormer：环境、资产与模型边界</summary>

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
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.molformer.environment import validate_molformer_environment; validate_molformer_environment(load_benchmark_config("configs/benchmarks/molformer.yaml"))'
```

MoLFormer的超长train row整行跳过且不进入scaler；valid/test显式截断到202 tokens并在结果中审计。训练前为unique SMILES建立run-local内存token cache，多组分合并为一次共享backbone forward；正式合同固定batch 128、encoder/head learning rate `5e-6/5e-5`、完整10 epochs、最终训练状态和TF32，OOM/NaN不自动缩批或回退。多GPU sweep可使用`--devices cuda:0,cuda:1,...`。

</details>

<details>
<summary>ILBERT：环境、资产与模型边界</summary>

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
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.ilbert.environment import validate_ilbert_environment; validate_ilbert_environment(load_benchmark_config("configs/benchmarks/ilbert.yaml"))'
```

若迁移已有且已核对来源的 checkout 后 Git 报 `dubious ownership`，只信任该精确目录：

```bash
git config --global --add safe.directory \
  "$PWD/artifacts/benchmarks/ilbert/upstream"
```

ILBERT普通离子液体输入为单条`cation.anion` AIS sequence；solvation/transfer只增加共享backbone的有序双view。所有输入固定padding/truncation到100 tokens并公开审计，数值条件保持registry顺序和原始物理单位。训练使用恒定`3e-5` learning rate完成10 epochs，validation不调整学习率。

</details>

<details>
<summary>SPMM：环境、资产与模型边界</summary>

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
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.spmm.environment import validate_spmm_environment; validate_spmm_environment(load_benchmark_config("configs/benchmarks/spmm.yaml"))'
```

若迁移已有且已核对来源的 checkout 后 Git 报 `dubious ownership`，只信任该精确目录：

```bash
git config --global --add safe.directory \
  "$PWD/artifacts/benchmarks/spmm/upstream"
```

SPMM只使用官方text-mode前6层和768维`[CLS]`表示；各分子component由同一encoder分别编码并合并为一次forward，随后只扩宽官方regression head第一层。模型输入去除立体信息，保留官方100-token tokenizer加首token切片路径，实际encoder上限为99；WordPiece单词字符上限固定为350，collision和truncation均公开审计。训练固定batch 128、完整10 epochs、最终训练状态、FP32+TF32及确定性sortish长度分桶；多GPU运行保持一张GPU一个job。旧SPMM输出不得与新合同混用。

</details>

<details>
<summary>LlaSMol：环境、资产与模型边界</summary>

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
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.llasmol.environment import validate_llasmol_environment; validate_llasmol_environment(load_benchmark_config("configs/benchmarks/llasmol.yaml"))'
```

普通IL使用带task marker的单条`cation.anion`sequence；solvation/transfer只增加whole-IL与partner的共享backbone双view。基座以NF4 double-quant冻结加载，并继续训练官方attention与MLP LoRA及`4096/8192 + conditions → 256 → 1`回归head。输入上限512 tokens并公开截断审计；target和numeric conditions只从train rows拟合scaler。训练使用batch 16、gradient accumulation 2固定完成10 epochs并保存最终状态。建议每张GPU仅运行一个job；OOM/NaN不自动缩批或回退。

</details>

<details>
<summary>AIonopedia：环境、资产与模型边界</summary>

AIonopedia 使用锁定的 PyTorch `2.9.0+cu128` 环境、Qwen3-0.6B 与 generic ionic-liquid
multimodal checkpoint，完整加载 released LoRA、GNN、projectors、graph merge、cross-modal
decoders 和 segment embeddings。当前正式 pretrained snapshot 是用户提供的本地 generic
pretraining export：

`artifacts/benchmarks/aionopedia/best_cosine(stable_ver)_qwen0.6b/qwen0.6b-pretrain_simple2.8m(itg_loss)`

配置逐文件固定 SHA-256 和字节数；运行不会读取同级的 property-specific 目录。本地
`adapter_config.json` 是旧 PEFT 0.14 元数据版本，不与 Hugging Face 当前文件逐字节相同，
但其 LoRA 语义被严格校验，且 adapter 始终加载到另行锁定的本地 Qwen base；作者机器路径
不会写入公开 run metadata。仓库不提交权重。
若以后需要从 gated Hugging Face revision 重新取得其余资产，可在网页获得权限并登录后执行：

```bash
conda env create -f benchmarks/aionopedia/environment.yml
conda run --no-capture-output -n ilume-aionopedia \
  python -m pip install --require-hashes \
  -r benchmarks/aionopedia/requirements-linux-x86_64-cu128.lock
conda run --no-capture-output -n ilume-aionopedia \
  python -m pip install --no-deps --no-build-isolation -e .

conda run --no-capture-output -n ilume-aionopedia hf auth login
mkdir -p artifacts/benchmarks/aionopedia/base \
  'artifacts/benchmarks/aionopedia/best_cosine(stable_ver)_qwen0.6b/qwen0.6b-pretrain_simple2.8m(itg_loss)'
conda run --no-capture-output -n ilume-aionopedia \
  hf download Qwen/Qwen3-0.6B \
  config.json model.safetensors tokenizer.json tokenizer_config.json \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir artifacts/benchmarks/aionopedia/base
conda run --no-capture-output -n ilume-aionopedia \
  hf download AIonopedia/AIonopedia \
  GNN_state_dict.pt adapter_model.safetensors \
  decoder1_state_dict.pt decoder2_state_dict.pt \
  embedding_property_state_dict.pt fc_out_state_dict.pt \
  graph_merge_encoder_state_dict.pt projector_gnn_state_dict.pt \
  projector_llm_state_dict.pt projector_temp_state_dict.pt segment_embeddings.pt \
  --revision 448236ef3532efd67b7956472df4e8f539b22629 \
  --local-dir 'artifacts/benchmarks/aionopedia/best_cosine(stable_ver)_qwen0.6b/qwen0.6b-pretrain_simple2.8m(itg_loss)'
```

上面的可选命令刻意不覆盖 `adapter_config.json`；正式配置要求保留已固定哈希的本地旧版
pretraining export 元数据，Hugging Face 当前元数据文件不能替换它。本地 snapshot 就位后，
先做一次环境、哈希、
Qwen config、LoRA tensor、全部官方模块和 71-output pretraining head 的只读结构验证：

```bash
PYTHONPATH=src:. ILUME_BENCHMARK_ENVIRONMENT=ilume-aionopedia \
conda run --no-capture-output -n ilume-aionopedia \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.aionopedia.environment import validate_aionopedia_environment; validate_aionopedia_environment(load_benchmark_config("configs/benchmarks/aionopedia.yaml"))'
```

AIonopedia prompt 只包含 composition 与原始单位 conditions，不包含 target 名；图路径保留官方
35D atom/11D edge、无 explicit-H preprocessing。Temperature 使用 `K/1000`；pressure 使用
fold train-only sample z-score；pressure、frequency 与 wavelength 均有独立随机初始化的 graph
projector/segment token。Target 同样只用 fold train rows 标准化，评估反归一化到 raw units。
训练固定 10 epochs，validation 每轮只报告，最终模型是 epoch 10 state。

</details>

<details>
<summary>ILTransR：环境、资产与模型边界</summary>

ILTransR 使用官方仓库 commit `ff5e55cfb8162b0706fc2d88ca5cd384705686b1` 的 generic
`pretraining/valid_best.params`，不下载或读取任何 property dataset 和 supervised
`*_best.params`。正式训练是 PyTorch `2.9.0+cu128` FP32；MXNet 1.9.1 只在 CPU 转换环境生成
encoder-only safetensors 与固定 parity reference。先建立两个独立环境：

```bash
conda env create -f benchmarks/iltransr/environment.yml
conda run --no-capture-output -n ilume-iltransr \
  python -m pip install --require-hashes \
  -r benchmarks/iltransr/requirements-linux-x86_64-cu128.lock

conda env create -f benchmarks/iltransr/conversion-environment.yml
```

新服务器能访问 GitHub 时，只下载固定 commit 的三个公开 generic-pretraining 文件：

```bash
mkdir -p artifacts/benchmarks/iltransr/pretraining
curl -fL \
  https://raw.githubusercontent.com/GuzhongChen/ILTransR/ff5e55cfb8162b0706fc2d88ca5cd384705686b1/pretraining/valid_best.params \
  -o artifacts/benchmarks/iltransr/pretraining/valid_best.params
curl -fL \
  https://raw.githubusercontent.com/GuzhongChen/ILTransR/ff5e55cfb8162b0706fc2d88ca5cd384705686b1/datasets/pubchem/vocab.random_smiles.json \
  -o artifacts/benchmarks/iltransr/pretraining/vocab.random_smiles.json
curl -fL \
  https://raw.githubusercontent.com/GuzhongChen/ILTransR/ff5e55cfb8162b0706fc2d88ca5cd384705686b1/datasets/pubchem/vocab.rdkit_canonical_smiles.json \
  -o artifacts/benchmarks/iltransr/pretraining/vocab.rdkit_canonical_smiles.json

sha256sum \
  artifacts/benchmarks/iltransr/pretraining/valid_best.params \
  artifacts/benchmarks/iltransr/pretraining/vocab.random_smiles.json \
  artifacts/benchmarks/iltransr/pretraining/vocab.rdkit_canonical_smiles.json
```

预期 SHA-256 依次为
`36a3401175dd372725bd2f5e4e041642a754eeaf3f46845b6ff949065871735a`、
`cfa1dca1642b105693981686b6bd300e6faea37de88f137069f131b3d85d5295`、
`856ad43fd6c13e7634c072ae3e58fbc92b641466184eb4577681bac74fd71116`。
若新服务器不能访问 GitHub，在旧服务器仓库根目录执行：

```bash
rsync -av --progress \
  artifacts/benchmarks/iltransr/pretraining/ \
  NEW_SERVER:/path/to/ILUME/artifacts/benchmarks/iltransr/pretraining/
```

原始三个文件就位后执行确定性转换；它只导出 source embedding/Transformer encoder，并把
decoder、one-step decoder、target embedding/projection 写入 ignored tensor 清单：

```bash
PYTHONPATH=src:. conda run --no-capture-output -n ilume-iltransr-convert \
  python -m benchmarks.iltransr.conversion \
  --checkpoint artifacts/benchmarks/iltransr/pretraining/valid_best.params \
  --source-vocab artifacts/benchmarks/iltransr/pretraining/vocab.random_smiles.json \
  --target-vocab artifacts/benchmarks/iltransr/pretraining/vocab.rdkit_canonical_smiles.json \
  --output artifacts/benchmarks/iltransr/pretraining/encoder.safetensors \
  --parity-reference artifacts/benchmarks/iltransr/pretraining/parity_reference.safetensors \
  --manifest artifacts/benchmarks/iltransr/pretraining/conversion_manifest.json

sha256sum \
  artifacts/benchmarks/iltransr/pretraining/encoder.safetensors \
  artifacts/benchmarks/iltransr/pretraining/parity_reference.safetensors \
  artifacts/benchmarks/iltransr/pretraining/conversion_manifest.json
```

预期转换 SHA-256 为
`dbfc010f21fd17e19b1ddc8fb45b86597655e1a7dbbc5b53c5a7768dad99c952` 和
`45e841123c9076369f3d55796166e08698e325ce94c8be68464c0d2efad750b6`，manifest SHA-256 为
`b1d386e8e501ffab171819a24008115e1704dfa72b2095f871fd2f1dd609619d`。正式运行前，validator
还会在 CPU 重放 MXNet/PyTorch parity、检查全部 40 个 encoder tensors，并在发现任意资产或
环境漂移时拒绝训练：

```bash
PYTHONPATH=src:. ILUME_BENCHMARK_ENVIRONMENT=ilume-iltransr \
conda run --no-capture-output -n ilume-iltransr \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.iltransr.environment import validate_iltransr_environment; print(validate_iltransr_environment(load_benchmark_config("configs/benchmarks/iltransr.yaml"))["pretrained_snapshot"]["structure"])'
```

ILTransR 对 partner task 使用共享 backbone 的 ordered multi-view fusion；所有 pretrained
embedding/Transformer 参数 full fine-tune。Condition 使用全五折加 test covariates 的 task-global
population z-score，这是显式 transductive feature scaling；target 仍只从当前 fold train rows 拟合，
normalized L1 训练后恢复 raw units。七个有同性质官方 notebook 的 task 使用其 epochs/batch/dropout，
其余任务使用登记的 10-epoch fallback；validation 只报告并始终发布 final epoch state。完整合同见
[ADR-0057](docs/adr/0057-iltransr-stage3-baseline.md)。

</details>

<details>
<summary>AIFC：环境、资产与模型边界</summary>

AIFC 使用作者公开的 fragment-level GNN、motif/junction graph 与 attention aggregation，fragment
dictionary 已作为小型公开资产提交并固定到历史官方 blob，因此不需要下载模型权重。正式运行使用
PyTorch `2.9.0+cu128` FP32；环境创建与 validator 命令如下：

```bash
conda env create -f benchmarks/aifc/environment.yml
conda run --no-capture-output -n ilume-aifc \
  python -m pip install --require-hashes \
  -r benchmarks/aifc/requirements-linux-x86_64-cu128.lock

PYTHONPATH=src:. ILUME_BENCHMARK_ENVIRONMENT=ilume-aifc \
conda run --no-capture-output -n ilume-aifc \
  python -c 'from benchmarks.common.config import load_benchmark_config; from benchmarks.aifc.environment import validate_aifc_environment; print(validate_aifc_environment(load_benchmark_config("configs/benchmarks/aifc.yaml"))["pretrained_snapshot"])'
```

普通 IL 生成一个 canonical `cation.anion` graph；solvation/transfer 分别编码 cation、anion、solute，
transfer_organic 分别编码 solute、solvent。所有 component 共用唯一 AIFC encoder，并按 registry slot
order concat；conditions 随后按 authoritative 顺序 concat，并与 representation 一起经过作者原有
ReLU。Target 和所有 conditions 都只用当前
fold train rows 做 population z-score。训练固定 seed 1000、batch 64、Adam `1e-3`、MSE、constant
LR 和 10 epochs，只发布 epoch 10 final state。

固定 DGL CPU golden reference 验证 prediction、representation 和 attention；当前 21-task 数据
审计的 11,282 个唯一 view 全部 fragmentation 成功。251,297 个原子中 5,618 个进入作者定义的
unknown motif（2.236%），不会删除样本。完整科学与审计边界见
[ADR-0060](docs/adr/0060-aifc-stage3-baseline.md)。

</details>

## 输出与结果汇总

新 train/evaluate 不覆盖既有输出，恢复必须显式请求。每个操作目录冻结 `run_config.yaml`，写入公开安全的 `metadata.json`，成功后生成 `summary.json`；checkpoint、训练日志和 tensor 默认不进入 Git。完整身份与 checkpoint 规则见 [ADR-0021](docs/adr/0021-identity-audit-contract-v1.md)。

全局 summarizer 只收录显式选中的目录。`--input` 提供一个或多个扫描根；可选 `--include` 是精确目录前缀白名单，省略时扫描全部 input。include 必须存在、位于某个 input 内并至少匹配一个 reporting candidate；重叠路径会去重，不支持 glob。

```bash
python scripts/benchmarks/summarize.py \
  --input outputs/v2 outputs/benchmarks \
  --include \
    outputs/v2/stage3/base \
    outputs/benchmarks/v3/mlp \
    outputs/benchmarks/fixed-budget-v1/ecfp_xgboost \
    outputs/benchmarks/v1/ilume_stage3_single_task_mlp \
  --output summary
```

只有 schema 完整且 comparison identity 兼容的 completed Stage 3 run 进入榜单；按 ADR-0031 允许 train-only normalization 不同，但 valid/test source 与其余协议必须一致。`stage3_{test,validation}_task_mae.csv` 以模型为行、registry task 为列，分别展示 test 原始 MAE 和 validation 五折 MAE 均值；对应的 `stage3_{test,validation}_task_rank.csv` 按每个任务的 MAE 从小到大给出模型排名。`ilume_scatter/validation/` 合并 validation 榜首 ILUME 的五折 raw prediction，`ilume_scatter/test/` 使用 test 榜首 ILUME 的 ensemble prediction，并按有样本的 task 输出 observed-vs-predicted SVG。旧 candidate 的 Stage 2 section 被忽略。其他 run 进入 health。损坏的选中正式结果或 prediction artifact 会使发布失败，已有 `summary/` 保持不变。详细 reporting 合同见 [ADR-0031/0043/0061](docs/adr/README.md)。

## 验证

```bash
pytest -q
```

测试只使用临时小数据，不执行正式 prepare、teacher cache、训练或五折 evaluation。
