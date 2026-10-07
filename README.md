# ILUME

ILUME 的正式 v4 流程是 **冻结双视图 Stage1 → Stage2-HoME → Stage3-HoME**。Stage1 只从 SMILES/2D graph 学习 task-agnostic1024D 表示；结构重建、双视图一致性、RDKit、离线Uni-Mol2与电子标签只在预训练中监督。角色2/2/1是 **loss权重，不是采样比例**；全量自然shuffle不变。下游拼接RDKit217，由ObjectEncoder内部投影1241→1024；Stage2/3永久冻结Stage1。Stage3保留20项实验任务与Phase2/3五项模拟辅助任务、Flat routing和固定末轮。合同见 [ADR-0089](docs/adr/0089-v4-frozen-dual-view-stage1.md)及 [ADR索引](docs/adr/README.md)。

命令从仓库根目录执行。安装依赖并准备 ILUME-Data 的 Stage1/2/3 输入；CSV 与输出不进入 Git。正式训练使用尚不存在的输出目录，旧 HoME 产物不能与新正式身份交叉加载。本页命令是运行手册，不属于自动验收。

```bash
python -m pip install -e ".[dev,tokenizers]"
```

## 正式 v4 主线

现役Stage1 Base为12层SMILES Transformer与8个独立residual图block，encoder-only约60.73M（2,048-token词表）；learned1024/atom512及下游冻结合同不变，见[ADR-0090](docs/adr/0090-stage1-v4-residual-encoder-capacity.md)。扩容直接复用既有v4 Stage1 corpus、统计和Uni-Mol缓存，但Stage1必须从头训练，下游Stage2/3重新生成表示并训练；旧输出不覆盖。

配置位于 `configs/v4/`，输出隔离到 `outputs/v4/`；Base与三个核心消融的完整命令，以及 Uni-Mol2 独立环境的创建、依赖安装和版本核验步骤，见 [v4运行手册](docs/v4-runbook.md)。先独立进行Uni-Mol2小样本成功率/吞吐/存储审计，再生成全量分片缓存；teacher使用版本锁定的独立环境，本仓库不会自动下载权重。Stage1训练导出仅encoder/fusion的 `stage1_encoder.pt`，下游不加载teacher或辅助头。旧v3 prepared/checkpoint不可直接复用，新Stage1/2/3均需prepare/train。

## v3 历史运行手册

下列v3主线、候选与核心消融命令保留用于历史追溯，不是v4入口，不覆盖旧结果；v3/legacy数值合同与代码路径未迁移。

现役 Stage3 以 `experiment/hydration` 替代 `experiment/transfer`（[ADR-0088](docs/adr/0088-stage3-hydration-replaces-transfer.md)）：单 solute + temperature_K、151 个体系、solvation GROUP、small 默认 PRIVATE 配方、random 五折、无 test。旧 transfer 产物保持历史身份；更换后的任务集合需要新的 Stage3 与 baseline 产物，禁止覆写既有输出。

Stage1 保持现有 v2 来源，完成 prepare 后训练：

```bash
python scripts/stage1/prepare.py --config configs/v2/stage1/base.yaml --output outputs/v3/stage1/base/prepare
python scripts/stage1/train.py --config configs/v2/stage1/base.yaml --output outputs/v3/stage1/base/train
```

Stage2 只准备九任务数据，不建立 teacher cache。physics-only HoME 使用逻辑 batch 256、微批 256，训练 10 轮并发布末轮完整九任务模型 `stage2_final.pt`、manifest 和供 Stage3 表示迁移的 `stage2_encoder.pt`。完整模型包括 Stage1 backbone、ObjectEncoder、全部 HoME owner、task routing/towers 与 atom adapter。一个逻辑 batch 只执行一次 optimizer/scheduler update；完整产物合同见 [ADR-0083](docs/adr/0083-stage2-home-full-artifact-evaluation.md)。

```bash
python scripts/stage2/prepare.py --config configs/v3/stage2/base.yaml --output outputs/v3/stage2/base/prepare
python scripts/stage2/train.py --config configs/v3/stage2/base.yaml --output outputs/v3/stage2/base/train
```

Stage2 不再提供独立 evaluate 入口或 validation/test 榜单（[ADR-0085](docs/adr/0085-retire-stage2-home-evaluation.md)）。完整九任务模型及 `SimulationHoME.predict` 保留；五项模拟任务在 Stage3 Phase2/3 继续训练，最终由 Stage3 模型预测。

Stage3 在 Phase1 适配 ObjectEncoder，Phase2/3 冻结它。Stage1 entity slots 冻结；15-epoch Phase1 仍只训练 20 项实验任务。Phase2 将五项模拟任务加入 thermophysical 与新增 electronic GROUP，Phase3 分别训练它们的 PRIVATE；Stage2 的模拟预测头和 atom adapter 进入最终 `three_phase_final.pt`。模拟任务使用 Stage2 prepared train 的原始样本逐轮覆盖，Phase2 共享 thermophysical GROUP 内模拟任务权重为 0.1、实验任务为 1.0，电子 GROUP 和 Phase3 不降权；电子 GROUP 沿用 thermophysical 的 4 轮预算，五项 PRIVATE 均按 Stage3 large 类训练。Stage3 实验 leaderboard 仍只汇总 20 项实验任务；模拟 validation 在 final 中单独记录。`--output` 是五折共同 root，实际训练位于 `foldN/`；并行时显式指定 GPU 槽。

```bash
python scripts/stage3/prepare.py --config configs/v3/stage3/base.yaml --output outputs/v3/stage3/base/prepare
python scripts/stage3/train.py --config configs/v3/stage3/base.yaml --fold 1 2 3 4 5 --output outputs/v3/stage3/base/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
python scripts/stage3/evaluate.py --config configs/v3/stage3/base.yaml --checkpoint-dir outputs/v3/stage3/base/train --split valid --fold 1 2 3 4 5 --output outputs/v3/stage3/base/valid
python scripts/stage3/evaluate.py --config configs/v3/stage3/base.yaml --checkpoint-dir outputs/v3/stage3/base/train --split test --ensemble-folds --output outputs/v3/stage3/base/test
```

只有身份一致且完整的 checkpoint 可以恢复；验证只记录，不选模型。先看五折 validation，再报告 test ensemble。Stage3 的正式最终文件名仍为 `three_phase_final.pt`；五项模拟任务可通过 `stage3.home.load_simulation_final()` 加载 final，并使用模型的 `predict_simulation()` 接口和 Stage2 的 packed 输入推理。

## v3 历史 HoME 候选

`base1-1` 至 `base1-10` 是十组隔离调参配置，差异和配对规则见 [ADR-0087](docs/adr/0087-stage2-stage3-home-base1-candidates.md)。先完成上方正式 Base 的 Stage2 prepare。`base1-1`～`base1-6` 各自训练 Stage2 并重新准备 Stage3；`base1-7`～`base1-10` 直接复用 Base 的 Stage2 和 Stage3 prepared 数据。下面分别示范 `base1-1` 与 `base1-7`；将编号替换为对应候选即可，不复用其他候选的 checkpoint 或输出。

```bash
python scripts/stage2/train.py --config configs/v3/stage2/candidates/base1-1.yaml --output outputs/v3/stage2/base1-1/train
python scripts/stage3/prepare.py --config configs/v3/stage3/candidates/base1-1.yaml --output outputs/v3/stage3/base1-1/prepare
python scripts/stage3/train.py --config configs/v3/stage3/candidates/base1-1.yaml --fold 1 2 3 4 5 --output outputs/v3/stage3/base1-1/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
python scripts/stage3/train.py --config configs/v3/stage3/candidates/base1-7.yaml --fold 1 2 3 4 5 --output outputs/v3/stage3/base1-7/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
```

评估时使用同名候选的 Stage3 配置、train root 和独立 valid/test 输出路径；模拟域沿用上方 `--domain simulation --ensemble-folds` 命令形式。候选不自动选优，不影响正式 Base。

## v3 历史核心消融

三个对照各自有独立身份与输出根，不与正式产物交叉加载；模型细节及解释边界见 [ADR-0082](docs/adr/0082-home-mainline-and-core-ablations.md)。

### w/o Stage1

用同结构、seed 42 随机初始化的 Stage1 编码器训练完整九任务 Stage2-HoME，再执行包含五项模拟任务 Phase2/3 的正式 Stage3 配方。它不读取 Stage1 预训练权重，但仍使用 Stage1 prepare 的 tokenizer/descriptor schema。

```bash
python scripts/stage2/prepare.py --config configs/ablations/no_stage1_stage2.yaml --output outputs/v3/ablations/no_stage1/stage2/prepare
python scripts/stage2/train.py --config configs/ablations/no_stage1_stage2.yaml --output outputs/v3/ablations/no_stage1/stage2/train
python scripts/stage3/prepare.py --config configs/ablations/no_stage1_stage3.yaml --output outputs/v3/ablations/no_stage1/stage3/prepare
python scripts/stage3/train.py --config configs/ablations/no_stage1_stage3.yaml --fold 1 2 3 4 5 --output outputs/v3/ablations/no_stage1/stage3/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
python scripts/stage3/evaluate.py --config configs/ablations/no_stage1_stage3.yaml --checkpoint-dir outputs/v3/ablations/no_stage1/stage3/train --split valid --fold 1 2 3 4 5 --output outputs/v3/ablations/no_stage1/stage3/valid
python scripts/stage3/evaluate.py --config configs/ablations/no_stage1_stage3.yaml --checkpoint-dir outputs/v3/ablations/no_stage1/stage3/train --split test --ensemble-folds --output outputs/v3/ablations/no_stage1/stage3/test
```

### w/o Stage2

从同一 Stage1 checkpoint 与 Stage2 seed 导出零更新 ObjectEncoder；Stage3 Phase1 仍适配它，且不接收 Stage2-HoME owner。按当前消融合同，该对照维持 20 项实验任务，不加入五项模拟辅助训练；因此与 Full ILUME 的差异不只包含 Stage2 权重。该对照依赖已完成的正式 Stage2 encoder，仅用于验证配对来源与身份。

```bash
python scripts/stage2/zero_update.py --config configs/v3/stage2/base.yaml --trained-encoder outputs/v3/stage2/base/train/stage2_encoder.pt --output outputs/v3/ablations/no_stage2/stage2_zero_update
python scripts/stage3/prepare.py --config configs/ablations/no_stage2_stage3.yaml --output outputs/v3/ablations/no_stage2/stage3/prepare
python scripts/stage3/train.py --config configs/ablations/no_stage2_stage3.yaml --fold 1 2 3 4 5 --output outputs/v3/ablations/no_stage2/stage3/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
python scripts/stage3/evaluate.py --config configs/ablations/no_stage2_stage3.yaml --checkpoint-dir outputs/v3/ablations/no_stage2/stage3/train --split valid --fold 1 2 3 4 5 --output outputs/v3/ablations/no_stage2/stage3/valid
python scripts/stage3/evaluate.py --config configs/ablations/no_stage2_stage3.yaml --checkpoint-dir outputs/v3/ablations/no_stage2/stage3/train --split test --ensemble-folds --output outputs/v3/ablations/no_stage2/stage3/test
```

### w/o Stage3-HoME

独立单任务 MLP 使用新正式 Stage3 prepared 的冻结 1024D 表示，各 task/fold 独立训练 `input → 1024 → 512 → 1`，固定 10 epochs、发布末轮模型。它维持 20 项实验任务，是整个 Stage3 后端对照；与 Full ILUME 的差异还包含五项模拟辅助训练，不能将差异单独归因于 HoME routing；不做预算匹配。需有 BF16-capable CUDA。

```bash
python scripts/benchmarks/sweep.py --config configs/ablations/no_stage3_home.yaml --output outputs/v3/ablations/no_stage3_home --max-workers 1
```

## 四项模拟性质比较

十个正式 baseline 另训 heat of vaporization、thermal expansion、HOMO、LUMO，各任务使用 catalog 的原有 train/valid/test，独立训练一次；不训练 partial charge，也不让模拟数据更新实验任务模型。神经 baseline 沿用各自 10 epochs、XGBoost 保持 1000 trees，发布末轮状态。内部单任务 MLP 消融及历史 split 配置不加入这些作业。合同见 [ADR-0086](docs/adr/0086-scalar-simulation-baselines-and-reporting.md)。

正式 baseline 的 sweep 默认运行实验任务和四项模拟任务；仅训练模拟任务时使用：

```bash
python scripts/benchmarks/sweep.py --config configs/benchmarks/mlp.yaml --benchmark simulation --output outputs/benchmarks/v4/mlp --max-workers 1
```

其余九个模型使用对应 `configs/benchmarks/<model>.yaml` 和独立 `outputs/benchmarks/v4/<model>`。全部任务可省略 `--benchmark`，只跑实验任务用 `--benchmark stage3`；新运行不覆盖旧 v3 输出；同一输出根固定使用同一种 sweep selector。模拟模型位于 `simulation/<task>/train/attempt-NNN/`，valid/test 在同一 task scope 下。单任务入口使用 `--benchmark simulation`，训练不传 `--fold`，评估使用 `--checkpoint .../train/attempt-NNN`（模型目录），不传 fold 或 ensemble 参数。

Full ILUME 与 no-Stage1 的模拟性质评估使用五个 Stage3 final 对同一固定 split 预测：每折先还原到原单位，再取预测均值计算指标；不对指标取均值。Full 命令为：

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --domain simulation --split valid --ensemble-folds --checkpoint-dir outputs/v4/stage3/base/train --output outputs/v4/stage3/base/simulation_valid
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --domain simulation --split test --ensemble-folds --checkpoint-dir outputs/v4/stage3/base/train --output outputs/v4/stage3/base/simulation_test
```

no-Stage1 使用其独立配置与输出根；no-Stage2 和单任务 MLP 消融没有这四项模拟预测分支，不支持该 domain。simulation 评估不接受 `--fold`、`--tasks` 或 `--checkpoint-epoch`。partial charge 仍在 Stage3 中训练并记录 validation，但不进入本次四项 scalar 比较。

统一 summarizer 发布独立 `simulation_{validation,test}_leaderboard.csv`、逐任务 metrics/MAE/rank 表。主指标为四任务等权 macro normalized MAE，归一尺度来自原始 train split 的 population standard deviation；原单位 MAE/RMSE/R² 同时发布。比较身份严格核对原始 split、完整行集合和尺度，不能静默取交集。模拟结果不进入实验榜单、wins、radar 或 scatter。Stage2 evaluate 继续退役。

## Baselines and Ablations

独立 baseline 的配置在 `configs/benchmarks/`，代码在 `benchmarks/`。训练预算、环境和模型合同从 [ADR 索引](docs/adr/README.md)查阅。以 D-MPNN 为例：

```bash
python scripts/benchmarks/sweep.py --config configs/benchmarks/dmpnn.yaml --output outputs/benchmarks/v4/dmpnn --max-workers 1
```

高级 baseline 的环境与资产步骤如下；各模型不自动安装或回退。

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

固定 DGL CPU golden reference 验证 prediction、representation 和 attention；当前 20-task 数据
审计的 11,282 个唯一 view 全部 fragmentation 成功。251,297 个原子中 5,618 个进入作者定义的
unknown motif（2.236%），不会删除样本。完整科学与审计边界见
[ADR-0060](docs/adr/0060-aifc-stage3-baseline.md)。

</details>

## 输出与结果汇总

新训练和评估输出不覆盖既有目录；每个操作记录 `run_config.yaml`、`metadata.json` 与完成后的 `summary.json`。全局 summarizer 只收录显式选中的目录，并对 prediction 完整性进行验证；Stage3 experimental 与 simulation 使用独立 leaderboard，Stage2 不发布榜单。见 [ADR-0021](docs/adr/0021-identity-audit-contract-v1.md)、[ADR-0031](docs/adr/0031-stage3-summary-normalization-relaxation.md)、[ADR-0085](docs/adr/0085-retire-stage2-home-evaluation.md)与 [ADR-0086](docs/adr/0086-scalar-simulation-baselines-and-reporting.md)。

```bash
python scripts/benchmarks/summarize.py --input outputs/v4 outputs/benchmarks/v4 --include outputs/v4/stage3/base/valid outputs/v4/stage3/base/test outputs/v4/stage3/base/simulation_valid outputs/v4/stage3/base/simulation_test outputs/benchmarks/v4 --output summary_v4
```

## 验证

```bash
pytest -q
```

测试只使用临时小数据，不启动正式 prepare、训练或五折评估。旧实验设计只从 [历史 ADR](docs/adr/README.md#冻结合同与历史) 与 Git history 追溯。
