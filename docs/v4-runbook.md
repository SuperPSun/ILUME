# v4 运行手册

本手册是现役 v4 Stage1及实体HoME Stage2/3运行入口。Stage1现役描述符选择与数值处理见[ADR-0112](adr/0112-v4-explicit-rdkit-211-selection.md)、[ADR-0113](adr/0113-v4-descriptor-validity-ring-log-review.md)、[ADR-0115](adr/0115-stage1-peoe-gasteiger-validity-mask.md)；Stage2/3见 [ADR-0095](adr/0095-v4-entity-home-without-object-encoder.md)，旧Object/v5运行需历史Git版本。


从仓库根目录执行，使用新的 `outputs/v4/` 路径，不覆盖历史v3输出。Stage1科研合同见 [ADR-0089](adr/0089-v4-frozen-dual-view-stage1.md)。CUDA示例使用四张设备。

## 数据 split 与路径

2026-11-04起，正式Stage2/3 Base采用 **G2_C01 / Stage2 007 / Stage3 032**，并显式配对 `data/3407`（[ADR-0114](adr/0114-g2-c01-main-model.md)）。共同训练seed42，Stage3按42+fold派生；数据划分seed与训练seed不得混淆。旧G2_C05的 `optimized_3407`、HPO、基线与消融产物保持原身份，历史五套划分结果不能重标为G2_C01。当前主模型命令见[下方运行步骤](#g2_c01-主模型)。

Stage1输入仍为 `data/stage1`；现役Base的Stage2/3输入分别为 `data/3407/stage2`、`data/3407/stage3`，共用 `data/3407/task_catalog.csv`。Stage2的 `data.data_root` 为split根目录，catalog内的 `stage2/...` 路径相对此根解析；Stage3的 `data.stage3_dir` 指向同一split的 `stage3`。Base的prepare审计显式写入各自 `outputs/v4/stage{2,3}/base/prepare/data_metadata.json`；旧split42 Base须使用历史Git和匹配来源复现，不能运行当前Base替代。

测试其他数据 split 时，使用独立的自包含 YAML，将 Stage2 的 `data_root/task_catalog_path` 与 Stage3 的 `stage3_dir/task_catalog` 同步指向对应 split，并为 prepare、训练和评估指定独立输出及对应的下游引用。目录名 `42` 只定位数据，不自动改变训练随机种子、split policy 或 cv repeat。不同内容的 split 必须重新 prepare，不能共用数据身份不一致的准备产物或检查点；Stage1 来源保持不变。历史配置及基线保留各自的数据合同，不由本次目录调整迁移。

五套数据目录为 `42`、`123`、`1234`、`3407`、`90756`，仓库正式Base YAML选择3407。各套的 Stage3 `stage2_final/simulation_artifacts_dir` 必须配对到对应 split 的 Stage2 来源。并行 prepare 使用各自的 `preparation.cache_dir`：冻结 Stage1 表示可按身份串行复用，但当前缓存写入不支持多个进程共同写入。prepare、train、evaluate的运行元数据 `data_metadata` 引用对应配置的数据审计路径；Base使用输出目录内的显式审计，不改写数据根的历史metadata。

各 split 分别汇总，用 `--input` 或 `--include` 限定到该 split 的结果，再写入独立汇总目录；现有 summarizer 不合并不同来源身份的 split。根目录 `data/task_catalog.csv` 保留旧42副本，其他 split 使用各自目录内的 catalog。`1234`、`90756` 附带的 `stage1/properties` 不属于现役共享 Stage1 输入；Stage1 仍显式读取 `data/stage1`。

当前Base采用十二层SMILES Transformer和八层双路2D DGT：词表2,048个token时encoder为77,986,987参数（[ADR-0116](adr/0116-stage1-dual-graph-transformer.md)）。learned1024、atom512及211维描述符接口不变。同policy的预处理语料、统计和完整Uni-Mol缓存可复用，但额外需要独立拓扑sidecar；旧D-MPNN不能恢复到DGT，须从头训练Stage1并重新生成Stage2/3表示及训练。切换Stage2/3 split本身不要求重训来源一致的Stage1。若训练/下游目录已占用，选择新路径并同步更新下游引用，或在运行前明确安排归档；这些命令不授权替换历史结果。

## RDKit 2026.9.1 主线升级

现役主项目依赖精确固定 `rdkit==2026.9.1`，实际导入运行时为 `2026.09.1`（[ADR-0111](adr/0111-rdkit-2026-09-1.md)）。该版本原始描述符仍为同名、同顺序217维。现役Base按[ADR-0112](adr/0112-v4-explicit-rdkit-211-selection.md)显式保留211列；learned1024、atom512不变，实体变为1235维，双槽位L1输入2478维。训练预算不变，筛选工具的候选清单不自动接入模型。新版构象生成的可选ETKDGv4不启用，继续ETKDGv3/MMFF或UFF。

主环境继续命名为 `ilume`，不另建版本后缀的正式环境。先确认没有进程正在使用其中的RDKit，并备份旧包及版本清单；只更新RDKit，保留当前Python、PyTorch/CUDA和NumPy，不使用无版本限制升级。若存在conda安装的 `rdkit/librdkit`，先卸载pip的RDKit，再对该环境执行conda的 `--force-remove rdkit librdkit`；执行前用 `--dry-run` 确认只移除这两个包，不连带更改其他依赖。pip-only环境无需conda移除步骤。完成旧包清理后安装并验证：

```bash
conda activate ilume
python -m pip install --upgrade --no-deps --only-binary=:all: rdkit==2026.9.1
python - <<'PY'
import importlib.metadata
from rdkit import rdBase
from rdkit.Chem import Descriptors
assert importlib.metadata.version("rdkit") == "2026.9.1"
assert rdBase.rdkitVersion == "2026.09.1", rdBase.rdkitVersion
assert len(Descriptors._descList) == 217
print("RDKit package/runtime: 2026.9.1 / 2026.09.1; descriptors: 217")
PY
```

不要直接覆盖conda/pip混装的旧包，也不要删除整个 `ilume` 环境来完成此次升级。升级后须启动新的Python进程/Jupyter kernel；已导入的扩展不会在原进程自动更新。基线及teacher独立环境不随主环境维护而自动更新；teacher用于新版缓存前须另按下文配方升级并验证。历史复现需要匹配历史Git及依赖，不再使用已升级的主环境。

**旧产物不迁移**：即使维数相同，2025.09.5语料、统计、tokenizer/图、teacher、partial-charge sidecar与检查点也不改SHA、不重标版本、不与新环境混用。重新prepare Stage1、准备电荷sidecar、在匹配2026.09.1的独立Uni-Mol环境重新审计/生成teacher缓存，再从头训练Stage1并重新prepare/train Stage2/3。原始CSV、split authority和已验证Uni-Mol权重文件可复用；不保证升级后的canonical、QC成员及描述符数值逐bit相同。

主线继续使用 `configs/v4/stage1/base.yaml`、`configs/v4/stage2/base.yaml`、`configs/v4/stage3/base.yaml`，不另设带RDKit版本后缀的Base配置。三者仍按 `outputs/v4/stage{1,2,3}/base/` 配对prepare、teacher、电荷sidecar与训练来源；本节版本升级不改变科研参数；后续Base的G2_C01容量、007/032超参及3407选择由[ADR-0114](adr/0114-g2-c01-main-model.md)另行确定。复用标准输出路径前，须按工作区规则完成用户明确确认的旧产物清理或归档；不能覆盖旧产物，也不能只删权重后在遗留metadata/log的运行目录中启动新训练。保留的旧电荷sidecar不能用于新版prepare，须先另行归档后重新生成。未经授权，不自动移动或删除历史结果。

## 描述符选择与历史审计

独立阈值筛选和处理后复审已于2026-11-06退役（[ADR-0110](adr/0110-stage1-descriptor-selection-audit.md)），相关脚本、实现、YAML与专属测试已移除。正式prepare仅按Base YAML和下节schema处理描述符，不再要求执行独立筛选命令；数值有效性、归一化和来源完整性校验继续保留。

描述符去留通过逐项人工评审后另行明确科研合同；当前仍为下述211列，197维人工候选尚未采纳。独立审计退役不改变现有prepared、teacher、电荷sidecar、training identity或模型兼容性，也不移除gradient audit、teacher或电荷审计。旧独立审计产物不再作为活动入口；移动或清理仍须先报告精确清单并获得确认。

## Base 的 211 维描述符选择

正式[Stage1 Base](../configs/v4/stage1/base.yaml)的`descriptor.excluded_names`固定排除`Ipc`、`Kappa3`、`SMR_VSA8`、`SlogP_VSA9`、`fr_Nhpyrrole`、`fr_amide`。保留12项partial-charge/BCUT2D描述符和17项高相关列，不采用审计默认的182列。原始计算仍为217列；同一schema选出的211列同时用于Stage1辅助目标和下游显式输入。标准化仅拟合train有限值，非有限项标准化后填零，Stage1 loss用逐维有效mask排除；分子归一化与角色2/2/1不变，下游不增加mask特征。

现役Base显式设置`descriptor.value_policy: phi_mask_ring_log1p_gasteiger_v2`（[ADR-0115](adr/0115-stage1-peoe-gasteiger-validity-mask.md)），保留[ADR-0113](adr/0113-v4-descriptor-validity-ring-log-review.md)的Phi与环计数处理。Phi仅对单连通、重原子数≥3、键数≥2且结果有限的结构有效；八项环计数统一先log1p再拟合train均值/标准差。另在分子副本上严格检查Gasteiger参数及所有原子/隐式氢电荷；计算失败或非有限时，整组14项PEOE_VSA不参与scaler拟合及描述符监督，即使RDKit返回有限PEOE值也会屏蔽。其他描述符仍按各自已有规则处理。

不适用或非有限项标准化后填0，下游与Stage1监督使用同一schema/scaler，不重复取log；不删分子、不加输入维度、不变loss或采样。schema format3内嵌policy，新v2和旧`phi_mask_ring_log1p_v1`分别保留各自行为；默认raw仍为历史format2并省略配置字段。维数仍为211，但新policy改变corpus/feature/training身份，旧v1 prepared、teacher、sidecar和checkpoint不能混用；须按原流程在新目录重新准备与训练，不覆盖当前产物。实施不启动正式运行。

learned1024和atom512、独立回归头输入不变；实体为1024+211=1235，GLOBAL/GROUP L1双槽输入为2478。Stage2/3读取来源schema/identity自动确定宽度，不在下游YAML再列一份维数。省略排除列表的旧配置仍为217维，旧产物必须与其原配置及来源配对。

**需要新准备和训练**：排除列表改变corpus/feature/training identity，旧217维prepared、teacher、电荷sidecar与checkpoint不能直接复用。按本手册依次重新prepare Stage1、生成匹配身份的teacher/sidecar、训练Stage1，再prepare/train Stage2和prepare/train/evaluate Stage3。Uni-Mol源权重、原始数据和split authority可复用；当前缓存即使teacher数学输入未变，也不能重标旧身份复用。已有标准路径中的217维prepared及独立审计结果保持只读，须先经用户另行确认归档/清理，或选择新输出并同步所有下游引用。本次代码修改不执行正式运行。

## 跨设备训练产物回传

4090作为统一训练产物归档设备。本规则适用于正式Stage1/2/3、独立回归头后训练、HPO、基线及其他训练实验；无论在哪台设备训练，都必须将相关训练结果与完整模型权重回传到4090的ILUME项目目录，保持原有相对目录结构，HPO仍归入 `outputs/HPO`。在4090本机训练时直接保留在对应目录。

回传包含完整运行目录：模型权重（包括 `.pt` 等文件）、final/阶段锚点/owner delta、周期检查点及恢复状态、`run_config.yaml`、metadata/summary/身份清单、metrics/diagnostics、日志，以及已生成的预测和评估报告。一并保留对应运行YAML及依赖的Prepare、特征/归一化统计和来源产物；4090已有相同且核验通过的来源可复用。失败或中断任务也回传已有现场，不只传成功摘要，不排除模型权重，不改变数据split或训练身份。

长任务可在完整轮或任务完成后分批回传；最终验收须确认4090归档齐全。不要把正在写入的日志/检查点当成完整快照：先停止相关写入或选择已固定的完成产物，再核验。目标目录先做差异预览；同路径不同身份的结果使用独立归档目录，禁止覆盖或混入既有运行，也不得将旧文件反向同步回正在训练的目录。

回传完成后按文件清单、大小和SHA核对源/目标，可用 `rsync -acni` 做只读差异检查，并通过项目现有身份/产物加载校验。确认权重、日志与检查点配套一致后才视为回传完成；核验前保留训练设备上的原始产物。实际传输地址以4090的SSH配置和项目路径为准；本规则不自动授权删除源产物或覆盖目标，也不要求程序自动同步。

## Stage1 数据准备与离线 teacher

先完成当前描述符Gasteiger v2对应的corpus与电荷sidecar；旧v1不能直接混用。以下标准路径已有历史产物时，须使用新路径并同步YAML引用，或先另行确认归档；不要直接覆盖。

```bash
python scripts/stage1/prepare.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/prepare
```

Teacher生成必须在**独立环境**执行。下方命令适用于Linux x86-64及兼容CUDA 12.8的驱动，使用Python 3.11，将teacher包的旧pandas/NumPy约束与正常训练环境隔离。其他CUDA运行时应按 [PyTorch官方说明](https://pytorch.org/get-started/previous-versions/) 安装匹配wheel，替代 `cu128`。Uni-Mol安装说明也要求其RDKit环境使用NumPy低于2（[官方安装指南](https://github.com/deepmodeling/Uni-Mol/blob/main/docs/source/installation.md)）。

创建环境，安装项目及锁定版本的离线teacher包。以 `--no-deps` 安装 `unimol-tools`，防止依赖解析器替换已选PyTorch、NumPy或RDKit版本；所需运行依赖显式安装：

```bash
conda create -n ilume-unimol2 python=3.11 pip -y
conda activate ilume-unimol2
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 pandas==1.5.3 rdkit==2026.9.1
python -m pip install -e .
python -m pip install scipy joblib addict scikit-learn numba
python -m pip install --no-deps unimol-tools==0.1.3.post1
```

从仓库根目录下载官方 [Uni-Mol2 84M检查点](https://huggingface.co/dptech/Uni-Mol2/blob/main/modelzoo/84M/checkpoint.pt)。发布文件约337 MB，下方校验SHA-256。项目明确关闭自动模型下载，因此文件必须位于配置指定路径：

```bash
mkdir -p assets/unimol2/modelzoo/84M
curl --fail --location --retry 3 --continue-at - \
  https://huggingface.co/dptech/Uni-Mol2/resolve/main/modelzoo/84M/checkpoint.pt \
  --output assets/unimol2/modelzoo/84M/checkpoint.pt
echo '5b9241630f1cf0b173fb06d1e76096e5daf5f91c3e51b066ba69524dafe60e35  assets/unimol2/modelzoo/84M/checkpoint.pt' | sha256sum --check -
```

校验和失败时不要生成缓存；获得已验证副本后再继续。随后检查环境及本地文件：

```bash
python - <<'PY'
from pathlib import Path
import importlib.metadata
import json
import numpy
import torch
from rdkit import rdBase
from unimol_tools.models import unimolv2
from stage1.identity import validate_feature_generation_runtime

checkpoint = Path("assets/unimol2/modelzoo/84M/checkpoint.pt")
assert checkpoint.is_file(), f"Missing local checkpoint: {checkpoint}"
assert importlib.metadata.version("unimol_tools") == "0.1.3.post1"
assert torch.__version__.split("+", 1)[0] == "2.9.0", torch.__version__
assert torch.cuda.is_available(), "CUDA is unavailable in the teacher environment"
assert numpy.__version__ == "1.26.4", numpy.__version__
assert importlib.metadata.version("rdkit") == "2026.9.1"
assert rdBase.rdkitVersion == "2026.09.1", rdBase.rdkitVersion
assert str(unimolv2.MODEL_CONFIG_V2["weight"]["84m"]) == "modelzoo/84M/checkpoint.pt"
metadata = json.loads(Path("outputs/v4/stage1/base/prepare/artifacts/metadata.json").read_text())
validate_feature_generation_runtime(metadata)
print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}; rdkit={rdBase.rdkitVersion}")
print(f"unimol_tools={importlib.metadata.version('unimol_tools')}; checkpoint={checkpoint}")
PY
```

验证器检查预处理语料的 `feature_generation_contract`，含RDKit运行时及分词器版本。须匹配 `rdBase.rdkitVersion`，不能只看 `pip show rdkit`：包元数据与实际导入运行时可能不同。当前新运行的训练/teacher两环境均应为 `2026.09.1`。旧语料记录其他版本时，按历史Git版本和独立历史环境复现；不要修改产物元数据或绕过检查。升级后的全新语料与teacher目录必须配对，不能用新版teacher读取旧Base prepare。

先生成独立小规模审计，检查 `manifest.json`（attempted/验证集/failed、吞吐及embedding字节数）；该审计不是训练缓存：

```bash
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 8 --limit 100 --output outputs/v4/stage1/base/teacher_audit
```

确认成功率、吞吐和存储后生成完整缓存。重复相同命令恢复不可变分片，不恢复任意部分行。OOM视为运行失败，不视为可mask分子。batch大小/设备和 `--workers` 是执行设置；canonical结构顺序及每分子构象种子固定。

Teacher默认推理batch为8，仅使用既有`--batch-size`控制。分片、结构顺序、构象及失败mask规则不变；batch不改变teacher recipe，已提交分片严格校验后原样复用，未完成分片重算。不同推理batch形状可能造成浮点舍入差异，不承诺新生成向量逐bit相同。单分子仍放不下时明确失败，不自动缩批、裁剪分子或把OOM记成无效3D目标。

```bash
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 8
```

GPU利用率低时可通过 `--workers` 在并行CPU进程准备构象及Uni-Mol特征。有界有序队列预取输入，父进程负责GPU推理；worker不加载teacher模型或使用CUDA。特征只算一次，不在推理前重复计算。默认 `--workers 1` 保留串行执行。worker增加RAM占用，不得超过作业分配的CPU资源。

Slurm作业可从 `#SBATCH --cpus-per-task=8`、一张GPU开始，在已分配作业内执行下方命令，保持已有独立teacher环境。先检查这次**新的**并行审计，再运行完整生成命令：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 8 --workers 8 --limit 100 --output outputs/v4/stage1/base/teacher_audit_parallel
python -u scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 8 --workers 8
```

重启同一缓存根前先停止旧作业；不支持并发写入。worker数变化仍可复用已完成分片。重定向/Slurm输出在启动、每个分片提交及推理batch之间约每30秒刷新 `teacher_progress` JSON。`processed/total` 包含内存中的工作；`committed` 统计安全发布的分子。单个慢构象/batch可能延迟下一条日志。可用 `tail -f slurm-<job-id>.out` 监控。

完整缓存完成后回到 `ilume` 训练环境。Uni-Mol仅用于缓存生成；Stage1训练读取缓存。Stage1拒绝部分或未绑定teacher缓存；损失按角色2/2/1加权，加载器仍为原始自然打乱。

```bash
conda deactivate
conda activate ilume
python scripts/stage1/train.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/train
```

现役Base全局batch256，由此前512减半以降低显存占用；单卡每步256条，四rank DDP每GPU为batch64。每rank仍使用8个DataLoader worker，四rank训练加载器共32个worker。LR保持1e-4，优化器步数及warmup/cosine预算按新batch重算；不能从旧512配置的checkpoint恢复。语料、teacher及匹配的sidecar仍可复用。worker数量仅属执行参数，本身不改变科研训练身份。

### Stage1 训练索引过滤与 H100 双卡

按[ADR-0122](adr/0122-stage1-graph-size-qc.md)，Base及`base-100h.yaml`使用`training.max_graph_nodes: 160`：启动训练前排除原子数或无向键数严格超过160的扩增训练样本，等于160保留；若过滤会排除原始训练样本则拒绝启动。当前仅14条中性扩增样本被排除，原始训练/验证及固定审计probe不变。运行目录`training_filter.json`记录来源、阈值、样本结构/大小和索引，清单纳入training identity。先过滤再按rank分配，不在DDP迭代中跳过batch。

**无需重跑prepare、teacher、电荷sidecar或拓扑。** 沿用现有全部缓存及原分词器、标准化统计，现有SHA检查不变；prepare已撤回最初的`data.max_graph_nodes`过滤，旧字段不再接受。训练启动会扫描manifest、核查少量相关分片，不进行3D构象生成或teacher推理。已有样本的统计保持原值，本方案不同于重新拟合筛选后统计。当前有效训练集5,176,579条，global84每轮61,626步；来源更新时重新解析清单与预算。

H100专用`configs/v4/stage1/base-100h.yaml`使用全局batch84、5轮；双rank每卡42，保持LR1e-4、query32及现有attention执行开关。沿用已有匹配缓存，在YAML来源路径配对且输出目录未占用后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 scripts/stage1/train.py --config configs/v4/stage1/base-100h.yaml --output outputs/v4/stage1/base/train_graph160_b84_ddp2
```

旧batch64或未过滤checkpoint不得续到此配置。筛除极端结构与DDP不能构成显存安全保证；不自动缩batch或跳过样本，当前训练不会被代码更新热切换。

启动阶段rank0会先扫描manifest、核对少量源分片位置，再校验teacher/sidecar身份；随后广播相同清单给两rank。此阶段可能暂时没有进度条或训练目录，不能仅据此判断卡死。2026-11-09本地只读检查中，过滤解析约72秒，完整training identity解析约135秒；NFS负载和缓存状态可能延长时间。进入更新后检查`metrics.jsonl`与`training_filter.json`，不要将启动校验视为新一轮teacher生成。

`--output`只决定训练写入位置，不会自动改写下游YAML。上例使用独立训练目录时，后续Stage2/3的`initialization.stage1_encoder`必须指向该目录生成的`stage1_encoder.pt`，`initialization.stage1_artifacts_dir`仍指向原prepare；不得误读标准`train/`下的历史encoder。若用户已确认标准`outputs/v4/stage1/base/train`未占用，也可将上例输出改为该标准路径。仅提供命令不代表自动停止、清理或启动训练。

<a id="stage1-dgt-topology"></a>

### Stage1 DGT 拓扑与恢复

当前Base图分支为8层双路DGT，宽512、8头、每路FFN1024；SMILES仍为12层/FFN2048。原子与键分别均值池化后投影512D，保持learned1024/atom512。纯2D最短路径与环关系不读取监督目标、3D、描述符或角色。当前global batch256、10轮，LR、loss和mask定义不变，不自动调整batch或截断原子。

完成上方匹配corpus的prepare后，在`ilume`环境额外执行一次：

```bash
python scripts/stage1/prepare.py --config configs/v4/stage1/base.yaml --dgt-topology-only --output outputs/v4/stage1/base/dgt_topology
```

YAML的`auxiliary.graph_topology_cache`指向该目录的`artifacts`。它按corpus分片生成独立拓扑缓存，提供分片进度；同身份重启可继续完整分片，损坏/缺失/错位明确失败，不覆盖未提交的孤立分片。普通prepare不会自动生成它，缺少缓存时DGT训练直接提示上述命令。该操作不修改原corpus、teacher或电荷sidecar。已匹配Gasteiger v2的teacher无需仅因DGT重做；旧v1来源仍须按0115重建。

拓扑worker使用普通NumPy数组传输，父进程重建原dtype的tensor；避免PyTorch共享storage为大量输入/输出tensor占用文件句柄，在终端默认1024句柄上限下也无需依赖提高`ulimit`。计算、样本顺序、缓存格式及身份不变。先前因`Too many open files`失败且没有孤立分片时，可重跑同一命令续做；不自动删除失败记录或已有分片。整段流水线应使用`&&`连接或逐步确认成功，避免准备失败后继续触发下游缺失来源报错。

训练DataLoader同样按batch合并局部键索引及六种DGT拓扑tensor的底层storage，保留逐分子视图，避免batch256在共享内存传输时耗尽1024句柄。此修复不改训练batch、worker、采样、mask、预测或梯度，不需要重建prepare/teacher/拓扑缓存。已经卡在旧代码中的进程需要停止后重新启动；没有完整轮checkpoint时不能使用`--resume`，原运行记录须保留，新输出目录和下游来源引用须配对，或按工作区规则确认归档后复用标准路径。

DGT查表反向采用dense embedding归约，并复用query循环中的padding/索引；拓扑在CPU整批打包后传至设备，减少逐分子小传输。模型结构、参数、batch、loss、mask及dropout随机数序列不变，无需重做prepare/teacher/电荷/拓扑缓存。BF16梯度的浮点累加顺序发生变化，不能承诺旧新实现训练轨迹逐bit相同。已经运行的进程不会自动切换实现；本优化不授权停止训练或清理运行目录，仍只支持既定完整轮恢复。

原子/键特征进一步使用按分子offset隔离的dense查表，空键池保持可微零贡献。现役Base显式启用`training.dgt_attention_compile: true`，只在CUDA局部编译逐通道attention的纯数学部分，dropout随机数在编译区外生成；全模型`compile`仍为false，CPU走原即时路径。首次及部分新shape会有编译等待，不能用第一步估算整轮；编译错误直接失败，不自动切回即时。只复用PyTorch自带编译后端，不安装新包；目标环境须支持该后端。浮点归约顺序可有变化，不承诺新旧预测/训练逐bit一致。

`training.dgt_query_chunk_size: 32`与`dgt_attention_checkpointing: true`控制attention分块重计算；这两项及局部编译开关均不进入科研identity，但恢复要求与检查点完全相同。共享默认局部编译false且序列化省略，旧检查点缺少此字段按false解析；不能用新的true配置直接resume旧即时检查点，须新运行，或显式恢复原执行配置。新启动进程才生效，当前训练不会自动切换，本次优化不授权停止/清理运行。不是普通FlashAttention替换，不支持OOM自动降batch。完整模型及encoder导出保存DGT和读出投影；旧D-MPNN只能按内嵌历史配置加载，不能resume到新Base。Stage1训练之后仍需重新prepare/train Stage2及prepare/train/evaluate Stage3，Stage1始终冻结。下游新分子由同一构建器按需生成拓扑，不要求其出现在预训练corpus缓存。

### Stage1 进度与输入停顿

普通 `python scripts/stage1/train.py ...` 只启动一个训练进程，不自动使用全部GPU。DDP需用 `torchrun`，例如在实际分配四张GPU的作业内执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/stage1/train.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/train
```

使用未占用输出目录。训练、快速验证和完整验证加载器各用配置的worker数；CUDA加载器首次使用后保留worker，因此开始验证时worker资源可能增加。训练加载器按实际最大训练分片行数S、全局batch B与每rank worker数W，令每worker `prefetch_factor = 2 + ceil(S / (B × W))`，使提交的任务窗口覆盖约一个完整分片，再保留原有每worker两个batch的余量；worker可在当前分片的batch仍被消费时准备下一分片。DDP使用全局batch计算，各rank读取原有分区，不增加或重排样本。快速/完整验证仍固定每worker两个batch；W=0保持同步读取。

例如8192行分片、全局batch64、8个worker时，每worker预取18个batch，每rank最多144个在途batch任务；不是将这144个batch放进GPU。代价是更多CPU/pinned内存和在途传输句柄，具体峰值取决于分子大小、worker及消费速度。仍逐项执行语料、拓扑、teacher和电荷来源/完整性校验，错误在读取对应batch时传播，不忽略或替换样本。首分片、持续慢于训练的存储/解包，以及审计/验证停顿不能保证被隐藏；该优化不消除不同worker重复解包。

深预取必须同时使用pin-memory句柄修复：打包图的局部键索引与DGT拓扑在pin线程中复制为主进程普通CPU storage，避免已pin好的batch仍持有worker共享storage句柄。旧代码可能在1024软限制下报`received 0 items of ancdata`，继而`Pin memory thread exited unexpectedly`；这是CPU IPC传输故障，不等于CUDA显存OOM。修复保留深预取、原batch、worker、pinning与完整性检查，不要求提高`ulimit`或重建缓存。

这只是数据加载时序优化，不修改配置schema、科研/缓存身份、batch边界、mask/RNG或训练/恢复规则；已有匹配缓存可复用。新进程才会使用新预取窗口，不能热更新运行中的进程，也不因此获得轮中途恢复能力。应按作业CPU/RAM配额及存储吞吐选择worker，而非仅按GPU数量；增加worker不保证吞吐提升。

预期停顿发生在每1,000次更新的rank0梯度审计（其他rank等待）、每5,000次更新的审计加快速验证，以及轮边界的完整验证/检查点导出。审计日志在审计完成后写入，不在开始时写入。这些周期操作不能解释发生在对应边界前的停顿。

若GPU利用率下降且加载器 worker在 `wait_on_page_bit_common` 进入 `D` 状态，应检查存储/页面读取。语料加载校验并反序列化分片，将样本关联到只读SQLite Uni-Mol索引及mmap teacher 分片。多个worker在NFS读取这些文件，即使RAM充足也可能使GPU等待。优先使用已验证字节一致的本地SSD语料/teacher副本，并在独立受控尝试比较worker数量；不要重生成标签、关闭hash校验、改变采样或覆盖运行中作业。这些是运行建议，不表示已实现本地缓存，也不保证8个worker能解决NFS停顿。归因前检查 `nvidia-smi`、`vmstat 1`、worker等待状态及 `metrics.jsonl` 时间戳。

输出包含保留辅助头的恢复检查点及仅编码器 `stage1_encoder.pt`。部署/Stage2不加载teacher或辅助头。

<a id="stage1-loss-and-gradient-audit"></a>

### Stage1 损失与梯度审计

当前Base采用SMILES/原子/键重建0.20/1/1、alignment0.10、RDKit0.25、Uni-Mol0.50、electronic0.25及独立partial-charge0.25。这是用户选定配方，替代 [ADR-0091](adr/0091-stage1-v4-loss-weights-gradient-audit.md) 和 [ADR-0092](adr/0092-stage1-atom-charge-and-frozen-regression-heads.md) 的早期系数；损失定义与审计隔离不变。这些损失变化要求新的Stage1运行，不从旧系数恢复；已有语料/统计/teacher缓存仍可复用。下游Stage2/3须使用新训练编码器和新输出。

训练每完成1,000次优化器更新自动追加 `gradient_audit.jsonl`。rank0使用固定32分子验证探针及评估mask，关闭dropout；其他rank等待。报告原有七项原始编码器范数，启用时增加 `partial_charge_grad_norm`，以及 `weighted_grad_norms`（损失系数绝对值乘范数）、有效目标覆盖、探针ID/hash、步及尝试。既有指标和检查点选择不变。

设置 `training.gradient_audit_interval_steps: 0` 或传入 `--gradient-audit-interval-steps 0` 可关闭；正CLI值覆盖YAML，省略则按YAML。`gradient_audit_batch_size` 控制探针大小（默认32）。诊断设置不改变科研身份。缺失标签目标记录 `null` 和 `no_valid_targets`；小自然探针没有电子/电荷标签不说明梯度弱。范数来自各自目标，不是向量和：不要仅凭大小自动调整系数。DGT审计固定使用未编译attention，避免多目标保留计算图与编译buffer donation冲突；审计结束（含异常）恢复正常训练的局部编译开关。审计增加一次前向及最多八次梯度计算；OOM明确失败，不自动缩小探针。监控命令：

```bash
tail -f outputs/v4/stage1/base/train/gradient_audit.jsonl
```

轮边界恢复追加带尝试标记的观察，不删除失败尝试行。审计绝不将验证导数用于参数更新，也不读取测试集。

<a id="partial-charge-sidecar-on-an-existing-corpus"></a>

### 在已有语料上准备部分电荷 sidecar

解析器接受已核验的 `@<TRIPOS>` 头别名：`MOLEMOLE/MOLMOLLE/MOMOLULE/MMOLCULE/MOLECMOL→MOLECULE`、`MOLM/AMOL→ATOM` 和 `MOLD/BMOL→BOND`。已核验前缀拼写 `@<TRMOLS>`、`@<TRIMOL>`、`@<TMOLOS>` 和 `@<MOLPOS>` 也按 `@<TRIPOS>` 接受。不猜测其他拼写错误；不要修改原始MOL2或清单 SHA。其他解析/完整性检查保持严格。完整段名/前缀别名表进入Stage1电荷来源身份：身份不同时保留旧运行并使用新sidecar输出/缓存路径。电荷解析器不消费SUBSTRUCTURE，无需纠正其拼写。

普通Stage1 prepare也准备独立原子标签sidecar。语料及teacher缓存已存在时只执行sidecar命令，不改变二者。现役Base通过 `auxiliary.simulation_dir` 读取 `data/stage1/properties/partial_atomic_charge/train.csv`，并验证 `auxiliary.partial_charge_manifest` 引用的MOL2资源。正式v4训练仍需现有电子来源。下方命令匹配当前YAML缓存路径；仅在目录未使用或已有兼容format2 sidecar时执行：

已知显示限制：映射进度条当前使用完整结构清单大小作分母，而仅处理选定训练集/验证集 CSV。因此可能低于100%结束（例如28,214资源中的22,530训练行显示80%），这本身不算失败。检查attempted/mapped/skipped审计数量及完成运行的汇总/元数据。修正进度总数属于独立代码任务。

```bash
python scripts/stage1/prepare.py --config configs/v4/stage1/base.yaml --partial-charge-only --output outputs/v4/stage1/base/partial_charge
```

仅精确canonical匹配获得标签；电荷统计包含匹配Stage1训练结构的全部来源观察。同结构电荷向量分别保留，不平均或按首/末行选择。预训练复用一次分子前向，但独立计算各观察。语料采样、其他损失及teacher缓存不变；不沿种子传播或读取测试集/验证集标签。资源损坏/缺失及解析错误仍失败；无同构记录按下方审计跳过策略处理。format2 sidecar及观察策略改变电荷监督训练身份，因此旧sidecar/检查点不能在此配方恢复。完整检查点保留 `partial_charge_head`；仅编码器导出不包含。若科学上关闭此目标，须在独立自包含YAML设 `loss.lambda_partial_charge: 0`（不是仅执行开关）。

默认输出已有旧或失败运行时保留它，准备到新目录，例如 `--output outputs/v4/stage1/base/partial_charge_observations`。训练前将Base的 `auxiliary.partial_charge_cache` 和回归的 `partial_charge_cache` 指向该目录的 `artifacts/` 子目录。不要重标hash或删除旧输出来绕过兼容检查。

无同构记录跳过并记录于 `artifacts/mapping_audit.json`（`status: skipped`、`reason: no_graph_isomorphism`、split/CSV行号/mol_id/SMILES/资源文件名/SHA）。`artifacts/metadata.json` 含 `attempted_observations`、`source_observations`（成功映射）及 `skipped_observations`。仅mask不可用电荷监督，不影响语料分子或其他损失。资源缺失/损坏及解析错误仍停止准备。跳过策略改变电荷身份；应准备到新目录，不能复用原映射失败即报错配方的产物。独立回归在输出根写入对应训练集/验证集映射审计。

### 独立冻结回归头训练

回归输入不能超过冻结来源检查点的 `max_smiles_tokens`。超长分子在表示生成与标准化拟合前跳过，不截断、不修改编码器；其他错误仍失败。输出根的 `input_filter_audit.json` 记录各目标/split过滤前、保留、排除样本数及逐条原因、token数和电荷mol_id；任务清单与summary也记录数量。相同电荷结构的源观察分别计数。过滤后训练集为空会报错；训练/验证结构重叠仍在过滤前拒绝。结果仅代表可编码的样本子集。失败尝试保留，修复后换用新输出目录重跑，无需重新训练Stage1/2/3。

部分电荷头训练将来源每行保留为独立样本，重复结构复用冻结原子表示；预训练及兼容性见 [sidecar合同](#partial-charge-sidecar-on-an-existing-corpus)。标量电子冲突及训练集/验证集重叠检查不变。现役回归 YAML 的 `simulation_dir` 为 `data/stage1/properties`，与 Stage2/3 的数据 split 目录独立。运行前核验来源路径和合同；这不授权移动或覆盖数据。

在Stage1末轮后显式执行；不会自动追加预训练。命令需完整末轮 `last.pt` 或末轮轮检查点，不能用 `stage1_encoder.pt`。在 `configs/v4/stage1/regression_heads.yaml` 配置数据/产物路径和预算。一次编码完整无mask结构，冻结编码器，独立训练13标量头加原子电荷。当前YAML显式采用MLP：各电子目标entity1024→512→256→1，partial charge为atom512→256→128→1，GELU、dropout0。各头按既有任务局部种子独立初始化（分别656,385 / 164,353参数），不共享权重。共享结构定义默认仍为Linear；省略预测器时从已训练行/头初始化。默认10轮、LR1e-4、恒定LR、batch128、AdamW/WD0.01、BF16、裁剪1。验证只报告原单位MAE/RMSE；final固定epoch10，绝不取最优。不读取测试集。此MLP配方使用新输出；已有Linear输出只读，无需重跑Stage1/2/3。

```bash
python scripts/stage1/regression.py --config configs/v4/stage1/regression_heads.yaml --checkpoint outputs/v4/stage1/base/train/last.pt --output outputs/v4/stage1/base/regression_mlp --device cuda:0
```

可选目标子集，也可使用没有原子头的旧完整v4检查点：

```bash
python scripts/stage1/regression.py --config configs/v4/stage1/regression_heads.yaml --checkpoint outputs/v4/stage1/base/train/last.pt --tasks HOMO_eV LUMO_eV --output outputs/v4/stage1/base/regression_mlp_homo_lumo --device cuda:0
```

默认任务为HOMO_eV/LUMO_eV、ESP_max/min/std/pos_frac、Dipole、Quadrupole、q_max/min/std/pos_frac、gap_eV及partial_atomic_charge。q_*摘要不是原子电荷预测。训练集/验证集 canonical重叠或标量电子冲突报错；partial-charge观察分别保留。新任务归一化仅拟合完整训练集；Linear初始权重/偏置经转换保留原单位预测。回归输出含来源绑定 `representations.pt`、逐任务 `metrics.jsonl`、`regression_head.pt/json` 及汇总。不是Stage1替代编码器或Stage3报告产物；仅头实验无需重跑Stage2/3。来源检查点、编码器及其他头只读。不覆盖已有输出；失败头运行在新目录重启。仅标量后训练不需要teacher执行或缓存；原子头初始化另须验证原sidecar/scaler身份。

预测器仅通过YAML配置；上方CLI参数不变。复制自包含回归YAML到新文件，仅修改所需配方。例如：

```yaml
predictor:
  type: linear
predictor_overrides:
  HOMO_eV:
    type: mlp
    hidden_dims: [512, 256]
    activation: gelu
    dropout: 0.1
  partial_atomic_charge:
    type: residual_mlp
    hidden_dims: [256, 256]
    activation: gelu
    dropout: 0.1
```

覆盖项替换完整共享预测器配方；未列出目标使用共享默认。全部目标用同一MLP时，设置 `predictor.type: mlp` 及隐藏维度/激活/dropout。类型为 `linear/mlp/residual_mlp`；激活为 `gelu/relu/silu`，默认GELU、dropout0。电子输入learned1024、原子输入atom512、输出1固定。MLP隐藏层为Linear→激活→Dropout。残差块采用输入→宽度→宽度、两次dropout及身份/投影捷径；无归一化或相加后激活。隐藏宽度须为正，非线性列表不能为空。不允许任意模块导入或覆盖输入/输出宽度。

Linear保留预训练初始化、数值行为及身份。非线性头按稳定任务局部种子随机初始化，不是预训练Linear的残差校准；初始原单位预测无需匹配来源。结构进入所选目标回归身份并要求新输出，但不改变Stage1/2/3身份。所有预测器保持相同训练/数据/冻结编码器规则，含来源无原子头时拒绝partial-charge请求。format2保存解析结构、输入宽度、参数量、初始化、种子及hash；`stage1.regression.load_regression_head(task_root, checkpoint_path)` 无需YAML即可从产物重建，也接受旧format1 Linear头。见 [ADR-0093](adr/0093-stage1-configurable-frozen-predictors.md)。

### 将已有 teacher 检查点复制到其他服务器

无需再次下载，可直接复制检查点。从来源仓库根执行，将 `SERVER` 替换为SSH别名（`h100` 或 `szx`），将 `/path/to/ILUME` 替换为该服务器仓库根。先检查目标已有文件；hash一致则跳过复制，不一致则保留检查并使用新目标，不覆盖。

```bash
sha256sum assets/unimol2/modelzoo/84M/checkpoint.pt
ssh SERVER 'mkdir -p /path/to/ILUME/assets/unimol2/modelzoo/84M'
rsync -avP --partial --append-verify \
  assets/unimol2/modelzoo/84M/checkpoint.pt \
  SERVER:/path/to/ILUME/assets/unimol2/modelzoo/84M/checkpoint.pt
ssh SERVER 'sha256sum /path/to/ILUME/assets/unimol2/modelzoo/84M/checkpoint.pt'
```

两个hash必须等于上方发布SHA-256。第二台服务器重复该过程。各服务器分别创建独立teacher环境，先进行环境检查及小审计，再生成完整缓存。

## Stage2 / Stage3：实体 HoME v4

现役专家与条件合同见 [ADR-0120](adr/0120-entity-token-attention-late-fusion.md)，数据及基础owner边界见 [ADR-0095](adr/0095-v4-entity-home-without-object-encoder.md)。Stage1保持本手册前述配方，GLOBAL/GROUP直接处理实体，没有ObjectEncoder。

当前Stage3 Base采用 [ADR-0096](adr/0096-stage3-batch-sample-weighted-gradients.md)：Phase1/2共享梯度按本step各实验任务真实batch长度计算样本加权均值，GLOBAL直接跨任务聚合、GROUP在组内聚合，PRIVATE不额外缩放。尾batch按实际长度计权，任务耗尽后不进分母；模拟仍仅更新自身PRIVATE。Raw Sampling、loss、owner预算及评估口径不变。新聚合标识及权重规则进入训练身份，旧等权Entity-HoME检查点不可跨合同恢复或加载，后续训练须使用新目录；身份一致的准备产物及Stage1可复用。

当前Stage2 Base也采用 [ADR-0097](adr/0097-stage2-batch-sample-weighted-gradients.md)：五项模拟全部参与共享预训练，每个联合step取各未耗尽任务一个真实batch，GLOBAL直接跨任务、GROUP组内按记录数加权，PRIVATE原始平均梯度；不使用旧任务权重补偿。原shuffle、完整行覆盖、256逻辑/微批、10轮、LR1e-4及整模裁剪不变。每轮optimizer/scheduler更新次数改为任务batch数最大值。旧Stage2 checkpoint不能恢复、旧final不能迁移到当前Stage3；需要用新目录从头训练Stage2，再以新final训练Stage3。数据/Stage1身份一致的Stage2/3 prepare可复用，历史产物只读。

训练诊断见 [ADR-0098](adr/0098-stage3-epoch-training-diagnostics.md)。各fold的 `phase_1/diagnostics.jsonl`、`phase_2/<scope>/diagnostics.jsonl` 和 `phase_3/<task>/diagnostics.jsonl` 每轮写一行：`diagnostics_version: 2` 的 `epoch_gradient_stats` 记录各owner实际梯度step数、裁剪前后范数mean/max及裁剪率；`shared_sample_contributions` 记录GLOBAL/GROUP的实际记录数、记录占比和逐step归一化权重累计/均值。任务耗尽及冻结不混入梯度均值分母，模拟只计PRIVATE。记录占比与权重均值含义不同，不能混用；无观测/分母时为null。新版替换原顶层最后一步范数字段，历史行不重写，恢复后可在同一日志追加新版行。诊断完全只读，不进入科研身份，本次诊断变更无需重训Stage2/3，不解除此前聚合算法变更的身份限制。

## 运行前提

使用已安装的 `ilume` 环境，不安装依赖或下载权重。运行前核验 `outputs/v4/stage1/base/train/stage1_encoder.pt` 与配对的 `outputs/v4/stage1/base/prepare/artifacts`；文件存在不代表身份兼容，仍以入口完整性检查为准。源数据不能在数据准备/训练/评估之间替换。数据审计元数据由正式 prepare 写入配置的数据目录，不手改 SHA。

截至2026-10-11，本地上述 Stage1 编码器及特征元数据已存在；默认 Stage2 `base/prepare`、`base/train` 已有旧运行，而配置指定的 `base/train/stage2_final.pt` 不存在。新运行需使用未占用的目录，并同步 prepare 产物和 Stage3 来源引用；本次仅核对文件状态，没有验证这些正式产物的完整加载兼容性或执行正式作业。其他数据 split 的正式运行仍由用户显式安排。

现役主模型入口为 `configs/v4/stage2/base.yaml` 和 `configs/v4/stage3/base.yaml`；`optimized_3407.yaml`仅保留历史G2_C05配方。Stage3 Phase2中的两项模拟任务只更新自身PRIVATE，不参与GLOBAL/GROUP梯度或共享梯度归一化；Phase3继续只更新PRIVATE。查看 `performance.jsonl` 的参数量、轮耗时及 `metrics.jsonl` 的实际 optimizer_updates。

当前Stage3 Base为G2_C01：GLOBAL一个专家，transport/thermophysical/phase_stability/dielectric_optical/solvation/biological的GROUP专家数为1/1/2/1/2/1。两电位任务归入phase_stability；PRIVATE FFN/Tower宽度、dropout与007/032配方的LR/预算保持不变。现役专家与条件融合按ADR-0120使用token注意力，旧FiLM不再创建。Stage2的GLOBAL及两个可迁移GROUP同步缩减，必须重新训练Stage2后再训练Stage3；旧G2_C05权重不能迁移或恢复到新容量。现役Stage1需先按211维新policy重建来源。

## 最终优化配方（3407）

本节保留历史G2_C05的显式复现命令；现役G2_C01入口见[下一节](#g2_c01-主模型)，不能把本节结果标为G2_C01。旧配方需要其原始Stage1与特征来源，若标准Stage1路径已升级须在独立YAML中配对历史来源，不得混用。

在 `outputs/v4` 重跑历史HoME `G2_C05` / Stage2 `007` / Stage3 `032`，使用独立配置 `configs/v4/stage2/optimized_3407.yaml` 与 `configs/v4/stage3/optimized_3407.yaml`。数据划分3407，训练seed42（Stage3实际42+fold）；完整模型结构、任务映射、LR、nominal Epoch及任务覆盖与完成五套复验时的配方一致。原Base YAML保留，不覆盖HPO结果。

Stage1不重训，保留 `outputs/v4/stage1/base` 中的编码器、Prepare及其他来源；本节也不重新训练14个独立后训练头。Stage2/3使用新的 `optimized_3407` 输出和prepare审计，来源严格配对，不读取旧split42缓存或旧Stage2权重。下列命令从仓库根目录依次执行；正式GPU运行由研究者启动。

```bash
conda activate ilume

python scripts/stage2/prepare.py --config configs/v4/stage2/optimized_3407.yaml --output outputs/v4/stage2/optimized_3407/prepare &&
python scripts/stage2/train.py --config configs/v4/stage2/optimized_3407.yaml --output outputs/v4/stage2/optimized_3407/train --device cuda:0 &&
python scripts/stage3/prepare.py --config configs/v4/stage3/optimized_3407.yaml --output outputs/v4/stage3/optimized_3407/prepare &&
python scripts/stage3/train.py --config configs/v4/stage3/optimized_3407.yaml --fold 1 2 3 4 5 --output outputs/v4/stage3/optimized_3407/train --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
```

Stage2在单卡训练；Stage3四卡并行四折，空闲后补第五折。新输出应不存在；中断时保留现场，Stage2训练追加 `--resume outputs/v4/stage2/optimized_3407/train`，Stage3训练追加布尔 `--resume`，不要删除中断产物或把整条首次运行命令当恢复入口。

训练完成后执行实验和两模拟的联合验证；模拟使用五个final的原单位预测均值，与HPO报告里的模拟逐折指标均值口径不同。

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/optimized_3407.yaml --checkpoint-dir outputs/v4/stage3/optimized_3407/train --split valid --fold 1 2 3 4 5 --output outputs/v4/stage3/optimized_3407/valid
```

需要发布锁参后的测试结果时，再由研究者执行一次完整测试批次，不据测试重新调参：

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/optimized_3407.yaml --checkpoint-dir outputs/v4/stage3/optimized_3407/train --split test --ensemble-folds --output outputs/v4/stage3/optimized_3407/test
python scripts/benchmarks/summarize.py --input outputs/v4/stage3/optimized_3407 --output outputs/v4/summary_optimized_3407
```

旧 `outputs/v4/stage2/base`、`outputs/v4/stage3/base` 可在报告准确文件数、大小并获确认后删除，但不能删除Stage1来源。旧split42消融配置仍引用这些Base产物，删除后须重新生成对应历史来源才能继续旧消融；3407新产物不能替代其身份。新运行使用独立路径，并不要求先删旧结果。其他设备上的代码与配置由研究者git pull；训练完成后按[回传规则](#跨设备训练产物回传)归档4090。

## G2_C01 主模型

当前Base按[ADR-0120](adr/0120-entity-token-attention-late-fusion.md)使用实体token注意力，1024D/8头/2层；L1不读条件，L2三个owner层级联合实体与条件，最后按三槽拼接。Stage1和匹配prepared保持原合同，需重新训练Stage2及Stage3；旧MLP权重不能续训或迁入。参数量与新kind见ADR，新输出使用独立`token_attention/`根，以下命令不覆盖旧`base/`结果。各步骤成功后再执行下一步。

使用 `configs/v4/stage{2,3}/base.yaml`，data/3407、训练seed42及007/032超参；依赖当前211维新policy的Stage1编码器与prepare。旧G2_C05来源及历史split42权重不能复用。新Stage2/3使用未占用的token_attention输出目录，保留旧Base；Stage1仍引用原配对来源，不需要为本次模型结构升级重新生成Stage1。

2026-11-09只读核验：配置指定的`outputs/v4/stage1/base/train/stage1_encoder.pt`尚不存在；正式运行前须完成或回传当前配对Stage1编码器，并由入口核验完整来源。Stage1 prepare的metadata存在不能替代编码器。新`token_attention/`输出根尚未创建，本次没有启动正式作业。

2026-11-06已按用户确认对齐当前数据协议（[ADR-0117](adr/0117-base-22-tasks-updated-solvation.md)）：Base排除平衡压力与水活度系数，保留22项实验和两项模拟辅助；solvation按真实更新的catalog声明3684体系，其余任务配方、GROUP和训练预算不变。严格任务/体系数校验仍保留。历史24任务的optimized_3407、HPO、消融及基线配置/结果不随此修改，不能与新22任务macro直接比较。Stage1新版DGT及Gasteiger v2来源须先按前文流程完成。

与Stage1一样，从仓库根目录在 `ilume` 环境执行。下面各步骤成功后再执行下一步，使用未占用的输出目录。入口会自动校验来源、数据SHA和模型身份，无需另写Python校验脚本。修改prepare输出路径时，须同步修改对应YAML的 `data.artifacts_dir`；修改训练输出路径时，须同步修改下游来源配置。

### Stage2：prepare 与 train

prepare生成五任务数据和冻结Stage1实体缓存；train固定训练10轮，发布 `stage2_final.pt` 和 `stage2_final.json`，不再导出独立编码器。

```bash
python scripts/stage2/prepare.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/token_attention/prepare
python scripts/stage2/train.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/token_attention/train --device cuda:2
```

### Stage3：prepare 与五折 train

Stage2训练完成后准备Stage3，再执行三阶段五折训练。下面使用GPU2/3、总并发两折；只用一张卡时改为单设备并将 `--max-parallel` 设为1。

```bash
python scripts/stage3/prepare.py --config configs/v4/stage3/base.yaml --output outputs/v4/stage3/token_attention/prepare
python scripts/stage3/train.py --config configs/v4/stage3/base.yaml --fold 1 2 3 4 5 --output outputs/v4/stage3/token_attention/train --devices cuda:2,cuda:3 --max-parallel 2
```

### Stage3：实验与模拟联合评估

默认 `--domain all`，每次调用完成同一split的实验与两项模拟评估。实验验证仍逐折评估，模拟验证仍使用全部五个final的原单位预测均值；测试集两类均按各自协议使用五模型集成。先查看验证结果，再报告测试集；hydration无测试集。

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/token_attention/train --split valid --fold 1 2 3 4 5 --output outputs/v4/stage3/token_attention/valid
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/token_attention/train --split test --ensemble-folds --output outputs/v4/stage3/token_attention/test
```

联合评估使用未占用的输出根；实验验证写入 `foldN/`，实验测试写入根目录，模拟结果写入同根的 `simulation/`。每个运行仍独立记录身份、metadata、summary和预测清单，汇总器递归读取，实验与模拟榜单继续独立。入口先核对模拟所需的全部五折final来源；实验评估失败时返回非零并停止后续模拟评估，模拟失败时保留已完成实验结果且返回非零，不覆盖已有产物。

只评估实验时显式使用 `--domain experimental`，其余参数沿用上方命令；只评估模拟时使用 `--domain simulation --ensemble-folds`，不传 `--fold`、`--tasks` 或 `--checkpoint-epoch`。联合评估中 `--fold`（验证）和 `--tasks` 仅选择实验任务/折，模拟始终评估全部两项任务及五模型集成；联合评估不接受 `--checkpoint-epoch`。

### 结果汇总

只扫描新的实体HoME输出，发布到独立目录，不与旧20/24实验或四模拟结果混排。

```bash
python scripts/benchmarks/summarize.py --input outputs/v4/stage3/token_attention --output outputs/v4/summary_token_attention_3407
```

与基线生成论文联表前，须先核验基线具有完整的22任务、相同split及匹配数据身份；历史24任务的`outputs/benchmarks/v4_3407`不能直接替代。基线补训和复用审计见[22任务补训手册](baseline-flft-22-retrain-runbook.md)，本次不运行基线或发布不完整联表。

新Base中断时，Stage2训练使用 `--resume outputs/v4/stage2/token_attention/train`，Stage3训练使用布尔 `--resume`；prepare和评估无此恢复步骤，已有输出不覆盖。

只在确实需要继续中断作业时按现有入口显式使用 `--resume`；必须保留同一配置、源 SHA、折/owner/训练状态和指标尾部。不要通过修改元数据、kind、格式、任务名称或宽松加载复用不匹配的历史产物。

<a id="group-l2-partner"></a>

## 独立试验：第三成分仅在GROUP L2接入

[ADR-0125](adr/0125-group-l2-partner-injection.md)对应`group_l2_partner.yaml`：GLOBAL/GROUP L1只读主体，GROUP L2加入槽位2；GLOBAL/PRIVATE L2保持主体路径。第三成分在L2复用对应GROUP投影，不增加可训练参数，其他配方不变。此试验独立于当前Base及现役消融，保留原`token_attention/`结果。

当前Base的Stage1和两阶段prepare已完成、身份匹配时，可直接只读复用，无需重复prepare。两份试验YAML已保留这些prepare路径，Stage3的Stage2来源改为新训练的final。4090仓库根依次执行：

```bash
python scripts/stage2/train.py --config configs/v4/stage2/group_l2_partner.yaml --output outputs/v4/stage2/group_l2_partner/train --device cuda:0 &&
python scripts/stage3/train.py --config configs/v4/stage3/group_l2_partner.yaml --fold 1 2 3 4 5 --output outputs/v4/stage3/group_l2_partner/train --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
```

完成后执行同口径独立验证，不据测试集选择配方：

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/group_l2_partner.yaml --checkpoint-dir outputs/v4/stage3/group_l2_partner/train --split valid --fold 1 2 3 4 5 --output outputs/v4/stage3/group_l2_partner/valid
```

中断后，Stage2训练追加`--resume outputs/v4/stage2/group_l2_partner/train`，Stage3训练追加布尔`--resume`。两种接入模式不能交叉续训或迁移，不能沿用已有token_attention模型作为本次新模式权重。

后续I2／I3／I4／S1的独立注入及分类型编码消融见[专用手册](partner-injection-ablation-runbook.md)和[ADR-0127](adr/0127-partner-injection-and-typed-encoding-ablations.md)；8份配对YAML位于`configs/ablations/partner_injection_3407/`，不替换上述Base或GROUP L2参照。

## 数据核验结论与限制

现役Base的任务集合由ADR-0117定义，以下24任务/120个分折文件描述是2026-10-08历史快照，不作为当前22任务的运行前提。

以下为2026-10-08本地核验快照；正式运行仍以加载和prepare的实时完整性检查为准。实现已通过临时CPU链路验证，正式训练和五折结果尚未生成。

当前任务目录的 gas_solubility 包含多种气体，不能视为旧 x_co2 的同义名称；本版本已按明确确认改变目标。新增水活度和焓任务有实际任务目录/分折来源，焓任务展示为汽化焓（Enthalpy of vaporization），仅使用temperature_K；常量phase列不进入模型。任务目录中的任务标识、目录、目标列及源SHA保持原样。hydration 继续 random/cv1 且无测试集，其余使用任务目录的 system split。Stage1电荷监督及其资源合同保持不变。参考压缩包不替代实时任务目录和文件身份。

Stage2五任务训练集/验证集来源齐全；两项辅助任务的训练集/验证集/测试集齐全。24实验的120个分折文件存在，当时10项有测试集文件，其余任务保留五折验证，不生成额外测试集划分；任务目录的实验测试集计数字段为空，不据此推测缺失文件的科学意图。当时正式运行的未满足前提是配置指定路径的 Stage1 编码器及配对特征产物；旧元数据需上述新 prepare 正常重建。2026-10-08的实现验收没有启动正式 GPU训练、正式五折或覆盖历史输出；后续HPO进度见[当前报告入口](hpo-simple-runbook.md#当前选型与报告入口)，运行前仍须核验当前配置指定的来源产物。

历史基线的20任务训练配方不扩展，模拟四任务只适配任务目录现有来源位置。其旧实验合同依据仍要求x_co2；当前任务目录缺少这一历史目标，运行旧实验基线需要对应历史任务目录/数据，不得用gas_solubility替代。两模拟实体 HoME v4与四模拟历史结果不能混入同一汇总。

## 独立静态 HPO

正式运行保持本手册原配方。独立调参使用 [HPO手册](hpo-simple-runbook.md) 和 [ADR-0099](adr/0099-simple-static-hpo.md)，配置与输出分别位于 `configs/HPO`、`outputs/HPO`，人工选型后分轮生成完整YAML。Stage2支持正整数Epoch及可选training.seed；默认10轮和data.seed沿用行为保留。HPO显式固定训练seed42，Prepare审计写入独立输出，不改写数据目录metadata或正式产物。

## 当前数据版本的基线

ADR-0117只调整正式Base；下面24任务基线配置保留原协议，尚未迁移到当前22任务catalog。配对重跑需另行更新注册表authority、科研配置及输出身份，不在本轮范围内。

基线现役catalog_v4_3407协议按[基线手册](baseline-catalog-v4-runbook.md)及[ADR-0102](adr/0102-baseline-split3407-full-retraining.md)运行：data/3407、24实验、三模拟，十模型全部重训共1,230作业，输出outputs/benchmarks/v4_3407。0101的split42复用安排及历史20实验/四模拟不再作为当前入口；旧baseline按用户确认范围归档/删除。Stage1/2/3数值合同不变。
