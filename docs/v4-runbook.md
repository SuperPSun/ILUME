# v4 运行手册

本手册是现役 v4 Stage1及实体HoME Stage2/3运行入口。Stage1合同不变；Stage2/3见 [ADR-0095](adr/0095-v4-entity-home-without-object-encoder.md)，旧Object/v5运行需历史Git版本。


从仓库根目录执行，使用新的 `outputs/v4/` 路径，不覆盖历史v3输出。Stage1科研合同见 [ADR-0089](adr/0089-v4-frozen-dual-view-stage1.md)。CUDA示例使用四张设备。

## 数据 split 与路径

2026-10-20起，后续新实验统一采用 `data/3407`。固定配方为HoME `G2_C05`、Stage2 `007`、Stage3 `032`，共同训练seed42，Stage3实际按42+fold派生；数据划分seed与训练seed不得混淆。已有正式Base、基线、消融配置及产物保留原split，不改写历史身份。最终五套划分结果与后续使用边界见 [HPO手册](hpo-simple-runbook.md#当前选型与报告入口)。

已有正式Base YAML的Stage1 输入仍为 `data/stage1`；Stage2/3 输入分别为 `data/42/stage2`、`data/42/stage3`，共用 `data/42/task_catalog.csv`。以下原Base命令仍复现split42；新实验须使用独立的3407 YAML与输出。Stage2 YAML 的 `data.data_root` 为 split 根目录，catalog 内的 `stage2/...` 路径相对此根解析；Stage3 YAML 的 `data.stage3_dir` 指向同一split的 `stage3`。prepare 的数据审计 `metadata.json` 写入各自的数据目录。

测试其他数据 split 时，使用独立的自包含 YAML，将 Stage2 的 `data_root/task_catalog_path` 与 Stage3 的 `stage3_dir/task_catalog` 同步指向对应 split，并为 prepare、训练和评估指定独立输出及对应的下游引用。目录名 `42` 只定位数据，不自动改变训练随机种子、split policy 或 cv repeat。不同内容的 split 必须重新 prepare，不能共用数据身份不一致的准备产物或检查点；Stage1 来源保持不变。历史配置及基线保留各自的数据合同，不由本次目录调整迁移。

五套数据目录为 `42`、`123`、`1234`、`3407`、`90756`，仓库正式 Base YAML 默认仅选择42。各套的 Stage3 `stage2_final/simulation_artifacts_dir` 必须配对到对应 split 的 Stage2 来源。并行 prepare 使用各自的 `preparation.cache_dir`：冻结 Stage1 表示可按身份串行复用，但当前缓存写入不支持多个进程共同写入。prepare、train、evaluate 的运行元数据 `data_metadata` 引用所选 split 的数据审计文件。

各 split 分别汇总，用 `--input` 或 `--include` 限定到该 split 的结果，再写入独立汇总目录；现有 summarizer 不合并不同来源身份的 split。根目录 `data/task_catalog.csv` 保留旧42副本，其他 split 使用各自目录内的 catalog。`1234`、`90756` 附带的 `stage1/properties` 不属于现役共享 Stage1 输入；Stage1 仍显式读取 `data/stage1`。

当前Base采用十二层SMILES Transformer和八个独立残差图块：词表2,048个token时仅编码器约60.73M参数（[ADR-0090](adr/0090-stage1-v4-residual-encoder-capacity.md)）。身份一致的 Stage1 编码器、预处理语料、统计和完整 Uni-Mol 缓存可复用；切换 Stage2/3 数据 split 不要求重新训练 Stage1。旧架构检查点不得恢复到当前架构。若训练/下游目录已占用，选择新路径并同步更新下游引用，或在运行前明确安排归档；这些命令不授权替换历史结果。

## RDKit 2026.9.1 主线升级

现役主项目依赖精确固定 `rdkit==2026.9.1`，实际导入运行时为 `2026.09.1`（[ADR-0111](adr/0111-rdkit-2026-09-1.md)）。该版本默认描述符仍为同名、同顺序217维，不改变learned1024、atom512、1241维实体、模型容量或训练预算；筛选清单仍按ADR-0110独立审计，不自动接入模型。新版构象生成的可选ETKDGv4不启用，继续ETKDGv3/MMFF或UFF。

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

主线继续使用 `configs/v4/stage1/base.yaml`、`configs/v4/stage2/base.yaml`、`configs/v4/stage3/base.yaml`，不另设带RDKit版本后缀的Base配置。三者仍按 `outputs/v4/stage{1,2,3}/base/` 配对prepare、teacher、电荷sidecar与训练来源；科研参数及原数据split不因路径整理改变。复用标准输出路径前，须按工作区规则完成用户明确确认的旧产物清理或归档；不能覆盖旧产物，也不能只删权重后在遗留metadata/log的运行目录中启动新训练。保留的旧电荷sidecar不能用于新版prepare，须先另行归档后重新生成。未经授权，不自动移动或删除历史结果。描述符筛选YAML继续指向Base的新prepare，旧版本来源仍被严格拒绝。

## RDKit 描述符只读筛选审计

该工具只生成候选清单，不改变217维Stage1监督、下游实体宽度、Base YAML或已有模型。规则见 [ADR-0110](adr/0110-stage1-descriptor-selection-audit.md)，阈值在独立 [筛选YAML](../configs/v4/stage1/descriptor_selection.yaml)。完成旧Base运行目录的清理或归档后，在 `ilume` 环境使用 [Stage1 Base](../configs/v4/stage1/base.yaml) 重新prepare；该命令也会生成新的电荷sidecar。筛选无需先训练Stage1或生成teacher。

```bash
conda activate ilume
python scripts/stage1/prepare.py \
  --config configs/v4/stage1/base.yaml \
  --output outputs/v4/stage1/base/prepare

python scripts/stage1/select_descriptors.py \
  --config configs/v4/stage1/descriptor_selection.yaml \
  --output outputs/v4/stage1/base/descriptor_selection
```

默认仅对prepared manifest的全部train重新计算原始float64 RDKit，使用16个CPU worker、不用GPU；不读valid/test或目标标签，也不从已标准化float32恢复原始值。约518万分子时输出矩阵约9GB，运行前确保新输出目录有空间和CPU资源；不覆盖、无resume。终端有进度条，重定向输出为JSON进度。若仅预审，在复制的独立YAML中设正整数`max_molecules`，结果会明确标为抽样，不能据此断言全量零方差或罕见信息不存在。

检查 `summary.json`的保留维数/原因，以及`artifacts/descriptor_statistics.csv`的逐列分位数、缺失和重尾比值；完整规则与名单在`artifacts/descriptor_selection.json`。默认排除超过1%非finite、数值平方不安全、明显重尾、常数/完全重复及`abs(Pearson)>0.98`的冗余列。IQR为零的稀疏列不会仅因此删除，不做自动变换。原始`.npy`保留复核。接入Stage1监督或下游输入需要另行实现并重训相应模型，不能把本清单替换旧schema或改SHA。

## 跨设备训练产物回传

4090作为统一训练产物归档设备。本规则适用于正式Stage1/2/3、独立回归头后训练、HPO、基线及其他训练实验；无论在哪台设备训练，都必须将相关训练结果与完整模型权重回传到4090的ILUME项目目录，保持原有相对目录结构，HPO仍归入 `outputs/HPO`。在4090本机训练时直接保留在对应目录。

回传包含完整运行目录：模型权重（包括 `.pt` 等文件）、final/阶段锚点/owner delta、周期检查点及恢复状态、`run_config.yaml`、metadata/summary/身份清单、metrics/diagnostics、日志，以及已生成的预测和评估报告。一并保留对应运行YAML及依赖的Prepare、特征/归一化统计和来源产物；4090已有相同且核验通过的来源可复用。失败或中断任务也回传已有现场，不只传成功摘要，不排除模型权重，不改变数据split或训练身份。

长任务可在完整轮或任务完成后分批回传；最终验收须确认4090归档齐全。不要把正在写入的日志/检查点当成完整快照：先停止相关写入或选择已固定的完成产物，再核验。目标目录先做差异预览；同路径不同身份的结果使用独立归档目录，禁止覆盖或混入既有运行，也不得将旧文件反向同步回正在训练的目录。

回传完成后按文件清单、大小和SHA核对源/目标，可用 `rsync -acni` 做只读差异检查，并通过项目现有身份/产物加载校验。确认权重、日志与检查点配套一致后才视为回传完成；核验前保留训练设备上的原始产物。实际传输地址以4090的SSH配置和项目路径为准；本规则不自动授权删除源产物或覆盖目标，也不要求程序自动同步。

## Stage1 数据准备与离线 teacher

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
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 32 --limit 100 --output outputs/v4/stage1/base/teacher_audit
```

确认成功率、吞吐和存储后生成完整缓存。重复相同命令恢复不可变分片，不恢复任意部分行。OOM视为运行失败，不视为可mask分子。batch大小/设备和 `--workers` 是执行设置；canonical结构顺序及每分子构象种子固定。

```bash
python scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 32
```

GPU利用率低时可通过 `--workers` 在并行CPU进程准备构象及Uni-Mol特征。有界有序队列预取输入，父进程负责GPU推理；worker不加载teacher模型或使用CUDA。特征只算一次，不在推理前重复计算。默认 `--workers 1` 保留串行执行。worker增加RAM占用，不得超过作业分配的CPU资源。

Slurm作业可从 `#SBATCH --cpus-per-task=8`、一张GPU开始，在已分配作业内执行下方命令，保持已有独立teacher环境。先检查这次**新的**并行审计，再运行完整生成命令：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 64 --workers 8 --limit 100 --output outputs/v4/stage1/base/teacher_audit_parallel
python -u scripts/stage1/teacher.py --config configs/v4/stage1/base.yaml --device cuda:0 --batch-size 64 --workers 8
```

重启同一缓存根前先停止旧作业；不支持并发写入。worker数变化仍可复用已完成分片。重定向/Slurm输出在启动、每个分片提交及推理batch之间约每30秒刷新 `teacher_progress` JSON。`processed/total` 包含内存中的工作；`committed` 统计安全发布的分子。单个慢构象/batch可能延迟下一条日志。可用 `tail -f slurm-<job-id>.out` 监控。

完整缓存完成后回到 `ilume` 训练环境。Uni-Mol仅用于缓存生成；Stage1训练读取缓存。Stage1拒绝部分或未绑定teacher缓存；损失按角色2/2/1加权，加载器仍为原始自然打乱。

```bash
conda deactivate
conda activate ilume
python scripts/stage1/train.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/train
```

现役Base全局batch512，每rank使用8个DataLoader worker。四rank DDP因此每GPU为batch128，训练加载器共32个worker。LR保持1e-4；此前batch128配方属于历史，不能恢复到改变后的训练身份。语料及teacher缓存仍可复用。worker数量仅属执行参数，本身不改变科研训练身份。

### Stage1 进度与输入停顿

普通 `python scripts/stage1/train.py ...` 只启动一个训练进程，不自动使用全部GPU。DDP需用 `torchrun`，例如在实际分配四张GPU的作业内执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/stage1/train.py --config configs/v4/stage1/base.yaml --output outputs/v4/stage1/base/train
```

使用未占用输出目录。训练、快速验证和完整验证加载器各用配置的worker数；CUDA 加载器首次使用后保留worker，每worker预取两个batch，因此开始验证时worker资源可能增加。应按作业CPU配额及存储吞吐选择worker，而非仅按GPU数量；增加worker不保证吞吐提升。

预期停顿发生在每1,000次更新的rank0梯度审计（其他rank等待）、每5,000次更新的审计加快速验证，以及轮边界的完整验证/检查点导出。审计日志在审计完成后写入，不在开始时写入。这些周期操作不能解释发生在对应边界前的停顿。

若GPU利用率下降且加载器 worker在 `wait_on_page_bit_common` 进入 `D` 状态，应检查存储/页面读取。语料加载校验并反序列化分片，将样本关联到只读SQLite Uni-Mol索引及mmap teacher 分片。多个worker在NFS读取这些文件，即使RAM充足也可能使GPU等待。优先使用已验证字节一致的本地SSD语料/teacher副本，并在独立受控尝试比较worker数量；不要重生成标签、关闭hash校验、改变采样或覆盖运行中作业。这些是运行建议，不表示已实现本地缓存，也不保证8个worker能解决NFS停顿。归因前检查 `nvidia-smi`、`vmstat 1`、worker等待状态及 `metrics.jsonl` 时间戳。

输出包含保留辅助头的恢复检查点及仅编码器 `stage1_encoder.pt`。部署/Stage2不加载teacher或辅助头。

<a id="stage1-loss-and-gradient-audit"></a>

### Stage1 损失与梯度审计

当前Base采用SMILES/原子/键重建0.20/1/1、alignment0.10、RDKit0.25、Uni-Mol0.50、electronic0.25及独立partial-charge0.25。这是用户选定配方，替代 [ADR-0091](adr/0091-stage1-v4-loss-weights-gradient-audit.md) 和 [ADR-0092](adr/0092-stage1-atom-charge-and-frozen-regression-heads.md) 的早期系数；损失定义与审计隔离不变。这些损失变化要求新的Stage1运行，不从旧系数恢复；已有语料/统计/teacher缓存仍可复用。下游Stage2/3须使用新训练编码器和新输出。

训练每完成1,000次优化器更新自动追加 `gradient_audit.jsonl`。rank0使用固定32分子验证探针及评估mask，关闭dropout；其他rank等待。报告原有七项原始编码器范数，启用时增加 `partial_charge_grad_norm`，以及 `weighted_grad_norms`（损失系数绝对值乘范数）、有效目标覆盖、探针ID/hash、步及尝试。既有指标和检查点选择不变。

设置 `training.gradient_audit_interval_steps: 0` 或传入 `--gradient-audit-interval-steps 0` 可关闭；正CLI值覆盖YAML，省略则按YAML。`gradient_audit_batch_size` 控制探针大小（默认32）。诊断设置不改变科研身份。缺失标签目标记录 `null` 和 `no_valid_targets`；小自然探针没有电子/电荷标签不说明梯度弱。范数来自各自目标，不是向量和：不要仅凭大小自动调整系数。审计增加一次前向及最多八次梯度计算；OOM明确失败，不自动缩小探针。监控命令：

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

合同见 [ADR-0095](adr/0095-v4-entity-home-without-object-encoder.md)。Stage1保持本手册前述配方，GLOBAL/GROUP直接处理实体，没有ObjectEncoder。

当前Stage3 Base采用 [ADR-0096](adr/0096-stage3-batch-sample-weighted-gradients.md)：Phase1/2共享梯度按本step各实验任务真实batch长度计算样本加权均值，GLOBAL直接跨任务聚合、GROUP在组内聚合，PRIVATE不额外缩放。尾batch按实际长度计权，任务耗尽后不进分母；模拟仍仅更新自身PRIVATE。Raw Sampling、loss、owner预算及评估口径不变。新聚合标识及权重规则进入训练身份，旧等权Entity-HoME检查点不可跨合同恢复或加载，后续训练须使用新目录；身份一致的准备产物及Stage1可复用。

当前Stage2 Base也采用 [ADR-0097](adr/0097-stage2-batch-sample-weighted-gradients.md)：五项模拟全部参与共享预训练，每个联合step取各未耗尽任务一个真实batch，GLOBAL直接跨任务、GROUP组内按记录数加权，PRIVATE原始平均梯度；不使用旧任务权重补偿。原shuffle、完整行覆盖、256逻辑/微批、10轮、LR1e-4及整模裁剪不变。每轮optimizer/scheduler更新次数改为任务batch数最大值。旧Stage2 checkpoint不能恢复、旧final不能迁移到当前Stage3；需要用新目录从头训练Stage2，再以新final训练Stage3。数据/Stage1身份一致的Stage2/3 prepare可复用，历史产物只读。

训练诊断见 [ADR-0098](adr/0098-stage3-epoch-training-diagnostics.md)。各fold的 `phase_1/diagnostics.jsonl`、`phase_2/<scope>/diagnostics.jsonl` 和 `phase_3/<task>/diagnostics.jsonl` 每轮写一行：`diagnostics_version: 2` 的 `epoch_gradient_stats` 记录各owner实际梯度step数、裁剪前后范数mean/max及裁剪率；`shared_sample_contributions` 记录GLOBAL/GROUP的实际记录数、记录占比和逐step归一化权重累计/均值。任务耗尽及冻结不混入梯度均值分母，模拟只计PRIVATE。记录占比与权重均值含义不同，不能混用；无观测/分母时为null。新版替换原顶层最后一步范数字段，历史行不重写，恢复后可在同一日志追加新版行。诊断完全只读，不进入科研身份，本次诊断变更无需重训Stage2/3，不解除此前聚合算法变更的身份限制。

## 运行前提

使用已安装的 `ilume` 环境，不安装依赖或下载权重。运行前核验 `outputs/v4/stage1/base/train/stage1_encoder.pt` 与配对的 `outputs/v4/stage1/base/prepare/artifacts`；文件存在不代表身份兼容，仍以入口完整性检查为准。源数据不能在数据准备/训练/评估之间替换。数据审计元数据由正式 prepare 写入配置的数据目录，不手改 SHA。

截至2026-10-11，本地上述 Stage1 编码器及特征元数据已存在；默认 Stage2 `base/prepare`、`base/train` 已有旧运行，而配置指定的 `base/train/stage2_final.pt` 不存在。新运行需使用未占用的目录，并同步 prepare 产物和 Stage3 来源引用；本次仅核对文件状态，没有验证这些正式产物的完整加载兼容性或执行正式作业。其他数据 split 的正式运行仍由用户显式安排。

仅提供 `configs/v4/stage2/base.yaml` 和 `configs/v4/stage3/base.yaml`；新增架构/任务集对照已移除。Stage3 Phase2中的两项模拟任务只更新自身PRIVATE，不参与GLOBAL/GROUP梯度或共享梯度归一化；Phase3继续只更新PRIVATE。查看 `performance.jsonl` 的参数量、轮耗时及 `metrics.jsonl` 的实际 optimizer_updates。

当前Stage3 Base的electrochemical GROUP采用L1/L2各2个专家、768D隐藏层、1024D输出。两项电位任务的PRIVATE/Tower/FiLM隐藏维度为512/256/128D，保留单PRIVATE专家、仅温度条件及dropout 0.1；Task Gate输入2048D、输出5个候选权重。electrochemical GROUP的Phase1保持2e-4/15轮，Phase2采用1e-4/4轮（原20轮）；两任务PRIVATE的Phase1/2/3预算仍为8/4/3轮，LR及三阶段机制不变。容量配方进入既有训练身份；容量或Phase2预算调整前的Stage3检查点不能恢复到当前Base，正式训练须使用未占用的新输出目录，Stage1/Stage2无需因本次容量调整重训。

## 最终优化配方（3407）

在 `outputs/v4` 重跑已锁定的HoME `G2_C05` / Stage2 `007` / Stage3 `032`，使用独立配置 `configs/v4/stage2/optimized_3407.yaml` 与 `configs/v4/stage3/optimized_3407.yaml`。数据划分3407，训练seed42（Stage3实际42+fold）；完整模型结构、任务映射、LR、nominal Epoch及任务覆盖与完成五套复验时的配方一致。原Base YAML保留，不覆盖HPO结果。

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

## 原Base（split42）复现步骤

与Stage1一样，从仓库根目录在 `ilume` 环境执行。下面各步骤成功后再执行下一步，使用未占用的输出目录。入口会自动校验来源、数据SHA和模型身份，无需另写Python校验脚本。修改prepare输出路径时，须同步修改对应YAML的 `data.artifacts_dir`；修改训练输出路径时，须同步修改下游来源配置。

### Stage2：prepare 与 train

prepare生成五任务数据和冻结Stage1实体缓存；train固定训练10轮，发布 `stage2_final.pt` 和 `stage2_final.json`，不再导出独立编码器。

```bash
python scripts/stage2/prepare.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/base/prepare
python scripts/stage2/train.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/base/train
```

### Stage3：prepare 与五折 train

Stage2训练完成后准备Stage3，再执行三阶段五折训练。下面使用一张已分配的GPU，五折串行执行；多GPU并行时才增加 `--max-parallel` 并扩展 `--devices`。

```bash
python scripts/stage3/prepare.py --config configs/v4/stage3/base.yaml --output outputs/v4/stage3/base/prepare
python scripts/stage3/train.py --config configs/v4/stage3/base.yaml --fold 1 2 3 4 5 --output outputs/v4/stage3/base/train --devices cuda:0
```

### Stage3：实验与模拟联合评估

默认 `--domain all`，每次调用完成同一split的实验与两项模拟评估。实验验证仍逐折评估，模拟验证仍使用全部五个final的原单位预测均值；测试集两类均按各自协议使用五模型集成。先查看验证结果，再报告测试集；hydration无测试集。

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/base/train --split valid --fold 1 2 3 4 5 --output outputs/v4/stage3/base/valid
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/base/train --split test --ensemble-folds --output outputs/v4/stage3/base/test
```

联合评估使用未占用的输出根；实验验证写入 `foldN/`，实验测试写入根目录，模拟结果写入同根的 `simulation/`。每个运行仍独立记录身份、metadata、summary和预测清单，汇总器递归读取，实验与模拟榜单继续独立。入口先核对模拟所需的全部五折final来源；实验评估失败时返回非零并停止后续模拟评估，模拟失败时保留已完成实验结果且返回非零，不覆盖已有产物。

只评估实验时显式使用 `--domain experimental`，其余参数沿用上方命令；只评估模拟时使用 `--domain simulation --ensemble-folds`，不传 `--fold`、`--tasks` 或 `--checkpoint-epoch`。联合评估中 `--fold`（验证）和 `--tasks` 仅选择实验任务/折，模拟始终评估全部两项任务及五模型集成；联合评估不接受 `--checkpoint-epoch`。

### 结果汇总

只扫描新的实体HoME输出，发布到独立目录，不与旧20实验/四模拟结果混排。

```bash
python scripts/benchmarks/summarize.py --input outputs/v4/stage3 --output outputs/v4/summary
```

只在确实需要继续中断作业时按现有入口显式使用 `--resume`；必须保留同一配置、源 SHA、折/owner/训练状态和指标尾部。不要通过修改元数据、kind、格式、任务名称或宽松加载复用不匹配的历史产物。

## 数据核验结论与限制

以下为2026-10-08本地核验快照；正式运行仍以加载和prepare的实时完整性检查为准。实现已通过临时CPU链路验证，正式训练和五折结果尚未生成。

当前任务目录的 gas_solubility 包含多种气体，不能视为旧 x_co2 的同义名称；本版本已按明确确认改变目标。新增水活度和焓任务有实际任务目录/分折来源，焓任务展示为汽化焓（Enthalpy of vaporization），仅使用temperature_K；常量phase列不进入模型。任务目录中的任务标识、目录、目标列及源SHA保持原样。hydration 继续 random/cv1 且无测试集，其余使用任务目录的 system split。Stage1电荷监督及其资源合同保持不变。参考压缩包不替代实时任务目录和文件身份。

Stage2五任务训练集/验证集来源齐全；两项辅助任务的训练集/验证集/测试集齐全。24实验的120个分折文件存在，当时10项有测试集文件，其余任务保留五折验证，不生成额外测试集划分；任务目录的实验测试集计数字段为空，不据此推测缺失文件的科学意图。当时正式运行的未满足前提是配置指定路径的 Stage1 编码器及配对特征产物；旧元数据需上述新 prepare 正常重建。2026-10-08的实现验收没有启动正式 GPU训练、正式五折或覆盖历史输出；后续HPO进度见[当前报告入口](hpo-simple-runbook.md#当前选型与报告入口)，运行前仍须核验当前配置指定的来源产物。

历史基线的20任务训练配方不扩展，模拟四任务只适配任务目录现有来源位置。其旧实验合同依据仍要求x_co2；当前任务目录缺少这一历史目标，运行旧实验基线需要对应历史任务目录/数据，不得用gas_solubility替代。两模拟实体 HoME v4与四模拟历史结果不能混入同一汇总。

## 独立静态 HPO

正式运行保持本手册原配方。独立调参使用 [HPO手册](hpo-simple-runbook.md) 和 [ADR-0099](adr/0099-simple-static-hpo.md)，配置与输出分别位于 `configs/HPO`、`outputs/HPO`，人工选型后分轮生成完整YAML。Stage2支持正整数Epoch及可选training.seed；默认10轮和data.seed沿用行为保留。HPO显式固定训练seed42，Prepare审计写入独立输出，不改写数据目录metadata或正式产物。

## 当前数据版本的基线

基线现役catalog_v4_3407协议按[基线手册](baseline-catalog-v4-runbook.md)及[ADR-0102](adr/0102-baseline-split3407-full-retraining.md)运行：data/3407、24实验、三模拟，十模型全部重训共1,230作业，输出outputs/benchmarks/v4_3407。0101的split42复用安排及历史20实验/四模拟不再作为当前入口；旧baseline按用户确认范围归档/删除。Stage1/2/3数值合同不变。
