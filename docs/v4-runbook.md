# v4 运行手册

本手册是现役 v4 Stage1及实体HoME Stage2/3运行入口。Stage1合同不变；Stage2/3见 [ADR-0095](adr/0095-v4-entity-home-without-object-encoder.md)，旧Object/v5运行需历史Git版本。


从仓库根目录执行，使用新的 `outputs/v4/` 路径，不覆盖历史v3输出。Stage1科研合同见 [ADR-0089](adr/0089-v4-frozen-dual-view-stage1.md)。CUDA示例使用四张设备。

当前Base采用十二层SMILES Transformer和八个独立残差图块：词表2,048个token时仅编码器约60.73M参数（[ADR-0090](adr/0090-stage1-v4-residual-encoder-capacity.md)）。相较此前约30M的v4 Base，**已有Stage1 预处理语料、统计和完成的Uni-Mol缓存可复用**；已完成时跳过下方生成步骤。从头训练新Stage1，然后重新生成Stage2表示并训练Stage2，再数据准备/训练/评估 Stage3。不要将旧Stage1检查点恢复到新架构。默认输出路径未变：若训练/下游目录已占用，选择新路径并同步更新下游引用，或在运行前明确安排归档；这些命令不授权替换历史结果。

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
python -m pip install numpy==1.26.4 pandas==1.5.3 rdkit==2025.9.5
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
assert str(unimolv2.MODEL_CONFIG_V2["weight"]["84m"]) == "modelzoo/84M/checkpoint.pt"
metadata = json.loads(Path("outputs/v4/stage1/base/prepare/artifacts/metadata.json").read_text())
validate_feature_generation_runtime(metadata)
print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}; rdkit={rdBase.rdkitVersion}")
print(f"unimol_tools={importlib.metadata.version('unimol_tools')}; checkpoint={checkpoint}")
PY
```

验证器检查预处理语料的 `feature_generation_contract`，含RDKit运行时及分词器版本。须匹配 `rdBase.rdkitVersion`，不能只看 `pip show rdkit`：包元数据与实际导入运行时可能不同。若与准备产物合同不同，先修复独立环境再继续；不要修改产物元数据或仅为绕过检查重建语料。安装示例对应运行时 `2025.09.5`；若语料记录其他版本，则安装其精确版本。

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

部分电荷头训练将来源每行保留为独立样本，重复结构复用冻结原子表示；预训练及兼容性见 [sidecar合同](#partial-charge-sidecar-on-an-existing-corpus)。标量电子冲突及训练集/验证集重叠检查不变。使用下方命令前核验回归YAML的 `simulation_dir`：当前写为 `data/stage2`，本地迁移后的电子/电荷来源在 `data/stage1/properties`。将自包含运行YAML指向实际合同目录；这不授权移动或覆盖数据。

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

## 运行前提

使用已安装的 `ilume` 环境，不安装依赖或下载权重。需先补齐 `outputs/v4/stage1/base/train/stage1_encoder.pt` 及它对应的 `outputs/v4/stage1/base/prepare/artifacts`；当前配置指定的正式编码器与特征产物尚未齐备。源数据不能在数据准备/训练/评估之间替换。现有过期 `data/stage*/metadata.json` 由下方正式 prepare 生成，不手改 SHA。

仅提供 `configs/v4/stage2/base.yaml` 和 `configs/v4/stage3/base.yaml`；新增架构/任务集对照已移除。Stage3 Phase2中的两项模拟任务只更新自身PRIVATE，不参与GLOBAL/GROUP梯度或共享梯度归一化；Phase3继续只更新PRIVATE。查看 `performance.jsonl` 的参数量、轮耗时及 `metrics.jsonl` 的实际 optimizer_updates。

## 按顺序执行

与Stage1一样，从仓库根目录在 `ilume` 环境执行。下面各步骤成功后再执行下一步，使用未占用的输出目录。入口会自动校验来源、数据SHA和模型身份，无需另写Python校验脚本。修改prepare输出路径时，须同步修改对应YAML的 `data.artifacts_dir`；修改训练输出路径时，须同步修改下游来源配置。

### Stage2：prepare 与 train

prepare生成五任务数据和冻结Stage1实体缓存；train固定训练10轮，发布 `stage2_final.pt` 和 `stage2_final.json`，不再导出独立编码器。

```bash
python scripts/stage2/prepare.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/prepare
python scripts/stage2/train.py --config configs/v4/stage2/base.yaml --output outputs/v4/stage2/train
```

### Stage3：prepare 与五折 train

Stage2训练完成后准备Stage3，再执行三阶段五折训练。下面使用一张已分配的GPU，五折串行执行；多GPU并行时才增加 `--max-parallel` 并扩展 `--devices`。

```bash
python scripts/stage3/prepare.py --config configs/v4/stage3/base.yaml --output outputs/v4/stage3/entity_home/prepare
python scripts/stage3/train.py --config configs/v4/stage3/base.yaml --fold 1 2 3 4 5 --output outputs/v4/stage3/entity_home/train --devices cuda:0
```

### Stage3：实验评估

先查看五折验证，再报告测试集的五模型集成；hydration无测试集。

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/train --split valid --fold 1 2 3 4 5 --output outputs/v4/stage3/valid
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/train --split test --ensemble-folds --output outputs/v4/stage3/test
```

### Stage3：两项模拟任务评估

验证和测试均取五个模型的原单位预测均值。

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/train --domain simulation --split valid --ensemble-folds --output outputs/v4/stage3/simulation_valid
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/train --domain simulation --split test --ensemble-folds --output outputs/v4/stage3/simulation_test
```

### 结果汇总

只扫描新的实体HoME输出，发布到独立目录，不与旧20实验/四模拟结果混排。

```bash
python scripts/benchmarks/summarize.py --input outputs/v4/stage3 --output outputs/v4/summary
```

只在确实需要继续中断作业时按现有入口显式使用 `--resume`；必须保留同一配置、源 SHA、折/owner/训练状态和指标尾部。不要通过修改元数据、kind、格式、任务名称或宽松加载复用不匹配的历史产物。

## 数据核验结论与限制

以下为2026-10-08本地核验快照；正式运行仍以加载和prepare的实时完整性检查为准。实现已通过临时CPU链路验证，正式训练和五折结果尚未生成。

当前任务目录的 gas_solubility 包含多种气体，不能视为旧 x_co2 的同义名称；本版本已按明确确认改变目标。新增水活度和焓任务有实际任务目录/分折来源，焓任务展示为汽化焓（Enthalpy of vaporization），仅使用temperature_K；常量phase列不进入模型。任务目录中的任务标识、目录、目标列及源SHA保持原样。hydration 继续 random/cv1 且无测试集，其余使用任务目录的 system split。Stage1电荷监督及其资源合同保持不变。参考压缩包不替代实时任务目录和文件身份。

Stage2五任务训练集/验证集来源齐全；两项辅助任务的训练集/验证集/测试集齐全。24实验的120个分折文件存在，当前10项有测试集文件，其余任务保留五折验证，不生成额外测试集划分；任务目录的实验测试集计数字段为空，不据此推测缺失文件的科学意图。当前正式运行的未满足前提是配置指定路径的 Stage1 编码器及配对特征产物；旧元数据也需上述新 prepare 正常重建。本次实现没有启动正式 GPU训练、正式五折或覆盖历史输出。

历史基线的20任务训练配方不扩展，模拟四任务只适配任务目录现有来源位置。其旧实验合同依据仍要求x_co2；当前任务目录缺少这一历史目标，运行旧实验基线需要对应历史任务目录/数据，不得用gas_solubility替代。两模拟实体 HoME v4与四模拟历史结果不能混入同一汇总。
