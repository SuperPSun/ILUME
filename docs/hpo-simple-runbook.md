# 扁平化 HPO 运行手册

本流程使用静态 YAML 和人工选型，不使用 Optuna，不自动晋级，不读取测试指标。正式 Base 入口与配方保持不变。合同见 [ADR-0099](adr/0099-simple-static-hpo.md)。所有训练、缓存、日志和报告位于 `outputs/HPO`，配置位于 `configs/HPO`。

HPO无论在哪台设备运行，训练结果与完整模型权重都须回传到4090的对应 `outputs/HPO`，保留配置、身份、日志和恢复产物；不得仅同步CSV/JSON或排除 `.pt`。共享缓存及来源按身份核验后复用或回传，具体遵循 [跨设备训练产物回传](v4-runbook.md#跨设备训练产物回传)。

使用安装项目依赖的 `ilume` 环境，从仓库根目录运行。需要完整末轮 `outputs/v4/stage1/base/train/last.pt`、配对的 `stage1_encoder.pt`、Stage1 Prepare 和电荷 sidecar。五套数据为42、123、1234、3407、90756，Stage1 仍共用 `data/stage1`。

## 当前选型与报告入口

2026-10-20，固定配方HoME `G2_C05` + Stage2 `007` + Stage3 `032`已完成五套split的Stage2与Stage3五折验证。研究者据验证结果决定：**后续新实验统一采用数据划分3407**；共同训练seed仍为42，Stage3实际按42+fold派生。已有正式Base、基线、消融配置及训练产物保留原split与身份，不批量迁移、不覆盖旧输出。Stage1输入及冻结来源不变。

报告总入口为 `outputs/HPO/reports/README.md`，最终汇总为 `outputs/HPO/reports/selection-split-20261020/selection.md`。其中按阶段链接HoME、Stage2、Stage3的选型记录及完整排名，并保留五套split的总分、逐折/逐任务指标、模拟独立指标和跨split统计。旧时间戳报告、资源审计和启动前状态仅作历史快照，不代表当前进度；报告归档不代表模型权重已全部回传。

最终报告按24项实验等权的五折normalized MAE比较，保留完整排名与逐任务结果；两模拟不进入该总分。五套数据有重叠，3407称为“验证选出的最佳划分”，不能把后续测试解读为完全独立于跨划分选择的泛化估计。

五套配置已在训练设备生成，3407参考为 `configs/HPO/final/3407/stage2.yaml`、`stage3.yaml`，完成模型为 `outputs/HPO/final/3407/stage2/` 与 `stage3/train/fold1/`～`fold5/`。新实验从配套配置派生到独立YAML及输出，重新绑定同split的Stage2来源和prepare身份；不得只替换 `data/42` 字符串后沿用旧缓存、generation绑定或恢复目录。下文保留完整HPO复现步骤，不需要重新启动已完成的搜索。

在 `outputs/v4` 独立复跑最终配方，使用 `configs/v4/stage2/optimized_3407.yaml` 与 `configs/v4/stage3/optimized_3407.yaml`；保持HPO选定参数及Stage1来源，Stage2/3准备产物和模型均写入新的v4路径，不再依赖HPO训练输出。命令见 [v4手册](v4-runbook.md#最终优化配方3407)。

## 四轮实验与人工选择

首轮已提供100份 Stage1 配置和100对 HoME 配置。可用以下命令再次核对并生成缺失的相同配置；已固定且不同的 YAML 不会被覆盖。

```bash
python scripts/HPO/stage1.py --generate
python scripts/HPO/stage1.py --ids 000 --dry-run
python scripts/HPO/home.py --ids G0_C05 --dry-run
```

正式运行由研究者显式启动。以下示例为四卡、总并发4；设为8则每卡两个任务，设为7则四卡分别分配2/2/2/1个任务槽。省略设备参数时仅使用 `cuda:0`；省略并发参数时默认每卡一个任务。

```bash
python scripts/HPO/stage1.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
python scripts/HPO/home.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4 &&
python scripts/HPO/collect.py --branch HoME
```

Stage1 每组训练所有14个目标。`000` 为原始 MLP Base；其余包括10个 Linear 配方及89个非线性配方。非线性结构为 MLP/Residual MLP，宽度倍率0.5/1/2，深度1/2/3，GELU；标量首层512、原子首层256，后续层逐次减半。五套训练组合为 `(LR,Epoch,Dropout,WD)`：`(1e-4,10,0,.01)`、`(3e-5,20,.1,.01)`、`(3e-4,5,0,0)`、`(1e-4,30,.2,.01)`、`(1e-5,50,.3,.1)`。Linear 不接受宽度/Dropout；初始化继承预训练头。训练 seed 保持42，记录每个任务的派生 seed。各目标独立看 MAE/RMSE，不能把原始 MAE 相加选统一赢家。

HoME 为 G0～G4 × C00～C19。C编号为 `(GLOBAL数量−1)×5+GROUP档位`；GLOBAL数量1/2/3/4，GROUP档位依次为原配方、专家数减1但不少于1、专家数加1、隐藏宽度乘0.5、隐藏宽度乘2。保留组间容量差异；G0_C05 对应原始 Base 容量。PRIVATE数量固定1。GLOBAL/GROUP的输出接口及 flat gate 不变。

分组映射如下；其余任务保留 Base 映射，完整24任务映射写在每份 Stage3 YAML 中。

| 方案 | 相对 G0 的变化 | 研究假设 |
|---|---|---|
| G0 | 不变 | 原始 Base |
| G1 | transport 全部归入 thermophysical | 对照传输与热物性共用物理响应专家 |
| G2 | electrochemical 全部归入 phase_stability | 对照稳定性任务共享专家 |
| G3 | dielectric_optical 全部归入 thermophysical | 对照介电/光学与宏观物性共享专家 |
| G4 | 合并 G1、G2、G3 的变化 | 对照更粗粒度的物理性质分组 |

归入任务使用目标 GROUP 的容量和预算，因此这是任务组织与实际 GROUP 配方的整体比较，不能归因于纯分组变量。Stage2 五模拟归属不变，thermophysical 和 solvation 保留，Interaction 仍在 solvation。相同容量下五种 G 的 Stage2 配方相同，引用 `outputs/HPO/HoME/stage2/G0_Cxx` 的单写入者训练结果；每一对仍有独立 Stage3 输出与记录。每对训练 Fold1、3，并只调用实验域验证。

100对结束后自行查看报告，复制 `configs/HPO/selected.example.yaml` 为 `configs/HPO/selected.yaml`，只填写你选定的 `home`（例如 `G2_C05`；示例不是程序推荐）。

```bash
python scripts/HPO/stage2.py --generate --selected configs/HPO/selected.yaml
python scripts/HPO/stage2.py --ids 000 --dry-run
python scripts/HPO/stage2.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
python scripts/HPO/collect.py --branch stage2
```

Stage2 是 Epoch5/10/15/20 × LR3e-5/6e-5/1e-4/2e-4/3e-4，编号按 Epoch 外层、LR 内层展开。每组在固定选定 HoME 的 Stage3 配方上训练 Fold1、3做下游验证。该配方不是额外搜索。独立 Stage2 输出在 `outputs/HPO/stage2/NNN/train`，下游输出在 `downstream`。

20组结束后手工填写选中的 `stage2_config`、`stage2_checkpoint`。完整静态配置在选型后生成；生成记录绑定人工选择、来源配置及存在的检查点 SHA，改动选择后不能沿用上一轮 YAML。

```yaml
home: G2_C05
stage2_config: configs/HPO/stage2/007.yaml
stage2_checkpoint: outputs/HPO/stage2/007/train/stage2_final.pt
stage3_config: null
```

```bash
python scripts/HPO/stage3.py --generate --selected configs/HPO/selected.yaml
python scripts/HPO/stage3.py --ids 000 --dry-run
python scripts/HPO/stage3.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
python scripts/HPO/collect.py --branch stage3
```

Stage3 50组依次为：Base1；GLOBAL LR倍率0.9/1.2/1.6/2 × Epoch倍率1/4:3/5:3，共12；GROUP Phase1 LR倍率0.95/1.05/1.1 × Phase2 Epoch倍率0.5/1/1.5/2，共12；PRIVATE LR倍率0.5/0.7/0.9/1.02/1.1 × `(Epoch倍率,宽度倍率)` 的 `(0.5,0.5)/(0.75,1)/(1,1.5)`，共15；PRIVATE Dropout0/0.05/0.1/0.15，4组；WD0/0.001/0.03/0.1，4组；两组 PRIVATE 组合，合计50。所有 PRIVATE 变化展开到每个实验任务，零轮保留0；共享 Dropout及模拟 PRIVATE 不变。轮数采用 `round`，正预算至少1。原生配置校验拒绝非法 LR 顺序与预算。

50组结束后填入选定的 `stage3_config`，再显式生成并运行五套 split 固定复验。

```bash
python scripts/HPO/final.py --generate --selected configs/HPO/selected.yaml
python scripts/HPO/final.py --ids 42 --dry-run
python scripts/HPO/final.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
python scripts/HPO/collect.py --branch final
```

五套 split 各训练相同配方的 Stage2 和 Stage3 Fold1～5。Stage2/3训练 seed 都为42；Stage3实际使用42+fold。先逐 split/任务统计五折，再报告五个 split 均值的均值、样本标准差、范围；缺折/非有限/失败不产生完整统计。程序不挑最佳 split，不再次搜索，不自动执行正式测试，不机械重复 Stage1 后训练；研究者在完成全部复验后采用3407的后续决定见上节。模拟只收集原生逐折验证，不能把逐折统计称为原单位五模型集成评估。不同 split 有样本重叠，不是五次独立外部验证/测试。

## 缓存、设备与恢复

Stage1 缓存绑定完整末轮检查点、特征、源数据、任务集合与过滤合同；保存映射后的每条观察、原始训练标签、审计及冻结实体/原子表示。各头自行拟合训练 scaler 并独立初始化，结构/LR/Epoch/Dropout/WD 不重复编码。

Stage2/3 Prepare 分别按 split 使用独立目录。数据身份相同且 Stage1 来源不变时，模型容量、训练参数与分组变动复用 Prepare；Stage3现有prepared身份已排除meta_group。缓存单写入者持文件锁，在临时目录生成，完整文件SHA核验后原子发布；失败临时目录和日志保留，不覆盖。缓存和 trial 身份按 JSON 表示比较，tuple/list 序列统一，不改变语义 hash 或文件 SHA 校验。共享 Stage3 entity bank 也只有一个写入者。HPO审计写在缓存的 `data_metadata.json`，不修改数据目录审计或正式产物。

日志/Dry Run 显示 `CREATE/REUSE/INVALID`。`INVALID` 保留现场并拒绝运行，由研究者指定新的配置根和缓存路径；不能改写身份来伪装合法。没有 Stage2 结果时，Dry Run 显示 `PENDING` 来源检查，实际训练前仍严格加载并检查来源、owner、shape、dtype、state hash。Dry Run 不创建缓存、锁、配置或训练输出。

所有运行器支持 `--device`、`--devices`、`--max-parallel`（别名 `--max_parallel`）、`--ids`、`--resume`、`--dry-run`。编号可以空格或逗号分隔。max-parallel 是全局并发上限，任务槽按设备列表轮流分配，允许一卡多任务；每个槽只运行一个作业，空槽供后续作业复用。底层折并行固定1，不放大外层并发。作业依次进行其 Stage2→Stage3→验证，完成后打印 `Progress N/M`；子命令详细进度写入作业日志。不会自动开始下一轮。

```bash
python scripts/HPO/home.py --ids G0_C05 G2_C05 --devices cuda:0,cuda:1 --max_parallel 2 --resume
```

Stage2/3 使用原生严格完整轮恢复；完成的正确产物校验后跳过或复用。Stage1 无中途恢复，中断 trial 在新的 `attemptN` 目录重跑，复用表示库并保留现场。原生评价无中途恢复，失败评价使用新 attempt 输出。身份不符直接报错；不覆盖完成检查点，不截断训练历史。失败作业保留记录，入口返回非零；其他独立作业可以完成。Ctrl+C取消尚未启动的任务，并终止本次控制器启动的子进程组，保留恢复现场。

### 继续失败任务与从头重跑

已有输出时，用同一配置加 `--resume` 继续；Prepare和完整Stage2 final经过身份与完整性校验后复用。修改调度器后，正在运行的控制器需要停止并重新启动才能使用新逻辑。

已完成Stage3折的零轮分支继承锚点，不要求该分支的训练目录、历史或epoch0检查点；正轮数分支仍须具有完整metrics/diagnostics和末轮检查点，final及拼接产物SHA校验不变。

```bash
python scripts/HPO/home.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4 --resume &&
python scripts/HPO/collect.py --branch HoME
```

若要从头重跑HoME训练，先停止相关进程，再在确认准确文件数、大小、备份目标和无覆盖冲突后，将 `outputs/HPO/HoME` 与存在的 `outputs/HPO/failures/HoME` 备份到独立位置。保持配置不变，保留已校验的 `outputs/HPO/cache` 和Stage1结果，然后不加 `--resume` 运行 `home.py`。这会重新训练Stage2及Stage3；只备份Stage3子目录仍会复用Stage2。若HoME输出目录已经不存在，直接按首次运行启动即可。

`outputs/HPO/failures` 只保存失败摘要，不包含模型、缓存或恢复检查点。误删后不影响训练/恢复，后续失败会自动重建；没有 `hpo_run.json` 的旧失败可能在汇总时显示为 `not_run`。日志和对应trial运行记录仍是排查依据。

## 报告与验证

`collect.py` 输出 `results.csv/json`、`task_metrics.csv`、`home_grid.csv`。缺任务、失败、非有限值明确标记；没有原生指标标为不可用。Stage2仅提供原生normalized MAE，Stage3提供原生实验指标及可用门控诊断；模拟逐折指标读取final清单中的 `simulation_validation.simulation_tasks`。不重定义指标、不按测试筛选、不自动排序或选择赢家。报告目录默认按时间新建；生成报告不代表训练成功，应先检查运行状态和任务完整性。旧报告保留生成时的快照，修复或重跑后重新汇总。

实现验收只使用临时小数据、pytest、各入口help、compileall与Dry Run。正式GPU搜索由研究者自行启动。历史Capacity配置缺失造成的既有测试失败单列，不恢复无关配置。
