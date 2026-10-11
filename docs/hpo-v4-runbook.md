# HPO v4 运行手册

本流程实施 [ADR-0099](adr/0099-hpo-v4-isolated-search-and-locked-test.md)。控制入口为 `scripts/stage3/hpo.py`，统一配置为 `configs/HPO_v4/study.yaml`。模型、缓存、日志和报告只写 `outputs/HPO_v4/<study_id>`；可编辑/生成 YAML 与锁定清单只写 `configs/HPO_v4`。现役正式 Base 仍按 [v4 手册](v4-runbook.md) 运行。

## 前提与输出

使用训练环境 `ilume`，Optuna 必须为4.7.0；可用 `python -m pip install -e '.[hpo]'` 安装固定依赖。首先核实已有完整末轮 `outputs/v4/stage1/base/train/last.pt`、`stage1_encoder.pt`、配对 prepare 特征和电荷 sidecar。HPO 不会重训 Stage1、不重建五套数据 split，也不会下载 teacher。

`study.yaml` 固定 split `{42,123,1234,3407,90756}`、搜索 seed `{11,23,42,71,101}`、重复 seed `{2027,2029,2039,2053}`、sampler seed、初筛 fold1/fold3、复核五折、完整预算和选模协议。运行前编辑预算/空间；study 一旦建立就冻结，改配方须换 `--study-id`。各阶段使用同一个 ID。Stage1 后训练是逐目标独立研究，Stage2/3 使用同一个共同训练 seed。

输出目录如下。候选记录同时给出研究阶段候选 ID 和可复用模型 ID；`model_id` 相同才可共享完成产物。

| 路径 | 内容 |
|---|---|
| `audit/` | Stage1/五 split 来源 SHA、1/2/4每卡并行的完整配方测量、独立数据审计 |
| `cache/` | 一次 train/valid 后训练表示、按 split 隔离的 Stage2/3 prepare |
| `studies/` | `candidates.json`、`jobs.json`、SQLite、采样器状态、固定空间、ask/tell事件及赢家索引 |
| `posttraining/<target>/<model_id>/` | 独立头、scaler、末轮验证、实际派生 seed |
| `joint/stage2_<id>/stage2/` | 可复用 Stage2 full final；不复用不同训练 seed/split/结构的来源 |
| `joint/<model_id>/` | 各折 attempt、只读 `train/foldN` 别名、验证预测及独立模拟验证 |
| `split_comparison/` | 五 split 独立比较 |
| `final/models/` | 锁定14头和五折模型的只读路径别名，不复制/改写源模型 |
| `final/` | 锁定镜像、重复统计、单次终测批次及独立预测/报告 |
| `reports/` | 候选 CSV、敏感性、成本估算、阶段结果 JSON 与中文报告 |

人工对照的完整 YAML 保存在 `controls/<study_id>`；实际不可变 worker YAML 保存在 `trials/<study_id>`。`locked/<study_id>/joint.yaml`、`heads/*.yaml` 和 `manifest.json` 共同提供完整最终配方与来源。

## 先检查配置，再运行

下面预览只生成完整配置和作业清单，不启动训练。预览使用单独 ID，正式运行另取 ID，避免混淆其审计快照。

```bash
cd /data/pengs/ILUME
conda activate ilume
python scripts/stage3/hpo.py prepare --study-id preview --dry-run
python scripts/stage3/hpo.py sensitivity --study-id preview --branch joint --dry-run
```

查看 `outputs/HPO_v4/preview/reports/*_jobs.json` 和 `configs/HPO_v4/controls/preview`。正式作业执行前设置 `study.yaml` 的 `execution.max_gpu_hours` 为实际 GPU 小时预算，或每个命令都显式传 `--max-gpu-hours`。默认 `null` 表示只限制候选数，不擅自猜测硬件预算。

以下步骤会运行真实完整末轮作业，应逐步检查结果再晋级。所有阶段串行启动控制器；同一 study 只能有一个控制进程。第一条 `audit --measure` 会核验来源，准备缓存，运行 Base 与容量边界候选，对每卡1/2/4作业分别测量；副本都保持 split42/训练seed42/fold1，仅用来测吞吐与资源。资源测量本身不进入240候选排名，也消耗 GPU 小时。四卡时两个容量配方在1/2/4额度各生成4/8/16个同配置副本，共56套完整 Stage2 与56个 Stage3 fold1，用以实际占满设备槽；应将这部分训练和完整检查点存储纳入资源预算。

```bash
cd /data/pengs/ILUME
conda activate ilume
set -e
python scripts/stage3/hpo.py audit --study-id hpo_v4_main --measure
python scripts/stage3/hpo.py sensitivity --study-id hpo_v4_main --resume
python scripts/stage3/hpo.py search --study-id hpo_v4_main --resume
python scripts/stage3/hpo.py search --study-id hpo_v4_main --branch joint --search-phase local --resume
python scripts/stage3/hpo.py confirm --study-id hpo_v4_main --confirm-phase folds --resume
python scripts/stage3/hpo.py confirm --study-id hpo_v4_main --branch joint --confirm-phase seeds --resume
python scripts/stage3/hpo.py splits --study-id hpo_v4_main --resume
python scripts/stage3/hpo.py lock --study-id hpo_v4_main --resume
python scripts/stage3/hpo.py repeat --study-id hpo_v4_main --resume
python scripts/stage3/hpo.py test --study-id hpo_v4_main --resume
python scripts/stage3/hpo.py report --study-id hpo_v4_main
```

每条命令失败就停止。来源审计输出 `audit/sources.json`；资源测量必须完成才允许 TPE。首次后训练缓存和 Stage2 各 split prepare 可单独用 `prepare` 完成；Stage3 prepare 在首个配对 Stage2 final 完成后生成，后续专家参数变动不重复编码。

敏感性以同一 split/seed/折的完整末轮结果作配对对照，按目标变化幅度排序。联合两结构各80 trial，前16启动探索，最多8个敏感数值变量；其余固定。局部32候选拆成两个固定小空间各16：先六类 Stage2/3 owner Weight Decay，再最多六个组/规模类别容量与 LR 参数，局部子 study 前4候选启动探索。后训练每目标最多18对照、两结构各32 trial，前8启动探索。Linear 不搜索 Dropout。生成器验证失败不会启动 worker，记录 FAIL 后继续采样，不占合法候选预算；合法配置发生训练失败则占对应预算。

`confirm --confirm-phase folds` 为初筛前8套加 Base 补齐五折，并完成14目标前三配方的五 seed 选择；之后 `confirm --branch joint --confirm-phase seeds` 为联合前3套枚举5个共同 seed。同配置完成的两折和 Stage2 自动复用，不把两折分数当成五折分数。

五 split 比较必须五套均完成才选择赢家；锁定后重复验证只报告均值、标准差、范围、逐任务稳定性。最终测试只用锁定最佳共同 seed 的五折模型。Stage1 只终测 HOMO、LUMO 和原子电荷，保留每条电荷观察，复用训练 scaler，其余11目标列明无 test。测试结果不能回流选模。跨 split 开发/测试体系有交叉，报告使用“验证选出的最佳划分”，不宣称无选择偏差的独立泛化估计。

## 设备、预算与恢复

默认四卡、每卡一个任务。可显式选择设备和全局并行上限，例如：

```bash
python scripts/stage3/hpo.py search --study-id hpo_v4_main --branch joint --resume \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --jobs-per-gpu 2 --max-parallel 8 \
  --gpu-memory-reserve-gb 6 --max-gpu-hours 2000
```

`--max-gpu-hours` 是整个 study 累积的物理设备占用小时上限；同卡多任务不重复收费。显式 `--max-parallel` 同样约束资源测量，测量副本数随实际全局额度减少。CPU/I/O、worker 树 RSS、实测峰值显存、耗时与共享并发额度单独记录；平台禁止读取 `/proc/pid/io` 时标明 `io_unavailable`，不影响科学训练。资源报告记录请求和 OOM 回退后的实际每卡额度。按测量结果选择并行度，不改 batch、AMP、调度配方或训练轮数。

队列对 trial 和折使用同一配额，Stage3 子入口固定 `--max-parallel 1`。OOM 在新 attempt 重跑同一配置并降至每卡一个，接受实际路径后重新生成下游配置；不会覆盖失败现场。非有限训练不产生有效分数；数据/来源/产物错误停止队列。GPU 预算耗尽退出2并终止活动 worker，保留已完成模型和中断 attempt；随后提高预算并用同一命令 `--resume` 继续。控制器恢复复用 hash 验证通过的完成作业，被中断的 attempt 在新目录完整重跑，原 Stage 入口的完整轮恢复合同继续保留。

只读报告可随时通过 `report` 生成。成本估算按实测配方外推 Stage2、Stage3折、后训练及缓存/检查点存储；未测量项是 `null`，不是零成本。它不自动缩减预算，不承诺未测量硬件上的时长或显存。技术性终测失败可按同一锁定清单 `test --resume` 继续，已完成域保留；不重新挑选 seed/split/头。

## 验证状态

实现验收只用临时 CPU 小数据和模拟 worker：覆盖默认合同、缓存一致性、owner 冻结与完整轮恢复、迁移与模拟隔离、配置展开、预测完整性、两折/五折隔离、调度/OOM、模型复用、锁定和终测顺序。真实四卡吞吐、正式 HPO 成绩、最终赢家及终测报告由上述正式作业产生，本次实现不生成虚构结果。

当前 checkout 的四个历史 Capacity YAML 已不存在，原 `test_capacity_formal_configs_remain_loadable` 因此失败；历史文件删除及该测试的后续处置不属于本次 HPO 数值实现。正式 v4 和 HPO 的验收分别记录，不能把这四项失败写成全套测试通过。
