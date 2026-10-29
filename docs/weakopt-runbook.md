# Stage3 WeakOpt 参数优化

合同见[ADR-0104](adr/0104-stage3-weakopt.md)。固定当前3407完整模型及Stage1/2来源，只调整实验任务PRIVATE的学习率、轮数和dropout。配置在 `configs/HPO/WeakOpt`，新增训练、日志、审计、临时文件与报告在 `outputs/HPO/WeakOpt`；原准备产物只读，不重建缓存。完整训练权重按项目规则保留在4090。

## 生成与运行

在仓库根目录、`ilume` 环境执行。首次生成会调用现有论文读取器，只审计验证来源；冻结baseline逐折MAE及共同尺度，并生成24份完整配置。再次生成只验证既有配置，不覆盖。

```bash
python scripts/HPO/weakopt.py generate
python scripts/HPO/weakopt.py run --ids 001 002 003 004 005 006 007 008 --fold 1 3 --devices cuda:2,cuda:3 --max-parallel 6 --dry-run
python scripts/HPO/weakopt.py run --ids 001 002 003 004 005 006 007 008 --fold 1 3 --devices cuda:2,cuda:3 --max-parallel 6
python scripts/HPO/weakopt.py report --fold 1 3
```

不设置外层 `CUDA_VISIBLE_DEVICES`；入口传递物理GPU编号，训练worker和评估子进程各自映射到分配的卡。默认且仅允许GPU2/3，每卡三个作业，总并发6。使用单卡时同时指定 `--devices cuda:2 --max-parallel 3`；显式较低并发仍兼容。跨控制器的三个GPU作业槽限制每卡最多三个作业，前两槽沿用旧GPU锁，因此可与在途旧控制器并存。2026-10-26此前已安排的GPU0/1作业全部跑完，后续不再追加。调整设备和并发不改变候选身份，不中断其他卡上的运行。省略 `--ids` 会运行全部已冻结候选。失败后执行相同命令继续，`--resume`可显式附加，原生严格恢复始终启用；不修改历史文件或配置SHA。

首批001–006依次为黏度、密度、折射率、表面张力、玻璃化转变温度、熔点的dropout=0.05；007–012同顺序只将PRIVATE LR乘0.7；013–018同顺序只将Phase3增加2轮。019/020为水活度dropout=0/0.05；021/022为静态介电Phase3=1/2；023/024为pEC50的LR乘0.7/dropout=0.05。其余值均与冻结参考一致。

### 每15分钟补位

```bash
python scripts/HPO/weakopt.py watch
```

该独立进程每15分钟只检查GPU2/3作业锁、候选锁和Fold1/3完成标记，在每卡三个作业的上限内补入未完成的已冻结候选。预留已启动但尚未取得GPU锁的子进程槽位，不重复启动正在运行的候选；失败候选保留现场，不自动重试。完成标记仅用于调度，最终排名仍执行完整身份和文件校验。检查记录追加到 `outputs/HPO/WeakOpt/watch.jsonl`，补位日志在 `outputs/HPO/WeakOpt/watch/`。不读取逐轮指标或测试指标，不生成新配置；锁定最终候选后停止检查。启用补位表示允许已有冻结候选在其他队列尚未结束时开始，GPU锁统一限制总并发。

## 追加候选及补折

每批先分析验证结果，再显式准备追加提案，例如 `configs/HPO/WeakOpt/proposals/batch02.yaml`：

```yaml
- description: Water activity dropout and static permittivity private budget
  overrides:
    experiment/water_activity_coefficient:
      model_overrides:
        private_dropout: 0.0
    experiment/static_relative_permittivity:
      phase3_private_epochs: 2
```

```bash
python scripts/HPO/weakopt.py generate --proposal configs/HPO/WeakOpt/proposals/batch02.yaml
python scripts/HPO/weakopt.py run --ids 025 --fold 1 3
```

提案按冻结参考写绝对参数值，不按前一候选复乘。只能改实验任务的 `phase1_private_lr`、`phase1_private_epochs`、`phase2_private_epochs`、`phase3_private_epochs` 和 `model_overrides.private_dropout`。宽度、专家数、任务集合、共享配方、loss、batch、seed和来源不允许修改；非法配方、重复配置及超过累计80组均拒绝。

入围编号由验证分析明确指定；示例以025为例，不表示它已入围：

```bash
python scripts/HPO/weakopt.py run --ids 025 --fold 2 4 5 --resume
python scripts/HPO/weakopt.py report --fold 1 2 3 4 5
```

仍训练同一个24任务完整模型，不从别的候选移植任务权重。补折不改变身份；再次请求完成折时，原生入口严格校验后跳过。报告保留不完整项，排名仅包含全部请求折完成且独立评估通过的候选。排序先比较第一名数，再比较共同尺度nMAE和编号；允许均值退化，所有任务均报告。两折分数不能当作五折达标，折间变化不代表显著性。

## 锁定及正式测试

五折验证目标是15/24项第一。达到目标或用完80组后，明确锁定五折完整候选中排名最高者。以下仍以025为示例：

```bash
python scripts/HPO/weakopt.py finalize --id 025
CUDA_VISIBLE_DEVICES=2 python scripts/stage3/evaluate.py --config configs/HPO/WeakOpt/trials/025.yaml --checkpoint-dir outputs/HPO/WeakOpt/trials/025/train --split test --domain experimental --ensemble-folds --output outputs/HPO/WeakOpt/trials/025/test
python scripts/benchmarks/summarize_paper.py --config configs/HPO/WeakOpt/trials/025.yaml --ilume outputs/HPO/WeakOpt/trials/025 --baselines outputs/benchmarks/v4_3407 --with-test --output outputs/HPO/WeakOpt/paper-final-001
```

测试只在锁定后运行一次五模型集成，目标6/10项第一。搜索入口不读取测试指标、不自动挑选或启动测试；目标未达到也保留结果，不返回测试驱动调参。当前3407及历史探索的选择边界需披露。相同名称的报告和锁定选择不得覆盖，失败评估目录保留后使用新的输出路径。

## 第二轮Stage2/Stage3联合调参

按[ADR-0108](adr/0108-weakopt-joint-stage2-stage3.md)启用独立80对预算，上一轮不改写。配置在 `configs/HPO/WeakOpt/Joint`，产物在 `outputs/HPO/WeakOpt/Joint`。

```bash
python scripts/HPO/weakopt_joint.py generate
python scripts/HPO/weakopt_joint.py run --fold 1 3 --devices cuda:2,cuda:3 --max-parallel 6 --dry-run
python scripts/HPO/weakopt_joint.py run --fold 1 3 --devices cuda:2,cuda:3 --max-parallel 6
python scripts/HPO/weakopt_joint.py report --fold 1 3
```

首批24对先训练12份唯一Stage2来源，再执行24份Stage3两折。重复run始终严格恢复，完成折不重训。后续用 `generate --proposal <提案.yaml>` 追加；提案列表每项明确 `description`、`stage2`、`stage3`。stage2仅接受epochs、task_head_learning_rate、weight_decay、warmup_fraction；stage3为上一轮格式的任务覆盖字典。所有数值相对冻结参考写绝对值，不改来源路径来复用其他权重。

补折用 `run --ids <编号> --fold 2 4 5`，然后 `report --fold 1 2 3 4 5`；锁定用 `finalize --id <五折最优编号>`。锁定后的测试和论文命令与第一轮相同，但配置、模型和输出均指向Joint内同编号路径；旧test不作为本轮产物，不按测试结果返回搜索。
