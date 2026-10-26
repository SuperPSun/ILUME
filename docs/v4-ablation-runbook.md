# 现役 v4 消融运行手册

科研合同见 [ADR-0100](adr/0100-v4-entity-home-ablations.md)。本手册所有命令在仓库根目录执行，使用 `ilume` 环境。配置位于 `configs/ablations/v4_3407/`，新产物位于 `outputs/ablations/v4_3407/`。正式训练由用户启动；每个新输出目录必须不存在。任何命令失败即停止后续步骤；不覆盖旧输出，不编辑 SHA，不自动下载 teacher或重建正式数据。

## 前置检查

本轮Base指 [最终优化配方3407](v4-runbook.md#最终优化配方3407)，对应 `configs/v4/stage2/optimized_3407.yaml` 和 `configs/v4/stage3/optimized_3407.yaml`。原 `configs/ablations/v4/` 与 `outputs/ablations/v4/` 仅保留split42历史身份。核心实验共享原Base Stage1 prepare；C1、C9、C10分别使用独立表示缓存，其他核心训练复用3407 Base Stage3 prepare及对应Stage2数据。Stage2来源必须满足ADR-0097新联合更新合同。Phase1/Phase2对照要求五个3407 Base折的阶段权重齐全；C0直接复用其final。

```bash
cd /data/pengs/ILUME
conda activate ilume
```

已有 `outputs/v4/stage2/optimized_3407/prepare` 与 `outputs/v4/stage3/optimized_3407/prepare` 且原生加载校验通过时直接复用。仅在对应目录不存在时执行下列prepare；Stage3 prepare还要求3407 Stage2 final已完成。不匹配的既有目录不覆盖、不改元数据，另配新路径及成套YAML。

```bash
python scripts/stage2/prepare.py --config configs/v4/stage2/optimized_3407.yaml --output outputs/v4/stage2/optimized_3407/prepare
python scripts/stage3/prepare.py --config configs/v4/stage3/optimized_3407.yaml --output outputs/v4/stage3/optimized_3407/prepare
```

C1、两项单视图和五项监督消融的prepare依赖各自新编码器；下文按其上游训练完成后依次生成，不可用完整模型的实体缓存替代。Stage1语料prepare、teacher和电荷sidecar仍按原身份复用，不重新生成全量teacher。

以下命令面向4090的四张GPU：Stage1/Stage2及prepare使用单卡，Stage3显式四折并发，第五折等待空闲设备；MLP最多四个作业并发。单卡执行将 `--max-parallel` 或 `--max-workers` 改为1，并将 `--devices` 改为 `cuda:0`。所有核心和扩展均使用 `data/3407`；训练seed仍为42，Stage3按42+fold派生，不把训练seed当作split seed。

Base五折final已完成时无需重新训练。汇总前需要按最终配方手册生成其验证/测试报告；若对应输出目录已存在且完整则复用。实验主表只含24任务，两模拟单列；分层使用优化配方的六个GROUP。

## C1：随机冻结 Stage1

先导出同结构的 seed42零训练编码器，再重新准备 Stage2/3实体、训练Stage2。导出不用 teacher前向、不更新语料，也不创建预训练末轮状态。

```bash
set -e
python scripts/stage1/train.py --config configs/ablations/v4_3407/core/no_stage1/stage1.yaml --export-random-encoder --output outputs/ablations/v4_3407/no_stage1/stage1/train
python scripts/stage2/prepare.py --config configs/ablations/v4_3407/core/no_stage1/stage2.yaml --output outputs/ablations/v4_3407/no_stage1/stage2/prepare
python scripts/stage2/train.py --config configs/ablations/v4_3407/core/no_stage1/stage2.yaml --output outputs/ablations/v4_3407/no_stage1/stage2/train --device cuda:0
python scripts/stage3/prepare.py --config configs/ablations/v4_3407/core/no_stage1/stage3.yaml --output outputs/ablations/v4_3407/no_stage1/stage3/prepare
python scripts/stage3/train.py --config configs/ablations/v4_3407/core/no_stage1/stage3.yaml --fold 1 2 3 4 5 --output outputs/ablations/v4_3407/no_stage1/stage3/train --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Stage1来源固定后不要重新导出或覆盖。

## C2～C6：下游核心对照

`no_stage2`、`no_global`、`no_group`、`no_private`、`uniform_gate` 都复用 Base Stage3实体prepare。C2只需要 Base Stage2 prepare与自包含初始化配置，不需要已训练 Stage2 final；模拟来源和零更新状态由后端严格验证。

```bash
set -e
for ablation_name in no_stage2 no_global no_group no_private uniform_gate; do
  python scripts/stage3/train.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
done
```

训练完成后核心六项分别评估。默认 `--domain all`同时输出实验及独立模拟报告；验证实验是五折，模拟验证是五个final模型的原单位集成。

```bash
set -e
for ablation_name in no_stage1 no_stage2 no_global no_group no_private uniform_gate; do
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --split valid --fold 1 2 3 4 5 --output "outputs/ablations/v4_3407/${ablation_name}/stage3/valid"
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --split test --ensemble-folds --output "outputs/ablations/v4_3407/${ablation_name}/stage3/test"
done
```

有失败/完整轮检查点时，只对相同配置和相同输出运行原命令并追加 `--resume`；身份或历史尾部不一致会失败，不截断日志。

## C7/C11：阶段增量

阶段增量按C11→C7→C0比较。两份阶段配置是评估专用，读取完整Base训练目录：`phase1`是仅Phase1固定末轮（C11），`phase2`是Phase1＋Phase2的拼接末状态（C7），完整三阶段直接复用C0的final验证/测试报告。C11沿用现有phase1配置、内部variant与输出路径，原评估命令继续适用。五折按同一协议比较阶段增量，无需新增训练；各阶段训练量不同，不能据此宣称等预算下三阶段最优。阶段评估不会创建/改写Base final或清单。

```bash
set -e
for ablation_name in phase1 phase2; do
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir outputs/v4/stage3/optimized_3407/train --split valid --fold 1 2 3 4 5 --output "outputs/ablations/v4_3407/${ablation_name}/stage3/valid"
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir outputs/v4/stage3/optimized_3407/train --split test --ensemble-folds --output "outputs/ablations/v4_3407/${ablation_name}/stage3/test"
done
```

## C8：独立实体 MLP

新注册名为 `ilume_stage3_single_task_mlp_entity_v4`，不使用历史Object表示或其权重。复用 Base实体prepare；完整输入含RDKit、角色和mask。共享基准比较入口负责24任务×5折及原单位五模型集成。

```bash
python scripts/benchmarks/sweep.py --config configs/ablations/v4_3407/core/single_task_mlp.yaml --benchmark stage3 --output outputs/ablations/v4_3407/single_task_mlp --max-workers 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
```

不支持中途权重恢复；失败任务由既有尝试机制从头训练，已完成尝试保留。MLP仅实验榜单，无模拟训练或榜单。

## C9/C10：移除 SMILES 或分子图

`no_smiles` 仅保留图编码器，关闭 SMILES 重建和双视图一致性；`no_graph` 仅保留 SMILES 编码器，关闭原子/键重建、电荷和双视图一致性。两项均从头训练，保留 RDKit/Uni-Mol/电子辅助监督和下游 RDKit217。缺失分支对应的512D槽固定填零，原1024D残差融合不变。剩余分支沿用原 masking 和 dropout。无图模型不提供原子表示，实体缓存保留行对齐的零宽原子张量。

本轮现役32份完整YAML，另保留4份已取消容量配置供追溯；所有训练seed固定42。Stage1语料、AIS分词、RDKit统计与teacher来源复用Base，不生成新语料或全量teacher。去掉分子图指移除图编码分支，并非完全不使用图衍生信息。每项独立Stage1/Stage2/五折Stage3，不能混用双视图来源。

```bash
set -e
for ablation_name in no_smiles no_graph; do
  CUDA_VISIBLE_DEVICES=0 python scripts/stage1/train.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage1.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage1/train"
  python scripts/stage2/prepare.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage2.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage2/prepare"
  python scripts/stage2/train.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage2.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage2/train" --device cuda:0
  python scripts/stage3/prepare.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage3/prepare"
  python scripts/stage3/train.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --split valid --fold 1 2 3 4 5 --output "outputs/ablations/v4_3407/${ablation_name}/stage3/valid"
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --split test --ensemble-folds --output "outputs/ablations/v4_3407/${ablation_name}/stage3/test"
done
```

两项新增成本合计2次Stage1、2次Stage2及10个Stage3折训练。结果解释为编码分支及其依赖监督的总体贡献，同时涉及容量减少。

## 扩展监督

五个监督名称为 `no_unimol`、`no_electronic`、`no_partial_charge`、`no_alignment`、`no_descriptor_loss`。每个只改变一项系数，全部预先注册，不能看结果再挑。相同语料/teacher/sidecar按原身份复用；各分支新的模型和实体prepare不可混用。

```bash
set -e
for ablation_name in no_unimol no_electronic no_partial_charge no_alignment no_descriptor_loss; do
  CUDA_VISIBLE_DEVICES=0 python scripts/stage1/train.py --config "configs/ablations/v4_3407/auxiliary/${ablation_name}/stage1.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage1/train"
  python scripts/stage2/prepare.py --config "configs/ablations/v4_3407/auxiliary/${ablation_name}/stage2.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage2/prepare"
  python scripts/stage2/train.py --config "configs/ablations/v4_3407/auxiliary/${ablation_name}/stage2.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage2/train" --device cuda:0
  python scripts/stage3/prepare.py --config "configs/ablations/v4_3407/auxiliary/${ablation_name}/stage3.yaml" --output "outputs/ablations/v4_3407/${ablation_name}/stage3/prepare"
  python scripts/stage3/train.py --config "configs/ablations/v4_3407/auxiliary/${ablation_name}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --max-parallel 4 --devices cuda:0,cuda:1,cuda:2,cuda:3
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/auxiliary/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --split valid --fold 1 2 3 4 5 --output "outputs/ablations/v4_3407/${ablation_name}/stage3/valid"
  python scripts/stage3/evaluate.py --config "configs/ablations/v4_3407/auxiliary/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/v4_3407/${ablation_name}/stage3/train" --split test --ensemble-folds --output "outputs/ablations/v4_3407/${ablation_name}/stage3/test"
done
```

2026-10-26已取消四项容量对照，不再训练或评估 `capacity_no_global`、`capacity_match_global`、`capacity_no_group`、`capacity_match_group`。保留其YAML供追溯，3407配置加载时会明确拒绝。此前复制的批量命令须移除整个容量循环；C1～C6评估、Phase1/Phase2阶段评估、C9/C10及五项辅助监督继续执行。C3/C4减少参数量的解释边界仍保留。

## 汇总与配对分析

明确包含 Base 的现役验证/测试与每个已完成对照，不扫描混入历史结果。下面生成两个全新报告目录；bootstrap为可选，开启后按原split体系单位重采样，固定42/10000次。

```bash
python scripts/benchmarks/summarize.py --input outputs/v4/stage3/optimized_3407/valid outputs/v4/stage3/optimized_3407/test outputs/ablations/v4_3407 --output summary_v4_3407_ablations --ablation-analysis-output summary_v4_3407_ablation_analysis --ablation-authority-config configs/v4/stage3/optimized_3407.yaml --ablation-bootstrap
```

所有实验固定训练 seed42，不追加 seed123/3407 稳定性训练。若输出根中存在历史重复实验，必须用 `--include` 显式列出本轮实际验证/测试目录及 MLP 输出，保证每次比较只有一个配对 Base；不删除历史结果。分析严格要求完整24任务验证、相同测试任务和来源；模拟只进入既有独立榜单。bootstrap 区间使用原单位、合并折预测的逐行MAE差，标记为 `pooled_row_mae`；它不是五折MAE等权均值的区间。`analysis.json`和两个差值CSV位于分析目录，现役v4快照文件集合不改。

每折训练保留 `training_cost`、resolved计划与整轮诊断；成本分析汇总GPU小时、显存、owner参数/更新和样本暴露。Stage1/2和MLP的原训练日志/权重也必须保留。没有完整权重不能仅靠报告认定运行可复现。

## 产物回传

无论在哪个设备训练，所有权重、配置、身份、日志和恢复产物都回传4090对应项目路径，不排除 `.pt`。按 [跨设备归档流程](v4-runbook.md#跨设备训练产物回传)完成只读校验，不覆盖现有运行或历史输出。
