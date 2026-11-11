# 当前DGT与实体token HoME消融运行手册

本轮仅更新已有实验；原Uniform Task Gate和ChemBERTa-2 MTR/MLM已删除。编号为C0～C10及E1～E5，定义见[ADR-0124](adr/0124-current-token-home-ablations.md)。旧[ADR-0100](adr/0100-v4-entity-home-ablations.md)及旧配置只供历史追溯，不能用于当前Base。正式运行由用户启动，任何一步失败即停止后续命令；新输出不可覆盖。

## 共同来源与检查

```bash
cd /data/pengs/ILUME
conda activate ilume
```

配对当前`configs/v4/stage{2,3}/base.yaml`：G2_C01、22任务、token注意力；使用RDKit2026.09.1及当前211列schema。Stage1引用`outputs/v4/stage1/base/train/stage1_encoder.pt`，本轮Stage1重训配置匹配其batch84/5轮及图上限160；不要改为标准Base的256/10轮或旧RDKit环境。相同来源的corpus/teacher/电荷/DGT拓扑复用，不重建全量teacher，不改metadata/SHA。

先按[v4手册](v4-runbook.md)完成新Base的Stage2、Stage3 prepare/train及valid/test；路径为`outputs/v4/stage{2,3}/token_attention/`。仅Stage3对照复用其Stage3 prepare，C2另需Base Stage2 prepare；缺少来源时先完成上游。每项新表示各自prepare，不能复制Base实体缓存替代。

## C1：随机冻结Stage1

```bash
set -e
python scripts/stage1/train.py --config configs/ablations/token_attention_3407/core/no_stage1/stage1.yaml --export-random-encoder --output outputs/ablations/token_attention_3407/no_stage1/stage1/train
CUDA_VISIBLE_DEVICES=2 python scripts/stage2/prepare.py --config configs/ablations/token_attention_3407/core/no_stage1/stage2.yaml --output outputs/ablations/token_attention_3407/no_stage1/stage2/prepare
python scripts/stage2/train.py --config configs/ablations/token_attention_3407/core/no_stage1/stage2.yaml --output outputs/ablations/token_attention_3407/no_stage1/stage2/train --device cuda:2
CUDA_VISIBLE_DEVICES=2 python scripts/stage3/prepare.py --config configs/ablations/token_attention_3407/core/no_stage1/stage3.yaml --output outputs/ablations/token_attention_3407/no_stage1/stage3/prepare
python scripts/stage3/train.py --config configs/ablations/token_attention_3407/core/no_stage1/stage3.yaml --fold 1 2 3 4 5 --output outputs/ablations/token_attention_3407/no_stage1/stage3/train --max-parallel 2 --devices cuda:2,cuda:3
```

## C2～C5：下游对照

```bash
set -e
for ablation_name in no_stage2 no_global no_group no_private; do
  python scripts/stage3/train.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/train" --max-parallel 2 --devices cuda:2,cuda:3
done
```

## C6：单任务实体MLP

```bash
python scripts/benchmarks/sweep.py --config configs/ablations/token_attention_3407/core/single_task_mlp.yaml --benchmark stage3 --output outputs/ablations/token_attention_3407/single_task_mlp --max-workers 2 --devices cuda:2,cuda:3
```

22任务×5折及原单位测试集成由该入口完成；不运行模拟。输入保留三槽实体、角色、mask和条件，固定10轮末轮发布，不支持中途权重恢复。

## C7/C8：两项单视图

下面先训Stage1，再逐项运行下游。Stage1使用当前DGT来源、相同batch84/5轮，不自动缩batch；显存需按目标设备核查。`no_smiles`保留DGT，`no_graph`删除DGT；无图仍保持同一训练索引过滤和语料。

```bash
set -e
for ablation_name in no_smiles no_graph; do
  CUDA_VISIBLE_DEVICES=2 python scripts/stage1/train.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage1.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage1/train"
  CUDA_VISIBLE_DEVICES=2 python scripts/stage2/prepare.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage2.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage2/prepare"
  python scripts/stage2/train.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage2.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage2/train" --device cuda:2
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/prepare.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage3.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/prepare"
  python scripts/stage3/train.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/train" --max-parallel 2 --devices cuda:2,cuda:3
done
```

## C9/C10：阶段增量

只评估同一新Base的Phase1末状态、Phase2拼接末状态；完整三阶段复用C0。不新增训练、不改写Base权重。

```bash
set -e
for ablation_name in phase1 phase2; do
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/evaluate.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir outputs/v4/stage3/token_attention/train --split valid --fold 1 2 3 4 5 --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/valid"
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/evaluate.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir outputs/v4/stage3/token_attention/train --split test --ensemble-folds --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/test"
done
```

## E1～E5：辅助监督

```bash
set -e
for ablation_name in no_unimol no_electronic no_partial_charge no_alignment no_descriptor_loss; do
  CUDA_VISIBLE_DEVICES=2 python scripts/stage1/train.py --config "configs/ablations/token_attention_3407/auxiliary/${ablation_name}/stage1.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage1/train"
  CUDA_VISIBLE_DEVICES=2 python scripts/stage2/prepare.py --config "configs/ablations/token_attention_3407/auxiliary/${ablation_name}/stage2.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage2/prepare"
  python scripts/stage2/train.py --config "configs/ablations/token_attention_3407/auxiliary/${ablation_name}/stage2.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage2/train" --device cuda:2
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/prepare.py --config "configs/ablations/token_attention_3407/auxiliary/${ablation_name}/stage3.yaml" --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/prepare"
  python scripts/stage3/train.py --config "configs/ablations/token_attention_3407/auxiliary/${ablation_name}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/train" --max-parallel 2 --devices cuda:2,cuda:3
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/evaluate.py --config "configs/ablations/token_attention_3407/auxiliary/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/token_attention_3407/${ablation_name}/stage3/train" --split valid --fold 1 2 3 4 5 --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/valid"
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/evaluate.py --config "configs/ablations/token_attention_3407/auxiliary/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/token_attention_3407/${ablation_name}/stage3/train" --split test --ensemble-folds --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/test"
done
```

## 核心训练后的验证与测试

```bash
set -e
for ablation_name in no_stage1 no_stage2 no_global no_group no_private no_smiles no_graph; do
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/evaluate.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/token_attention_3407/${ablation_name}/stage3/train" --split valid --fold 1 2 3 4 5 --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/valid"
  CUDA_VISIBLE_DEVICES=2 python scripts/stage3/evaluate.py --config "configs/ablations/token_attention_3407/core/${ablation_name}/stage3.yaml" --checkpoint-dir "outputs/ablations/token_attention_3407/${ablation_name}/stage3/train" --split test --ensemble-folds --output "outputs/ablations/token_attention_3407/${ablation_name}/stage3/test"
done
```

默认一次调用输出实验及两模拟独立报告；验证五折、测试五模型原单位集成。末状态固定，验证不选模。Stage2只有训练产物，不单独发布榜单。命令中`--device/--devices`使用物理GPU编号，会重新设置CUDA mask；不要在Slurm分配环境照搬物理编号。

## 汇总、恢复与回传

只包含新Base及本套件已完成结果，不扫描历史消融根或重复尝试。bootstrap可选，固定seed42、10000次；不追加seed重复。

```bash
python scripts/benchmarks/summarize.py --input outputs/v4/stage3/token_attention/valid outputs/v4/stage3/token_attention/test outputs/ablations/token_attention_3407 --output summary_token_attention_3407_ablations --ablation-analysis-output summary_token_attention_3407_analysis --ablation-authority-config configs/v4/stage3/base.yaml
```

主表22实验、六GROUP及规模分层，两模拟独立。目录存在时不覆盖；只对身份一致的完整轮中断按原入口追加明确的`--resume`，不重导出C1、不修改历史日志。完整产物范围及源/目标SHA、权重严格重载按[跨设备回传规则](v4-runbook.md#跨设备训练产物回传)执行，统一归档4090。
