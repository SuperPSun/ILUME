# 第三成分注入与离子／中性分子独立编码消融手册

科研定义见[ADR-0127](adr/0127-partner-injection-and-typed-encoding-ablations.md)。本套件独立于[现役核心消融](v4-ablation-runbook.md)，复用当前Stage1与token_attention prepare，不改Base、不重训Stage1、不新增训练seed。以下正式命令由用户运行。

## 变体与来源

| 编号 | 配置目录 | 第三成分进入的L2分支 | 第三成分编码 |
|---|---|---|---|
| I0 | 既有`configs/v4/stage{2,3}/base.yaml` | 原GLOBAL／GROUP／PRIVATE | 原L1编码、路由及残差；完整方案参照 |
| I1 | 既有`configs/v4/stage{2,3}/group_l2_partner.yaml` | GROUP | 仅原GROUP投影及归一化 |
| I2 | `configs/ablations/partner_injection_3407/I2/` | GROUP＋PRIVATE | 原投影，PRIVATE用主体GROUP路由加权 |
| I3 | `configs/ablations/partner_injection_3407/I3/` | GLOBAL＋GROUP＋PRIVATE | 原投影，GLOBAL使用自身投影 |
| I4 | `configs/ablations/partner_injection_3407/I4/` | GROUP | 对第三成分独立执行原GROUP L1注意力 |
| I5 | `configs/ablations/partner_injection_3407/I5/` | PRIVATE | 与I2相同的GROUP投影、归一化与主体路由加权汇聚 |
| S1 | `configs/ablations/partner_injection_3407/S1/` | GROUP | 离子／中性独立投影与L1注意力，再于GROUP L2交互 |

各变体固定I1的主体L1、门控与残差边界。比较I1→I2、I2→I3、I1→I4、I4→S1；I5与I2比较关闭GROUP接收，与I1比较仅PRIVATE／仅GROUP接收。I0不作为纯注入时机对照。S1按原形式电荷角色分流，槽位0的中性溶剂也走中性分支，带电溶质走离子分支；所有对象在L1仍分别编码。

运行前已有来源应为：

- Stage1：`outputs/v4/stage1/base/train/stage1_encoder.pt`及配套`prepare/artifacts`。
- Stage2 prepare：`outputs/v4/stage2/token_attention/prepare/artifacts`。
- Stage3 prepare：`outputs/v4/stage3/token_attention/prepare/artifacts`，含匹配的data/3407与211维来源。

10份YAML已保留上述prepare路径；每项Stage3严格指向自己的新Stage2 final。加载器会校验数据、Stage1、模型合同和张量hash，不复制或修改审计元数据。缺少匹配prepare时先按[v4手册](v4-runbook.md#g2_c01-主模型)补齐，不能跳过来源检查。

## 训练与五折验证

从4090仓库根目录执行。下面依次运行五种变体；每项Stage2使用一张卡，随后Stage3最多同时运行四折。没有外层任务并行，不会与内层折调度叠加。

```bash
conda activate ilume
set -e
for ablation_id in I2 I3 I4 S1 I5; do
  python scripts/stage2/train.py --config "configs/ablations/partner_injection_3407/${ablation_id}/stage2.yaml" --output "outputs/ablations/partner_injection_3407/${ablation_id}/stage2/train" --device cuda:0
  python scripts/stage3/train.py --config "configs/ablations/partner_injection_3407/${ablation_id}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/partner_injection_3407/${ablation_id}/stage3/train" --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
  python scripts/stage3/evaluate.py --config "configs/ablations/partner_injection_3407/${ablation_id}/stage3.yaml" --checkpoint-dir "outputs/ablations/partner_injection_3407/${ablation_id}/stage3/train" --split valid --fold 1 2 3 4 5 --output "outputs/ablations/partner_injection_3407/${ablation_id}/stage3/valid" --study-id partner_injection_3407
done
```

固定新增5次Stage2及25个Stage3折次（含补充I5的1次Stage2及5折）。每个trial仍运行完整预设末轮，不缩短预算、不启用早停。模型合同不同，禁止借用I1或其他变体的Stage2权重与检查点。

### 仅补跑I5

已有I2／I3／I4／S1结果时不重复运行上方循环。I5仅PRIVATE L2读取第三成分，投影／归一化／汇聚仍复用原GROUP参数并按原owner冻结，不新增参数。执行下列三步，新输出根须尚不存在：

```bash
python scripts/stage2/train.py --config configs/ablations/partner_injection_3407/I5/stage2.yaml --output outputs/ablations/partner_injection_3407/I5/stage2/train --device cuda:0
python scripts/stage3/train.py --config configs/ablations/partner_injection_3407/I5/stage3.yaml --fold 1 2 3 4 5 --output outputs/ablations/partner_injection_3407/I5/stage3/train --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
python scripts/stage3/evaluate.py --config configs/ablations/partner_injection_3407/I5/stage3.yaml --checkpoint-dir outputs/ablations/partner_injection_3407/I5/stage3/train --split valid --fold 1 2 3 4 5 --output outputs/ablations/partner_injection_3407/I5/stage3/valid --study-id partner_injection_3407
```

I1已有完整独立验证时直接复用。I0若尚无独立验证，使用原权重补评估；目标目录已存在时跳过此命令，不能覆盖：

```bash
python scripts/stage3/evaluate.py --config configs/v4/stage3/base.yaml --checkpoint-dir outputs/v4/stage3/token_attention/train --split valid --fold 1 2 3 4 5 --output outputs/v4/stage3/token_attention/valid
```

I0／I1只读复用原权重，不复制到新套件。验证默认同时输出22实验五折及两模拟独立五模型集成报告；模拟不进入实验主分数。

## 完整轮恢复

以I2为例，仅在已存在且身份匹配的中断输出上执行：

```bash
python scripts/stage2/train.py --config configs/ablations/partner_injection_3407/I2/stage2.yaml --output outputs/ablations/partner_injection_3407/I2/stage2/train --device cuda:0 --resume outputs/ablations/partner_injection_3407/I2/stage2/train
python scripts/stage3/train.py --config configs/ablations/partner_injection_3407/I2/stage3.yaml --fold 1 2 3 4 5 --output outputs/ablations/partner_injection_3407/I2/stage3/train --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4 --resume
```

Stage3会跳过完整且身份一致的折；不截断日志、不猜测轮中恢复。评估没有`--resume`，失败现场保留；需要重评时使用新的完整评估输出根，汇总只指向其中一套有效完整五折，避免重复折歧义。清理旧现场仍先报告准确范围并确认。

## 验证对照报告

全部五折验证完成后，汇总I0、I1及五种新变体。I1的study标识从既有报告读取，不硬编码来源hash：

```bash
ablation_base_run=$(python -c 'import json; from pathlib import Path; d=json.loads(Path("outputs/v4/stage3/group_l2_partner/valid/fold1/summary.json").read_text()); print("ilume@study:"+d["reporting"]["study_id"])')
python scripts/benchmarks/summarize.py --input outputs/v4/stage3/token_attention/valid outputs/v4/stage3/group_l2_partner/valid outputs/ablations/partner_injection_3407/{I2,I3,I4,S1,I5}/stage3/valid outputs/ablations/partner_injection_3407/{I2,I3,I4,S1,I5}/stage2/train --output outputs/ablations/partner_injection_3407/reports/validation --ablation-analysis-output outputs/ablations/partner_injection_3407/reports/vs_I1 --ablation-authority-config configs/v4/stage3/group_l2_partner.yaml --ablation-base-run "$ablation_base_run"
```

S1与I4另作配对比较；下面的run ID对应上方验证命令固定的study ID：

```bash
python scripts/benchmarks/summarize.py --input outputs/ablations/partner_injection_3407/{I4,S1}/stage3/valid outputs/ablations/partner_injection_3407/{I4,S1}/stage2/train --output outputs/ablations/partner_injection_3407/reports/typed_validation --ablation-analysis-output outputs/ablations/partner_injection_3407/reports/S1_vs_I4 --ablation-authority-config configs/v4/stage3/group_l2_partner.yaml --ablation-base-run ilume_v4_partner_encoded@study:partner_injection_3407
```

命令中的花括号由Bash展开；输入限定验证目录与Stage2成本来源，不混入后续测试或其他评估尝试。只有通过现有完整性与来源检查的评估结果进入snapshot。分析输出`analysis.json`、`task_differences.csv`和`stratified_differences.csv`：含22任务等权主分数、逐任务MAE／RMSE／R²、差值、改善折数、六GROUP／规模类别／体系拓扑分层。两个模拟维持独立榜单。缺模型／任务／折时检查覆盖报告，不将不完整结果当作零误差或完整实验。

按`ionic_liquid`、`ionic_liquid_partner`、`solute_solvent`、`single_molecule`的现有任务拓扑分层，不改变原数据筛选。fold胜负只由五份验证summary计算，不使用训练曲线或测试预测；五折标准差不是多训练seed显著性。

## 训练成本、覆盖与阶段变化

汇总的`training_costs`保留Stage3既有owner参数、更新步数、样本暴露及GPU时间，并从扫描到的Stage2 `performance.jsonl`提取成本、峰值显存和owner参数。Stage2只报告成本及训练覆盖，不发布准确率榜单。Stage2 GPU小时口径为GPU运行时各轮wall time之和，包含轮内验证与保存；不包含初始化等轮外时间，不宣称与Stage3计时范围完全一致。

S1的Stage2 `performance.jsonl`另有`entity_type_training`：`samples`为本轮含该类型的原始行数，同一行可含两种类型；`entity_occurrences`为实体出现数；`optimizer_steps`为本轮该类型参数实际收到梯度并参加优化的联合step数。它们只用于审计，不改变梯度分母。solvation组没有匹配类型样本时，其相应分支不能称为已预训练。

Phase1／2／3的验证变化、门控与冻结诊断沿用每折`phase_1/`、`phase_2/`、`phase_3/`中的`metrics.jsonl`及`diagnostics.jsonl`；各owner的末轮由固定预算确定，不从曲线挑选最佳轮。模型总量与各owner参数绑定在`resolved_training_plan.json`。I2～I5参数量相同，S1较I4下游Stage3增加22.46%，不能将其收益全部归因于分离处理。

## 测试与跨设备回传

先完成固定验证比较，再明确选定配方。若需要测试，例如选定S1后执行：

```bash
python scripts/stage3/evaluate.py --config configs/ablations/partner_injection_3407/S1/stage3.yaml --checkpoint-dir outputs/ablations/partner_injection_3407/S1/stage3/train --split test --ensemble-folds --output outputs/ablations/partner_injection_3407/S1/stage3/test --study-id partner_injection_3407
```

不据测试结果更换变体、预算或训练seed。原测试集已在开发过程中被查看，后续报告披露这一点，不将再次评估称为从未触碰的独立终测。

其他设备运行时，先同步完整代码、配置、数据、Stage1及两阶段prepare，保持相同仓库相对路径；完成后将本套件的完整训练目录、配置、日志、身份和全部权重回传4090。回传及核验遵循[v4手册](v4-runbook.md#跨设备训练产物回传)，不能只同步报告或排除检查点。
