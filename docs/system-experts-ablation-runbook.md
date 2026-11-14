# IL／非IL体系隔离专家消融S2

定义见[ADR-0128](adr/0128-system-isolated-experts.md)。S2以I2为参照，共享冻结Stage1；含IL体系与非IL体系使用独立的GLOBAL／GROUP专家及共享条件编码参数，PRIVATE等逐任务参数保持独立。两套L2 GLOBAL保留且仅接收主体，GROUP／PRIVATE L2按任务注入溶质。hydration只输入本身，不新增水实体。

分流按任务声明的阳／阴离子双槽与单实体主体确定，不按溶质是否带电。IL solvation GROUP没有Stage2预训练数据，在Stage3本折随机初始化，不复制非IL权重。保留原owner样本加权分母、裁剪及训练日历，参数隔离不等同于两套独立优化协议；参数量及预训练覆盖差异见ADR。

## 来源与运行

在4090仓库根目录使用`ilume`环境。只读复用：

- `outputs/v4/stage1/base/train/stage1_encoder.pt`及匹配的Stage1特征。
- `outputs/v4/stage2/token_attention/prepare/artifacts`。
- `outputs/v4/stage3/token_attention/prepare/artifacts`。

配对配置为`configs/ablations/partner_injection_3407/S2/stage2.yaml`和`stage3.yaml`，固定3407、训练seed42、原折派生规则及007/032预算。S2需要自己的完整Stage2来源，不能借用I2／S1的final，也不修改旧输出。首次运行各输出目录须尚不存在；三个命令按成功顺序执行：

```bash
conda activate ilume
set -e
python scripts/stage2/train.py --config configs/ablations/partner_injection_3407/S2/stage2.yaml --output outputs/ablations/partner_injection_3407/S2/stage2/train --device cuda:0
python scripts/stage3/train.py --config configs/ablations/partner_injection_3407/S2/stage3.yaml --fold 1 2 3 4 5 --output outputs/ablations/partner_injection_3407/S2/stage3/train --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
python scripts/stage3/evaluate.py --config configs/ablations/partner_injection_3407/S2/stage3.yaml --checkpoint-dir outputs/ablations/partner_injection_3407/S2/stage3/train --split valid --fold 1 2 3 4 5 --output outputs/ablations/partner_injection_3407/S2/stage3/valid --study-id partner_injection_3407
```

Stage2单卡，Stage3最多并行四折，不叠加外层并行。无需重跑prepare或其他变体；新增工作量为1次Stage2和5个Stage3折。正式训练由用户启动。

身份一致的中断输出可以恢复：Stage2原命令追加`--resume outputs/ablations/partner_injection_3407/S2/stage2/train`；Stage3追加布尔`--resume`。保持完整轮与日志检查，不猜测、不截断。评估失败使用新的空输出目录，不覆盖已有现场。

## 审计与比较

检查每折`resolved_training_plan.json`中的`ablation.system_experts`：`task_systems`为任务分支，`shared_parameter_counts`为共享模块按分支计数，`random_il_solvation_parameters`为未从Stage2迁移的全部IL solvation参数。总参数、owner预算和实际更新仍由原训练成本与性能日志报告；无本分支输入的参数不应产生梯度或更新。

完整独立五折验证后，与I2做配对比较：

```bash
python scripts/benchmarks/summarize.py --input outputs/ablations/partner_injection_3407/{I2,S2}/stage3/valid outputs/ablations/partner_injection_3407/{I2,S2}/stage2/train --output outputs/ablations/partner_injection_3407/reports/system_validation --ablation-analysis-output outputs/ablations/partner_injection_3407/reports/S2_vs_I2 --ablation-authority-config configs/ablations/partner_injection_3407/I2/stage3.yaml --ablation-base-run ilume_v4_partner_group_private@study:partner_injection_3407
```

仅以22实验任务等权、五折平均nMAE比较；同时保留solvation、gas_solubility、transfer_organic、hydration及其他任务的结果，不按任务拼装变体。两模拟独立报告，Stage2只作成本与来源审计，不进入准确率榜单。I2来源或study ID不同须先读取既有summary核对；汇总输出已存在时使用新的空报告目录。

S2同时改变共享范围、容量及IL solvation预训练覆盖，不能单独归因为“更晚交互”或等预算最优。固定末轮，不增加seed、不根据测试更换配方。其他设备运行时，完整配置、日志、身份、权重及恢复检查点按[跨设备回传规则](v4-runbook.md#跨设备训练产物回传)归档4090。
