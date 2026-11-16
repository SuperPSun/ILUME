# L2交互算子与条件通路四格消融

科研定义见[ADR-0132](adr/0132-l2-operator-condition-factorial.md)。参照为既有I2：注意力＋条件token。新增A1（MLP混合＋条件token）、A2（注意力＋FiLM）、A3（MLP混合＋FiLM）。三项均固定I2的GROUP／PRIVATE第三成分接入和跨体系共享，不改变Stage1、split3407、seed42、五折或007/032预算。

## 来源与运行

从4090仓库根目录执行，主环境为`ilume`。先确认这些配套来源完整：

- `outputs/v4/stage1/base/train/stage1_encoder.pt`及对应`outputs/v4/stage1/base/prepare/artifacts`。
- `outputs/v4/stage2/token_attention/prepare/artifacts`。
- `outputs/v4/stage3/token_attention/prepare/artifacts`。
- 参照I2已有独立五折验证：`outputs/ablations/partner_injection_3407/I2/stage3/valid/fold1`至`fold5`。

不重新prepare，不复制来源或修改metadata／SHA。三个新输出根应不存在；按下列顺序执行，每一步成功后才继续。无需重跑I2。

```bash
conda activate ilume
set -e
for ablation_id in A1 A2 A3; do
  python scripts/stage2/train.py --config "configs/ablations/l2_factorial_3407/${ablation_id}/stage2.yaml" --output "outputs/ablations/l2_factorial_3407/${ablation_id}/stage2/train" --device cuda:0
  python scripts/stage3/train.py --config "configs/ablations/l2_factorial_3407/${ablation_id}/stage3.yaml" --fold 1 2 3 4 5 --output "outputs/ablations/l2_factorial_3407/${ablation_id}/stage3/train" --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-parallel 4
  python scripts/stage3/evaluate.py --config "configs/ablations/l2_factorial_3407/${ablation_id}/stage3.yaml" --checkpoint-dir "outputs/ablations/l2_factorial_3407/${ablation_id}/stage3/train" --split valid --fold 1 2 3 4 5 --output "outputs/ablations/l2_factorial_3407/${ablation_id}/stage3/valid" --study-id l2_factorial_3407
done
```

Stage2单卡，Stage3最多四折并行，不叠加外层并发。以上公开入口同时支持这些研究配置；无需新的调度器或HPO入口。额外GPU占用及存储以实际日志为准，不因MLP名义上更简单而自动放大batch或并行。

中断时只恢复失败的阶段：Stage2在该项原命令后追加`--resume outputs/ablations/l2_factorial_3407/A1/stage2/train`（按实际编号替换）；Stage3追加布尔`--resume`。不从头重跑整个首次执行循环，不删除现场、不跨格借用final；评估失败使用新的空输出目录。

## 汇总与解读

四格完整验证后，使用未占用的输出目录汇总：

```bash
python scripts/benchmarks/summarize.py \
  --input outputs/ablations/partner_injection_3407/I2/stage3/valid outputs/ablations/l2_factorial_3407/{A1,A2,A3}/stage3/valid outputs/ablations/l2_factorial_3407/{A1,A2,A3}/stage2/train \
  --output outputs/ablations/l2_factorial_3407/reports/validation \
  --ablation-analysis-output outputs/ablations/l2_factorial_3407/reports/analysis \
  --ablation-authority-config configs/v4/stage3/group_l2_partner.yaml \
  --ablation-base-run ilume_v4_partner_group_private@study:partner_injection_3407
```

参照的study ID须与I2既有summary一致；若不同，读取实际身份后使用对应run ID，不重新标记I2。报告目录已存在时换新编号。authority只提供一致的22任务及分组，研究模型由其已审计产物读取。

主要查看I2→A1的算子差值、I2→A2的条件通路差值，以及两条边A1→A3和A2→A3；交互量为`Q(A3)-Q(A1)-Q(A2)+Q(I2)`，只描述该固定配方下的四格差异，不作为统计显著性。主表保持全部22任务及五折，两模拟单列；列出参数量、GPU时间与全部退化，不把最佳任务来自不同配方的拼接当作一个模型。

本轮先做验证，不自动执行测试；后续选定配方后再明确测试范围。旧测试集已参与开发反馈，不能将后续测试称为从未触碰的独立终测。

在其他设备运行时，保持相同仓库相对路径，将完整训练结果、配置、身份、日志、全部权重及恢复检查点回传4090，按[v4回传规则](v4-runbook.md#跨设备训练产物回传)核验。不会自动写入3090，其他服务器由用户拉取代码。
