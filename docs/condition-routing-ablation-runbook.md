# PRIVATE FiLM与工况Task Gate消融

定义见[ADR-0133](adr/0133-private-film-and-conditioned-task-gate.md)。两项分别以5轮Stage1来源I2为参照，不叠加、不等待或混用10轮复跑。

| 编号 | 唯一指定变化 |
|---|---|
| F1 | GLOBAL／GROUP保留条件token，仅PRIVATE改用FiLM |
| G1 | 专家保持I2，Task Gate增加原物理条件与有效位，新增条件权重零初始化 |

## 来源与运行

在4090仓库根目录使用`ilume`环境，核对现有来源：

- `outputs/v4/stage1/base/train/stage1_encoder.pt`及匹配`outputs/v4/stage1/base/prepare/artifacts`，是5轮编码器。
- `outputs/v4/stage2/token_attention/prepare/artifacts`。
- `outputs/v4/stage3/token_attention/prepare/artifacts`。
- I2已完成的五折验证`outputs/ablations/partner_injection_3407/I2/stage3/valid/`。

两项只读共用prepare，无需重prepare或重训Stage1。每项必须训练自己的Stage2，不借用旧final；保持3407、seed42、007/032预算和原batch。首次输出根须不存在。下面示例使用cuda:1、cuda:3，执行前确认均空闲；其他设备仅调整仓库根与设备编号，代码由用户自行git pull。

在tmux窗口执行，括号内失败只停止本段：

```bash
cd /data/pengs/ILUME
conda activate ilume
(
set -e
for ablation_id in F1 G1; do
  python scripts/stage2/train.py \
    --config "configs/ablations/condition_routing_3407/${ablation_id}/stage2.yaml" \
    --output "outputs/ablations/condition_routing_3407/${ablation_id}/stage2/train" \
    --device cuda:3

  python scripts/stage3/train.py \
    --config "configs/ablations/condition_routing_3407/${ablation_id}/stage3.yaml" \
    --fold 1 2 3 4 5 \
    --output "outputs/ablations/condition_routing_3407/${ablation_id}/stage3/train" \
    --devices cuda:1,cuda:3 --max-parallel 2

  CUDA_VISIBLE_DEVICES=3 python scripts/stage3/evaluate.py \
    --config "configs/ablations/condition_routing_3407/${ablation_id}/stage3.yaml" \
    --checkpoint-dir "outputs/ablations/condition_routing_3407/${ablation_id}/stage3/train" \
    --split valid --fold 1 2 3 4 5 \
    --output "outputs/ablations/condition_routing_3407/${ablation_id}/stage3/valid" \
    --study-id condition_routing_3407
done
)
```

Stage2单卡，Stage3两卡并行两折，验证固定cuda:3，不叠加外层并发。中断只恢复对应阶段：Stage2原命令追加`--resume outputs/ablations/condition_routing_3407/F1/stage2/train`（按实际编号替换），Stage3追加布尔`--resume`；不重跑整个首次循环，不删除现场。评估失败使用新的空输出目录。

默认验证同时包括22实验五折与两模拟独立五模型集成。本轮不自动测试；测试不用于选择配置。F1的FiLM恒等初始化不意味着起始预测与I2相同，因为PRIVATE注意力不再读取条件token；G1起始门控与I2相同，随后从训练学习工况依赖。

## 汇总与回传

全部验证完成后，使用未占用的新报告目录。以I2为主参照，A2作为F1的辅助参照：

```bash
python scripts/benchmarks/summarize.py \
  --input outputs/ablations/partner_injection_3407/I2/stage3/valid outputs/ablations/l2_factorial_3407/A2/stage3/valid outputs/ablations/condition_routing_3407/{F1,G1}/stage3/valid outputs/ablations/condition_routing_3407/{F1,G1}/stage2/train \
  --output outputs/ablations/condition_routing_3407/reports/validation \
  --ablation-analysis-output outputs/ablations/condition_routing_3407/reports/analysis \
  --ablation-authority-config configs/v4/stage3/group_l2_partner.yaml \
  --ablation-base-run ilume_v4_partner_group_private@study:partner_injection_3407
```

参照study ID必须与既有I2产物一致，不能重新标记历史结果。只比较完整五折和同数据／归一尺度的22任务总分，保留全部退化、参数量及成本；两模拟独立。缺产物时先补齐实际失败阶段，不能将缺失分数视为零。

在3090或其他设备完成后，将`outputs/ablations/condition_routing_3407/`完整训练目录、权重、恢复检查点、日志、配置及身份回传4090，按[v4回传规则](v4-runbook.md#跨设备训练产物回传)核验，不只传报告，不自动删除远端来源。
