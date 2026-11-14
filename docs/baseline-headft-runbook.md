# 冻结预训练模块的六模型 baseline

本轮按[ADR-0129](adr/0129-frozen-pretrained-baselines.md)，冻结通用预训练模块，训练原有任务侧模块。六模型各113次训练，合计678次；正式训练由用户启动，既有FLft微调结果继续保留。本次实现验证不执行正式prepare、训练或模型评估。

## 冻结范围与解读

| 模型配置名 | 可训练任务模块 |
|---|---|
| molformer | fusion、classifier |
| ilbert | TextCNN、predictor |
| spmm | predictor |
| llasmol | predictor；Mistral与已发布LoRA一起冻结 |
| aionopedia | 当前任务使用的新增条件投影及segment、fc_out；完整官方多模态模块与已发布LoRA一起冻结 |
| iltransr | TextCNN、predictor |

MLP、D-MPNN、AIFC没有通用预训练权重，本轮不比较随机特征；不纳入XGBoost或ILUME。保留原来的模型结构、输入和条件通路、种子及10轮任务侧配方；不统一任务头宽度或学习率。ILBERT使用训练行条件z-score，ILTransR保持已披露的传导式条件统计。LlaSMol仍用现有NF4/BF16数值实现，但不再更新LoRA。

因此该结果是“冻结预训练模块、训练原生任务模块”的比较，不能称为统一线性探针或仅由表示质量决定的排名；新报告公开任务模块和可训练参数量。

## 环境与启动

沿用六模型已有的独立环境、hash-lock和通用预训练资产，不安装新依赖或下载新权重。环境准备步骤见[README](../README.md#baselines-and-ablations)。在目标服务器先拉取已提交的代码和配置，数据及预训练资产仍需按原步骤单独传输。

在4090使用仓库`/data/pengs/ILUME`；3090使用`/mnt/backup8t/pengs/ILUME`。进入对应仓库后激活主环境，sweep会调用模型自己的独立环境。下例GPU编号仅为示例，执行前改为实际空闲卡。

单模型，先运行训练/验证：

```bash
conda activate ilume
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/Headftv4_3407/molformer.yaml \
  --output outputs/benchmarks/Headftv4_3407/molformer \
  --devices cuda:0 --max-workers 1 --validation-only
```

正式固定配方核验后，在相同输出根去掉`--validation-only`补齐测试评估，已完成且身份/完整性检查通过的训练与验证会跳过：

```bash
python scripts/benchmarks/sweep.py \
  --config configs/benchmarks/Headftv4_3407/molformer.yaml \
  --output outputs/benchmarks/Headftv4_3407/molformer \
  --devices cuda:0 --max-workers 1
```

运行全部六模型，模型之间顺序执行，每个模型内部两张卡各一个独立任务，不使用DDP：

```bash
for model in molformer ilbert spmm llasmol aionopedia iltransr; do
  python scripts/benchmarks/sweep.py \
    --config "configs/benchmarks/Headftv4_3407/${model}.yaml" \
    --output "outputs/benchmarks/Headftv4_3407/${model}" \
    --devices cuda:0,cuda:1 --max-workers 2 || break
done
```

沿用task/fold/attempt布局。不能在同一模型输出根同时启动多个sweep；失败在新attempt从头重跑，没有部分训练resume。不要指定旧FLft目录，不使用复用清单。

## 审计与回传

每个checkpoint的完整性清单包含`frozen_audit.json`，记录参数白名单、优化器成员、可训练参数量和冻结状态前后指纹。冻结模块eval、任务模块train；AIonopedia条件模块仍能通过冻结融合模块获得梯度。冻结状态变化、参数归属错误、审计缺失/篡改或跨协议加载均明确失败。

回传完整运行与权重，不仅是summary。按[v4回传规则](v4-runbook.md#跨设备训练产物回传)，例如在4090拉取3090的新结果：

```bash
cd /data/pengs/ILUME
rsync -aH --ignore-existing --partial-dir=.rsync-partial --info=progress2 \
  3090:/mnt/backup8t/pengs/ILUME/outputs/benchmarks/Headftv4_3407/ \
  outputs/benchmarks/Headftv4_3407/
rsync -rHcni \
  3090:/mnt/backup8t/pengs/ILUME/outputs/benchmarks/Headftv4_3407/ \
  outputs/benchmarks/Headftv4_3407/
```

校验命令应无差异输出；若已有同名冲突文件，不覆盖、不删除，先核对来源。远端原件保留。既有资产按原环境/资产合同归档。

## 汇总

六模型全部训练及评估完成并回传后，在4090主环境执行：

```bash
python scripts/benchmarks/summarize.py \
  --baseline-config configs/benchmarks/Headftv4_3407/reports/summary_22.yaml \
  --output outputs/benchmarks/reports/Headftv4_3407_22
```

报告目录必须不存在；以后重新生成需换新目录。该配置只接受本轮六个完成sweep的明确来源，拒绝缺折、缺冻结审计、旧微调替代及仅验证未测试的结果。

`overview.md`展示任务模块/参数量范围及四张榜单；`stage3_validation/`覆盖22实验五折，`stage3_test/`覆盖实际有测试样本的10实验，另外两节分别列三模拟验证/测试。无测试集任务显式列出，不伪造结果。实验与模拟分开任务等权，使用完整训练集共同尺度；不包含ILUME，不混入旧FLft报告。

`source_manifest.json`保存原训练身份、冻结审计和已核验文件SHA；`summary.json`、`manifest.json`绑定本轮比较身份和报告完整性。报告只读取原单位预测重算指标，不调用模型前向，不修改历史报告。
