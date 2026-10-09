# ILUME Capacity v1 操作手册

> 历史手册：根据 [ADR-0070](adr/0070-stage3-retire-pcgrad.md)，当前 Stage 3 v1/Capacity 训练与恢复入口已退役，以下 Stage 3 训练命令不可执行；已有最终产物仍可只读评估。

本文只给出正式运行命令；实现验收不会执行这些数据准备/训练/评估。所有命令从仓库
根目录运行。开始前必须确认 Git 干净、没有仍在写入的现役 Stage 作业，并保留全部
`outputs/v1` 与 `summary/`。

## 1. Stage1 Base 选择

只用 Base 配置 prepare 一次实验语料；R1–R8 选择只使用 Base 第10轮检查点：

```bash
python scripts/stage1/prepare.py \
  --config configs/experiments_v1/stage1/base.yaml \
  --output outputs/experiments_v1/stage1/prepare
```

prepare 完整成功后，先训练 Base：

Capacity Stage 1 的 batch、LR 和训练周期以各自 YAML 为准；当前 Base 为全局batch
128、LR `1e-4`、10 轮。其他规模必须读取对应配置，不从显存大小推算或临时改写。
OOM/NaN/发散时停止研究，不改 batch、LR、梯度检查点或训练周期后续跑。

训练字段不进入语料身份，因此数据身份不变时可复用共享语料；检查点恢复
仍严格绑定训练配置。运行前检查目标输出；新训练不得覆盖已有目录。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage1/train.py --config configs/experiments_v1/stage1/base.yaml --output outputs/experiments_v1/stage1/base/train
```

确认 Base 第10轮完整验证与检查点完整。OOM/NaN 时停止研究，不改 batch 或
梯度检查点后续跑。S/L/XL 只在 Base 胜出方案后按另行冻结的晋级配置训练。

## 2. Stage2 Base 选择的数据准备与8次运行

先以 R4 调用一次可复用 prepare 根目录，物化共享数据与 Stage 1 Base 编码器的 teacher 缓存：

```bash
python scripts/stage2/prepare.py --config configs/experiments_v1/stage2/base-e09-r4.yaml --output outputs/experiments_v1/stage2/prepare
```

随后对 R1–R8 的八个 YAML 分别运行：

```bash
python scripts/stage2/train.py \
  --config configs/experiments_v1/stage2/base-e09-rN.yaml \
  --output outputs/experiments_v1/stage2/base/rN/train
```

每次先完成 10 个联合训练轮，再完成 YAML 指定的 10 个四任务仅预测头精调
轮；连续发布轮 1–20 历史检查点、联合训练第10轮边界的
`stage2_encoder.pt` 和 Stage 2 自身的 `taskwise_refined.pt`。Stage 3 只消费编码器，
Stage 2 验证不淘汰候选。

## 3. Stage3 数据准备、探针与自动选择 Base 配方

对八个 `base-rN` 配置逐一准备：

```bash
python scripts/stage3/prepare.py \
  --config configs/experiments_v1/stage3/probe/base-rN.yaml \
  --output outputs/experiments_v1/stage3/prepare/base-rN
```

每个候选跑第1/2折：

```bash
python scripts/stage3/train.py \
  --config configs/experiments_v1/stage3/probe/base-rN.yaml \
  --fold 1 2 \
  --output outputs/experiments_v1/stage3/probe/base/rN \
  --max-parallel 2 \
  --devices cuda:0,cuda:1
```

全部完成后生成只读探针报告：

```bash
python scripts/stage3/capacity.py \
  --manifest configs/experiments_v1/stage3/probe-report.yaml \
  --output outputs/experiments_v1/reports/probe
```

报告的 `scale_winners` 含唯一的 Base 胜出方案，按各折逐任务精调后拼接后的验证
和 R4→R3→R5→R2→R6→R1→R7→R8 同分规则选出。
该命令自动汇总主指标、任务/组指标、折样本-SD 和原始运行路径；参数量、峰值
显存、吞吐、墙钟耗时与 Stage 1/2 稳定性不会由报告器推断，必须从同类硬件上的训练
日志和监控记录中另行取证，并随人工决策一起保留。
超参数搜索、五折确认、种子配置物化和基于试验输出生成 final 配方的能力已于
2026-09-06 退役。不要创建新的 HPO study，也不要尝试用当前代码恢复旧 SQLite study。
既有 HPO 输出只可作为历史证据读取，不再是运行输入。

## 4. 冻结的四规模正式配置

Stage 3 正式配方已经冻结在 `configs/experiments_v1/stage3/formal/{s,base,l,xl}.yaml`。
这些 YAML 是唯一现役 Capacity Stage 3 训练输入；不得从旧 HPO 产物重新生成或修改其
模型、优化器、种子、轮与数据身份。对每个 `<scale>` 先准备数据：

```bash
python scripts/stage3/prepare.py \
  --config configs/experiments_v1/stage3/formal/<scale>.yaml \
  --output outputs/experiments_v1/stage3/formal/<scale>/prepare
```

然后运行五折训练：

```bash
python scripts/stage3/train.py \
  --config configs/experiments_v1/stage3/formal/<scale>.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/experiments_v1/stage3/formal/<scale>/train \
  --max-parallel 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

四个规模全部完成后，使用提交到仓库的固定清单汇总验证：

```bash
python scripts/stage3/capacity.py \
  --manifest configs/experiments_v1/stage3/formal-report.yaml \
  --output outputs/experiments_v1/reports/formal-validation
```

再把同类硬件上的参数量、峰值显存、吞吐和墙钟耗时证据附入决策记录。在查看测试集
前写入 `outputs/experiments_v1/decisions/main-scale.yaml`，记录所选规模、验证/资源
Pareto 理由和正式 report。然后才允许对四个规模各执行一次测试集集成：

```bash
python scripts/stage3/evaluate.py \
  --config configs/experiments_v1/stage3/formal/<scale>.yaml \
  --checkpoint-dir outputs/experiments_v1/stage3/formal/<scale>/train \
  --split test \
  --ensemble-folds \
  --study-id capacity-v1-<scale> \
  --output outputs/experiments_v1/stage3/test/<scale>
```

测试只发布四点容量趋势，不得修改主要规模、冻结配方或 refined 产物。任何正式
命令失败时先保存原日志和元数据；同配置最多原样重跑一次，不做动态 rescue。
