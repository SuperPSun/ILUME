# ILUME Capacity v1 操作手册

本文只给出正式运行命令；实现验收不会执行这些 prepare/train/evaluate。所有命令从仓库
根目录运行。开始前必须确认 Git clean、没有仍在写入的现役 Stage job，并保留全部
`outputs/v1` 与 `summary/`。

## 1. Stage 1 四规模

只用 Base 配置 prepare 一次实验 corpus；S、Base、L、XL 四个训练配置都读取该 artifact，
不得分别 prepare：

```bash
python scripts/stage1/prepare.py \
  --config configs/experiments_v1/stage1/base.yaml \
  --output outputs/experiments_v1/stage1/base/prepare
```

prepare 完整成功后，四个训练命令可分别显式绑定一张同型 GPU：

四份 Capacity Stage 1 YAML 已共同冻结为 global batch 512、LR `4e-4`。这是从原
batch 128 约 20GB 显存占用线性估算得到的 84GB 单卡配置，目标峰值约 80GB；它不是
自动调参，也不适用于 48GB 卡。正式运行前确认目标 GPU 空闲且为同类 84GB 硬件，并在
训练日志中记录实际 peak VRAM、吞吐和稳定性。若出现 OOM/NaN/divergence，停止研究，
不得改 batch、LR、gradient checkpointing 或 horizon 后续跑。

现有共享 corpus 仍可直接复用：它在此前的 prepare run 中记录了 batch 128/LR `1e-4`，
但这两个训练字段不进入 corpus identity。不得仅因本次 batch/LR 变化重新 prepare；反之，
任何不同 global batch 的 Stage 1 checkpoint 都不能 resume。本仓库当前没有 Capacity Stage 1
checkpoint，因此四个训练均从 epoch 0 开始。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage1/train.py --config configs/experiments_v1/stage1/s.yaml --output outputs/experiments_v1/stage1/s/train
CUDA_VISIBLE_DEVICES=1 python scripts/stage1/train.py --config configs/experiments_v1/stage1/base.yaml --output outputs/experiments_v1/stage1/base/train
CUDA_VISIBLE_DEVICES=2 python scripts/stage1/train.py --config configs/experiments_v1/stage1/l.yaml --output outputs/experiments_v1/stage1/l/train
CUDA_VISIBLE_DEVICES=3 python scripts/stage1/train.py --config configs/experiments_v1/stage1/xl.yaml --output outputs/experiments_v1/stage1/xl/train
```

不要并入同一 shell runner；分别启动、监控并确认四者 epoch 15 full validation 与 checkpoint
完整。OOM/NaN 时停止研究，不改 batch 或 gradient checkpointing 后续跑。

## 2. Stage 2 prepare 与 12 runs

先用每个 scale 的 Default 配置依次调用同一 reusable prepare root。第一次物化共享 data，
四次分别补充对应 Stage 1 encoder 的 teacher cache：

```bash
python scripts/stage2/prepare.py --config configs/experiments_v1/stage2/s-default.yaml --output outputs/experiments_v1/stage2/prepare
python scripts/stage2/prepare.py --config configs/experiments_v1/stage2/base-default.yaml --output outputs/experiments_v1/stage2/prepare
python scripts/stage2/prepare.py --config configs/experiments_v1/stage2/l-default.yaml --output outputs/experiments_v1/stage2/prepare
python scripts/stage2/prepare.py --config configs/experiments_v1/stage2/xl-default.yaml --output outputs/experiments_v1/stage2/prepare
```

随后对 `s/base/l/xl` 与 `conservative/default/aggressive` 的 12 个 YAML 分别运行：

```bash
python scripts/stage2/train.py \
  --config configs/experiments_v1/stage2/<scale>-<recipe>.yaml \
  --output outputs/experiments_v1/stage2/<scale>/<recipe>/train
```

每次发布历史 epoch-10 checkpoint、boundary-frozen `stage2_encoder.pt` 和 Stage 2 自身的
`taskwise_refined.pt`；Stage 3 只消费 encoder，Stage 2 validation 不淘汰 candidate。

## 3. Stage 3 prepare、probe 与自动选 recipe

对 12 个 `<scale>-<recipe>` 配置逐一准备：

```bash
python scripts/stage3/prepare.py \
  --config configs/experiments_v1/stage3/probe/<scale>-<recipe>.yaml \
  --output outputs/experiments_v1/stage3/prepare/<scale>-<recipe>
```

每个 candidate 跑 folds 1/2：

```bash
python scripts/stage3/train.py \
  --config configs/experiments_v1/stage3/probe/<scale>-<recipe>.yaml \
  --fold 1 2 \
  --output outputs/experiments_v1/stage3/probe/<scale>/<recipe> \
  --max-parallel 2 \
  --devices cuda:0,cuda:1
```

全部完成后生成只读 probe 报告：

```bash
python scripts/stage3/capacity.py \
  --manifest configs/experiments_v1/stage3/probe-report.yaml \
  --output outputs/experiments_v1/reports/probe
```

报告的 `scale_winners` 已按各 fold taskwise-refined stitched validation 和
Default→Conservative→Aggressive tie-break 选出。
该命令自动汇总主指标、task/group 指标、fold sample-SD 和原始 run 路径；参数量、峰值
显存、吞吐、wall time 与 Stage 1/2 稳定性不会由报告器推断，必须从同类硬件上的训练
日志和监控记录中另行取证，并随人工 decision 一起保留。
人工 Pareto 只能从这四者中选 anchor。创建
`outputs/experiments_v1/decisions/anchor.yaml`：

```yaml
schema_version: 1
kind: anchor
selected_candidate: l-default
selected_config: configs/experiments_v1/stage3/probe/l-default.yaml
probe_report: outputs/experiments_v1/reports/probe/summary.json
reason: >-
  在 validation、fold 波动、参数量、峰值显存、吞吐和 wall time 的 Pareto 证据下选择。
```

`selected_candidate` 和 `selected_config` 必须替换为真实决定；HPO 会验证它确实是一个
scale winner，并拒绝缺失 decision/reason/probe report 的运行。

## 4. Anchor HPO 与五折 confirmation

安装项目的 HPO extra 后，传入 anchor 对应 probe config。控制器使用四张 GPU、同步两
trial wave、SQLite resume、一次 fold retry，并自动完成 Top-5+Base 的 folds 3/4/5：

```bash
python -m pip install -e '.[hpo]'
python scripts/stage3/train.py \
  --config configs/experiments_v1/stage3/probe/l-default.yaml \
  --study-config configs/experiments_v1/stage3/hpo.yaml \
  --output outputs/experiments_v1/stage3/hpo \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

中断后原命令追加 `--resume`。结果位于 `confirmation_report.json`；人工选择一个完成
五折 confirmation 的 trial，并创建 `final-recipe.yaml`：

```yaml
schema_version: 1
kind: final_recipe
hpo_output: outputs/experiments_v1/stage3/hpo
trial_number: 17
reason: >-
  根据五折 stitched validation 主指标、fold sample-SD、per-task 指标与完整曲线选择。
scale_configs:
  s: configs/experiments_v1/stage3/probe/s-default.yaml
  base: configs/experiments_v1/stage3/probe/base-conservative.yaml
  l: configs/experiments_v1/stage3/probe/l-default.yaml
  xl: configs/experiments_v1/stage3/probe/xl-aggressive.yaml
seed_output_root: outputs/experiments_v1/stage3/seed-robustness
formal_output_root: outputs/experiments_v1/stage3/formal
```

`trial_number` 与四个 `scale_configs` 必须替换为真实决定。物化完整 seed/formal YAML：

```bash
python scripts/stage3/capacity.py \
  --recipe-decision outputs/experiments_v1/decisions/final-recipe.yaml \
  --output-dir outputs/experiments_v1/decisions/final-recipe
```

## 5. Seed robustness

seed 42 的 folds 1/2 已由 HPO 复用。其余四个 seed 分别运行：

```bash
python scripts/stage3/train.py \
  --config outputs/experiments_v1/decisions/final-recipe/seed/seed<seed>.yaml \
  --fold 1 2 \
  --output outputs/experiments_v1/stage3/seed-robustness/seed<seed> \
  --max-parallel 2 \
  --devices cuda:0,cuda:1
```

其中 `<seed>` 依次为 `10042/20042/30042/40042`。生成 robustness 报告：

```bash
python scripts/stage3/capacity.py \
  --manifest outputs/experiments_v1/decisions/final-recipe/robustness-report.yaml \
  --output outputs/experiments_v1/reports/seed-robustness
```

人工复核并写 `decisions/seed-acceptance.yaml`，至少包含 `schema_version: 1`、
`kind: seed_acceptance`、`accepted: true|false`、报告路径和非空 reason。`false` 时停止，
不得运行 formal comparison。

## 6. 四规模 50-epoch formal comparison 与 test

只有 seed acceptance 为 true 后，四规模分别运行生成的 self-contained formal YAML：

```bash
python scripts/stage3/train.py \
  --config outputs/experiments_v1/decisions/final-recipe/formal/<scale>.yaml \
  --fold 1 2 3 4 5 \
  --output outputs/experiments_v1/stage3/formal/<scale> \
  --max-parallel 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

四者完成后生成 validation 汇总：

```bash
python scripts/stage3/capacity.py \
  --manifest outputs/experiments_v1/decisions/final-recipe/formal-report.yaml \
  --output outputs/experiments_v1/reports/formal-validation
```

再把同类硬件上的参数量、峰值显存、吞吐和 wall time 证据附入决策记录。在查看 test
前写入 `decisions/main-scale.yaml`，记录所选 scale、validation/resource Pareto 理由和
formal report。然后才允许对四个 scale 各执行一次 test ensemble：

```bash
python scripts/stage3/evaluate.py \
  --config outputs/experiments_v1/decisions/final-recipe/formal/<scale>.yaml \
  --checkpoint-dir outputs/experiments_v1/stage3/formal/<scale> \
  --split test \
  --ensemble-folds \
  --study-id capacity-v1-<scale> \
  --output outputs/experiments_v1/stage3/test/<scale>
```

Test 只发布四点 capacity trend，不得修改 main scale、recipe 或 refined artifact。任何正式命令
失败时先保存原日志和 metadata；同配置最多原样重跑一次，不做动态 rescue。
