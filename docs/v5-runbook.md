# v5 Stage2 / Stage3 运行手册

Stage1 沿用 [v4 手册](v4-runbook.md)与既有 encoder，不需要为本次改动重新预训练。Stage2 正式方案为 Unary + Pair / 五任务，Stage3 为22项实验加两项模拟辅助。合同见 [ADR-0094](adr/0094-v5-unary-pair-five-task-stage2.md)。旧输出只读，所有新目录使用 `outputs/v5/`。

## 前提与对照

使用已安装的 `ilume` 环境，不安装依赖或下载权重。需先补齐 `outputs/v4/stage1/base/train/stage1_encoder.pt` 及它对应的 `outputs/v4/stage1/base/prepare/artifacts`；当前配置指定的正式 encoder 尚不存在。源数据及 charge resources 不能在 prepare/train/evaluate 之间替换。现有过期 `data/stage*/metadata.json` 由下方正式 prepare 生成，不手改 SHA。

| 变量 `NAME` | Stage2/Stage3 配置相对路径 | 编码器 | Stage2任务 |
|---|---|---|---|
| base | base.yaml | Unary + Pair | 5 |
| transformer5 | controls/transformer5.yaml | Transformer | 5 |
| transformer9 | controls/transformer9.yaml | Transformer | 9 |
| unary_pair9 | controls/unary_pair9.yaml | Unary + Pair | 9 |

四组均使用22实验、两模拟辅助；分别 prepare/train 到自己的输出，不复用其他组的 Stage3缓存/checkpoint。九任务四项电子来源保持 catalog 的 `stage1/properties/` 路径、原 train/valid/test 与资源来源。缺失即停止。组内架构比较具有相同数据暴露/更新数；五对九任务比较包含预算变化，查看 `performance.jsonl` 的参数量、epoch耗时以及 `metrics.jsonl` 的实际 optimizer_updates。

## 按顺序执行

下面是用户正式运行的完整命令；实现验证不会自动执行。选择一组，逐步检查成功后继续。默认只占一张已分配的 GPU、串行五折；若分配多张 GPU，可显式改变已有 Stage3入口的 `--max-parallel` 和 `--devices`，不超过实际资源。

```bash
set -euo pipefail
cd /data/pengs/ILUME
source /home/pengs/softwares/conda/etc/profile.d/conda.sh
conda activate ilume
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
NAME=base
case "$NAME" in
  base) CFG=base ;;
  transformer5|transformer9|unary_pair9) CFG="controls/$NAME" ;;
  *) exit 1 ;;
esac
S2="outputs/v5/stage2/$NAME"
S3="outputs/v5/stage3/$NAME"
C2="configs/v5/stage2/$CFG.yaml"
C3="configs/v5/stage3/$CFG.yaml"

# 1. 完整性/运行版本校验；加载严格核对 Stage1 来源、特征和 tensor hash。
python - <<'PY'
from stage1.model import load_stage1_model
loaded = load_stage1_model(
    "outputs/v4/stage1/base/train/stage1_encoder.pt",
    "outputs/v4/stage1/base/prepare/artifacts", device="cpu", backbone_dropout=0.0,
)
assert loaded.config.is_dual_view and loaded.model.entity_dim == 1024
print("Stage1 encoder and feature contracts verified")
PY

# 2. 新 Stage2 prepare；生成 metadata/SHA 及永久冻结 Stage1 特征缓存。
test ! -e "$S2/prepare"
python scripts/stage2/prepare.py --config "$C2" --output "$S2/prepare"

# 3. 固定10轮；验证完整 final/manifest 和 encoder 身份及严格重载。
test ! -e "$S2/train"
python scripts/stage2/train.py --config "$C2" --output "$S2/train"
python - "$S2/train" <<'PY'
from pathlib import Path
import sys
from stage2.home_artifact import load_home_final
from stage2 import load_frozen_object_encoder
root = Path(sys.argv[1])
payload, model, _ = load_home_final(root / "stage2_final.pt")
encoder = load_frozen_object_encoder(root / "stage2_encoder.pt", device="cpu")
assert payload["kind"] == "ilume_stage2_home_final_v5"
assert encoder.embedding_dim == 1024 and encoder.entity_input_dim == 1241
assert len(model.registry.task_ids) in (5, 9)
print("Stage2 final, manifest and encoder verified")
PY

# 4. 配对 Stage3 prepare；22任务、类别词表、train-fold统计及 Stage2来源校验。
test ! -e "$S3/prepare"
python scripts/stage3/prepare.py --config "$C3" --output "$S3/prepare"

# 5. 三阶段五折；只使用已分配的 cuda:0。
test ! -e "$S3/train"
python scripts/stage3/train.py --config "$C3" --fold 1 2 3 4 5 \
  --output "$S3/train" --max-parallel 1 --devices cuda:0

# 6. 实验 validation 五折、test 五模型 ensemble（hydration 无 test）。
python scripts/stage3/evaluate.py --config "$C3" --checkpoint-dir "$S3/train" \
  --split valid --fold 1 2 3 4 5 --output "$S3/valid"
python scripts/stage3/evaluate.py --config "$C3" --checkpoint-dir "$S3/train" \
  --split test --ensemble-folds --output "$S3/test"
# 两项模拟任务 valid/test 都是五模型原单位预测均值。
python scripts/stage3/evaluate.py --config "$C3" --checkpoint-dir "$S3/train" \
  --domain simulation --split valid --ensemble-folds --output "$S3/simulation_valid"
python scripts/stage3/evaluate.py --config "$C3" --checkpoint-dir "$S3/train" \
  --domain simulation --split test --ensemble-folds --output "$S3/simulation_test"

# 7. 只扫描此组新输出；独立发布，避免与旧20任务/四模拟榜单混排。
test ! -e "outputs/v5/summary/$NAME"
python scripts/benchmarks/summarize.py --input "$S3" --output "outputs/v5/summary/$NAME"
```

只在确实需要继续中断作业时按现有入口显式使用 `--resume`；必须保留同一配置、源 SHA、fold/owner/训练状态和指标尾部。不要通过修改 metadata、kind、format、任务名称或宽松加载复用其他组/历史产物。

## 数据核验结论与限制

当前 catalog 的 gas_solubility 包含多种气体，不能视为旧 x_co2 的同义名称；本版本已按明确确认改变目标。新增水活度和焓任务有实际 catalog/分折来源，焓 phase 当前只允许 `Liquid | Gas`。hydration 继续 random/cv1 且无 test，其余使用 catalog 的 system split。九任务所需28,214个电荷文件已通过大小/SHA检查，正式 prepare 仍严格验证资源与映射。参考压缩包不替代实时 catalog 和文件身份。

当前九任务train/valid共18个来源均存在；缺少test的density、heat_capacity、QM和transfer_organic不参与本协议的模拟test报告，两项辅助任务的train/valid/test齐全。22实验的110个分折文件存在，当前10项有test文件，其余任务保留五折validation，不生成额外test划分；catalog的实验test计数字段为空，不据此推测缺失文件的科学意图。当前正式运行的未满足前提是配置指定路径的 Stage1 encoder；旧 metadata 也需上述新 prepare 正常重建。本次实现没有启动正式 GPU训练、正式五折或覆盖历史输出。

历史baseline的20任务训练配方不扩展，模拟四任务只适配catalog现有来源位置。其旧实验authority仍要求x_co2；当前catalog缺少这一历史目标，运行旧实验baseline需要对应历史catalog/数据，不得用gas_solubility替代。两模拟v5与四模拟历史结果不能混入同一summary。
