# ILUME workspace rules

## 科学边界

仓库重构不得改变现役数据筛选与 split、tokenizer、descriptor、fingerprint、masking、模型、loss、优化顺序、验证指标、早停和 Stage 1/2/3 数值训练行为。发现科研问题时单独记录，不在结构整理中顺手修改。

Stage 1 现役合同以 ADR-0013/0014/0015/0017 为准：自然频率全量 epoch、cation/anion/molecule 的 element-level loss 权重 2/2/1、五模态等权、global batch 128、默认 eager 执行、单一 Base、原生单卡/DDP、corpus/checkpoint 既定 kind 与 `format_version=2`。只支持 epoch 边界恢复；不得恢复 45/45/10 coverage sampler、augmentation multiplier、多容量正式配置、mid-epoch cursor/RNG 恢复或旧 artifact/checkpoint 兼容。

Stage 2 现役合同以 ADR-0019 为准：九个 catalog task 共享 `ObjectEncoder`，registry 与 Stage 1 派生的 model contract 分离；保留 QM 部分标签 mask、entity teacher、逐行完整覆盖和只作用于 physics loss 的 task compensation。第一 epoch object/interaction 走 cached-CLS 快路径，atom task仍运行冻结 Stage 1 获取 fusion atom states；后四个 epoch 联合微调。Object v3 强制 `gradient_accumulation_steps == 1`，只支持完整 epoch checkpoint 恢复、三组 LR、固定 TF32/CUDA fused AdamW 和始终开启的 artifact 校验；partial charge 使用带显式 fallback audit 的 MOL2 graph mapping。旧 v2 不迁移，不恢复双实体编码器、体系采样、渐进解冻、early stopping、best/last 或 step checkpoint。Stage 3 保持 `il21`/`aux6` 双域的模型与训练状态完全隔离；现役决定以 ADR-0011/0012 为准。Stage 3 到 `stage2_encoder.pt` 的表示迁移尚未完成，prepare 必须在写 artifact 前明确拒绝 Object v3 checkpoint 与 encoder artifact。

## 结构与入口

- 实现只位于 `src/common`、`src/stage1`、`src/stage2`、`src/stage3`。
- `common` 只接收至少两个 Stage 实际复用的原子功能；禁止建立 `utils.py`。
- Stage 可以导入其他 Stage 的公开 contract，但 `src/stageN` 禁止从其他 Stage 导入 `_private_symbol` 或使用 `import *`。
- 唯一运行入口是 `scripts/stage{1,2,3}/*.py`；不恢复 console CLI、shell runner、smoke 或 matrix runner。
- 科研参数与 prepare 执行参数都写入完整自包含 YAML；`preparation` 不进入实验身份。正式 v1 配置位于 `configs/v1`；首次真实消融时再建立 `configs/experiments/<stage>`。
- `--output`、`--resume`、Stage 3 fold 和 evaluation selector 是运行参数，不进入科研 config schema。

## 数据、输出与恢复

- 数据本体不进入 Git；prepare 自动写 `data/stage*/metadata.json`。
- 每次操作必须冻结 `run_config.yaml`，并写公开安全的 `metadata.json` 和成功后的 `summary.json`；只有 reusable Stage 1 prepare 可在科研配置一致时刷新 `preparation` 执行参数。禁止用户名、hostname、私有绝对路径。
- 新 train/evaluate 输出不可覆盖；resume 必须显式且严格校验阶段、fold/domain、有效配置、step/epoch/cycle、optimizer、scheduler 与 AMP。Stage 1 仅从完整 epoch 恢复并允许改变 world size；Stage 2 严格校验 Object v3 的完整 epoch、registry、model contract、RNG、数据、teacher、任务规模、数学精度与 optimizer implementation 合同；Stage 3 继续严格校验其 RNG 与 sampler/cursor 合同。
- 保留 prepared artifact 自身的 SHA 和完整性校验；不得恢复跨阶段 checkpoint SHA lineage 强制绑定。
- 周期 checkpoint 全部保留。Stage 1/3 以 `last.pt` 表示最新完整恢复状态；Stage 2 只保存不可覆盖的 `checkpoint_epoch_XXXXX.pt`，最终模型固定为 epoch 5。

## 验证与清理

修改后运行 `pytest -q`，并按风险检查七个 script 的 `--help`、`compileall`、`git diff --check`、Markdown 链接、ignore 白名单与旧入口搜索。只使用临时小数据；除非用户明确要求，不执行正式 prepare、教师缓存、训练或五折 evaluation。

`trash/` 不进入 Git。移动旧 artifact、旧 YAML、未消费数据或删除机器缓存前，必须先报告精确文件数、大小、目标和冲突策略，并等待用户明确确认。不得覆盖、重排或删除既有 `trash/` 内容。
