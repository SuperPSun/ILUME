# ILUME workspace rules

## 科学边界

仓库重构不得改变数据筛选与 split、tokenizer、descriptor、fingerprint、masking、45/45/10 sampling、模型、loss、优化顺序、验证指标、早停和 Stage 1/2/3 数值训练行为。发现科研问题时单独记录，不在结构整理中顺手修改。

Stage 2 保持五任务、PairEncoder、任务内体系采样、部分标签 mask 与 10% 渐进解冻。Stage 3 保持 `il21`/`aux6` 双域的模型与 optimizer、scheduler、AMP、RNG、BN、早停和选优状态完全隔离；现役决定以 ADR-0011/0012 为准。

## 结构与入口

- 实现只位于 `src/common`、`src/stage1`、`src/stage2`、`src/stage3`。
- `common` 只接收至少两个 Stage 实际复用的原子功能；禁止建立 `utils.py`。
- Stage 可以导入其他 Stage 的公开 contract，但 `src/stageN` 禁止从其他 Stage 导入 `_private_symbol` 或使用 `import *`。
- 唯一运行入口是 `scripts/stage{1,2,3}/*.py`；不恢复 console CLI、shell runner、smoke 或 matrix runner。
- 科研参数只写入完整自包含 YAML。正式配置位于 `configs/formal`；首次真实消融时再建立 `configs/experiments/<stage>`。
- `--output`、`--resume`、Stage 3 fold 和 evaluation selector 是运行参数，不进入科研 config schema。

## 数据、输出与恢复

- 数据本体不进入 Git；prepare 自动写 `data/stage*/metadata.json`。
- 每次操作必须冻结 `run_config.yaml`，并写公开安全的 `metadata.json` 和成功后的 `summary.json`；禁止用户名、hostname、私有绝对路径。
- 新 train/evaluate 输出不可覆盖；resume 必须显式且严格校验阶段、fold/domain、有效配置、step/epoch/cycle、optimizer、scheduler、AMP、RNG、sampler/cursor。
- 保留 prepared artifact 自身的 SHA 和完整性校验；不得恢复跨阶段 checkpoint SHA lineage 强制绑定。
- 周期 checkpoint 全部保留，`last.pt` 始终是最新完整恢复状态。

## 验证与清理

修改后运行 `pytest -q`，并按风险检查七个 script 的 `--help`、`compileall`、`git diff --check`、Markdown 链接、ignore 白名单与旧入口搜索。只使用临时小数据；除非用户明确要求，不执行正式 prepare、教师缓存、训练或五折 evaluation。

`trash/` 不进入 Git。移动旧 artifact、旧 YAML、未消费数据或删除机器缓存前，必须先报告精确文件数、大小、目标和冲突策略，并等待用户明确确认。不得覆盖、重排或删除既有 `trash/` 内容。
