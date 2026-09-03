# ILUME workspace rules

## 科学边界

仓库重构不得改变现役数据筛选与 split、tokenizer、descriptor、fingerprint、masking、模型、loss、优化顺序、验证指标、早停和 Stage 1/2/3 数值训练行为。发现科研问题时单独记录，不在结构整理中顺手修改。

Stage 1 现役主线以 ADR-0039 和 `configs/v2/stage1/base.yaml` 为准，训练执行继续以 ADR-0013/0014/0015/0017 为准，identity/audit 边界以 ADR-0021 为准：SMILES、Graph、固定 217 维 whole-vector RDKit 三模态，四项 reconstruction lambda 等于 1，cation/anion/molecule 的 element-level loss 权重 2/2/1，Fusion token 512，`[CLS; RDKit]` entity 1024，global batch 128、默认 eager、单一 Base、原生单卡/DDP、corpus/checkpoint `format_version=3`。`encode()` 仍只返回 512 维 CLS；跨 Stage 必须使用 `encode_entity()`。只支持 epoch 边界恢复；不得加入 fingerprint、grouped descriptor token、降维 projection、45/45/10 coverage sampler、augmentation multiplier、多容量正式配置、mid-epoch cursor/RNG 恢复或旧 artifact/checkpoint 兼容。legacy `configs/v1` 与 Capacity v1 继续冻结五模态、format v2 合同，不得与 v2 artifact 交叉加载。

Stage 2 现役合同以 ADR-0019、ADR-0025、ADR-0027、ADR-0039 和 `configs/v2/stage2/base.yaml` 为准，identity/audit 边界以 ADR-0021 为准：九个 catalog task 共享 1024 维 `ObjectEncoder`（8 heads、2 layers、FFN 2048）；teacher/student 使用完整 1024 维 entity MSE，Partial Charge 只在 task head 内把 object context 投影到 512 维 atom space。HOMO/LUMO 是各自 pooling cation/anion 的独立 scalar task，并分别使用 pooled train global scaler。Registry 与 Stage 1 派生的 model contract 分离；prepared data不绑定Stage 2 model contract，teacher cache只绑定Stage 1 encoder与entity artifact，model contract只属于训练checkpoint与encoder artifact。QM mask、entity teacher、逐行完整覆盖和 joint phase 只补偿 physics loss 的语义必须保持。Stage 2 先完成 YAML 的全部 joint epochs，再额外对 heat of vaporization、HOMO、LUMO、partial atomic charge 做 10 epochs refinement；refinement 冻结 backbone/ObjectEncoder、保持 live student forward，只以原始 task loss 独立优化当前 task head。Object v3 强制一个 batch 对应一个 optimizer step，只从完整 epoch 恢复；旧 v2、旧 orbital contract 和缺少现役 preparation/extraction contract 的开发期 v3 不迁移。不得恢复双实体编码器、体系采样、渐进解冻、early stopping、best/last、step checkpoint、PCGrad 或 accumulation window。Stage 3 现役合同以 ADR-0020、ADR-0027、ADR-0039 和 `configs/v2/stage3/base.yaml` 为准、identity/audit 边界以 ADR-0021 为准：1024 维冻结 Stage 2 Object v3 表示、21 个 sparse-label observation task、6 个 meta-group、动态 HoME、joint phase ownership-aware hierarchical PCGrad 和完整 epoch checkpoint；末期 refinement 冻结 GLOBAL/GROUP，只更新 PRIVATE:<task> 且不使用 PCGrad/task weight。Base 的 training/validation microbatch 上限为 1024 且属于严格 training identity。ADR-0034 的隔离消融只以 RDKit 2D + 两个 GLOBAL Linear→LayerNorm adapter 替换 frozen Object representation，其他 Stage3 数值合同保持不变；不得加载 Stage1/2 checkpoint、使用 plugin、增加 HPO 或与 Object artifact/checkpoint 交叉恢复。ADR-0036 的 No-Stage1 隔离消融以共享 RDKit-217 MLP 替换 Stage1 backbone，保留 Stage2 ObjectEncoder 与 Stage3 Base；Stage2 仅训练八个 object/interaction task，禁止 Stage1/teacher artifact，Partial Charge/Full 为 unsupported，且不得与现役 Stage2 artifact/checkpoint 交叉加载。ADR-0010～0012 仅为历史。

Capacity v1 是 ADR-0026/0027 和 `configs/experiments_v1/{stage1,stage2,stage3}` 限定的预注册端到端 legacy 实验；它不修改 `configs/v1` 或现役 v2 Stage 1/2/3 合同，也不得被解释为正式多容量配置、strict scaling law 或 encoder-only effect。该实验只用 Stage 1 Base 配置 prepare 一次，四规模共享 `outputs/experiments_v1/stage1/prepare/artifacts`；Stage 3 选择统一读取 taskwise-refined stitched validation，不读取末轮均值或 test。

## 结构与入口

- 现役 Stage 实现只位于 `src/common`、`src/stage1`、`src/stage2`、`src/stage3`；论文对比 baseline 位于顶层 `benchmarks/`，内部消融位于顶层 `ablations/`，二者均不得被 Stage 1/2/3 导入。
- `common` 只接收至少两个 Stage 实际复用的原子功能；禁止建立 `utils.py`。
- Stage 可以导入其他 Stage 的公开 contract，但 `src/stageN` 禁止从其他 Stage 导入 `_private_symbol` 或使用 `import *`。
- 唯一运行入口是 `scripts/stage{1,2,3}/*.py` 与 `scripts/benchmarks/*.py`；不恢复 console CLI、shell runner、smoke 或 matrix runner。
- 科研参数与 prepare 执行参数都写入完整自包含 YAML；`preparation` 不进入实验身份。现役主线配置位于 `configs/v2`，legacy 配置位于 `configs/v1`；预注册实验配置位于 `configs/experiments_v1/<stage>`；内部消融配置位于 `configs/ablations`。
- `--output`、`--resume`、Stage 3 fold 和 evaluation selector 是运行参数，不进入科研 config schema。
- Stage 3 train 的 `--fold` 可接收一个或多个 fold，`--output` 始终是共同 root，实际 run contract 位于 `foldN/`。多 fold 只允许由该入口使用 spawn worker 和显式设备槽调度；不得把并发下沉到 `src/stage3`，也不得恢复独立 fold/matrix runner。
- Baseline 与消融以 ADR-0022/0028/0029/0030/0031/0032/0033/0034/0035/0037/0038/0040 为准，只复用现役 registry、split、canonical SMILES、condition/target 与评估口径；不得改变或被解释为 Stage 1/2/3 数值训练合同。消融专用实现不得放回 `benchmarks/`。Stage3 Single-task MLP 是同时移除 HoME routing、跨任务共享、PCGrad 与 composite sampling 的整体消融，不得表述为单组件归因。Baseline v1 不支持 resume，失败由 sweep 在新 attempt 目录完整重跑。高级 baseline 可使用独立 hash-lock 环境，但不得修改主环境依赖、自动安装或静默回退。
- Evaluation reporting 与汇总以 ADR-0023/0024/0025/0028 为准：Stage 2 固定分为 Core、Partial Charge、Full 三榜；Core 是 heat of vaporization、HOMO、LUMO 三任务等权，Full 再与 Partial Charge 组成四单元等权。缺少 `stage2-benchmark-suite-v2` 的旧 Stage 2 reporting 只进入 health，禁止跨 run 拼接 Full。D-MPNN 的 Partial/Full 为 supported；MLP、ECFP+XGBoost、MoLFormer、ILBERT、SPMM 与 LlaSMol 为 unsupported。

## 数据、输出与恢复

- 数据本体不进入 Git；prepare 自动写 `data/stage*/metadata.json`。
- 每次操作必须冻结 `run_config.yaml`，并写公开安全的 `metadata.json` 和成功后的 `summary.json`；reusable Stage 1/2 prepare 可在各自数据身份不变时刷新允许忽略的执行或模型配置，Stage 2 的具体边界以 ADR-0019 为准。禁止用户名、hostname、私有绝对路径。
- 新 train/evaluate 输出不可覆盖；resume 必须显式且严格校验阶段、fold、有效配置、step/epoch、optimizer、scheduler 与 AMP。Stage 1 仅从完整 epoch 恢复并允许改变 world size；Stage 2 严格校验 Object v3 的完整 epoch、registry、model contract、RNG、数据、teacher、任务规模、数学精度与 optimizer implementation 合同；Stage 3 严格校验 resolved plan、ownership、Stage 2 SHA、数据与 normalization，并由 seed/epoch/task 重建 virtual sampler。
- Stage 3 的布尔 `--resume` 只 skip identity 一致且 summary、最终 checkpoint、metrics/diagnostics 完整的 fold；其他 fold 只从 checkpoint 与两份 epoch history 尾部完全一致的位置恢复，不截断、不猜测。
- 保留 prepared artifact 自身的 SHA 和完整性校验。跨 Stage 只绑定 ADR-0021 定义的 semantic identity 与必要 state hash，不绑定完整 checkpoint SHA。
- 周期 checkpoint 全部保留。Stage 1 以 `last.pt` 表示最新完整恢复状态。Stage 2 保存不可覆盖的 `checkpoint_epoch_XXXXX.pt`，Stage 3 按 YAML interval 保存不可覆盖的完整 epoch checkpoint；普通 final checkpoint 都表示真实历史恢复状态，不生成 `best.pt` 或 `last.pt`。Stage 2/3 的最终评估模型是独立的 `taskwise_refined.pt` 与 manifest，不得伪装成某个 epoch checkpoint。

## 验证与清理

修改后运行 `pytest -q`，并按风险检查九个 Stage script 与四个 benchmark script 的 `--help`、`compileall`、`git diff --check`、Markdown 链接、ignore 白名单与旧入口搜索。只使用临时小数据；除非用户明确要求，不执行正式 prepare、教师缓存、训练或五折 evaluation。

测试默认不随代码修改自动增长。新增测试前必须确认：它保护此前未覆盖且会造成实质损失的科研行为、checkpoint/resume、artifact/identity、CLI/reporting 或高风险调度合同，并且现有测试无法合理覆盖。优先复用、修改或合并现有测试；只有答案明确时才新增一个最小行为测试。不要为 private helper 返回值、内部调用关系、函数搬家、简单重构、coverage 数字或近似重复参数组合增加测试。`tests/` 按 Stage、benchmark、common 与 architecture 集中组织；`conftest.py` 只放真正跨文件复用的小型 fixture，不建立通用测试框架。

`trash/` 不进入 Git。移动旧 artifact、旧 YAML、未消费数据或删除机器缓存前，必须先报告精确文件数、大小、目标和冲突策略，并等待用户明确确认。不得覆盖、重排或删除既有 `trash/` 内容。
