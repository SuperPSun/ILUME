# ADR-0021：Identity / Audit Contract v1

- 状态：Accepted
- 日期：2026-08-20
- 取代范围：ADR-0013、0014、0015、0019、0020 中分散的 identity、artifact lineage、run reuse/resume 与 audit 规则

> 2026-08-26：Stage 2/3 refinement identity、checkpoint 版本、taskwise-refined artifact 与 evaluation selector 由 [ADR-0027](0027-late-taskwise-refinement.md) 增补；本文四层 identity 模型继续有效。

## 背景

旧实现混合了资源位置、文件完整性、科学语义和运行环境：移动 artifact、改变 worker 或 Git dirty 状态可能阻断复用，而 metadata 文件重新序列化又可能改变跨 Stage 身份。完整 checkpoint SHA 还把 optimizer、RNG 和序列化细节错误带入模型身份。与此同时，AMP dtype、TF32 策略与 optimizer implementation 会改变数值训练，不能降为普通运行记录。

本 ADR 只重构身份与审计边界，不改变数据筛选、split、feature、模型、loss、采样、优化顺序或数值训练行为。

## 决定

所有新 artifact、cache、checkpoint 和 run 使用 `identity_contract_version: 1`，并区分四层：

- `locator`：仓库内安全相对路径，只用于定位。
- `integrity`：逻辑文件 ID 对应的 SHA256、size 和结构合同；不一致 HARD FAIL。
- `semantic`：由 Stage builder 显式选择的内容、模型和训练数学语义，使用 domain-separated canonical hash。
- `provenance`：Git、软件、硬件、worker、设备、路径和计时，只记录或告警。

恢复状态与四层正交。checkpoint 继续严格保存并验证完整 epoch/step、optimizer、scheduler、AMP、RNG、metrics 历史和各 Stage 现役几何合同。AMP dtype、TF32 策略、optimizer implementation 等数值执行合同仍属于严格 training semantic identity。

`common.identity` 是唯一通用 hash/compare/integrity 实现。semantic payload 禁止 NaN、`Path` 和不稳定对象；Stage builder 不自动删除疑似路径字段。`open_run_directory()` 始终冻结完整 `run_config.yaml`，但 reuse/resume 只比较调用方提供的 semantic identity。每次尝试追加到 `attempts.jsonl`，保留 started/completed/failed、runtime provenance 和 resume locator。

## Stage 1

Corpus identity 绑定源内容、split/seed、augmentation、QC、tokenizer、descriptor、fingerprint、token 上限和 feature-generation contract；`shard_size` 只进入独立 sampler-layout identity，`shard_cache_size` 只属执行参数。Training identity 绑定 corpus、layout、模型、masking、loss、global batch、optimizer/scheduler、epoch、seed 和 precision，排除路径、worker、device、compile 与 logging/quick-validation 频率。

Feature identity 独立描述 tokenizer、descriptor schema/scaler、fingerprint 与 feature-generation contract。Encoder identity 只绑定 feature identity、encoding API/architecture 与 encoding-only state hash。读取已物化 tensor 不检查当前 RDKit 字符串；生成新 feature 时严格检查 feature-generation contract。

## Stage 2

配置显式提供 `data.target_materialization_modes`。默认 `require_complete`；QM Base 使用 `allow_partial_drop_all_missing`，并与 `loss.task_loss_modes` 做相容性校验。Prepare 不再从 loss reduction 暗推 materialization。

Data identity 绑定源内容、Stage 1 feature identity、registry、tensor/scaler/normalization、partial-charge mapping 与 materialization contract，排除路径、worker、entity shard size 和 Stage 2 model contract。Teacher identity 只绑定 entity identity 与 Stage 1 encoder identity；teacher embeddings SHA 另作严格完整性与 resume 连续性检查。Training identity 绑定 data、teacher、Stage 1 encoder、registry、完整 model contract、task loss/weight、batch/freeze/epoch 几何、optimizer/scheduler 和 precision。

`stage2_encoder.pt` 是 Stage 3 的唯一 Stage 2 模型输入。它内嵌 tokenizer、descriptor schema/scaler 等小型 encoding 依赖，保存 encoding-only contract、Stage 1 backbone 与 ObjectEncoder state/hash、role mapping 和 `stage2_encoder_identity`；不含 physics heads、task registry、optimizer、scheduler、RNG 或路径。完整 Stage 2 checkpoint 仅用于 Stage 2 resume。

## Stage 3

配置使用 `initialization.stage2_encoder`。Object cache key 固定绑定 `stage2_encoder_identity + object_encoding_contract + ObjectKey`。Prepared identity 绑定源内容、resolved registry、split/CV、normalization、对象顺序、encoder identity 与 encoding contract，排除 encoder/cache 路径、batch size 和硬件。

Resolved training plan 同时保存 semantic plan 和 execution 审计；run、plan 与 checkpoint 从同一 semantic plan 生成 training identity。线程、device、debug、checkpoint interval 和路径不阻断恢复；模型、ownership、allocation、virtual sampling、PCGrad、loss、optimizer/scheduler、microbatch、precision 与 normalization 必须一致。

Plugin 绑定源 training identity、模型 state hash、normalization、load/adaptation scopes 与 ownership，不绑定 plugin 路径或源 optimizer/RNG。Evaluate 绑定 prepared identity、checkpoint training identity、模型 state hash、normalization 和 selector，不绑定完整 checkpoint SHA 或训练状态。

## 迁移与后果

Corpus v2、Stage 1 checkpoint v2、Stage 2 data v3、Stage 2 encoder v1 与 Stage 3 prepared v1 的物理格式保持不变；ADR-0027 将 Stage 2 checkpoint 升级为 v4、Stage 3 checkpoint 升级为 v2，并新增两 Stage 各自的 taskwise-refined artifact v1。缺少现役 identity/refinement contract 的旧 artifact/cache/checkpoint 一律明确拒绝；不推导旧 identity，不生成 sidecar，不静默迁移。

启用本合同需要按 Stage 1 prepare/train、Stage 2 prepare/teacher/train、Stage 3 prepare/five-fold train/evaluate 的顺序正式重跑。归档旧正式输出前必须先报告精确路径、文件数和大小，并等待用户单独确认；目标使用全新且不冲突的 `trash/pre-identity-contract-v1-<timestamp>/`。实现验收只使用临时小数据。
