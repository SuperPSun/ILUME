# ADR-0021：身份与审计合同 v1

- 状态：已接受
- 日期：2026-08-20
- 取代范围：ADR-0013、0014、0015、0019、0020 中分散的身份、产物 lineage、运行 reuse/恢复与审计规则

> 2026-08-26：Stage 2/3 精调身份、检查点版本、逐任务精调后产物与评估 selector 由 [ADR-0027](0027-late-taskwise-refinement.md) 增补；本文四层身份模型继续有效。

## 背景

旧实现混合了资源位置、文件完整性、科学语义和运行环境：移动产物、改变 worker 或 Git dirty 状态可能阻断复用，而元数据文件重新序列化又可能改变跨 Stage 身份。完整检查点 SHA 还把优化器、RNG 和序列化细节错误带入模型身份。与此同时，AMP 数据类型、TF32 策略与优化器实现会改变数值训练，不能降为普通运行记录。

本 ADR 只重构身份与审计边界，不改变数据筛选、split、特征、模型、loss、采样、优化顺序或数值训练行为。

## 决定

所有新产物、缓存、检查点和运行使用 `identity_contract_version: 1`，并区分四层：

- `locator`：仓库内安全相对路径，只用于定位。
- `integrity`：逻辑文件 ID 对应的 SHA256、size 和结构合同；不一致 HARD FAIL。
- `semantic`：由 Stage builder 显式选择的内容、模型和训练数学语义，使用域-separated canonical hash。
- `provenance`：Git、软件、硬件、worker、设备、路径和计时，只记录或告警。

恢复状态与四层正交。检查点继续严格保存并验证完整轮/步、优化器、调度器、AMP、RNG、metrics 历史和各 Stage 现役几何合同。AMP 数据类型、TF32 策略、优化器实现等数值执行合同仍属于严格训练语义身份。

`common.identity` 是唯一通用 hash/compare/integrity 实现。semantic payload 禁止 NaN、`Path` 和不稳定对象；Stage builder 不自动删除疑似路径字段。`open_run_directory()` 始终冻结完整 `run_config.yaml`，但 reuse/恢复只比较调用方提供的语义身份。每次尝试追加到 `attempts.jsonl`，保留 started/completed/failed、运行时来源记录和恢复 locator。

## Stage 1

Corpus 身份绑定源内容、split/种子、augmentation、QC、分词器、描述符、指纹、token 上限和特征-generation 合同；`shard_size` 只进入独立 sampler-layout 身份，`shard_cache_size` 只属执行参数。Training 身份绑定语料、layout、模型、masking、loss、全局batch、优化器/调度器、轮、种子和 precision，排除路径、worker、device、compile 与 logging/quick-验证频率。

Feature 身份独立描述分词器、描述符结构定义/scaler、指纹与特征-generation 合同。Encoder 身份只绑定特征身份、encoding API/architecture 与 encoding-仅状态hash。读取已物化张量不检查当前 RDKit 字符串；生成新特征时严格检查特征-generation 合同。

## Stage 2

配置显式提供 `data.target_materialization_modes`。默认 `require_complete`；QM Base 使用 `allow_partial_drop_all_missing`，并与 `loss.task_loss_modes` 做相容性校验。Prepare 不再从 loss reduction 暗推 materialization。

Data 身份绑定源内容、Stage 1 特征身份、注册表、张量/scaler/归一化、partial-charge mapping 与 materialization 合同，排除路径、worker、实体分片 size 和 Stage 2 模型合同。Teacher 身份只绑定实体身份与 Stage 1 编码器身份；teacher embeddings SHA 另作严格完整性与恢复连续性检查。Training 身份绑定数据、teacher、Stage 1 编码器、注册表、完整模型合同、任务 loss/权重、batch/冻结/轮几何、优化器/调度器和 precision。

`stage2_encoder.pt` 是 Stage 3 的唯一 Stage 2 模型输入。它内嵌分词器、描述符结构定义/scaler 等小型 encoding 依赖，保存 encoding-仅合同、Stage 1 backbone 与 ObjectEncoder 状态/hash、角色 mapping 和 `stage2_encoder_identity`；不含物理性质预测头、任务注册表、优化器、调度器、RNG 或路径。完整 Stage 2 检查点仅用于 Stage 2 恢复。

## Stage 3

配置使用 `initialization.stage2_encoder`。Object 缓存 key 固定绑定 `stage2_encoder_identity + object_encoding_contract + ObjectKey`。准备数据身份绑定源内容、resolved 注册表、split/CV、归一化、对象顺序、编码器身份与 encoding 合同，排除编码器/缓存路径、batch大小和硬件。

Resolved 训练 plan 同时保存 semantic plan 和 execution 审计；运行、plan 与检查点从同一 semantic plan 生成训练身份。线程、device、debug、检查点 interval 和路径不阻断恢复；模型、参数归属、allocation、virtual 采样、PCGrad、loss、优化器/调度器、microbatch、precision 与归一化必须一致。

Plugin 绑定源训练身份、模型状态hash、归一化、加载/适配范围与参数归属，不绑定插件路径或源优化器/RNG。Evaluate 绑定准备产物身份、检查点训练身份、模型状态hash、归一化和 selector，不绑定完整检查点 SHA 或训练状态。

## 迁移与后果

Corpus v2、Stage 1 检查点 v2、Stage 2 数据 v3、Stage 2 编码器 v1 与 Stage 3 准备产物 v1 的物理格式保持不变；ADR-0027 将 Stage 2 检查点升级为 v4、Stage 3 检查点升级为 v2，并新增两 Stage 各自的逐任务精调后产物 v1。缺少现役身份/精调合同的旧产物/缓存/检查点一律明确拒绝；不推导旧身份，不生成 sidecar，不静默迁移。

启用本合同需要按 Stage 1 数据准备/训练、Stage 2 数据准备/teacher缓存生成/训练、Stage 3 数据准备/五折训练/评估的顺序正式重跑。归档旧正式输出前必须先报告精确路径、文件数和大小，并等待用户单独确认；目标使用全新且不冲突的 `trash/pre-identity-contract-v1-<timestamp>/`。实现验收只使用临时小数据。
