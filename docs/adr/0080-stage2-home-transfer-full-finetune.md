# ADR-0080：Stage2-HoME 迁移表示编码器全量微调

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-27
- 范围：独立组合消融，不替代 ADR-0079 或正式 Base

## 决定

直接复用已完成的 Stage2-HoME 产物；不重新训练 Stage1/Stage2。加载源表示编码器、GLOBAL 与 thermophysical/solvation 两个 GROUP，其他 GROUP、实验 PRIVATE/任务门控/FiLM/归一化/tower按原迁移消融的种子规则初始化。模拟 PRIVATE、电子 GROUP与源预测模块不迁移。

Stage3 保持当前20-任务 Flat HoME、three-phase、原始样本采样、owner 存续期、loss、AdamW及 `weighted_owner_raw_v1`。Phase 1共15 轮，可微地重算Stage1→ObjectEncoder表示：Stage1编码器与ObjectEncoder的独立owner LR为 `5e-6` / `1.5e-5`，5% 更新预热、余弦 floor 0.1，各自clip至norm 1。重建头与模拟预测头不参与优化。microbatch固定128，保持原逻辑任务 batch、原始 exposure和优化器更新预算。

2026-09-27 因用户报告 microbatch256 显存不足，将 microbatch 统一为 128，退役 8/64/256 的现役训练设置；旧微批检查点与新训练身份不兼容。准备产物/特征数学合同不变，新输出目录须先运行 prepare 再训练/评估；代码不自动降低 microbatch，也不补齐不足128行的任务 batch。

独立prepare从当前Base数据合同生成源编码器对应的对象表示和分子输入，校验五折训练/验证行顺序、ObjectKey、条件与目标归一化。Phase 1末缓存final编码器生成的表示，绑定Phase 1 模型 hash；Phase 2/3编码器冻结eval，各分支使用同源锚点、固定末轮拼接。验证只报告，测试集不得驱动训练或配置选择。

## 接口与产物

配置为 `configs/ablations/stage2_home_transfer_full_finetune.yaml`，入口为 `scripts/stage3/home_transfer_full_finetune.py prepare|train|evaluate`。必填 `--source-dir` 指已有源实验根（含 `stage2/`），`--output` 指独立新实验根。默认输出 `outputs/ablations/stage2_home_transfer_full_finetune/`，下含 `prepare/`、`features/`、`train/foldN/`、`valid/foldN/`和 `test/`。

训练支持spawn多折设备槽、进度条和严格第边界恢复；尚未启动的折可在恢复时新建。组合消融使用训练身份合同 11、解析后的计划格式 9及独立 `ilume_stage3_home_transfer_full_finetune_three_phase_*` kind。身份绑定特征 SHA、源编码器与HoME 产物 SHA、源训练身份及精确迁移参数清单。final完整保存两级编码器与HoME状态和hash，禁止与正式Base、普通迁移或普通全量微调交叉恢复/加载；它们原身份不变。

评估发布标准元数据、汇总和预测 CSV，study为 `ilume-stage2-home-transfer-full-finetune-v1`，显示名为 `ILUME (HoME transfer full fine-tune)`。沿用七项门控诊断和五折测试集集成；统一summarizer直接识别。

## 比较限制

按用户决定复用已有ADR-0079结果作对照，不重训同微批的paired control。可微编码路径、dropout随机数消耗和微批可能不同，因此结果代表完整微调方案，不能把全部差异严格归因于Stage1解冻。所有历史输出只读；实现和测试不执行正式prepare、训练或评估。
