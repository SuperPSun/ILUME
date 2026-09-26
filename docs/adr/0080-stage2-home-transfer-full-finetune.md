# ADR-0080：Stage2-HoME Transfer 表示编码器全量微调

- 状态：Experimental
- 日期：2026-09-27
- 范围：独立组合消融，不替代 ADR-0079 或正式 Base

## 决定

直接复用已完成的 Stage2-HoME artifact；不重新训练 Stage1/Stage2。加载源表示编码器、GLOBAL 与 thermophysical/solvation 两个 GROUP，其他 GROUP、实验 PRIVATE/task gate/FiLM/normalization/tower按原迁移消融的种子规则初始化。simulation PRIVATE、电子 GROUP与源预测模块不迁移。

Stage3 保持当前20-task Flat HoME、three-phase、raw sampling、owner lifetime、loss、AdamW及 `weighted_owner_raw_v1`。Phase 1共15 epochs，可微地重算Stage1→ObjectEncoder表示：Stage1编码器与ObjectEncoder的独立owner LR为 `5e-6` / `1.5e-5`，5% update warmup、cosine floor 0.1，各自clip至norm 1。重建头与simulation预测头不参与优化。microbatch固定8，保持原逻辑task batch、raw exposure和optimizer update预算。

独立prepare从当前Base数据合同生成源encoder对应的对象表示和分子输入，校验五折训练/validation行顺序、ObjectKey、条件与目标归一化。Phase 1末缓存final编码器生成的表示，绑定Phase 1 model hash；Phase 2/3编码器冻结eval，各分支使用同源anchor、固定末轮stitch。validation只报告，test不得驱动训练或配置选择。

## 接口与产物

配置为 `configs/ablations/stage2_home_transfer_full_finetune.yaml`，入口为 `scripts/stage3/home_transfer_full_finetune.py prepare|train|evaluate`。必填 `--source-dir` 指已有源实验根（含 `stage2/`），`--output` 指独立新实验根。默认输出 `outputs/ablations/stage2_home_transfer_full_finetune/`，下含 `prepare/`、`features/`、`train/foldN/`、`valid/foldN/`和 `test/`。

训练支持spawn多fold设备槽、进度条和严格epoch-boundary resume；尚未启动的fold可在resume时新建。组合消融使用训练identity contract 11、resolved plan format 9及独立 `ilume_stage3_home_transfer_full_finetune_three_phase_*` kind。身份绑定feature SHA、源encoder与HoME artifact SHA、源训练身份及精确迁移参数清单。final完整保存两级编码器与HoME状态和hash，禁止与正式Base、普通迁移或普通全量微调交叉恢复/加载；它们原身份不变。

评估发布标准metadata、summary和prediction CSV，study为 `ilume-stage2-home-transfer-full-finetune-v1`，显示名为 `ILUME (HoME transfer full fine-tune)`。沿用七项gate diagnostics和五折test ensemble；统一summarizer直接识别。

## 比较限制

按用户决定复用已有ADR-0079结果作对照，不重训microbatch8的paired control。可微编码路径、dropout随机数消耗和微批可能不同，因此结果代表完整微调方案，不能把全部差异严格归因于Stage1解冻。所有历史输出只读；实现和测试不执行正式prepare、训练或evaluation。
