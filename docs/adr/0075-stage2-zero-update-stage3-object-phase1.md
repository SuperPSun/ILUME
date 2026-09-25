# ADR-0075：Stage 2 零训练对照与 Stage 3 ObjectEncoder Phase 1 适配

- 状态：Experimental
- 日期：2026-09-25
- 范围：Object-backed v2 Stage 3 与隔离的 No-Stage2 对照

## 决定

以相同 Stage 1 checkpoint、Stage 2 模型结构和初始化 seed 导出一个零 optimizer update 的 Stage 2 encoder。它保留随机初始化的 ObjectEncoder；正式对照则使用已完成 Stage 2 训练的 encoder。两组 Stage 3 使用相同 20-task Flat HoME、split、归一化、初始化 seed、owner 预算与 fixed-final-state 协议。正式对照必须重新运行；历史冻结 ObjectEncoder Base 不是配对 control。实验测量完整 Stage 2 训练带来的初始化收益，包含其对 Stage 1 编码权重的更新，不能单独归因于 physics loss。

Stage 3 prepare 为每个 ObjectKey 保存按原顺序排列的 Stage 1 entity slots、roles 和完整性 SHA。Stage 1 在所有阶段冻结。Phase 1 每批对 frozen slots 可微运行 ObjectEncoder，并与 HoME 共同训练 15 epochs。ObjectEncoder 是独立 `ENCODER_STAGE2` owner：LR `1.5e-5`，实际 update 前 5% warmup、cosine 到 0.1，梯度按 GLOBAL 的 task/group 权重汇总并单独裁剪到 norm 1。Phase 1 末冻结 ObjectEncoder，并从其 final state 缓存 object 表示；Phase 2/3 的同源分支与 validation/test 都使用该 final state，不得回退到 prepare 时的 embedding。

零更新 encoder 导出不得读取 Stage 2 标签或运行 optimizer；其 manifest 绑定 Stage 1 checkpoint SHA、Stage 2 prepared identity、seed、初始 shared-state hash、零 update 和正式 Stage 2 encoder SHA。两组 Stage 3 prepared、训练、checkpoint、final artifact使用新合同和独立输出。checkpoint 保存完整 ObjectEncoder、optimizer/scheduler/RNG/update/freeze state；final manifest记录来源 encoder、slots、Phase 1 final ObjectEncoder state和最终表示 hash。旧 prepared/checkpoint 与新合同不可交叉加载或 resume。

## 边界

Base、random/system/individual、base1/base2/base3 系列与 No-Stage1 Object-backed Stage 3使用此合同。RDKit-HoME 无 ObjectEncoder，不适用；现有全量微调消融保留独立合同；预计算 bank 的 transfer-knowledge 消融保留历史冻结表示公式。legacy/Capacity、Stage 1/2 正式训练合同不变。实现和测试阶段不执行正式 prepare、训练或评估；先比较两组 system-split 5CV task-equal macro NMAE与逐任务，再报告test ensemble，不用test选预算。
