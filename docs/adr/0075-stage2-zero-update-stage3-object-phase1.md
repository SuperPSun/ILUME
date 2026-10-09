# ADR-0075：Stage 2 零训练对照与 Stage 3 ObjectEncoder Phase 1 适配

- 状态：部分由 [ADR-0082](0082-home-mainline-and-core-ablations.md) 取代；Phase1机制保留
- 日期：2026-09-25
- 范围：Object-backed v2 Stage 3 与隔离的 No-Stage2 对照

## 决定

以相同 Stage 1 检查点、Stage 2 模型结构和初始化种子导出一个零优化器更新的 Stage 2 编码器。它保留随机初始化的 ObjectEncoder；正式对照则使用已完成 Stage 2 训练的编码器。两组 Stage 3 使用相同 20-任务 Flat HoME、split、归一化、初始化种子、owner 预算与固定-末轮状态协议。正式对照必须重新运行；历史冻结 ObjectEncoder Base 不是配对 control。实验测量完整 Stage 2 训练带来的初始化收益，包含其对 Stage 1 编码权重的更新，不能单独归因于 physics loss。

Stage 3 prepare 为每个 ObjectKey 保存按原顺序排列的 Stage 1 实体 slots、roles 和完整性 SHA。Stage 1 在所有阶段冻结。Phase 1 每批对冻结 slots 可微运行 ObjectEncoder，并与 HoME 共同训练 15 轮。ObjectEncoder 是独立 `ENCODER_STAGE2` owner：LR `1.5e-5`，实际更新前 5% 预热、余弦到 0.1，梯度按 GLOBAL 的任务/组权重汇总并单独裁剪到 norm 1。Phase 1 末冻结 ObjectEncoder，并从其末轮状态缓存 object 表示；Phase 2/3 的同源分支与验证/测试集都使用该末轮状态，不得回退到 prepare 时的 embedding。

零更新编码器导出不得读取 Stage 2 标签或运行优化器；其清单绑定 Stage 1 检查点 SHA、Stage 2 准备产物身份、种子、初始 shared-状态hash、零更新和正式 Stage 2 编码器 SHA。两组 Stage 3 准备产物、训练、检查点、final 产物使用新合同和独立输出。检查点保存完整 ObjectEncoder、优化器/调度器/RNG/更新/冻结状态；final 清单记录来源编码器、slots、Phase 1 final ObjectEncoder 状态和最终表示 hash。旧准备产物/检查点与新合同不可交叉加载或恢复。

## 边界

Base、random/system/individual、base1/base2/base3 系列与 No-Stage1 Object-backed Stage 3使用此合同。RDKit-HoME 无 ObjectEncoder，不适用；现有全量微调消融保留独立合同；预计算 bank 的 transfer-knowledge 消融保留历史冻结表示公式。历史实现/Capacity、Stage 1/2 正式训练合同不变。实现和测试阶段不执行正式 prepare、训练或评估；先比较两组 system-split 5CV 任务等权 macro NMAE与逐任务，再报告测试集集成，不用测试集选预算。
