# ADR-0073：Stage 3 表示编码器全量微调消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-24
- 范围：`configs/ablations/stage3_full_finetune.yaml`，不替代 v2 Flat Base

## 决定

本实验从 Base 所用的 `stage2_encoder.pt` 初始化 Stage 1 三模态表示编码器与 Stage 2 ObjectEncoder。仅在 Stage 3 Phase 1 的 15 个原始轮中，使用 Stage 3 标签反向更新这两级编码器；Stage 1 重建头与 Stage 2 物理性质预测头不参与前向或优化。Phase 2/3 将编码器以 `requires_grad=False` 冻结并置为 eval，六个 GROUP 与二十个 PRIVATE 分支继续从共同锚点独立训练、固定末轮拼接。Flat HoME、owner 存续期、`weighted_owner_raw_v1`、任务/组权重、loss 和验证报告-仅保持 Base 配置。

Stage 1/ObjectEncoder 分别是两个独立的 Phase 1 owner，LR 为 `5e-6`/`1.5e-5`，各使用 5% 更新预热、余弦到初始值的 0.1，并各自裁剪到 norm 1。两级共享表示梯度按与 GLOBAL 相同的任务/组权重汇总。行级 microbatch 固定为 128；每任务的 `B_t`、轮 exposure和优化器更新次数仍由 Base 原始 allocation 决定。

2026-09-27 因用户报告 microbatch256 显存不足，将 microbatch 统一为 128，退役 8/64/256 的现役训练设置。该字段进入训练身份，旧检查点不可续训，新训练使用独立输出目录；准备产物/特征数学合同不变，但入口要求新输出根中具备相应输入，HoME-迁移全量微调须先运行 prepare。浮点累加和 dropout 随机数路径可能变化，不保证复现旧微批结果。

消融 prepare 只把 Base ObjectKey 转成 Stage 1 分词器/graph/RDKit 输入，并绑定 Base 准备产物身份、Stage 2 编码器 SHA 与 ObjectKey 顺序。Phase 1 不缓存可微 embedding；结束后可缓存绑定 Phase 1 模型状态的只读 embedding 供 Phase 2/3 使用。Base 准备产物不修改、不覆盖。

周期检查点、训练/评估身份和 final 产物 kind与正式 Base隔离。Phase 1 full 检查点含两级编码器、HoME、AdamW/调度器/RNG/更新/冻结状态；Phase 2/3仍使用 immutable-锚点 owner delta。final完整保存编码器与HoME及各自hash。旧Base或其他消融检查点不可交叉恢复/evaluate。

多折训练可由独立入口以 `spawn` 进程和显式 CUDA 设备槽并发执行。`--max-parallel`、`--devices` 仅改变执行调度，不进入训练身份；每个折仍使用自己的完整模型和原有轮边界恢复合同。

## 比较与限制

正式比较使用相同20-任务 system-split五折验证任务等权 macro NMAE和逐任务分数；测试集集成只在验证分析后报告，不反向选参。按本轮决定，只使用历史Base 汇总作参照，不重训新可微输入路径下的冻结 control。因此差异代表整个消融方案，不可完全归因于解冻编码器。

现役 Stage 1/2 与 Base Stage 3训练器默认路径均保持不变；本 ADR 不授权在实现或测试阶段执行正式prepare、五折训练或评估。
