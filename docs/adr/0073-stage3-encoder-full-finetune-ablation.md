# ADR-0073：Stage 3 表示编码器全量微调消融

- 状态：Experimental
- 日期：2026-09-24
- 范围：`configs/ablations/stage3_full_finetune.yaml`，不替代 v2 Flat Base

## 决定

本实验从 Base 所用的 `stage2_encoder.pt` 初始化 Stage 1 三模态表示编码器与 Stage 2 ObjectEncoder。仅在 Stage 3 Phase 1 的 15 个 raw epochs 中，使用 Stage 3 标签反向更新这两级编码器；Stage 1 重建头与 Stage 2 physics heads 不参与前向或优化。Phase 2/3 将编码器以 `requires_grad=False` 冻结并置为 eval，六个 GROUP 与二十个 PRIVATE 分支继续从共同 anchor 独立训练、固定末轮 stitch。Flat HoME、owner lifetime、`weighted_owner_raw_v1`、task/group weight、loss 和 validation reporting-only 保持 Base 配置。

Stage 1/ObjectEncoder 分别是两个独立的 Phase 1 owner，LR 为 `5e-6`/`1.5e-5`，各使用 5% update warmup、cosine 到初始值的 0.1，并各自裁剪到 norm 1。两级共享表示梯度按与 GLOBAL 相同的 task/group 权重汇总。行级 microbatch 固定为 8；每 task 的 `B_t`、epoch exposure和optimizer update次数仍由 Base raw allocation 决定。

消融 prepare 只把 Base ObjectKey 转成 Stage 1 tokenizer/graph/RDKit 输入，并绑定 Base prepared identity、Stage 2 encoder SHA 与 ObjectKey 顺序。Phase 1 不缓存可微 embedding；结束后可缓存绑定 Phase 1 模型状态的只读 embedding 供 Phase 2/3 使用。Base prepared artifact不修改、不覆盖。

周期 checkpoint、训练/评估身份和 final artifact kind与正式 Base隔离。Phase 1 full checkpoint含两级编码器、HoME、AdamW/scheduler/RNG/update/freeze状态；Phase 2/3仍使用 immutable-anchor owner delta。final完整保存编码器与HoME及各自hash。旧Base或其他消融checkpoint不可交叉resume/evaluate。

## 比较与限制

正式比较使用相同20-task system-split五折validation task-equal macro NMAE和逐任务分数；test ensemble只在validation分析后报告，不反向选参。按本轮决定，只使用历史Base summary作参照，不重训新可微输入路径下的frozen control。因此差异代表整个消融方案，不可完全归因于解冻编码器。

现役 Stage 1/2 与 Base Stage 3训练器默认路径均保持不变；本 ADR 不授权在实现或测试阶段执行正式prepare、五折训练或评估。
