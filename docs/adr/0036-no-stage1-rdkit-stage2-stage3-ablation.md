# ADR-0036：No-Stage1 RDKit 2D → Stage2 → Stage3 HoME 消融

- 状态：Accepted
- 日期：2026-09-01

## 背景

现役 ILUME 依次使用 Stage1 multimodal pretrained backbone、Stage2 ObjectEncoder 和
Stage3 HoME。ADR-0034 已同时移除 Stage1/2 representation；为了单独研究 Stage1
pretraining 的贡献，需要保留 Stage2 supervision 与 Stage3 全部训练合同，只替换 Stage2
接收的分子表示。

该实验比较：

`Stage1 pretrained backbone → Stage2 ObjectEncoder → Stage3 HoME`

与：

`RDKit 217D → shared Stage2-supervised MLP → Stage2 ObjectEncoder → Stage3 HoME`。

## 决定

1. 配置固定为 `configs/ablations/no_stage1_rdkit_stage2.yaml` 与
   `configs/ablations/no_stage1_rdkit_stage3.yaml`，输出根为
   `outputs/ablations/no_stage1_rdkit_stage2_stage3`。现役 `configs/v1` 与正式输出不变。
2. Stage2 不读取 Stage1 prepare artifact、checkpoint、teacher embedding 或 learned state；
   只复用 `stage1.descriptors` 的既定 RDKit-217 名称、顺序与计算实现。
3. 全部 component role 共享一个 descriptor encoder：
   `Linear(retained_width, 1024) → GELU → Dropout(0.10) → Linear(1024, 512)
   → LayerNorm(512)`。role 仍由 ObjectEncoder 的 role embedding 表达；IL slot 顺序、
   interaction、condition 与 task head 路径保持不变，condition 不进入 descriptor encoder。
4. preprocessing 在八个 supported Stage2 task 的全部 train-row component occurrence 联合池
   上拟合，重复 occurrence 保留权重。全无有限训练值的列删除，非有限值按 train median
   填充，按 population mean/std 标准化，近常量 scale 设为 1，最终 clip 到 `[-10, 10]`。
   validation/test 不参与拟合。
5. `simulation/partial_atomic_charge` 不受分子级 RDKit descriptor 支持，从本消融的 registry、
   joint training 与 refinement 中移除。其余八任务按现役 task-weight 规则重新归一化。
6. joint phase 不生成 teacher cache，loss 只含原 physics loss。共享 MLP 与 ObjectEncoder 从
   epoch 1 共同训练，学习率均为 `3e-5`；5 个 joint epochs、batch schedule、单 batch 单
   optimizer step、BF16、optimizer 和 scheduler 沿用 Base。
7. 额外 10 个 refinement epochs 仅覆盖 heat of vaporization、HOMO 与 LUMO；共享 MLP 和
   ObjectEncoder 冻结，只更新当前 task head。普通 checkpoint、taskwise-refined artifact 与
   encoder 使用 RDKit 专属 kind，拒绝与现役 Stage2 artifact 双向交叉加载。
8. `stage2_encoder.pt` 内嵌 descriptor schema、RDKit version、preprocessor、共享 MLP、
   ObjectEncoder、role mapping 与 state identity，不包含 Stage1 或 teacher identity。Stage3
   prepare 直接使用该 frozen encoder，不重新拟合 preprocessing。
9. Stage3 继续覆盖 21 tasks、5 folds，并完整沿用 Base HoME、PCGrad、composite sampling、
   virtual oversampling、100-epoch 80/20 joint/refinement、loss、scheduler 与 evaluation；不做
   HPO 或 fallback。
10. reporting identity 固定为 Stage2 `rdkit_2d_stage2` / `RDKit 2D MLP + Stage2`，Stage3
    `rdkit_2d_stage2_home` / `RDKit 2D MLP + Stage2 + HoME`。Stage2 Core supported，Partial
    Charge 与 Full unsupported；Stage3 validation 要求 `21×5`，test 仅评价实际非空 test
    split 并对五折 raw prediction 逐样本 ensemble。结果进入现有 summarizer，不预设胜负阈值。

## 后果

- 多层 MLP 会从 Stage2 supervision 学习 representation，因此该实验衡量的是 Stage1
  multimodal pretraining 相对于 handcrafted descriptors + Stage2-supervised MLP 的贡献，
  不是“learned representation 与完全固定特征”的比较。
- 实际 MLP 输入宽度可以因全无效列移除而小于 217；preprocessor、retained width、RDKit
  version 与两个模型 state hash 都进入 artifact identity。
- Stage2 八任务 validation 指标继续作为训练诊断；统一 leaderboard 只发布 Core，避免将
  unsupported 的 atom-level Partial Charge 伪装成缺失结果。

## 备选方案

- 拒绝保留 partial atomic charge：分子级 descriptor 无法提供 atom-wise state，增加专用 atom
  encoder 会改变本消融问题。
- 拒绝每 role/task 独立 MLP：它会引入额外 representation capacity，并破坏共享 encoder 语义。
- 拒绝从 Stage1 descriptor artifact 读取标准化值：这会重新引入 Stage1 artifact 依赖并使
  preprocessing 不再严格 train-only。
- 拒绝为本路径单独 HPO：实验应复用正式 Base recipe，而不是比较不同优化预算。

## 参考

- [ADR-0019：Stage2 Object v3](0019-stage2-catalog-object-v3.md)
- [ADR-0020：Stage3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0023：统一 reporting](0023-unified-evaluation-reporting.md)
- [ADR-0027：late task-wise refinement](0027-late-taskwise-refinement.md)
- [ADR-0034：RDKit 2D → HoME representation 消融](0034-rdkit-2d-home-representation-ablation.md)
