# ADR-0036：No-Stage1 RDKit 2D → Stage2 → Stage3 HoME 消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-01

> 2026-09-07：Stage 3 采样与联合训练裁剪已随现役 Base 由 [ADR-0046](0046-stage3-ownership-clipping-raw-sampling.md) 修订；本消融继续只替换 Stage 1 表示。
>
> 2026-09-07：Stage 3 四阶段训练与 final 产物随现役 Base 由
> [ADR-0047](history.md#adr-0047) 修订；本消融仍不改变 Stage 3 优化合同。
>
> 2026-09-10：现役 Stage 3 优化合同进一步由
> [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 取代为 owner-存续期三阶段训练；
> 本消融继续继承 Base 的 Stage 3 数值合同。

## 背景

现役 ILUME 依次使用 Stage1 多模态预训练 backbone、Stage2 ObjectEncoder 和
Stage3 HoME。ADR-0034 已同时移除 Stage1/2 表示；为了单独研究 Stage1
预训练的贡献，需要保留 Stage2 supervision 与 Stage3 全部训练合同，只替换 Stage2
接收的分子表示。

该实验比较：

`Stage1 pretrained backbone → Stage2 ObjectEncoder → Stage3 HoME`

与：

`RDKit 217D → shared Stage2-supervised MLP → Stage2 ObjectEncoder → Stage3 HoME`。

## 决定

1. 配置固定为 `configs/ablations/no_stage1_rdkit_stage2.yaml` 与
   `configs/ablations/no_stage1_rdkit_stage3.yaml`，输出根为
   `outputs/ablations/no_stage1_rdkit_stage2_stage3`。现役 `configs/v1` 与正式输出不变。
2. Stage2 不读取 Stage1 prepare 产物、检查点、teacher embedding 或 learned 状态；
   只复用 `stage1.descriptors` 的既定 RDKit-217 名称、顺序与计算实现。
3. 全部组分角色共享一个描述符编码器：
   `Linear(retained_width, 1024) → GELU → Dropout(0.10) → Linear(1024, 512)
   → LayerNorm(512)`。角色仍由 ObjectEncoder 的角色embedding 表达；IL 槽位顺序、
   交互、条件与任务预测头路径保持不变，条件不进入描述符编码器。
4. 预处理在八个支持 Stage2 任务的全部训练集-行组分 occurrence 联合池
   上拟合，重复 occurrence 保留权重。全无有限训练值的列删除，非有限值按训练集 median
   填充，按 population 均值/std 标准化，近常量规模设为 1，最终 clip 到 `[-10, 10]`。
   验证/测试集不参与拟合。
5. `simulation/partial_atomic_charge` 不受分子级 RDKit 描述符支持，从本消融的注册表、
   联合训练与精调中移除。其余八任务按现役任务-权重规则重新归一化。
6. 联合训练 phase 不生成 teacher 缓存，loss 只含原 physics loss。共享 MLP 与 ObjectEncoder 从
   轮 1 共同训练，学习率均为 `3e-5`；5 个联合训练轮、batch schedule、单 batch 单
   优化器步、BF16、优化器和调度器沿用 Base。
7. 额外 10 个精调轮仅覆盖汽化热、HOMO 与 LUMO；共享 MLP 和
   ObjectEncoder 冻结，只更新当前任务预测头。普通检查点、逐任务精调后产物与
   编码器使用 RDKit 专属 kind，拒绝与现役 Stage2 产物双向交叉加载。
8. `stage2_encoder.pt` 内嵌描述符结构定义、RDKit version、preprocessor、共享 MLP、
   ObjectEncoder、角色 mapping 与状态身份，不包含 Stage1 或 teacher 身份。Stage3
   prepare 直接使用该冻结编码器，不重新拟合预处理。
9. Stage3 继续覆盖 21 任务、5折，并完整沿用 Base HoME、PCGrad、复合采样、
   虚拟过采样、100轮 80/20 联合训练/精调、loss、调度器与评估；不做
   HPO 或 fallback。
10. 报告身份固定为 Stage2 `rdkit_2d_stage2` / `RDKit 2D MLP + Stage2`，Stage3
    `rdkit_2d_stage2_home` / `RDKit 2D MLP + Stage2 + HoME`。Stage2 Core 支持，Partial
    Charge 与完整 unsupported；Stage3 验证要求 `21×5`，测试集仅评价实际非空测试集
    split 并对五折原始预测逐样本集成。结果进入现有 summarizer，不预设胜负阈值。

## 后果

- 多层 MLP 会从 Stage2 supervision 学习表示，因此该实验衡量的是 Stage1
  多模态预训练相对于 handcrafted descriptors + Stage2-监督训练 MLP 的贡献，
  不是“learned 表示与完全固定特征”的比较。
- 实际 MLP 输入宽度可以因全无效列移除而小于 217；preprocessor、retained 宽度、RDKit
  version 与两个模型状态hash 都进入产物身份。
- Stage2 八任务验证指标继续作为训练诊断；统一榜单只发布 Core，避免将
  unsupported 的原子-level Partial Charge 伪装成缺失结果。

## 备选方案

- 拒绝保留 partial atomic charge：分子级描述符无法提供原子-wise 状态，增加专用原子
  编码器会改变本消融问题。
- 拒绝每角色/任务独立 MLP：它会引入额外表示 capacity，并破坏共享编码器语义。
- 拒绝从 Stage1 描述符产物读取标准化值：这会重新引入 Stage1 产物依赖并使
  预处理不再严格仅训练集。
- 拒绝为本路径单独 HPO：实验应复用正式 Base 配方，而不是比较不同优化预算。

## 参考

- [ADR-0019：Stage2 Object v3](0019-stage2-catalog-object-v3.md)
- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0023：统一报告](0023-unified-evaluation-reporting.md)
- [ADR-0027：后期逐任务精调](0027-late-taskwise-refinement.md)
- [ADR-0034：RDKit 2D → HoME 表示消融](0034-rdkit-2d-home-representation-ablation.md)
