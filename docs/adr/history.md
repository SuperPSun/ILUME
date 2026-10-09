# 已取代 ADR 简史

本页合并已取代的设计与配方试验过程，只用于追溯理由，不提供运行命令或兼容入口。现役决定从 [ADR 索引](README.md) 查找；仍约束历史实现/Capacity 的 ADR 不在本次合并范围。

0006～0047 条目的原始全文保存在 Git 提交 `1edf5d853bb736ffc59fe042ce17f54cf5e0a8a2`。需要精确复盘时，在仓库根运行 `git show 1edf5d8:docs/adr/<原文件名>`；原文件名见各条目。编号不重用。

<a id="adr-0006"></a>

## ADR-0006：覆盖型轮

2026-07-26 · 已被取代 · 替代合同：[ADR-0013](0013-stage1-full-corpus-ddp.md)。

为落实 45/45/10 角色比例，曾采用覆盖型轮、Base/Large/XLarge 与完整轮恢复。自然频率全量轮、单一 Base 取代该采样和容量设计；不得复用其旧检查点版本说明。

原文件：`0006-coverage-epochs-and-experiment-configs.md`。

<a id="adr-0007"></a>

## ADR-0007：五任务物性与冻结教师

2026-08-03 · 已被取代 · 替代合同：[ADR-0019](0019-stage2-catalog-object-v3.md)。

曾用冻结教师离线 CLS、五任务回归与固定 20-步任务块约束学生表示。离线教师的动机延续；五任务、步检查点和验证选优均不是现役 Object v3 合同。

原文件：`0007-stage2-property-alignment.md`。

<a id="adr-0008"></a>

## ADR-0008：Base batch 收敛

2026-08-03 · 已被取代 · 替代合同：[ADR-0013](0013-stage1-full-corpus-ddp.md)。

曾将三个容量的 effective batch 统一到 256，以消除当时 YAML、检查点与文档的分叉。该数值不是现役默认，也不授权自动调整 batch 或累积步数。

原文件：`0008-base-training-profile.md`。

<a id="adr-0009"></a>

## ADR-0009：体系采样与 PairEncoder

2026-08-04 · 已被取代 · 替代合同：[ADR-0019](0019-stage2-catalog-object-v3.md)。

为降低多温度点体系的采样偏重，曾按体系均匀采样、使用双 PairEncoder 并渐进解冻。QM 部分标签 mask 的需求保留；体系采样、双编码器和步恢复被逐行覆盖的 Object 设计取代。

原文件：`0009-stage2-system-sampling-and-progressive-unfreezing.md`。

<a id="adr-0010"></a>

## ADR-0010：延后溶质引入与阶段扩展

2026-08-05 · 已被取代 · 替代合同：[ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md)。

曾从 21 项 IL 任务扩展到 27 项任务，并在第二层加入 solute，避免改变直接 IL 的共享表示。该阶段扩展、旧 PairEncoder 缓存和最优检查点已退役。任务局部折不能解释为跨所有任务的联合冷启动。

原文件：`0010-stage3-home-and-late-solute.md`。

<a id="adr-0011"></a>

## ADR-0011：IL21/Aux6 双域隔离

2026-08-05 · 已被取代 · 替代合同：[ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md)。

曾以独立模型、优化器、RNG、早停和最优状态隔离 IL21 与 Aux6，防止辅助任务改变 IL 更新。现役使用任务目录驱动的稀疏 HoME；旧双域、aux6 与 Object v3 拒绝声明不再适用。

原文件：`0011-stage3-single-stage-isolated-domains.md`。

<a id="adr-0012"></a>

## ADR-0012：双域吞吐优化

2026-08-05 · 已被取代 · 替代合同：[ADR-0020](0020-stage3-v1-sparse-home-pcgrad.md)。

曾限制 CPU 线程、让张量常驻 GPU，并按域聚合反向，以保持双域预算和状态隔离。旧 blocks、batch 降级方案、域反向与 reference.yaml 不再是运行入口。

原文件：`0012-stage3-throughput-and-budget.md`。

<a id="adr-0016"></a>

## ADR-0016：Object v1

2026-08-13 · 已被取代 · 替代合同：[ADR-0019](0019-stage2-catalog-object-v3.md)。

引入统一 ObjectEncoder、逐行完整覆盖、任务 compensation 和完整轮检查点，以分离任务重要性与数据规模。固定五任务、旧拓扑/预测头路由与旧产物已由任务目录 Object v3 取代；不提供迁移。

原文件：`0016-stage2-universal-object-modeling.md`。

<a id="adr-0018"></a>

## ADR-0018：Object v2 吞吐

2026-08-14 · 已被取代 · 替代合同：[ADR-0019](0019-stage2-catalog-object-v3.md)。

通过内容寻址教师缓存、实体 preload、冻结快路径和跨任务累加窗口减少重复编码。Object v3 取代旧缓存/检查点与 window loss，现役一个 batch 对应一个优化器步；不可恢复 Object v2。

原文件：`0018-stage2-object-v2-throughput.md`。

<a id="adr-0047"></a>

## ADR-0047：四阶段训练

2026-09-07 · 已被取代 · 替代合同：[ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md)。

为分离 GLOBAL/GROUP/PRIVATE 学习率并取消验证最优选择，曾采用 A/B/C/D 四阶段、同源分支和锚点+owner delta。2026-09-10 被 owner-存续期三阶段完全取代；four_phase_final 不可加载、恢复或评估。历史实现 v1/Capacity 的 ADR-0027 合同未被替换。

原文件：`0047-stage3-deterministic-four-phase-training.md`。

<a id="adr-0051"></a>

## ADR-0051～0053：PRIVATE 配方试验与回滚

2026-09-12～13 · 历史数值已由 [ADR-0055](0055-stage3-pec50-phase3-single-variable-rollback.md)
汇总；0051 引入且仍有效的 dropout、Phase 2 零预算与身份机制已并入
[ADR-0050](0050-stage3-task-specific-owner-budget-and-private-capacity.md)。

| 决策 | 原因与主要变化 |
|---|---|
| 0051 弱任务定向正则化 | tiny 验证过拟合与部分任务测试集泛化偏弱；缩小 PRIVATE、缩短预算、引入任务 dropout，未改变 GLOBAL/GROUP 或训练算法。volume expansion Phase 2 先设为 0，thermal conductivity Phase 3 设为 4，pEC50 比例设为 0.75；部分值后续回滚。 |
| 0052 定向回滚与轮 cleanup | 对无改善或退化的 specialization 回滚：volume expansion Phase 2 恢复 3、pEC50 比例恢复 1.0、refractive index 比例恢复 0.75；thermal conductivity/自扩散/melting point/热分解缩短 Phase 3，viscosity/热分解 dropout 回到 0.10。 |
| 0053 干净 hybrid | 复核约 0.21173 的五折 macro 归一化MAE 基线后，只将 pEC50 比例回调 0.75、volume expansion Phase 2 延长至 4。0054 后再次将后者归零；最终规则不需通过这些中间态推导。 |

<a id="adr-0052"></a>
<a id="adr-0053"></a>

以上数值试验当时需要新的 Stage 3 训练身份与输出目录；Stage 1/2、准备产物数据、
基线与历史结果保持不变。该历史说明不要求当前纯结构重构重新训练。

原文保存在 Git 提交 `9fdc35d1c7286a9efda55a4fc9aa120a1c09755e`，使用 `git show 9fdc35d1c7286a9efda55a4fc9aa120a1c09755e:docs/adr/<原文件名>`：

- `0051-stage3-weak-task-private-regularization.md`
- `0052-stage3-task-recipe-rollback-and-epoch-cleanup.md`
- `0053-stage3-clean-hybrid-task-recipe.md`

ADR-0094 的 v5 Unary/Pair 及旧 v4 ObjectEncoder/消融合同已由 ADR-0095 替代；历史复现使用对应 Git 版本，历史输出只读。
