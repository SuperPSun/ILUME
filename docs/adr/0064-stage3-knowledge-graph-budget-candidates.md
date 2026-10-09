# ADR-0064：Stage 3 知识图谱分组容量与预算候选

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-21
- 配置：`configs/v2/stage3/base1_1.yaml` 至 `base1_5.yaml`

## 背景

沿用 [ADR-0063](0063-stage3-knowledge-graph-grouping-candidate.md) 的六组和20个任务归属；
任务集合由 [ADR-0067](0067-stage3-twenty-task-catalog.md) 修订。
已有 base1 五折日志中，thermophysical/interfacial 的 Phase 2 任务等权 NMAE
在轮 3 约为0.20855、轮 4约为0.20993；static singleton 的对应三轮均值
约为0.52327、0.53504、0.54172。这些现象用于提出固定预算假设，不证明容量不足或过拟合的因果。
本轮不延长大组 Phase 2，而通过四个诊断候选与一个预先固定的组合检验改进空间。

## 决定

以下全部是相对 `base1.yaml` 的差异，五份配置均自包含，不能按编号逐级继承：

| 配置 | 固定改动 | 检验假设 |
|---|---|---|
| base1_1 | thermophysical_interfacial_response 专家 2→3 | 大组容量 |
| base1_2 | 该组 Phase 1 轮 10→15 | GLOBAL继续训练时GROUP的适配寿命 |
| base1_3 | 该组 Phase 1 LR 1.5e-4→2e-4，Phase 2 LR 7.5e-5→1e-4 | GROUP学习强度 |
| base1_4 | static_dielectric Phase 1 LR 1e-4→5e-5；Phase 2 LR 5e-5→2.5e-5、轮 3→1；static PRIVATE Phase 1 LR 4e-5→2e-5 | singleton较保守的local训练预算 |
| base1_5 | 合并以上四项 | 组合效果 |

1. thermophysical/interfacial Phase 2固定4 轮；GLOBAL固定2 专家、15 轮。
   其他组、全部任务容量、dropout与PRIVATE 轮逐项保持base1。
2. `base1_4/5`唯一的PRIVATE LR例外为static，其Phase 2/3 LR由既有规则自动变为
   `1e-5/5e-6`；PRIVATE 轮仍为`4/1/0`、三种比例仍为0.25。
   static GROUP Phase 1仍为8 轮，专家数和隐藏宽度比例不变。
3. `base1_1/5`增加目标组各一个L1/L2 专家，八个任务门控宽度由5变6，L1 组
   门控宽度由2变3。其余专家、FiLM、归一化和tower容量不变。
   改变模块形状可能改变后续随机初始化的取样位置；相同种子不保证未改形状模块的初值逐bit相同。
4. Phase 1严格维持GLOBAL > GROUP > PRIVATE nominal LR，GROUP Phase 2起始LR
   等于Phase 1 terminal LR。Flat、原始样本采样、PCGrad、按参数归属裁剪、优化器、
   loss、固定-末轮状态和诊断合同不变，不修改运行时/结构定义或身份版本号。
5. base1_3按LR连续性联动两阶段；base1_4是LR与存续期的联合正则化候选，不能单独归因到
   一个标量。base1_5用于组合验证，不以其表现反向解释单项因果。

## 比较与产物

复用Base 准备产物及Stage 2 编码器。每份完整配方生成独立训练身份；
从头训练五折，输出各自 `outputs/v2/stage3/base1_N/train`，不跨候选恢复/加载。
历史Base/base1产物只读，不覆盖。实现验收只使用小型测试，不执行正式训练或评估。

先比较五个候选的system-split五折任务等权 macro NMAE、逐折/逐任务结果与五折方差，
重点审计static和thermophysical/interfacial八任务、门控权重占比变化。每个运行固定使用末轮，
不early stop或验证最优。候选比较属于开发集调参，不能将选中候选的5CV当作未选择的
无偏估计。根据5CV确定一个候选后才运行测试集集成，不以测试集在五个候选之间选择。

## 关联

- [ADR-0048：三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：owner 配方与LR连续性](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0054：门控诊断](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
