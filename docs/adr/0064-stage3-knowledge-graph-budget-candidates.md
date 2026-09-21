# ADR-0064：Stage 3 知识图谱分组容量与预算候选

- 状态：Accepted（独立候选实验，未替代 Base/base1）
- 日期：2026-09-21
- 配置：`configs/v2/stage3/base1_1.yaml` 至 `base1_5.yaml`

## 背景

沿用 [ADR-0063](0063-stage3-knowledge-graph-grouping-candidate.md) 的六组和21个任务归属。
已有 base1 五折日志中，thermophysical/interfacial 的 Phase 2 task-equal NMAE
在 epoch 3 约为0.20855、epoch 4约为0.20993；static singleton 的对应三轮均值
约为0.52327、0.53504、0.54172。这些现象用于提出固定预算假设，不证明容量不足或过拟合的因果。
本轮不延长大组 Phase 2，而通过四个诊断候选与一个预先固定的组合检验改进空间。

## 决定

以下全部是相对 `base1.yaml` 的差异，五份配置均自包含，不能按编号逐级继承：

| 配置 | 固定改动 | 检验假设 |
|---|---|---|
| base1_1 | thermophysical_interfacial_response experts 2→3 | 大组容量 |
| base1_2 | 该组 Phase 1 epochs 10→15 | GLOBAL继续训练时GROUP的适配寿命 |
| base1_3 | 该组 Phase 1 LR 1.5e-4→2e-4，Phase 2 LR 7.5e-5→1e-4 | GROUP学习强度 |
| base1_4 | static_dielectric Phase 1 LR 1e-4→5e-5；Phase 2 LR 5e-5→2.5e-5、epochs 3→1；static PRIVATE Phase 1 LR 4e-5→2e-5 | singleton较保守的local训练预算 |
| base1_5 | 合并以上四项 | 组合效果 |

1. thermophysical/interfacial Phase 2固定4 epochs；GLOBAL固定2 experts、15 epochs。
   其他group、全部task容量、dropout与PRIVATE epochs逐项保持base1。
2. `base1_4/5`唯一的PRIVATE LR例外为static，其Phase 2/3 LR由既有规则自动变为
   `1e-5/5e-6`；PRIVATE epochs仍为`4/1/0`、三种ratio仍为0.25。
   static GROUP Phase 1仍为8 epochs，expert数和hidden ratio不变。
3. `base1_1/5`增加目标组各一个L1/L2 expert，九个task gate宽度由5变6，L1 group
   gate宽度由2变3。其余expert、FiLM、normalization和tower容量不变。
   改变模块形状可能改变后续随机初始化的取样位置；相同seed不保证未改形状模块的初值逐bit相同。
4. Phase 1严格维持GLOBAL > GROUP > PRIVATE nominal LR，GROUP Phase 2起始LR
   等于Phase 1 terminal LR。Flat、raw sampling、PCGrad、ownership clipping、optimizer、
   loss、fixed-final-state和诊断合同不变，不修改runtime/schema或identity版本号。
5. base1_3按LR连续性联动两阶段；base1_4是LR与lifetime的联合正则化候选，不能单独归因到
   一个标量。base1_5用于组合验证，不以其表现反向解释单项因果。

## 比较与产物

复用Base prepared artifact及Stage 2 encoder。每份完整recipe生成独立training identity；
从头训练五折，输出各自 `outputs/v2/stage3/base1_N/train`，不跨候选resume/load。
历史Base/base1产物只读，不覆盖。实现验收只使用小型测试，不执行正式训练或evaluation。

先比较五个候选的system-split五折task-equal macro NMAE、逐fold/逐task结果与五折方差，
重点审计static和thermophysical/interfacial九任务、gate mass变化。每个run固定使用末轮，
不early stop或validation-best。候选比较属于开发集调参，不能将选中候选的5CV当作未选择的
无偏估计。根据5CV确定一个候选后才运行test ensemble，不以test在五个候选之间选择。

## 关联

- [ADR-0048：三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：owner recipe与LR连续性](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0054：gate diagnostics](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
