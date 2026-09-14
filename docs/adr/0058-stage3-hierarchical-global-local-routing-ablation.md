# ADR-0058：Stage 3 hierarchical GLOBAL-vs-LOCAL routing 消融

- 状态：Accepted
- 日期：2026-09-14
- 关联：ADR-0054 的 task-gate diagnostics、ADR-0056 的 inference-only routing ablation

## 背景

现役 Stage 3 在 L2 将 GLOBAL、当前 GROUP 和当前 PRIVATE experts 作为平级 candidates，
由一个 task-specific gate 统一 softmax。为单独检验显式建模 GLOBAL/LOCAL 决策是否改善
OOD 泛化，需要建立与现役 Base 训练合同一致的结构消融。

## 决定

1. `model.routing.type` 支持 `flat` 与 `hierarchical`，默认 `flat`。默认字段在序列化和
   resolved plan 中省略，现役 flat checkpoint、identity 和数值路径保持不变。
2. hierarchical 模式为每个 task 建立三个使用相同 `[z_global; local]` 输入的线性 gate：
   GLOBAL/LOCAL family gate、GLOBAL internal gate，以及当前 GROUP+PRIVATE 的 LOCAL
   internal gate。三个模块均属于 `PRIVATE:<task>`。
3. 最终 GLOBAL candidate weights 为 `alpha_global * global_internal`；最终 GROUP/PRIVATE
   weights 为 `alpha_local * local_internal`。LOCAL 不得访问其他 GROUP 的参数或输出。
4. 该结构固有地比 flat gate 每个 task 多两个输出神经元；这是唯一允许的参数增量。
   expert、Tower、FiLM、normalization、dropout、三阶段训练、LR、owner lifetime、PCGrad、
   clipping、sampling、optimizer和loss全部保持。
5. 仅新增 `configs/ablations/stage3_hierarchical_routing.yaml`，复用 v2 Base prepared
   artifact与Stage 2 checkpoint。random/system/individual、RDKit-HoME和No-Stage1不扩展。
6. hierarchical 在原 gate diagnostics 之外报告 family mass及分位数、GLOBAL/LOCAL
   internal entropy。entropy使用归一化Shannon定义，单candidate family定义为0。
7. ADR-0056 的 `--routing-mode` 继续只作用于最终candidate weights。forced mode下旧mass
   字段反映干预结果，hierarchical专属字段保留learned router决策。

## 后果

- hierarchical routing进入现有training identity contract v5和plan format v3；与flat
  checkpoint不可交叉恢复或评估，但prepared artifact可直接复用。
- hierarchical evaluation默认study ID带`model-routing-hierarchical`后缀；prediction CSV
  与common reporting comparison schema不变。
- 该消融必须重新进行Stage 3五折训练、validation和test；无需重跑Stage 1、Stage 2或baseline。

## 关联

- [ADR-0048：owner-lifetime三阶段训练](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0054：弱任务微调与task-gate diagnostics](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
- [ADR-0056：inference-only task-gate routing ablation](0056-stage3-inference-only-routing-ablation.md)
