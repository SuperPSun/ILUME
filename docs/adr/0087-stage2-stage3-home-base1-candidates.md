# ADR-0087：十组 Stage2/Stage3 HoME Base 候选

- 状态：已接受（隔离的调参候选；不替代正式 Base）
- 日期：2026-09-28
- 依赖：[ADR-0082](0082-home-mainline-and-core-ablations.md)、[ADR-0084](0084-stage3-simulation-phase2-phase3.md)

## 决定

`configs/v3/stage3/candidates/base1-1.yaml` 至 `base1-10.yaml` 是十组自包含的 Stage3 候选。改变 Stage2 来源的前六组各有同名 Stage2 配置；其余四组复用正式 Stage2 来源和 Stage3 准备产物数据。每一组都从正式 Base 独立派生，不按编号继承。任务、数据 split、loss、owner 聚合、优化顺序和末轮发布方式不变。

| 候选 | 相对正式 Base 的配方变化 |
|---|---|
| base1-1 | Stage2/3 GLOBAL 专家 2→3 |
| base1-2 | Stage2/3 thermophysical 专家 2→3、solvation 3→4；Stage3 electronic_structure 2→3 |
| base1-3 | Stage2/3 GLOBAL expert_hidden_ratio 2.0→2.5 |
| base1-4 | Stage2/3 thermophysical、solvation expert_hidden_ratio 1.5→2.0；Stage3 electronic_structure 同步 1.5→2.0 |
| base1-5 | Stage2 轮 10→8 |
| base1-6 | Stage2 轮 10→12 |
| base1-7 | Stage3 Phase1 GLOBAL 与 ObjectEncoder 轮 15→18 |
| base1-8 | Stage3 phase_stability GROUP Phase2 轮 20→24 |
| base1-9 | Stage3 large PRIVATE 类 Phase3 轮 8→10，五项模拟任务同步 |
| base1-10 | Stage3 Phase1 GLOBAL LR 2.5e-4→3.0e-4 |

Stage2 HoME 仍使用九任务仅物理监督和 256 逻辑 batch/微批。候选允许 Stage2 在 8、10、12 轮之一按配置指定末轮发布；`stage2_final.pt`、清单、`stage2_encoder.pt` 的内容与 SHA/owner 身份合同不变。最终检查点及清单的 `fixed_final_epoch` 必须等于产物内训练配方的轮，Stage3 来源加载同样核对。现有正式 Base 固定 10 轮且保持原身份。

前六组各自训练 Stage2，使用共享的正式九任务准备产物数据，Stage3 用对应编码器和完整 final 重新 prepare，再以独立输出根目录训练五折。后四组只训练 Stage3，直接使用正式 Stage2 来源和正式 Stage3 准备产物数据。所有候选的训练、验证集、测试集、模拟评估均输出到 `outputs/v3/stage3/base1-N/` 下的独立目录；前六组的 Stage2 训练输出到 `outputs/v3/stage2/base1-N/train`。各组产物不交叉加载或恢复。

候选属于开发集调参；本次只发布配置，不运行正式训练、不自动选择胜者。用户统一训练后自行比较，不能把挑选候选所用五折验证当成未参与选择的无偏测试结果。
