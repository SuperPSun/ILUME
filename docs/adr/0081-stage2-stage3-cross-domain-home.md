# ADR-0081: Stage2–Stage3 HoME 跨域专家共享

## Status

Historical / Superseded by [ADR-0082](0082-home-mainline-and-core-ablations.md). 原跨域共享实验已退出活跃代码。

## Context

初始化迁移不能保证 simulation 专家在实验适配后仍作为 physics prior。
本实验保留完整 simulation HoME，并在 experimental HoME 的 GLOBAL/GROUP
内部进行两域融合。它同时改变专家使用方式、L2 候选结构和 replay，不能将
结果单独归因于 replay，也不能将 test 用于选择训练配置。

## Decision

- 独立配置为 `configs/ablations/stage2_stage3_cross_domain_home.yaml`，入口为
  `scripts/stage3/cross_domain_home.py export-source|prepare|train|evaluate`。
- 来源必须是 ADR-0079 已完成的完整 epoch10 checkpoint；只读导出表示编码器、
  全部 simulation HoME 与 atom adapter。旧部分迁移 artifact 不足以构造源分支，
  禁止随机补齐。导出绑定原训练身份、checkpoint SHA、架构与逐块 tensor hash。
- experimental 专家与任务模块按 Base seed 规则初始化，不从源复制 GLOBAL/GROUP。
  L1 GLOBAL sim/exp 各自内部归一化，随后由 object-input 两输出 domain gate 融合；
  L1 GROUP 当前实验组与全部 simulation GROUP experts 分别汇总再融合。
  L2 selectors/domain gates 使用 `concat(z_global, local)`。最终 task gate 仅有
  一个融合 GLOBAL、一个融合 GROUP 和当前实验 PRIVATE pool。
- 新 GLOBAL routers 属于实验 GLOBAL，GROUP routers 属于当前实验 GROUP，
  沿用其 LR/lifetime。三个 simulation GROUP 全可见，无人工映射。
  Simulation PRIVATE、源 normalization/FiLM/tower 和 atom adapter 仅服务原模拟任务。
- Stage1 全程冻结。ObjectEncoder 与 simulation GLOBAL/GROUP 的实验梯度按
  `weighted_owner_raw_v1` 的 GLOBAL 口径聚合；不恢复 PCGrad。
  Phase1 15 raw epochs，simulation owners LR=1e-5、5% owner-update warmup、
  cosine 至0.1倍；simulation PRIVATE 仅其 replay 时推进 scheduler。
- 每四个实验 updates 在同一个逻辑步添加一个 train-only replay batch，九任务按
  ID稳定排序轮转，各自无放回 permutation/cycle，真实短尾、不补齐。
  逻辑 batch256、microbatch128。任务系数为配置 weight 除九任务平均 weight，
  无额外规模补偿；沿用 QM masked-target-macro 与 partial-charge 分子等权 SmoothL1。
  权重 `0.1 * (1 - (u - 1) / (15K - 1))`，零权重不消费 batch。
  实验梯度加 replay 梯度后逐 owner clip=1，只有一次 AdamW update。
  Replay 使用独立 torch RNG，退出恢复实验 RNG。
- Phase1 后 simulation 全部模块与 ObjectEncoder 冻结 eval；其专家仍参与预测。
  Phase2/3 无 replay，只 stitch 原实验 GROUP/PRIVATE owner，固定末轮发布。
  Validation 只报告，不选 checkpoint；test 只做最终五折 ensemble。
- Prepare 另外保存冻结 Stage1 replay entity/atom cache；新 prepared authority
  绑定 source、普通 Stage3 prepared identity 与 cache SHA/hash。split、ObjectKey
  顺序和 normalization 与20-task Base逐项核验，不写回 Base prepared。
- 独立训练 identity contract12、plan format10、checkpoint/final kind，与 Base、
  HoME Transfer、全量微调互斥。完整 Phase1 checkpoint 保存 optimizer、owner
  scheduler、replay 游标/cycle、独立 RNG、实验 RNG、update/freeze 与日志尾。
  完整 epoch resume 校验所有状态；损坏不截断、不猜测。Final包含两域 HoME、
  ObjectEncoder、来源合同与 hashes，依赖只读源和冻结输入进行评估。
- 原七项 gate diagnostics 按最终 family candidates 汇总；额外报告 L1/L2 sim/exp
  domain mass、归一化 entropy、各 simulation GROUP selector mass、replay
  task/update/sample/loss/weight，以及 `final family mass × sim domain mass`
  的 effective contribution。Selector group mass为条件比例，不等同最终贡献。
  Test 按 fold×sample池化；prediction CSV与公共 reporting schema不变。

## Consequences

新增可持续使用的 physics expert prior，同时明显增加计算量、显存和恢复状态。
不采用超大扁平候选池、不共享 simulation PRIVATE、不引入 prior loss或OOD detector。
正式 Base 数值路径与所有既有结果不变。先比较 system-split 五折 macro/逐任务 NMAE
及实际 prior 使用量，再报告 test；实现时不执行正式训练或 evaluation。

## References

- [ADR-0079](0079-stage2-home-stage3-home-transfer-ablation.md)
- [ADR-0075](0075-stage2-zero-update-stage3-object-phase1.md)
- [ADR-0070](0070-stage3-retire-pcgrad.md)
- [ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
