# ADR-0081: Stage2–Stage3 HoME 跨域专家共享

## 状态

历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md). 原跨域共享实验已退出活跃代码。

## 背景

初始化迁移不能保证模拟专家在实验适配后仍作为物理先验。
本实验保留完整模拟 HoME，并在实验 HoME 的 GLOBAL/GROUP
内部进行两域融合。它同时改变专家使用方式、L2 候选结构和重放，不能将
结果单独归因于重放，也不能将测试集用于选择训练配置。

## 决策

- 独立配置为 `configs/ablations/stage2_stage3_cross_domain_home.yaml`，入口为
  `scripts/stage3/cross_domain_home.py export-source|prepare|train|evaluate`。
- 来源必须是 ADR-0079 已完成的完整 epoch10 检查点；只读导出表示编码器、
  全部模拟 HoME 与原子 adapter。旧部分迁移产物不足以构造源分支，
  禁止随机补齐。导出绑定原训练身份、检查点 SHA、架构与逐块张量 hash。
- 实验专家与任务模块按 Base 种子规则初始化，不从源复制 GLOBAL/GROUP。
  L1 GLOBAL sim/exp 各自内部归一化，随后由 object-input 两输出域门控融合；
  L1 GROUP 当前实验组与全部模拟 GROUP 专家分别汇总再融合。
  L2 selectors/域 gates 使用 `concat(z_global, local)`。最终任务门控仅有
  一个融合 GLOBAL、一个融合 GROUP 和当前实验 PRIVATE 池。
- 新 GLOBAL routers 属于实验 GLOBAL，GROUP routers 属于当前实验 GROUP，
  沿用其 LR/存续期。三个模拟 GROUP 全可见，无人工映射。
  Simulation PRIVATE、源归一化/FiLM/tower 和原子 adapter 仅服务原模拟任务。
- Stage1 全程冻结。ObjectEncoder 与模拟 GLOBAL/GROUP 的实验梯度按
  `weighted_owner_raw_v1` 的 GLOBAL 口径聚合；不恢复 PCGrad。
  Phase1 15 原始轮，模拟 owners LR=1e-5、5% 按owner更新次数预热、
  余弦至0.1倍；模拟 PRIVATE 仅其重放时推进调度器。
- 每四个实验次更新在同一个逻辑步添加一个仅训练集重放 batch，九任务按
  ID稳定排序轮转，各自无放回随机排列/cycle，真实短尾、不补齐。
  逻辑 batch256、microbatch128。任务系数为配置权重除九任务平均权重，
  无额外规模补偿；沿用 QM 带mask-目标-macro 与 partial-charge 分子等权 SmoothL1。
  权重 `0.1 * (1 - (u - 1) / (15K - 1))`，零权重不消费 batch。
  实验梯度加重放梯度后逐 owner clip=1，只有一次 AdamW 更新。
  重放使用独立 torch RNG，退出恢复实验 RNG。
- Phase1 后模拟全部模块与 ObjectEncoder 冻结 eval；其专家仍参与预测。
  Phase2/3 无重放，只拼接原实验 GROUP/PRIVATE owner，固定末轮发布。
  验证只报告，不选检查点；测试集只做最终五折集成。
- Prepare 另外保存冻结 Stage1 重放实体/原子缓存；新准备产物合同依据
  绑定来源、普通 Stage3 准备产物身份与缓存 SHA/hash。split、ObjectKey
  顺序和归一化与20-任务 Base逐项核验，不写回 Base 准备产物。
- 独立训练身份合同12、plan format10、检查点/final kind，与 Base、
  HoME 迁移、全量微调互斥。完整 Phase1 检查点保存优化器、owner
  调度器、重放游标/cycle、独立 RNG、实验 RNG、更新/冻结与日志尾。
  完整轮恢复校验所有状态；损坏不截断、不猜测。Final包含两域 HoME、
  ObjectEncoder、来源合同与 hashes，依赖只读源和冻结输入进行评估。
- 原七项门控诊断按最终 family 候选汇总；额外报告 L1/L2 sim/exp
  域权重占比、归一化熵、各模拟 GROUP selector 权重占比、重放
  任务/更新/样本/loss/权重，以及 `final family mass × sim domain mass`
  的 effective contribution。Selector 组权重占比为条件比例，不等同最终贡献。
  测试按折×样本池化；预测 CSV与公共报告结构定义不变。

## 后果

新增可持续使用的 physics 专家 prior，同时明显增加计算量、显存和恢复状态。
不采用超大扁平候选池、不共享模拟 PRIVATE、不引入 prior loss或OOD detector。
正式 Base 数值路径与所有既有结果不变。先比较 system-split 五折 macro/逐任务 NMAE
及实际 prior 使用量，再报告测试集；实现时不执行正式训练或评估。

## 参考

- [ADR-0079](0079-stage2-home-stage3-home-transfer-ablation.md)
- [ADR-0075](0075-stage2-zero-update-stage3-object-phase1.md)
- [ADR-0070](0070-stage3-retire-pcgrad.md)
- [ADR-0054](0054-stage3-task-gate-diagnostics-and-weak-task-tuning.md)
