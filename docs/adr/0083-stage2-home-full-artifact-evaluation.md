# ADR-0083：完整 Stage2-HoME 产物与独立评估

- 状态：已接受（完整产物合同）；评估与独立榜单部分由 ADR-0085 取代
- 日期：2026-09-27
- 修订：扩展 ADR-0082 的 `stage2_final.pt` 合同；在正式 Stage2-HoME 范围取代 ADR-0043 的 Stage2 评估/报告退役决定。旧 Stage2 基线/Core/Partial/完整榜单仍为历史。

> 后续边界：[ADR-0084](0084-stage3-simulation-phase2-phase3.md) 扩展模拟 GROUP/PRIVATE 迁移；[ADR-0085](0085-retire-stage2-home-evaluation.md) 退役本篇独立 evaluator/报告。以下评估描述仅保留为历史，完整九任务产物与预测接口仍有效。

## 决定

第 10 轮末轮的 `stage2_final.pt` 是可独立加载的完整九任务模拟模型，而不只是迁移桥。它保存 Stage1 backbone、ObjectEncoder、GLOBAL/GROUP/PRIVATE HoME、路由、任务 towers、原子 adapter 的完整状态，并嵌入九任务注册表、正式配方、Stage1 特征快照与 Stage2 仅训练集 scaler。清单绑定来源身份、完整状态hash、owner 清单、产物 SHA 与最终轮次。旧 v1 transfer-仅 kind 拒载；周期检查点仅用于恢复训练和来源记录，Stage3 不再依赖它加载 final。

Stage3 仍只接收原有 GLOBAL 与 thermophysical、solvation GROUP，以及相同的表示编码器。加载器从完整模型严格校验并提取这些张量，要求与 final 中的 transferable 快照逐张量一致。模拟 electronic GROUP、PRIVATE、路由、tower 与原子 adapter 不迁移。

正式 `scripts/stage2/evaluate.py` 仅加载 `stage2_final.pt`，提供验证集/测试集两种只读评估。模型层继续提供九任务 `SimulationHoME.predict`，此阶段的正式报告固定为汽化热、热膨胀、HOMO、LUMO、partial atomic charge 五项；simulated QM electrostatic/HF 仍保留完整权重，但不进入本版 evaluator。验证集使用准备产物 tensors，测试集按任务目录原始 split、正式特征 QC、仅训练集 scaler 和确定性 MOL2 映射构建输入；原子 CSV 使用 canonical RDKit 原子 index。未映射或特征 QC 失败的测试集行进入审计，不由预测误差决定筛选。

逐任务发布原单位 MAE/RMSE/R² 与归一化MAE/RMSE。partial charge 的主 MAE 和归一化MAE 先在分子内聚合，再对分子等权；另记原子-micro 诊断。headline 为已评估任务的等权 macro 归一化MAE，不聚合原单位 MAE。当前热膨胀测试集 split 已补齐，测试集纳入五项；若某项 split 缺失或为空，评估器仍只纳入实际存在的任务。任务集合、测试来源 hash、准备产物身份、scaler 和末轮状态进入评估身份，不同任务集合不混排。

统一 summarizer 接收 Stage2 结果，但以 `stage2_property` 比较身份和 `stage2_{test,validation}_*` 文件单独发布；增加输出文件与 JSON section，因此汇总快照结构定义升为 v2，单个评估的报告结构定义保持 v1。Stage2 不参与 Stage3 榜单、wins、radar 或 scatter。完成状态、预测文件 SHA/行数和任务覆盖必须完整才能发布；旧 Stage2 报告继续作为历史结果，不恢复旧基线 sweep。

## 验证边界

只用临时小数据测试完整状态重载、迁移张量一致、拒载旧 kind/篡改状态、标量/原子指标、测试集任务集合和报告隔离。不运行正式 prepare、训练、测试集或五折评估，也不迁移或覆盖旧输出。
