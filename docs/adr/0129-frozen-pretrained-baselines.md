# ADR-0129：冻结预训练模块的独立基线协议

- 状态：已接受
- 日期：2026-11-14
- 范围：六个预训练 baseline；不修改现有微调结果及 Stage 1/2/3

## 决定

新增 `headft_v4_3407_22`，配置位于 `configs/benchmarks/Headftv4_3407/`，显式声明 `training.adaptation_mode: frozen_pretrained`。输出为 `outputs/benchmarks/Headftv4_3407/<model>/`。六模型各训练22项实验五折及热膨胀、HOMO、LUMO三项模拟，共678次训练。hydration继续random/cv1、无测试集，其他实验遵循当前registry；HOMO/LUMO仍读取共享`data/stage1/properties/`。

| 模型 | 冻结部分 | 可训练任务模块 |
|---|---|---|
| MoLFormer | 已发布backbone | fusion、classifier |
| ILBERT | RoBERTa | TextCNN、predictor |
| SPMM | 已发布文本编码器 | predictor |
| LlaSMol | Mistral与已发布LoRA | predictor |
| AIonopedia | Qwen／已发布LoRA、官方GNN、投影层、拓扑embedding及segment、graph merge、两个decoder | 当前任务使用的新增条件投影及segment、fc_out |
| ILTransR | 已发布embedding／Transformer | TextCNN、predictor |

MLP、D-MPNN、AIFC没有本地通用预训练初始化，不纳入；ILUME与XGBoost也不纳入。所有模型从各自通用预训练资产初始化，不读取现有下游监督训练checkpoint。LlaSMol保留NF4/BF16实现和已发布adapter，冻结后不再更新LoRA，不改成纯Mistral；AIonopedia同样保留完整官方多模态初始化，不改成纯Qwen。

## 训练边界

沿用各模型现役固定10轮、种子、batch、任务侧学习率、损失、权重衰减、预设调度器及末轮发布规则；只移除冻结参数的优化器组。ILBERT采用0119训练行条件z-score，ILTransR保留0057公开登记的传导式条件统计，其他预处理及输入/条件/目标不变。验证只报告，不早停或选择轮次；失败由新attempt从头重跑，不新增恢复支持。

冻结模块始终eval，只有任务侧模块train。AIonopedia新增条件模块位于冻结融合模块之前，必须允许梯度穿过后者回传到条件模块，不能对整个路径使用no_grad。未使用的新增条件模块冻结；官方温度通路保留并冻结。

新协议及adaptation_mode进入原训练身份。旧配置省略此字段且继续原行为；新旧协议拒绝交叉加载，不改变旧hash或原生全量微调审计。LlaSMol的学习率历史将冻结LoRA的学习率记为null，任务头仍按原head学习率调度。

## 冻结审计与保存

每次训练保存`frozen_audit.json`，绑定训练身份、完整参数角色/形状/类型、可训练参数量、任务模块白名单、优化器参数名单及冻结参数/全部buffer的训练前后指纹。冻结参数不能有梯度，优化器必须恰好包含任务侧参数且不得重复，冻结模块不能处于train模式；冻结状态变化立即失败，不发布成功checkpoint。

审计纳入checkpoint完整性清单；现有完整模型保存方式保留，LlaSMol仅保存可训练任务头，并继续依赖SHA锁定的原始基座及官方adapter。评估先验证身份、审计文件及SHA，再严格恢复，比较恢复后的冻结状态与审计指纹；禁止缺审计或跨协议替代。完整运行、日志、审计、权重及依赖声明回传4090。

## 独立报告

报告协议为`headft_v4_3407_22_report`，沿用`scripts/benchmarks/summarize.py --baseline-config`。读取六模型完成sweep的固定`source_runs`及评估checkpoint来源，不扫描挑选有利attempt，不复用FLft审计或预测。逐项核验配置、数据SHA、冻结审计、完整模型与预测、行集合/顺序及五模型集成；缺模型、任务、折、审计或测试评估时拒绝发布。

输出22实验验证、实际10实验测试、三模拟验证/测试四张独立任务等权榜单，归一化保持0123的共同尺度。新比较身份绑定六模型协议和来源，展示任务侧模块及可训练参数量范围，不混入旧微调榜单、ILUME或随机特征对照。

该实验衡量“固定预训练表示在原生任务侧模块及原配方下的可用性”，不是统一线性探针，也不能排除任务模块容量与优化难度差异。固定10轮不是各模型冻结适配的最优预算声明。

验证使用临时小数据和小型编码器替身，覆盖六模型原生前向、冻结/梯度/优化器、AIonopedia条件梯度、LoRA保留及保存恢复、跨协议拒绝和报告完整性。正式训练/模型评估由用户启动；操作见[运行手册](../baseline-headft-runbook.md)。
