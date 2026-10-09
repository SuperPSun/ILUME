# ADR-0027：Stage 2/3 后期逐任务精调

- 状态：已接受
- 日期：2026-08-26
- 修订：ADR-0019、ADR-0020、ADR-0021、ADR-0026 中的末期训练、最终模型选择、检查点与 HPO 评分合同

> 2026-09-06：HPO 执行能力已退役。下文 HPO 相关段落仅记录历史评分语义；精调、
> 拼接后的验证、产物与普通训练合同继续有效。
>
> 2026-09-07：现役 v2 Stage 3 与对应消融的精调、选择和 final 产物已由
> [ADR-0047](history.md#adr-0047) 取代；本文相应条款仅继续约束
> 历史实现 v1 与 Capacity v1。
>
> 2026-09-10：ADR-0047 的现役合同已被
> [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 取代；本文仍只约束历史实现 v1
> 与 Capacity v1。

## 背景

联合多任务训练能在共享参数中迁移知识，但训练末期继续更新 shared 状态与使用跨任务
梯度平衡，会限制单任务参数独立收敛。部分任务因此不能超过对应单任务 MLP 基线。
本决定把精调纳入正式训练合同：Stage 2 先完成原定完整联合训练周期，再额外对
核心物理任务做仅预测头精调；Stage 3 继续在既定总轮内进行末期精调。

## 决定

### 共同训练几何

Stage 2 的 `training.epochs` 只表示完整联合训练轮，并显式配置
`training.refinement_epochs`、`training.refinement_tasks` 与
`training.refinement_lr_multiplier`。现役 v1 为 5 联合训练 + 10 精调；Capacity Stage 2
为 10 联合训练 + 10 精调。Stage 3 继续使用 `training.refinement_ratio: 0.20`：
30-轮探针为 24+6，50-轮正式为 40+10，Stage 3 v1 为 80+20。相关字段进入
训练身份，但不进入 HPO 搜索空间。

联合训练 phase 的预热+余弦在边界前的真实更新数内完整走完。精调不
预热；进入边界后，为每个任务新建互相独立的 AdamW 与余弦调度器，清空
原优化器的一阶、二阶动量。初始 LR 是相应原 LR 的 0.1 倍，Stage 2 余弦终点为
零，Stage 3 终点沿用 `min_lr_ratio`。每个调度器只按所属任务的真实更新数推进。

### Stage 2

完整联合训练后冻结 backbone 与共享 ObjectEncoder，并将共享模块置为 eval；只对
`simulation/heat_of_vaporization`、`simulation/homo`、`simulation/lumo` 与
`simulation/partial_atomic_charge` 逐任务调度，且只将当前任务预测头置为训练集。HEAD
参数归属来自注册表-backed 模型 API，必须互不重叠，不按参数名推断。冻结梯度与
前向路径分离：四个任务都继续运行边界时的 student backbone -> ObjectEncoder，
禁止使用 teacher embedding 替代路径。每个任务仅对原始 physics loss 反向，不使用任务
compensation 权重，也不加入 teacher loss。

`stage2_encoder.pt` 继续只导出边界后不再变化的 backbone/ObjectEncoder，并记录
精调边界、shared 状态hash 与来源记录；预测头精调不改变 Stage 3 表示。

### Stage 3

边界后严格通过参数归属 API 冻结所有 GLOBAL 与 GROUP，仅允许当前
PRIVATE:<task> 更新。整模先置 eval，仅当前任务的 PRIVATE 模块置训练集。virtual 采样、
batch allocation、microbatch、归一化 SmoothL1 与确定性顺序不变；每个复合步
逐任务直接更新其优化器，不调用 PCGrad，也不使用任务/组权重。诊断记录
`pcgrad_applied=false`、不适用矩阵、任务 loss、LR、梯度范数与更新 count。

### 选择与产物

Stage 2 四个精调任务与 Stage 3 每个任务的候选为边界状态
（精调轮 0）与每个精调轮结束后的 PRIVATE 状态。每个候选都必须有
完整且有限的验证主指标；严格变小时才替换，精确并列保留更早候选。
Partial Charge 使用分子-macro 归一化MAE；HOMO/LUMO 使用 pooled 样本-micro
原始 MAE；其余 Stage 2 与全部 Stage 3 任务使用任务验证归一化MAE。测试集不得
参与选择。

最后一个普通轮检查点先保存真实历史状态；随后将同一冻结 shared 状态与
精调任务的验证最优 PRIVATE 状态拼接，未精调的 Stage 2 预测头
保持联合训练边界状态，重新运行完整验证，并原子发布：

- `taskwise_refined.pt`：独立 kind（Stage 2 格式 v2、Stage 3 格式 v1）、完整拼接后的
  模型、来源训练身份、shared/PRIVATE hashes、边界、选择记录与拼接后的
  验证；
- `taskwise_refinement.json`：公开安全清单，记录边界/最优 metric、候选、是否严格
  改善、产物 hash 与拼接后的验证。

Stage 2 检查点升级到格式 v5，Stage 2 refined 产物升级到格式 v2；Stage 3
检查点维持格式 v2，保存 phase、
逐任务优化器/调度器、更新 counters 与最优-状态缓存。支持边界、
mid-精调和 finalization 恢复；旧检查点不迁移。evaluate 默认选择
逐任务精调后产物；只有显式提供 `--checkpoint-epoch N` 才选择
对应普通历史检查点。正式模型选择与测试集使用 refined 产物，普通检查点
只保留历史、恢复与显式诊断评估语义。

### Capacity 与 HPO

所有探针、HPO、确认、种子稳健性与正式运行都执行精调。HPO
原七维搜索空间不变，试验 LR 同时决定精调 LR。Capacity study/report 结构定义升级为
v2，删除 `tail_epochs`；探针胜出方案、HPO、确认、稳健性和正式比较
统一读取拼接后的验证的 `macro_task_equal.normalized_mae.value`。测试集只能在
配方与规模冻结后由显式 evaluate 执行。

精调改变训练身份。旧训练、旧 HPO SQLite/study 目录不能续跑；必须使用
不冲突的新输出目录。Stage 2 准备产物数据、teacher 缓存、编码器物理格式与 Stage 3
准备产物物理格式不因本决定改变，但新的 Stage 2 编码器身份要求重新准备 Stage 3。

## 后果

- Stage 2 每个运行在完整联合训练周期后额外增加 10 轮的四任务预测头-仅计算；
  Stage 3 总轮预算不变。
- 最终评估产物不对应单一历史轮；不同任务可来自不同精调轮，但共享
  完全相同的冻结状态。
- 普通检查点与逐任务精调后产物的用途明确分离，恢复、完整性与报告管理更
  复杂，但可以审计每个任务是否真正改善。
- Stage 2 预测头精调只改善 Stage 2 自身任务，不向 Stage 3 提供额外表示
  improvement；Stage 3 依靠自己的 PRIVATE 精调收尾。

## 备选方案

- 拒绝继续用 Stage 2 的 80/20 切分：会缩短 shared 表示的联合训练训练周期。
- 拒绝在精调继续联合训练优化器/PCGrad：残留优化器动量和共享梯度平衡会
  破坏任务独立收尾语义。
- 拒绝只取最后轮：不同任务的最佳 PRIVATE 状态不必出现在同一候选轮。
- 拒绝把 GROUP 当作 PRIVATE：GROUP 服务多个任务，继续更新会破坏统一冻结 shared 状态。

## 参考

- [ADR-0019：Stage 2 Object v3](0019-stage2-catalog-object-v3.md)
- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：身份与审计合同 v1](0021-identity-audit-contract-v1.md)
- [ADR-0026：Capacity v1](0026-capacity-v1-pipeline-study.md)
