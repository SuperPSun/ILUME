# ADR-0020：Stage 3 v1 稀疏标签动态 HoME 与分块 PCGrad

- 状态：已接受
- 日期：2026-08-20
- 取代：ADR-0010、ADR-0011、ADR-0012 的全部现役 Stage 3 决定

> 2026-08-20：本文的 Stage 2 检查点 SHA、插件 lineage、运行恢复与评估身份规则已由 [ADR-0021](0021-identity-audit-contract-v1.md) 取代；模型、采样、PCGrad 与数值训练合同不变。
>
> 2026-08-26：本文的全程 PCGrad、固定 final 轮评估与检查点 v1 合同已由 [ADR-0027](0027-late-taskwise-refinement.md) 修订；联合训练 phase 的 HoME、采样与分层 PCGrad 合同保持不变。
>
> 2026-09-07：现役 v2 与对应 ADR-0034/0036 消融的联合训练梯度裁剪和采样已由 [ADR-0046](0046-stage3-ownership-clipping-raw-sampling.md) 修订；历史实现 v1 与 Capacity v1 保留本文合同。
>
> 2026-09-07：现役 v2 与上述两个消融的固定统一容量、单一联合训练优化器/调度器、
> 精调和 final 产物合同已由 [ADR-0047](history.md#adr-0047)
> 修订；本文与 ADR-0027 的对应条款仅继续约束历史实现 v1 与 Capacity v1。
>
> 2026-09-10：上述现役四阶段条款已由
> [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 的 owner-存续期三阶段合同取代；
> 历史实现 v1 与 Capacity v1 语义不变。

## 背景

旧 Stage 3 将任务固定拆成 `il21` 与 `aux6`，依赖固定条件/phase 表示、BatchNorm 专家、早停和最优检查点，且尚未迁移到 Stage 2 Object v3。新数据合同由 `task_catalog.csv` 给出标量目标、身份、条件、拓扑、物化路径和合法 split；consumer 不应复制这些事实，也不能将稀疏 observation 拼成稠密目标 table。

## 决定

1. Stage 3 v1 使用独立产物/检查点 kind；准备产物保持 `format_version=1`，训练检查点按 ADR-0027 升级为 `format_version=2`，逐任务精调后产物独立使用格式 v1。旧 Stage 3 产物、YAML 或检查点不兼容。正式配置为 `configs/v1/stage3/base.yaml`，输出根为 `outputs/v1/stage3/base`。
2. 注册表合并任务目录数据事实与 YAML 模型侧分组、拓扑 slots、启用状态和权重。Base 包含 21 个 observation 任务与 transport、thermophysical、phase_stability、dielectric_optical、solvation、biological 六组。默认 split 为 `prefer_il`，但任务目录声明且已物化的任意合法 strategy 与 repeat 均可逐任务显式选择；绝不自动退回 random。
3. 身份、目标、条件与 SMILES 必须逐行合法且有限。归一化只使用 held-out 折之外四折的 population 均值/std；目标零方差失败，恒定条件记为规模 1。当前上游条件缺失由 ILUME-Data 修复，Stage 3 不填充、不删行。
4. Stage 2 提供公开冻结 Object v3 检查点加载器。prepare 严格加载完整模型、注册表与模型合同，并生成内容寻址的 FP32 object 缓存；partner 始终保持冻结 embedding，只有 primary 进入 GroupMoE/FiLM。产物全部成功后原子发布，训练/评估只读产物并强校验 Stage 2 检查点 SHA。
5. 动态 HoME 由 L1 Global Experts 与唯一 L1 Global Gate、每组 L1/L2 Group Experts、可选任务 FiLM、组内共享 partner 交互、L2 Global Experts、任务-PRIVATE 专家、唯一 unified TaskGate、任务残差/tower 组成。不存在 L2 Global Gate。专家数、dropout、激活与隐藏宽度 ratios 是可配置且进入实验身份的模型超参，不是不可变架构常量。
6. 每个参数由显式 API 唯一标记为 GLOBAL、GROUP 或 PRIVATE；训练器不得从名称反推。GLOBAL 包含 L1 Global Experts/Gate 与 L2 Global Experts；GROUP 包含组专家/门控/残差/交互；PRIVATE 包含 PRIVATE 专家、TaskGate、FiLM、任务残差/tower。
7. virtual 轮使用 `N'_t=max(N_t,1000)`、总复合 allocation 2048 和稳定 SHA-256 shuffle。每步按任务取得固定 `B_t`，Base 拆成至多 1024 的 microbatch，以 `loss_sum/B_t` 累积完整 FP32 任务梯度，完成全部任务后才执行 PCGrad、weighting、梯度 assembly、global 裁剪、优化器与调度器步。`microbatch_size` 同时约束验证前向分块，保留为显式 YAML 参数并进入训练身份；不增加独立验证 batch 参数，不自动选择或在 OOM 后降低。
8. 分层 PCGrad 将 GLOBAL 与每个 GROUP 作为独立 logical block，绝不拼接。组内任务-level GLOBAL 与 GROUP 投影使用独立可复现顺序；任务权重在投影后生效。GROUP 在组内聚合，GLOBAL 先组内聚合再做组-level PCGrad 并按组权重聚合；PRIVATE 不投影。诊断使用 NaN 与 applicability mask 表示不适用位置。
9. Base 使用归一化 SmoothL1、AdamW、5% 预热后余弦到 5% base LR、global norm 裁剪 1.0、100 轮。BF16 不可用时失败，只能通过 YAML 显式选择 FP32/none。每个折 worker 只支持单进程单 CUDA GPU，CPU 仅用于测试；唯一训练入口可以使用 spawn worker 在显式设备槽中并行调度多个彼此隔离的折，这属于 execution orchestration，不改变单折数值合同。
10. 每轮完整验证；科学指标在原单位与归一化单位报告，并同时给出任务等权与组-equal macro。Base 每 10 轮保存不可覆盖检查点，额外保存非整除 final；轮 100 是固定普通历史检查点，不 early stop，不生成最优/last。最终评估模型按 ADR-0027 发布为独立逐任务精调后产物。
11. 插件加载与 adaptation 分离。默认加载并冻结来源 GLOBAL、匹配 GROUP/PRIVATE，只训练新任务 PRIVATE 或新组 GROUP+PRIVATE；但 YAML 可显式 adaptation GLOBAL、已有 GROUP、已有 PRIVATE，并可让任意 enabled 任务重新参与复合 batching 与 PCGrad。目标注册表可等于或扩展来源注册表，结构、参数归属、shape、Stage 2 SHA 和归一化严格校验；插件新运行不继承优化器、调度器、轮或 RNG。

## 后果

- 每个任务保留独立稀疏 observation dataset，训练成本由冻结的复合 plan 决定，不再依赖稠密-label 缺失模式。
- 旧 `il21/aux6`、AdaTT、IndependentTaskHead、FeatureGate/SelfGate、BatchNorm 专家、phase embedding、模拟/quantum Stage 3 任务、早停与最优检查点从现役实现删除。
- 检查点 interval 是执行默认值并写入运行配置，不是模型架构或训练数学身份；模型超参和全部 resolved 宽度则进入 plan 与检查点严格校验。
- 多折调度的请求顺序、最大并发数和设备分配只属于运行参数；每个折继续持有独立运行 directory、身份、日志、检查点和恢复状态。
- 正式 prepare、100 轮训练与五折评估必须由用户显式执行；仓库验收只使用临时小数据与短训练。运行产物状态属于 README/元数据的操作事实，不改变本 ADR 的科学合同。
