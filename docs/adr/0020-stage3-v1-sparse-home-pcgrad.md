# ADR-0020：Stage 3 v1 sparse-label 动态 HoME 与分块 PCGrad

- 状态：Accepted
- 日期：2026-08-20
- 取代：ADR-0010、ADR-0011、ADR-0012 的全部现役 Stage 3 决定

> 2026-08-20：本文的 Stage 2 checkpoint SHA、plugin lineage、run resume 与 evaluation identity 规则已由 [ADR-0021](0021-identity-audit-contract-v1.md) 取代；模型、采样、PCGrad 与数值训练合同不变。

## 背景

旧 Stage 3 将任务固定拆成 `il21` 与 `aux6`，依赖固定 condition/phase 表示、BatchNorm expert、早停和 best checkpoint，且尚未迁移到 Stage 2 Object v3。新数据合同由 `task_catalog.csv` 给出 scalar target、identity、condition、topology、物化路径和合法 split；consumer 不应复制这些事实，也不能将 sparse observation 拼成 dense target table。

## 决定

1. Stage 3 v1 使用独立 artifact/checkpoint kind 与 `format_version=1`，不兼容任何旧 Stage 3 artifact、YAML 或 checkpoint。正式配置为 `configs/v1/stage3/base.yaml`，输出根为 `outputs/v1/stage3/base`。
2. registry 合并 catalog 数据事实与 YAML 模型侧分组、topology slots、启用状态和权重。Base 包含 21 个 observation task 与 transport、thermophysical、phase_stability、dielectric_optical、solvation、biological 六组。默认 split 为 `prefer_il`，但 catalog 声明且已物化的任意合法 strategy 与 repeat 均可逐 task 显式选择；绝不自动退回 random。
3. identity、target、condition 与 SMILES 必须逐行合法且有限。normalization 只使用 held-out fold 之外四折的 population mean/std；target 零方差失败，constant condition 记为 scale 1。当前上游 condition 缺失由 ILUME-Data 修复，Stage 3 不填充、不删行。
4. Stage 2 提供公开 frozen Object v3 checkpoint loader。prepare 严格加载完整模型、registry 与 model contract，并生成内容寻址的 FP32 object cache；partner 始终保持 frozen embedding，只有 primary 进入 GroupMoE/FiLM。artifact 全部成功后原子发布，train/evaluate 只读 artifact 并强校验 Stage 2 checkpoint SHA。
5. 动态 HoME 由 L1 Global Experts 与唯一 L1 Global Gate、每组 L1/L2 Group Experts、可选 task FiLM、group-shared partner interaction、L2 Global Experts、task-private experts、唯一 unified TaskGate、task residual/tower 组成。不存在 L2 Global Gate。expert 数、dropout、activation 与 hidden ratios 是可配置且进入实验身份的模型超参，不是不可变架构常量。
6. 每个参数由显式 API 唯一标记为 GLOBAL、GROUP 或 PRIVATE；训练器不得从名称反推。GLOBAL 包含 L1 Global Experts/Gate 与 L2 Global Experts；GROUP 包含组 expert/gate/residual/interaction；PRIVATE 包含 private expert、TaskGate、FiLM、task residual/tower。
7. virtual epoch 使用 `N'_t=max(N_t,1000)`、总 composite allocation 2048 和稳定 SHA-256 shuffle。每 step 按 task 取得固定 `B_t`，拆成至多 128 的 microbatch，以 `loss_sum/B_t` 累积完整 FP32 task gradient，完成全部 task 后才执行 PCGrad、weighting、gradient assembly、global clipping、optimizer 与 scheduler step。
8. hierarchical PCGrad 将 GLOBAL 与每个 GROUP 作为独立 logical block，绝不拼接。组内 task-level GLOBAL 与 GROUP 投影使用独立可复现顺序；task weight 在投影后生效。GROUP 在组内聚合，GLOBAL 先组内聚合再做 group-level PCGrad 并按 group weight 聚合；PRIVATE 不投影。diagnostics 使用 NaN 与 applicability mask 表示不适用位置。
9. Base 使用 normalized SmoothL1、AdamW、5% warmup 后 cosine 到 5% base LR、global norm clipping 1.0、100 epochs。BF16 不可用时失败，只能通过 YAML 显式选择 FP32/none。正式训练只支持单进程单 CUDA GPU，CPU 仅用于测试。
10. 每 epoch 完整 validation；科学指标在原单位与 normalized 单位报告，并同时给出 task-equal 与 group-equal macro。Base 每 10 epoch 保存不可覆盖 checkpoint，额外保存非整除 final；epoch 100 是固定最终模型，不 early stop，不生成 best/last。
11. plugin load 与 adaptation 分离。默认加载并冻结 source GLOBAL、匹配 GROUP/PRIVATE，只训练新 task PRIVATE 或新 group GROUP+PRIVATE；但 YAML 可显式 adaptation GLOBAL、已有 GROUP、已有 PRIVATE，并可让任意 enabled task 重新参与 composite batching 与 PCGrad。target registry 可等于或扩展 source registry，结构、ownership、shape、Stage 2 SHA 和 normalization 严格校验；plugin 新 run 不继承 optimizer、scheduler、epoch 或 RNG。

## 后果

- 每个 task 保留独立 sparse observation dataset，训练成本由冻结的 composite plan 决定，不再依赖 dense-label 缺失模式。
- 旧 `il21/aux6`、AdaTT、IndependentTaskHead、FeatureGate/SelfGate、BatchNorm expert、phase embedding、simulation/quantum Stage 3 task、early stopping 与 best checkpoint 从现役实现删除。
- checkpoint interval 是执行默认值并写入 run config，不是模型架构或训练数学身份；模型超参和全部 resolved width 则进入 plan 与 checkpoint 严格校验。
- 正式 prepare、100 epoch 训练与五折评估由用户在 ILUME-Data condition 完整产物发布后执行；仓库验收只使用临时小数据与短训练。
