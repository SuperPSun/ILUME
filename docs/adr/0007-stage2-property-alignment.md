# ADR-0007：Stage 2 多任务物性监督与冻结教师对齐

- 状态：Accepted
- 日期：2026-08-03
- 部分取代：ADR-0009 已取代任务内行级采样、共享 IL 回归 trunk、立即训练 backbone 和完整 QM 标签假设；冻结教师与五任务总体设计仍有效。

## 决定

Stage 2 从一个 checkpoint v3 初始化两条逻辑分支：冻结教师 A 以完整、无掩码四模态输入离线生成 FP32 `fused_cls`；学生 B 从相同权重开始，训练全部编码路径和新增回归头。教师缓存以 `(checkpoint SHA-256, Stage 2 data metadata SHA-256)` 标识，正式训练不把教师模型驻留在显存。

首版联合五项任务：density、heat capacity、thermal expansion 使用有序的 cation/anion CLS 与 train-only 标准化温度；QM 使用一个 neutral CLS 预测11项标签；transfer 使用有序的 solute/solvent neutral CLS。所有目标只用各自训练集拟合 scaler。首版不输入压力，也不加入显式热力学恒等式。

监督项使用标准化目标上的 SmoothL1。对齐项直接计算输入实体学生 CLS 与冻结教师 CLS 的逐维 MSE；二元输入先在两个实体间平均。Stage 1 reconstruction heads 不进入 Stage 2 optimizer。学生编码路径关闭随机 dropout，使初始化时的对齐损失为零；新增回归头保留 dropout。

reference 任务调度使用确定性打乱的20-step块，每块固定7个 QM、4个 density、3个 heat capacity、3个 thermal expansion 和3个 transfer step。正式采样消融还提供4/4/4/4/4和2/6/5/5/2两种整数配额，并通过调整总step让三种配置都抽样900,096条transfer训练行。各任务内部无放回抽样并在耗尽后重排。Stage 2 使用独立的 step checkpoint 格式，保存任务游标和 RNG，可从任意保存 step 恢复；这不改变 Stage 1 的覆盖型 epoch 或 checkpoint v3。

## 理由

冻结教师只依赖实体与 checkpoint，离线缓存与逐 batch 在线教师在确定性输入下等价，但显著降低显存和重复计算。直接约束共享 CLS 避免新增投影层吸收偏移。五类数据量相差近两个数量级，显式任务块比按自然行数合并更能落实既定实验权重，也比把 Stage 1 的全覆盖 epoch 套用到多任务数据更清晰。

## 后果

Stage 2 data artifact 绑定 Stage 1 tokenizer、descriptor schema/scaler 和 fingerprint 合同；未进入 Stage 1 manifest 的实体仍可在通过相同 QC 后编码。更换 checkpoint 只需重建教师缓存，若预处理合同不变则复用实体与任务工件。当前10个实体 shard 在正式配置中全部缓存，避免随机批次触发重复反序列化。

验证以五任务等权宏平均 normalized MAE 选 checkpoint，同时报告原单位 MAE/RMSE/R²、CLS MSE/cosine 和参数相对漂移。正式 Base λ 扫描及 XLarge 训练由用户运行；代码交付阶段只执行临时小数据测试，不生成完整缓存或启动长任务。
