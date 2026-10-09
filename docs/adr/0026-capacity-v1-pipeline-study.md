# ADR-0026：ILUME Capacity v1 端到端容量研究

- 状态：已接受
- 日期：2026-08-25
- 适用范围：`configs/experiments_v1/{stage1,stage2,stage3}` 与 `outputs/experiments_v1`

> 2026-08-26：Stage 2/3 后期逐任务精调、Capacity report 结构定义 v2 与拼接后的验证评分由 [ADR-0027](0027-late-taskwise-refinement.md) 修订；试验 wave、搜索空间、人工决策与种子规则不变。
>
> 2026-08-27：Capacity Stage 1 的目标硬件为单卡 84GB。基于原全局batch 128 约占
> 20GB 的线性显存估算，四规模共同改为全局batch 512、LR `4e-4`，估计峰值约 80GB；
> 不通过脚本自动探测或回退。该估算只适用于同类空闲 84GB GPU，首次正式运行须记录实际
> 峰值显存、吞吐和结果稳定性。
>
> 2026-09-06：Anchor HPO、确认、种子配置物化及其 Optuna 实现已退役；下文相关
> 条款只保留历史。当前 Stage 3 正式运行直接读取已冻结的
> `configs/experiments_v1/stage3/formal/*.yaml`，不得从旧 study 重新生成配方。

## 背景

本研究的目标是选择容量合适、下游迁移有效且训练稳定的 ILUME 主模型，并形成
S、Base、L、XL 四点容量趋势；它不是严格 scaling-law 研究，也不隔离 Stage 1
编码器的单独因果效应。Stage 2 ObjectEncoder 和 Stage 3 HoME 宽度均随 Stage 1
`d_model` 自然变化，因此结论只能表述为端到端 pipeline 容量趋势。

本决定不取代 ADR-0013/0017、ADR-0019/0025 或 ADR-0020。现役 `configs/v1`、
既有 5-轮 Stage 1/2 和 100轮 Stage 3 Base 继续保持原合同；capacity-v1 使用
独立配置、身份和输出，不恢复或覆盖现役运行。

## 决定

### Stage 1 与 Stage 2

1. Stage 1 固定四规模：
   - S：`d_model=384`、6 注意力头、SMILES/Fusion 6/6、FFN 1536、描述符隐藏宽度 768；
   - Base：512、8 注意力头、8/8、2048、1024；
   - L：640、10 注意力头、10/10、2560、1280；
   - XL：768、12 注意力头、12/12、3072、1536。
2. 四规模均固定 graph depth 6、预测头 dim 64、描述符 blocks 2，并保持现役数据、
   masking、loss、优化器、dropout 和数值合同。共同全局batch 为 512，LR 为
   `4e-4`（相对 128/`1e-4` 线性缩放）；每个规模从头训练 10 轮。所有轮
   检查点保留，但 Stage 2 只消费轮 10。
3. Stage 2 hyperparameter 选择固定只使用 Stage 1 Base 第10轮编码器，跑 R1–R8
   八个配方。`object_layers=2`、`object_ffn_dim=1024`、dropout 0.1，均先完成 10 个联合训练
   轮，再对 ADR-0027 的四个核心物理任务额外执行 10 个仅预测头精调轮。
   Stage 3 只消费联合训练第10轮边界导出的编码器。LR 顺序为 backbone/ObjectEncoder/任务预测头：
   - R1：冻结 4，LR `3e-6/1e-5/3e-5`，teacher λ 0.30；
   - R2：冻结 3，LR `5e-6/1.5e-5/5e-5`，teacher λ 0.20；
   - R3：冻结 2，LR `7e-6/2e-5/7e-5`，teacher λ 0.15；
   - R4：冻结 1，LR `1e-5/3e-5/1e-4`，teacher λ 0.10；
   - R5：冻结 0，LR `1e-5/3e-5/1e-4`，teacher λ 0.10；
   - R6：冻结 0，LR `1.5e-5/4.5e-5/1.5e-4`，teacher λ 0.075；
   - R7：冻结 0，LR `2e-5/6e-5/2e-4`，teacher λ 0.05；
   - R8：冻结 0，LR `3e-5/9e-5/3e-4`，teacher λ 0.03。
4. 仅用 `configs/experiments_v1/stage1/base.yaml` prepare 一次 Stage 1 语料；Base选择的
   Stage 2 prepare 与八个配方共用 `outputs/experiments_v1/stage1/prepare/artifacts`、一个
   Stage 2 数据产物和同一内容寻址 teacher 缓存。

### Probe、HPO 与正式比较

5. 8 个 Base Stage 2 候选全部跑 Stage 3 第1/2折 × 30 轮。每折的 proxy
   固定为逐任务精调后拼接后的验证的
   `macro_task_equal.normalized_mae.value`；候选再跨折平均并选择唯一 Base 胜出方案。
   精确并列时依次优先 R4、R3、R5、R2、R6、R1、R7、R8。
6. Base 胜出方案是唯一可进入 Anchor HPO 的配置。锚点决策仍须记录验证、曲线、
   参数量、显存、吞吐、墙钟耗时与失败证据；HPO 入口拒绝未选中的 Base 配方。测试集不参与。
7. Anchor HPO 固定 40 attempted trials、Optuna seeded TPE、试验 0 Base、前 10 个
   startup trials、不 pruning。搜索 7 个变量：global/组/PRIVATE 专家数、专家
   隐藏宽度比例、dropout、LR 和权重衰减。两试验为同步 wave，每试验第1/2折，
   四张 GPU 各运行一个折；wave 完成后按试验 number 写回结果。
8. 每个失败折只允许同配置再运行一次；第二次失败使试验 failed 并消耗预算。
   Top-5 加 Base 补 folds 3/4/5；Base 已在 Top-5 时不重复。五折均按拼接后的验证
   评分，最终配方由人工确认并冻结。
9. Stage 3 配置新增可选 `training.seed`。`null` 保持旧 `data.seed` RNG 和旧训练
   plan 形状；显式值只改变模型初始化、虚拟采样器、任务 order 与 PCGrad RNG，
   并进入训练身份，不进入准备产物身份。稳健性固定 seeds
   `42/10042/20042/30042/40042`、第1/2折 × 20 轮，结果由人工复核；拒绝时停止，
   不自动换配方、加种子或重启 HPO。
10. Base选择胜出方案不隐式复用于 S/L/XL：进入四规模正式比较前，必须另行冻结每个
    规模的 Stage 2 训练、编码器产物与对应 Stage 3 配置，再物化 final 配方。
    四规模从头跑 5折 × 50 轮、种子 42，固定使用各折逐任务精调后产物；
    30-轮运行不恢复到 50 轮。测试集不得反向改变规模、配方或产物；本轮不追加
    100轮训练。

### 失败、身份与报告

11. 正式开始前冻结干净 commit、数据身份、完整 YAML、试验清单和同类硬件。
    OOM、NaN、发散或不完整验证不触发 batch、LR、checkpointing 或训练周期
    自动调整。若要改变合同，必须形成新的预注册决定。
12. 报告包含任务/组指标、完整曲线、折/种子波动、参数量、显存、吞吐、wall
    time、失败和人工理由。结论固定称为“经 Stage 1 Base 上的 Stage 2 配方选择、共享
    Stage 3 配方后的端到端容量趋势”。

## 后果与取舍

- Final-仅将 Stage 2 选择候选从 24 减为 8，删除两层主观中间检查点选择，
  也无需增加任意 Stage 2 检查点导出接口。
- 每个 Stage 2 候选的共享表示仍比较同一 10轮联合训练周期；额外 10 轮
  只更新四个核心任务预测头，不进入 Stage 3 编码器。
- 30-轮探针/HPO 和 50-轮正式是不同调度器 trajectory，不能恢复或
  解释为同一训练的前后段；现役 100轮 Base 也只作历史/健康参考。
- Anchor HPO 不为其他规模各自优化 Stage 3，因此四点回答的是受控、可负担的主模型
  选择问题，而不是每个规模的性能上界。
- 人工 Pareto、配方决策和种子复核保留科学判断，但每个判断的可见证据、时点和禁止
  使用的测试集信息均被冻结并写入决策 record。

## 备选方案

- 拒绝每规模两个 Stage 1 检查点与 50-轮全候选探针：成本高且增加
  检查点选择自由度。
- 拒绝固定-宽度 Stage 3 adapter：它能更接近仅编码器因果比较，但改变本研究的
  端到端主模型目标和现役 HoME 接口。
- 拒绝以 30 轮作为正式终点或在选定规模后追加 100 轮：前者证据不足，后者
  超出本轮已接受预算并形成第三条调度器合同。

## 参考

- [ADR-0019：Stage 2 任务目录Object v3](0019-stage2-catalog-object-v3.md)
- [ADR-0020：Stage3稀疏HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：身份与审计合同 v1](0021-identity-audit-contract-v1.md)
- [Capacity v1 操作手册](../capacity-v1-runbook.md)
