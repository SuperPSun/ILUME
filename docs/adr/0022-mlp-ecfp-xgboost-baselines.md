# ADR-0022：MLP 与 ECFP-XGBoost 对比基线

- 状态：已接受
- 日期：2026-08-21
- 修订：2026-09-21 以固定 21 维/组分的 Basic molecular statistics 完整取代
  RDKit 2D 描述符 MLP；`mlp` 身份与显示名 `MLP` 不变

> Early stopping、验证最优 iteration 与检查点语义由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代；其余决定保持有效。

> 第 6 条中“不建立 Stage 2 aggregate”的决定先由 [ADR-0023](0023-unified-evaluation-reporting.md) 取代，并由 [ADR-0024](0024-stage2-partial-charge-benchmark-suite.md) 扩展为 Core、Partial Charge、完整三榜；其余训练与评估数值合同保持有效。
>
> Stage 2 orbital 任务与 Core 单元定义由 [ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 取代；MLP 归一化-目标 MSE 与 XGBoost 原始-目标标量 regressor 合同保持不变。

## 背景

ILUME 需要以简单、可审计的单任务模型作为 Stage 3 property 基准比较和 Stage 2 physics 基准比较的论文对照。对照实验必须复用任务目录、任务拓扑、canonical SMILES、既有 split、仅训练集归一化与测试集评价边界，但不能进入或改变 Stage 1/2/3 的训练数值合同。

## 决定

1. 基线实现隔离在顶层 `benchmarks/`，入口仅为 `scripts/benchmarks/{train,evaluate,sweep}.py`，两份正式配置为 `configs/benchmarks/{mlp,ecfp_xgboost}.yaml`。Stage 包不得导入基线。
2. Stage 3 从现役配置和任务目录动态解析 21 个任务、槽位、条件、拓扑与五折；四折训练、一折验证，独立测试集只由 evaluate 读取。Stage 2 只纳入汽化热、PBE/TZVP 阳离子 orbitals 和阴离子 orbitals，直接使用既有训练集/验证集/测试集。
3. MLP 按注册表 `identity_columns` 顺序拼接每个组分的固定 Basic 特征，随后
   按 authoritative `condition_columns` 顺序追加条件。每个组分的 21 项依次为：
   average molecular 权重、heavy 原子 count、包含隐式氢的 total 原子 count、C/N/O/F/P/S/
   Cl/Br/I count、正式 charge、原始分子图键 count、ring count、aromatic 原子/ring
   count、strict rotatable 键 count、H-键 donor/acceptor count 与 fraction C sp3。
   结构定义固定为 `basic-molecular-statistics-v1`，不得调用完整 RDKit 描述符 list、
   指纹或清单外 engineered 描述符。非有限值使用训练集 median，训练集整列无效时
   删除；全部保留列使用训练集总体z-score。条件、目标和 canonical SMILES
   非法时硬失败，不删样本或填充条件。
4. XGBoost 按同一槽位顺序拼接各组分的 ECFP4（radius 2、2048 bits）与原始条件，不缩放特征；每个标量目标使用独立 regressor，HOMO/LUMO 因而各有一个模型。XGBoost 拟合原始目标；MLP 拟合归一化目标。
5. 原始分子特征缓存以 canonical SMILES、特征合同、显式特征结构定义/version
   和 RDKit version 内容寻址。折预处理、目标 statistics 与任何 split label 不缓存。
6. 验证集只用于早停。Stage 3 正式测试集指标来自五个折模型逐样本预测平均后的集成，并另存各折诊断；五折验证报告均值/样本-std。Stage 2 每个目标单独报告，不建立 Stage 2 aggregate 或跨阶段总分。
7. Checkpoint 与运行使用身份合同 v1，绑定注册表、源内容、特征/预处理、目标 statistics、模型、训练数学、种子与模型完整性。Basic 结构定义/version 同时写入缓存 key、训练身份与检查点清单；旧 RDKit2D-MLP 产物必须被判为 incompatible。基线不支持恢复；sweep 保留成功与失败尝试，并在新尝试从头重跑。

## 后果

- 正式 `MLP` 只表示“basic molecular statistics + 条件 + shallow MLP”；旧完整
  RDKit2D-MLP 不再是现役基线，也不注册为正式 ablation。
- MLP 与 XGBoost 同时改变特征 family 与 estimator，因此只解释为完整基线 pipeline 对照，不用于归因单一组件。
- Stage 3 声明条件的缺失继续硬失败；当前压力缺失由上游数据修复，基线不建立例外。
- 正式 Stage 3 训练运行与测试集评估由用户执行；代码验收只使用临时小数据，不运行正式实验。
