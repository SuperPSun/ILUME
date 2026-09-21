# ADR-0022：MLP 与 ECFP-XGBoost 对比基线

- 状态：Accepted
- 日期：2026-08-21
- 修订：2026-09-21 以固定 21 维/component 的 Basic molecular statistics 完整取代
  RDKit 2D descriptor MLP；`mlp` identity 与显示名 `MLP` 不变

> Early stopping、validation-best iteration 与 checkpoint 语义由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代；其余决定保持有效。

> 第 6 条中“不建立 Stage 2 aggregate”的决定先由 [ADR-0023](0023-unified-evaluation-reporting.md) 取代，并由 [ADR-0024](0024-stage2-partial-charge-benchmark-suite.md) 扩展为 Core、Partial Charge、Full 三榜；其余训练与 evaluation 数值合同保持有效。
>
> Stage 2 orbital task 与 Core 单元定义由 [ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 取代；MLP normalized-target MSE 与 XGBoost raw-target scalar regressor 合同保持不变。

## 背景

ILUME 需要以简单、可审计的单任务模型作为 Stage 3 property benchmark 和 Stage 2 physics benchmark 的论文对照。对照实验必须复用 catalog、任务 topology、canonical SMILES、既有 split、train-only normalization 与 test 评价边界，但不能进入或改变 Stage 1/2/3 的训练数值合同。

## 决定

1. Baseline 实现隔离在顶层 `benchmarks/`，入口仅为 `scripts/benchmarks/{train,evaluate,sweep}.py`，两份正式配置为 `configs/benchmarks/{mlp,ecfp_xgboost}.yaml`。Stage 包不得导入 baseline。
2. Stage 3 从现役配置和 catalog 动态解析 21 个任务、slot、condition、topology 与五折；四折训练、一折 validation，独立 test 只由 evaluate 读取。Stage 2 只纳入 heat of vaporization、PBE/TZVP cation orbitals 和 anion orbitals，直接使用既有 train/valid/test。
3. MLP 按 registry `identity_columns` 顺序拼接每个 component 的固定 Basic feature，随后
   按 authoritative `condition_columns` 顺序追加 conditions。每个 component 的 21 项依次为：
   average molecular weight、heavy atom count、包含隐式氢的 total atom count、C/N/O/F/P/S/
   Cl/Br/I count、formal charge、原始分子图 bond count、ring count、aromatic atom/ring
   count、strict rotatable bond count、H-bond donor/acceptor count 与 fraction C sp3。
   schema 固定为 `basic-molecular-statistics-v1`，不得调用完整 RDKit descriptor list、
   fingerprint 或清单外 engineered descriptor。非有限值使用 train median，train 整列无效时
   删除；全部保留列使用 train population z-score。condition、target 和 canonical SMILES
   非法时硬失败，不删样本或填充 condition。
4. XGBoost 按同一 slot 顺序拼接各 component 的 ECFP4（radius 2、2048 bits）与原始 conditions，不缩放 feature；每个 scalar target 使用独立 regressor，HOMO/LUMO 因而各有一个模型。XGBoost 拟合原始 target；MLP 拟合 normalized target。
5. 原始分子 feature cache 以 canonical SMILES、feature contract、显式 feature schema/version
   和 RDKit version 内容寻址。fold preprocessing、target statistics 与任何 split label 不缓存。
6. valid 只用于 early stopping。Stage 3 正式 test 指标来自五个 fold 模型逐样本预测平均后的 ensemble，并另存各 fold 诊断；五折 validation 报告 mean/sample-std。Stage 2 每个 target 单独报告，不建立 Stage 2 aggregate 或跨阶段总分。
7. Checkpoint 与 run 使用 identity contract v1，绑定 registry、源内容、feature/preprocessing、target statistics、模型、训练数学、seed 与模型完整性。Basic schema/version 同时写入 cache key、training identity 与 checkpoint manifest；旧 RDKit2D-MLP artifact 必须被判为 incompatible。Baseline 不支持 resume；sweep 保留成功与失败 attempt，并在新 attempt 从头重跑。

## 后果

- 正式 `MLP` 只表示“basic molecular statistics + conditions + shallow MLP”；旧完整
  RDKit2D-MLP 不再是现役 baseline，也不注册为正式 ablation。
- MLP 与 XGBoost 同时改变 feature family 与 estimator，因此只解释为完整 baseline pipeline 对照，不用于归因单一组件。
- Stage 3 声明 condition 的缺失继续硬失败；当前压力缺失由上游数据修复，baseline 不建立例外。
- 正式 Stage 3 训练 run 与 test evaluation 由用户执行；代码验收只使用临时小数据，不运行正式实验。
