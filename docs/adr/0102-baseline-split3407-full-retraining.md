# ADR-0102：基线统一3407划分并全量重训

- 状态：已接受
- 日期：2026-10-20
- 范围：十个正式基线；替代ADR-0101的现役split42补训与复用安排。

## 数据与训练

新增十份自包含配置`configs/benchmarks/catalog_v4_3407/<model>.yaml`，显式使用`protocol: catalog_v4_3407`、`data/3407`及其task catalog。Stage3注册表与split authority指向`configs/v4/stage3/optimized_3407.yaml`，只读取任务、输入、目标和划分合同，不采用该文件的HoME模型或训练预算。

仍为24实验任务、cv1五折；hydration单solute加温度、random/cv1且无测试集，其余遵循当前system规则。模拟仅thermal_expansion、homo、lumo；热膨胀读取data/3407/stage2，HOMO/LUMO按Stage1共用合同读取data/stage1/properties的原始训练/验证/测试集，不伪造随实验seed变化的电子划分。三模拟模型也全部重训。水活度、gas_solubility、汽化焓及两项电位的输入和目标沿用0101。

十个模型全部从头训练，不复用任何split42训练结果或预测。每模型120个实验折次加3个模拟作业，共123个；总计1,200个实验折次、30个模拟作业，即1,230个训练作业。各模型原结构、预训练资产、训练种子和配方保持不变：AIFC seed1000，其余按各自原配置；3407是数据划分，不替换训练种子。神经模型10轮、XGBoost 1000 trees、发布末轮，验证只报告。

## 身份与输出

新协议进入全部任务的训练身份，包括内容可能与42相同的模拟数据。3407协议明确拒绝`--prepare-reuse-from`与`--reuse-manifest`；原有checkpoint校验保持严格。历史YAML和身份不改写。

输出独立使用`outputs/benchmarks/v4_3407/<model>/`，特征缓存使用该根的独立cache；禁止把旧summary、权重、预测或迁移清单放入新根。失败训练在新attempt从头重跑，已完成作业仅在配置、数据与产物身份完整校验后跳过。实验/三模拟及与现役Stage3的thermal_expansion比较规则沿用0101，汇总仅扫描3407配套结果。

## 旧产物清理与调度

用户要求4090旧`outputs/benchmarks/`整体移入`trash/outputs/benchmarks-split42-20261020/`，szx旧`outputs/benchmarks/`删除；按工作区规则先报告精确清单、文件数、字节数、目标和冲突策略，获得确认后执行。trash目标已存在则停止，不覆盖或合并；szx删除范围包含其旧v2/v3/v4与特征缓存。保留预训练`artifacts/benchmarks/`、环境、原始数据、其他Stage/HPO/消融产物及Slurm脚本和日志。

szx的两份现有sbatch保留在`/home/sunp/slurm/baseline/`，固定`gpu_6000D`的`gpu05`：九个GPU模型使用数组，每项1卡、最多4项并发、每个sweep内部max-workers1；XGBoost单独申请8 CPU、不占GPU。保留Slurm的CUDA_VISIBLE_DEVICES，不传会覆盖它的`--devices`。运行命令见[基线手册](../baseline-catalog-v4-runbook.md)。

代码交付不启动正式训练/评估。完成后将完整权重、配置、清单、日志和预测回传4090相同3407输出目录。
