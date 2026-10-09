# ADR-0076：AIonopedia 标量预测头容量对照

- 状态：已退役（2026-09-26）
- 日期：2026-09-25
- 范围：AIonopedia 基准比较的隔离回归头容量对照

## 退役决定

用户反馈该消融对模型性能的削弱未达到预期，因此停止将其作为现役对照。删除独立可运行 YAML
并移除 128维宽度回归头实现和配置校验分支；正式 AIonopedia 与新增图侧条件通路消融均只使用
`Linear(512,1024) → ReLU → Linear(1024,1)`。旧结果如存在应保持只读，不与现役结果合并。
本决定不根据主观预期推断该消融的具体指标或统计显著性。

以下保留实验设立时的历史合同，不再作为运行入口。

## 历史决定

曾新增 `configs/benchmarks/aionopedia_head128.yaml`，将每个任务/折新建的标量预测头从官方宽度
`Linear(512,1024) → ReLU → Linear(1024,1)` 改为
`Linear(512,128) → ReLU → Linear(128,1)`。

除预测头隐藏宽度外，通用预训练 assets、Qwen LoRA、全部已发布多模态模块及其全量下游微调方式、
新增条件 modules、数据/划分、优化器、各组学习率、batch、精度和 10轮训练预算均与
[ADR-0049](0049-aionopedia-multimodal-baseline.md) 相同。原始 `configs/benchmarks/aionopedia.yaml`
保持官方 1024-wide 预测头。

## 历史身份与输出

该变体曾由 `model.scalar_head_hidden_dim: 128` 进入训练身份，输出使用独立根
`outputs/benchmarks/model-native-v1/aionopedia-head128/`。它只能作为预测头-capacity 对照解释，不能与
官方宽度 AIonopedia 结果合并或替代正式基线。

## 历史验证边界

配置和模型结构测试只验证变体注册、回归头宽度以及其余训练/数据/预训练 asset 配置保持一致，
不构成性能评估。该配置现已删除。
