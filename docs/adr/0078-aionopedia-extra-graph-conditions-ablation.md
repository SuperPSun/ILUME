# ADR-0078：AIonopedia 新增图侧条件通路消融

- 状态：Retired
- 日期：2026-09-26
- 范围：AIonopedia baseline 的独立单变量消融

## 退役决定

用户反馈该消融未达到预期的性能削弱效果，因此移除独立 YAML、图侧通路关闭开关及相关正向测试。
现役 AIonopedia 保留压力、频率、波长的文本和图侧输入，以及官方温度通路和 1024 宽回归头。
历史训练与评估产物保持只读，不与现役结果混用。此反馈不作为具体指标或统计显著性的结论。
以下为历史合同，不再提供运行入口。

## 历史决定

新增 `configs/benchmarks/aionopedia_no_extra_graph_conditions.yaml`。压力、频率、波长仍按
[ADR-0049](0049-aionopedia-multimodal-baseline.md) 写入文本 prompt，但不进入图侧融合；变体不创建
这三项的 projector 和 segment token。官方已有的温度 projector、温度文本、四种 topology 与
`Linear(512,1024) → ReLU → Linear(1024,1)` 回归头保持不变。官方公开 prompt 只支持温度文本；
本消融有意保留 ILUME 新增的三项文本条件，以单独衡量新增图侧通路的贡献。

通用 pretrained assets、released Qwen LoRA、多模态模块的下游微调、数据与 fold、loss、优化器、
各组学习率、batch、精度、10-epoch 预算及 final-state 选择均沿用 ADR-0049。原始
`aionopedia.yaml` 不变。

## 历史身份与输出

`model.extra_graph_conditions: false` 和对应输入合同进入训练身份；输入审计记录空的
`active_graph_conditions`，保留 registry 条件列及文本 prompt hash。checkpoint 的随机初始化
模块清单仅含 `fc_out`；三项新增图侧模块不存在，评估不能跨变体加载 checkpoint。

输出使用独立根 `outputs/benchmarks/model-native-v1/aionopedia-no-extra-graph-conditions/`，
不能与正式 AIonopedia 结果混用。该结果只解释图侧通路的作用；
不能解释为完全恢复官方输入，因为文本仍包含压力、频率和波长。

## 历史验证边界

配置、prompt、图侧 token、checkpoint 与身份由小规模行为测试验证。正式训练与评估由用户
使用独立配置另行运行；新增变体本身不启动 sweep。
