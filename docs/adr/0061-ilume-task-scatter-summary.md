# ADR-0061：ILUME 任务预测散点图汇总

- 状态：已接受
- 日期：2026-09-16
- 修订：扩展 ADR-0043 第 5 条的统一汇总输出集合

## 背景

现有统一汇总只发布指标表、overview 和 radar，无法直接检查 ILUME 各任务的
observed-vs-predicted 分布。逐样本预测已由 Stage 3 评估按报告
合同发布，无需重新训练或评估。

## 决定

1. 统一汇总在 `ilume_scatter/validation/` 为验证榜单中排名最高的
   ILUME 合并五折预测，并按任务发布 SVG；在 `ilume_scatter/test/` 为测试集
   榜单中排名最高的 ILUME 使用集成预测发布有测试集样本的任务。
2. 验证使用原始 `target`/`prediction`，测试集使用原始
   `target`/`prediction_ensemble`。图的两个坐标轴共享范围并显示 `y=x`，不改变任何
   metric、榜单、选择或 `summary.json` 结构定义。散点尺寸随该图样本数连续缩放，
   并限制在 1.125～6.0 SVG user units，避免稠密任务遮挡或稀疏任务难以辨认。
3. 绘图前严格校验预测清单的任务、相对路径、行数与 SHA256，以及 CSV
   必需列和有限数值。失败沿用 summarizer 的原子发布语义，不替换已有 `summary/`。
4. SVG 由标准库确定性生成，不增加主环境依赖，也不复制预测 CSV。

## 后果

- 同一输入仍产生确定性汇总快照，但固定顶层输出新增 `ilume_scatter/`。
- 验证与测试集独立选择榜首 ILUME，因而可能来自不同 variant；并列时沿用现有
  榜单的确定性运行顺序。
- 已有 Stage 3 结果只需重新运行 summarizer，不需要重训或重新评估。
