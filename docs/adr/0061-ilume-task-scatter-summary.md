# ADR-0061：ILUME task prediction 散点图汇总

- 状态：Accepted
- 日期：2026-09-16
- 修订：扩展 ADR-0043 第 5 条的统一汇总输出集合

## 背景

现有统一汇总只发布指标表、overview 和 radar，无法直接检查 ILUME 各任务的
observed-vs-predicted 分布。逐样本 prediction 已由 Stage 3 evaluation 按 reporting
合同发布，无需重新训练或评估。

## 决定

1. 统一汇总在 `ilume_scatter/validation/` 为 validation leaderboard 中排名最高的
   ILUME 合并五折 prediction，并按 task 发布 SVG；在 `ilume_scatter/test/` 为 test
   leaderboard 中排名最高的 ILUME 使用 ensemble prediction 发布有 test 样本的 task。
2. validation 使用 raw `target`/`prediction`，test 使用 raw
   `target`/`prediction_ensemble`。图的两个坐标轴共享范围并显示 `y=x`，不改变任何
   metric、leaderboard、selection 或 `summary.json` schema。散点尺寸随该图样本数连续缩放，
   并限制在 1.125～6.0 SVG user units，避免稠密任务遮挡或稀疏任务难以辨认。
3. 绘图前严格校验 prediction manifest 的 task、相对路径、行数与 SHA256，以及 CSV
   必需列和有限数值。失败沿用 summarizer 的原子发布语义，不替换已有 `summary/`。
4. SVG 由标准库确定性生成，不增加主环境依赖，也不复制 prediction CSV。

## 后果

- 同一输入仍产生确定性 summary snapshot，但固定顶层输出新增 `ilume_scatter/`。
- validation 与 test 独立选择榜首 ILUME，因而可能来自不同 variant；并列时沿用现有
  leaderboard 的确定性 run 顺序。
- 已有 Stage 3 结果只需重新运行 summarizer，不需要重训或重新 evaluation。
