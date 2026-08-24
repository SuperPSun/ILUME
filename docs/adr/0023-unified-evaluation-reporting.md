# ADR-0023：统一 evaluation reporting 与结果汇总

- 状态：Accepted
- 日期：2026-08-24

## 背景

完整科研运行产物分散在 `outputs/`，ILUME、MLP、ECFP+XGBoost 与后续 baseline 缺少统一、可审计且适合论文比较的结果入口。Stage 2 也缺少与 baseline test 合同对齐的独立 evaluator。

## 决定

1. `outputs/` 保留完整运行、prediction、checkpoint 与审计材料；Git 可跟踪的 `summary/` 只保留 overview、三张 leaderboard、三张明细表、health 表和机器可读 summary，不复制训练或 prediction payload。
2. Evaluation summary 使用 `reporting_schema_version=1`，显式记录模型显示名、protocol、study ID、comparison identity 与 prediction manifest。全局 summarizer 只按 `stage/operation/schema` 解析，不按模型名分支。
3. Stage 2 正式 test 固定为 heat of vaporization 与 cation/anion PBE/TZVP HOMO/LUMO，共 3 task、5 scalar target。Test 实体在 evaluation 进程内按现役 Stage 1 feature contract 构建，不写入 prepared artifact、teacher cache 或训练 identity。
4. 所有 scalar target 报告 MAE、RMSE、R²、train-only population-std normalized MAE/RMSE。Stage 2 主指标为五个 target 等权 macro normalized MAE；不得聚合 raw MAE。本条取代 ADR-0022 中“不建立 Stage 2 aggregate”的决定。
5. Stage 3 test 保持五折 raw prediction 逐样本平均后计分。Test 排名要求覆盖 registry 中实际存在非空 test 的任务；当前为 11/21 enabled task。Validation 要求全部 enabled task 的五折结果，并报告 mean/sample standard deviation。
6. Prediction 按 task 写入 `predictions/<task>.csv`，保留 source row、registry identity/condition、raw target、prediction 与 absolute error。Stage 3 test 同时保留五个 fold prediction 和 ensemble prediction。
7. `completed` 且 schema 完整的 run 才能排名；running、failed、legacy 与 incomplete 只进入 health。声称当前 schema 的 completed run 若损坏，summarizer 必须失败且保持已有 `summary/` 不变。不同 comparison identity 不进入同一榜单。
8. Summarizer 在同级 staging 目录完成全套校验后交换 `summary/`，相同输入产生确定性结果。不得按 mtime、文件名或路径顺序猜测正式 run；alternative run 全部保留。

## 后果

- 旧 evaluation output 仍是有效历史证据，但需复用原 checkpoint 重新 evaluation 后才能进入新榜单；不要求重新训练。
- Prediction 会增加 evaluation output 体积，但不会复制到 `summary/`。
- 新模型只需生产统一 reporting contract，即可由同一 summarizer 收录。
