# ADR-0023：统一评估报告与结果汇总

- 状态：已接受
- 日期：2026-08-24

> Stage 2 的输出文件集、ranking 和 eligibility 已由
> [ADR-0024](0024-stage2-partial-charge-benchmark-suite.md) 扩展为 Core、Partial Charge、完整
> 三榜；本 ADR 的 Stage 3 合同保持不变。
>
> 2026-08-25：Stage 2 的 orbital、Core 与子合同版本由 [ADR-0025](0025-stage2-homo-lumo-scalar-tasks.md) 取代；通用报告结构定义 v1 与 Stage 3 合同保持不变。

> 现役边界：Stage 2 评估/报告已由 [ADR-0043](0043-retire-stage2-evaluation-and-v2-refinement.md) 退役，下文 Stage 2 榜单与执行步骤仅作历史记录。Stage 3 与通用合同按 [ADR 索引](README.md) 的后续修订读取。

## 背景

完整科研运行产物分散在 `outputs/`，ILUME、MLP、ECFP+XGBoost 与后续基线缺少统一、可审计且适合论文比较的结果入口。Stage 2 也缺少与基线测试集合同对齐的独立 evaluator。

## 决定

1. `outputs/` 保留完整运行、预测、检查点与审计材料；Git 可跟踪的 `summary/` 只保留 overview、三张榜单、三张明细表、health 表和机器可读汇总，不复制训练或预测 payload。
2. Evaluation 汇总使用 `reporting_schema_version=1`，显式记录模型显示名、protocol、study ID、比较身份与预测清单。全局 summarizer 只按 `stage/operation/schema` 解析，不按模型名分支。
3. Stage 2 正式测试集固定为汽化热与阳离子/阴离子 PBE/TZVP HOMO/LUMO，共 3 任务、5 标量目标。测试实体在评估进程内按现役 Stage 1 特征合同构建，不写入准备产物、teacher 缓存或训练身份。
4. 所有标量目标报告 MAE、RMSE、R²、仅训练集 population-std 归一化MAE/RMSE。Stage 2 主指标为五个目标等权 macro 归一化MAE；不得聚合原始 MAE。本条取代 ADR-0022 中“不建立 Stage 2 aggregate”的决定。
5. Stage 3 测试集保持五折原始预测逐样本平均后计分。测试排名要求覆盖注册表中实际存在非空测试集的任务；当前为 11/21 enabled 任务。验证要求全部 enabled 任务的五折结果，并报告均值/样本 standard deviation。
6. Prediction 按任务写入 `predictions/<task>.csv`，保留来源行、注册表身份/条件、原始目标、预测与 absolute error。Stage 3 测试集同时保留五个折预测和集成预测。
7. `completed` 且结构定义完整的运行才能排名；running、failed、历史实现与 incomplete 只进入 health。声称当前结构定义的 completed 运行若损坏，summarizer 必须失败且保持已有 `summary/` 不变。不同比较身份不进入同一榜单。
8. Summarizer 必须显式接收一个或多个 input 根目录；可选 include 是精确目录前缀，必须属于某个 input 且至少匹配一个报告候选，不支持 glob。过滤先于元数据解析，未选择目录中的损坏结果不阻塞发布。
9. 选中范围内的 alternative 运行全部保留，并按真实运行路径去重。Summarizer 在同级 staging 目录校验后原子交换 `summary/`；相同输入产生确定性结果，不按 mtime、文件名或路径顺序猜测正式运行。

## 后果

- 旧评估输出仍是有效历史证据，但需复用原检查点重新评估后才能进入新榜单；不要求重新训练。
- Prediction 会增加评估输出体积，但不会复制到 `summary/`。
- 新模型只需生产统一报告合同，即可由同一 summarizer 收录。
