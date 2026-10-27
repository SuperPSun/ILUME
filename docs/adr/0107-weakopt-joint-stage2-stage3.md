# ADR-0107：WeakOpt 第二轮 Stage2 / Stage3 联合参数优化

- 状态：已接受
- 日期：2026-10-27
- 范围：用户授权的新一轮最多80对候选；独立于已结束的 ADR-0104。

## 决定

上一轮配置与结果完整保留，在 `configs/HPO/WeakOpt/Joint`、`outputs/HPO/WeakOpt/Joint` 建立独立研究。固定 data/3407、训练seed42、Stage1编码器、全部结构/宽度/专家数、任务分组、Stage2五模拟和Stage3的24实验/两模拟。Stage3仍按fold派生seed。正式Base和Stage科研算法不改。

Stage2仅允许调整 `training.epochs`、`task_head_learning_rate`、`weight_decay`、`warmup_fraction`。Stage1永久冻结，因此不搜索backbone LR；当前task/group权重不参与该路径，不搜索这些无效字段。保留256逻辑/微批、真实batch样本加权共享梯度、PRIVATE原始梯度、完整轮覆盖及预定末轮发布。GROUP内的Stage3预算不是Stage2训练参数。

Stage3沿用上一轮的实验任务PRIVATE LR、三阶段轮数、dropout白名单。只增加必须严格验证的来源变化：每个候选绑定注册的Stage2完整final。GLOBAL/GROUP结构与共享dropout保持不变，两个模拟PRIVATE迁移合同不变。不拼接不同候选任务权重，不以Stage2训练误差替代下游验证。

每个候选是一对完整自包含YAML的语义身份，编号001–080。相同Stage2配方共享规范来源S001等，只训练一次并严格检查完整训练身份。Stage3绑定来源权重及清单SHA，补折/恢复发现来源变化必须拒绝；不同配方重新训练，不跨配置续训。准备产物只读复用，来源与原始数据审计冻结，不运行prepare。

## 搜索与调度

首批24对：Stage2的5/15/20轮与HoME LR6e-5/1e-4/2e-4/3e-4组合，各配Stage3原参考或上一轮五折验证优胜038的PRIVATE参数，共12份Stage2来源。后续由授权Goal分析验证后分批追加静态提案，累计最多80对；先验证整批合法性和去重再写配置，不使用Optuna。

沿用第一名数降序、共同尺度nMAE升序、编号升序。Fold1/3筛选、入围补五折；目标为实验五折15/24、测试集成6/10。与相同折十个baseline按原单位MAE比较，同MAE并列。80对用尽即停止新增，即使未达标也完整报告收益与退化。测试不参与搜索，锁定五折优胜后只执行一次测试集成。上一轮测试已被研究者看到，须披露探索历史，不解释为未触及的独立测试。

独立入口 `scripts/HPO/weakopt_joint.py` 提供generate/run/report/finalize，复用原生Stage2训练、Stage3训练/独立验证、严格恢复和队列。已有WeakOpt验证读取函数只增加显式目录参数，旧默认行为不变。run先执行请求候选依赖的唯一Stage2集合，全部成功后执行Stage3队列；失败保留现场，不启动依赖失败来源的Stage3。

仅物理GPU2/3及其子集，每卡3作业、总计6，底层折并发1。共享原WeakOpt的三个GPU槽锁，避免跨控制器超额。只在完成后核对，不实时读取逐轮指标；全部权重、日志、身份、报告保留在4090。

运行见[WeakOpt手册](../weakopt-runbook.md)。验证来源去重、参数保护、80对上限、补折绑定、来源SHA变化拒载；共用原GPU槽、排名与验证身份测试，并执行项目pytest及入口/编译/链接/diff检查。
