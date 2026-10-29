# ADR-0109：WeakOpt第三轮共享owner定向参数搜索

- 状态：已接受
- 日期：2026-10-29
- 范围：用户授权自行设计预算并执行的新Goal；独立于已结束的0104、0108，不改变正式Base。

## 决定

依据第二轮只读验证诊断，新增 `configs/HPO/WeakOpt/Shared`、`outputs/HPO/WeakOpt/Shared` 和独立入口 `scripts/HPO/weakopt_shared.py`。固定Joint/068的架构、data/3407、训练seed42及fold派生、Stage1、任务分组、条件、loss、原始采样、batch、owner裁剪、梯度规则、优化器和三个阶段机制。严格绑定既有Stage2 S030完整final及清单SHA，本轮不重新训练Stage2，也不改变共享表示结构。

在原实验PRIVATE LR、三阶段预算、dropout白名单之外，只允许现有Stage3 GLOBAL的LR/存续期及活跃GROUP的Phase1/2 LR/存续期。维持GLOBAL > GROUP > PRIVATE的Phase1 nominal LR顺序及GROUP Phase2 LR等于Phase1 terminal LR；warmup、floor、共享dropout、宽度、专家数、任务权重、模拟PRIVATE及其他参数保持不变。不搜索未被任务使用的GROUP，不恢复路由干预或梯度算法候选。各owner及PRIVATE预算上限30轮，配方进入原生训练身份，不跨配方恢复。

## 预算与选择

最多48份新完整配置，先各训练Fold1/3，最多8份补Fold2/4/5；唯一请求训练折上限120。预算预约通过共享锁持久化，重复请求只作严格恢复，不增加新名额；失败现场与已预约名额保留。正式输出、模型权重和周期检查点全部保留，不以清理换预算。

首批16份：三份PRIVATE短预算组合；在第一份组合上分别测试GLOBAL LR乘0.9/1.2、GLOBAL寿命20轮；phase_stability GROUP LR乘0.85/1.15、Phase2寿命12/28轮；thermophysical GROUP LR乘0.75/1.25、Phase2寿命8轮；dielectric_optical GROUP LR乘0.75/1.25、Phase2寿命6轮。GROUP两阶段LR同步缩放，架构不变。保留068密度/黏度的较长Phase3，其余短预算组合检验减少无收益末期适配的假设。

后续最多两批各16份，由完成后的验证分析明确写成静态提案再执行，不恢复Optuna。补折由已完成的验证结果明确指定，最多8份覆盖整个新Goal。只按第一名数降序、共同尺度nMAE升序、编号升序比较；两折不算五折达标，不从不同候选拼接逐任务权重或检查点。

目标仍为验证15/24项第一、测试五模型集成6/10项第一。达到五折验证目标后不再生成候选；预算耗尽也停止新增。锁定后最多执行一次新测试集成，测试不反馈搜索。若新配方未超过既有068的五折验证优先级，则保留既有模型并如实报告，不强行替换或根据测试改选。旧测试已被查看，属于探索性研究，不能表述为完全独立的泛化确认。

## 执行与验证

仅物理GPU2/3，每卡三个作业槽、总并发六，底层折并发一；沿用前两轮相同GPU槽锁。队列自动补位，完成后核对，不实时轮询逐轮指标。所有配置/来源/数据/预测/恢复检查沿用原生严格校验；新run目录和报告不覆盖历史。

最小测试覆盖新增共享参数白名单、保护字段与来源SHA拒绝、重复/超预算拒绝、跨队列补折上限、锁定后禁止追加、显式新路径和旧入口兼容。执行项目pytest、现役入口帮助、编译、Markdown链接及diff检查；运行方式见[WeakOpt手册](../weakopt-runbook.md#第三轮共享owner定向搜索)。
