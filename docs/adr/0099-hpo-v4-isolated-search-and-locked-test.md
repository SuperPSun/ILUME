# ADR-0099：独立 HPO v4 搜索与锁定终测

状态：采用。日期：2026-10-11。

## 决策与边界

新增 `hpo_v4` 研究协议，仅作用于 `configs/HPO_v4` 和 `outputs/HPO_v4`。正式 `configs/v4` Base 默认配方、接口、身份和训练行为保持不变。此 ADR 只在 HPO 范围内扩展 0093、0095、0096、0097 的可搜索容量与训练预算；不恢复历史 ObjectEncoder、PCGrad、训练早停、验证最佳检查点或路由校准。

Stage1 完整末轮 `last.pt` 提供独立14目标后训练；仅编码器导出及配对特征提供 Stage2/3 表示。Encoder/Fusion 永久冻结。后训练头不影响下游表示。实体1024D、原子512D、专家输出1024D及有序实体槽位合同不变。三个 Stage 的 HPO 头/检查点/final 用独立 kind 和身份，禁止跨协议恢复或重标旧产物。

配置默认省略新增的可选字段，保留正式身份。Optuna 仅由 `scripts/stage3/hpo.py` 控制，数值训练继续通过公开 Stage 入口执行。实现位于 `src/stage3/hpo`；跨 Stage 只调用公开函数。

## 数值扩展

Stage2 的 `training.seed` 独立于 `data.seed`。HPO `owner_training` 对 GLOBAL/GROUP/PRIVATE 分别提供 LR、Epoch、Weight Decay。总轮数等于最长 owner 寿命，各 owner 达到预设寿命后冻结且保持 eval；五任务继续覆盖每轮原始行。联合 step、真实 batch 样本数加权的共享梯度、PRIVATE 原始平均梯度、逻辑/微批256及余批规则不变。owner 调度沿用余弦 warmup，以各自寿命换算总步数；逐轮及 final 记录每个 owner 实际获得梯度的更新次数，完整轮恢复校验寿命与累计更新数。

Stage3 新增 GLOBAL/GROUP 独立 Dropout、三类 owner Weight Decay；PRIVATE 任务/规模类别仍完整展开。HPO 可以改变专家数量和隐藏宽度，flat 门控和接口不变。PRIVATE 的隐藏宽度与 GLOBAL 分离，Tower/FiLM 同步缩放。GLOBAL、GROUP、PRIVATE Phase1 LR 严格递减；后续 LR 保持原连续性公式；PRIVATE Phase1 不超过 GLOBAL，原零轮覆盖项保持零。计划记录实际容量、参数量、nominal/effective 预算和更新步数。

迁移结构和训练来源分别检查。共享模型结构、迁移组结构、模拟 PRIVATE 张量形状须匹配；Stage2/3 训练预算可以不同。训练身份仍绑定完整来源配方、数据、同一共同训练 seed、Stage1 SHA、owner 清单及状态 hash。每个 split 的 Stage2 final 与其模拟数据严格配对。准备产物仍只绑定数据/冻结来源，不绑定专家配方。

Stage1 冻结输入缓存绑定来源 SHA、特征、输入过滤、train/valid 数据内容、全部目标与表示张量 hash。头训练各自独立初始化、Dropout、优化器与样本次序；seed 记录任务派生值。Linear 保留预训练头的等价标准化换算，不能称为随机初始化搜索。

## 选择合同

联合目标为24实验任务的折内 normalized MAE 先按折平均、再按任务等权平均。归一化沿用每折训练统计。模拟独立报告，不进入目标。缺任务、预测行覆盖不完整或主指标非有限时不能排名。后训练逐目标用末轮验证 MAE，原子电荷保持每条观察等权、观察内原子平均；14项指标不相加。

先人工配对对照，再两套结构的小空间 TPE。初筛仅 fold1/fold3；所有训练到预定末轮，不剪枝。五折复核、共同训练 seed 枚举、五套现存 split 比较都用完整五折目标。不同折集合不进入同一 Optuna study。按更小参数量、更低实测计算成本、更早候选顺序打破精确平局。完整配方和来源一致的模型可复用，评分记录按研究阶段隔离。

联合初筛上限240：最多48对照、160 TPE、32局部候选。局部预算拆为16个六维 Weight Decay 候选及16个最多六维的组/规模类别容量与 LR 候选。每个子空间先冻结，恢复不重新用最新赢家重建空间。生成失败记录为 FAIL，不占合法候选预算；没有合法配置就不启动 worker，不给虚假惩罚分数。后训练每目标最多18对照、两结构各32 TPE、前三配方五 seed。

## 运行、锁定和终测

prepare 的数据审计使用显式 `preparation.data_metadata_path`，HPO 写入 `audit/data/splitN/stage*/metadata.json`，与 prepare 自身运行元数据分开，禁止写 `data/*/metadata.json`。配置、来源审计、试验索引、SQLite、采样器和作业事件持久化；单控制进程持排他锁，worker 不写 Optuna。采用 Optuna 4.7.0 `TPESampler(multivariate=True, constant_liar=True)` 的 [ask/tell 接口](https://optuna.readthedocs.io/en/v4.7.0/tutorial/20_recipes/009_ask_and_tell.html)。每卡配额和全局配额共同限制 trial/折，不嵌套并发；逻辑 batch 和数学精度不因并行度变化。

OOM 自动降为每卡一个作业，在新 attempt 完整重跑同一配方，旧目录保留。接受路径传递给后续配置与折模型别名。非有限训练记 FAIL；数据、身份、损坏产物和其他异常停止队列。GPU 小时耗尽则中断在运行的进程并保留现场，退出码2；不减少折数或改变标准。`--resume` 校验配置和已接受输出 hash，再复用完成作业；被中断的 attempt 保留，新 attempt 完整执行，既有 Stage 完整轮恢复接口保持可用。

锁定清单绑定14头、胜出配方、共同训练 seed、验证选出的 split、五折模型、源码/Git版本和来源文件 SHA。锁定之后仅在胜出 split 运行最佳 seed 与 `{2027,2029,2039,2053}` 的重复五折，不重新选 seed。终测入口强制锁定清单，使用最佳共同 seed 的五折原单位预测均值；技术失败可按同一清单续跑。Stage1 独立测试只允许 HOMO、LUMO、原子电荷，加载训练 scaler、冻结头和原有观察/原子映射；其余11目标标明无 test。

保持 split `{42,123,1234,3407,90756}` 不重建。五套测试集合与其他划分的开发体系交叉，因此胜者称为“验证选出的最佳划分”；报告披露跨划分选择，不能当作完全独立泛化估计。此限制属于本研究协议，依据 [Cawley 与 Talbot 的选择偏差分析](https://www.jmlr.org/beta/papers/v11/cawley10a.html)。不同 split 的 HPO 结果独立展示，模型报告使用 `ilume_hpo_v4`，comparison 身份增加 `research_protocol=hpo_v4`，不混入要求来源一致的正式榜单。

运行、资源测量和产物定位见 [HPO 手册](../hpo-v4-runbook.md)。验收只使用临时小数据与模拟 worker，不自动运行正式作业。
