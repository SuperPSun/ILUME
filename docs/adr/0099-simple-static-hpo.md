# ADR-0099：独立静态配置 HPO 与人工选型

- 状态：已接受
- 日期：2026-10-12
- 范围：独立 `configs/HPO`、`scripts/HPO`、`outputs/HPO`；替代已撤回的复杂 HPO 系统，不恢复 HPO_v4/Optuna。

## 决定

允许独立研究流程使用静态完整 YAML 比较预测器、HoME容量/分组和训练配方。Stage1后训练100组；HoME100对（五分组×二十容量）；人工选型后生成Stage2二十组；再次人工选型后生成Stage3五十组；最后锁住人工指定配方在五套现有split重复五折验证。研究者负责逐任务比较和选择，程序不做选模、自动晋级或测试。详见 [运行手册](../hpo-simple-runbook.md)。

Stage1完整末轮及仅编码器/特征/sidecar只读；表示库可绑定身份后复用，不改变过滤、原子映射、重复观察、角色/分子权重、初始化和训练scaler。缓存独立于预测器及训练配方，训练身份仍绑定完整配方。各owner优化、共享梯度聚合、采样、三阶段冻结、原始行覆盖、末轮发布全部保留。正式v4 YAML不增加容量候选。

Entity-HoME Stage2 Epoch可为任意正整数（历史非Entity枚举合同保留），调度器依据实际联合步预算；默认仍10。可选training.seed只在Entity-HoME有效，缺省沿用data.seed并省略序列化，保留既有身份；显式seed控制初始化和排列并进入身份。HPO显式42，独立于split。Stage3沿用已有training.seed及fold派生。

迁移按结构/语义而非完整训练预算比较；Stage3 GROUP LR/Epoch与优化参数可独立改变。GLOBAL与两个可迁移GROUP的结构、共享Dropout、Activation、接口和两模拟PRIVATE仍须匹配；实验PRIVATE在任务覆盖配方中独立改变。仍严格检查来源SHA、Stage1/特征身份、完整模型/owner、共享张量集合、shape、dtype、状态hash与加载；无部分加载或跨配方恢复。

复用原生Prepare身份：Stage2模型不绑定数据准备；Stage3prepared身份已排除meta_group，不新增身份拆分平台。单写入者、临时目录、SHA与原子完成标记保障缓存；损坏与不匹配拒绝复用。HPO使用preparation.data_metadata_path，将数据审计写入独立输出；默认正式prepare路径不变。所有配置静态固定，生成记录绑定人工选择，禁止覆盖已固定不同配置。

单机固定设备队列，max-parallel决定全局任务槽数，按设备列表轮流分配，允许每卡多个任务；默认槽数等于设备数。不嵌套折并行。Stage2/3保留原生恢复；Stage1失败trial在新attempt重跑并复用表示。无数据库、自动采样、自动重要性或动态显存调度。五套split只报告固定训练seed的验证统计；程序不挑最佳split，不自动终测，披露样本重叠。

## 2026-10-20：后续新实验采用3407

完成固定配方G2_C05 / 007 / 032的五套split五折复验后，研究者明确指定后续新实验统一使用3407；训练seed仍为42，Stage3沿用fold派生。已有正式Base、基线、消融配置及产物保留原split与身份，新实验使用独立配置、配对缓存和输出。此决定不修改HPO程序的自动选型/测试行为，也不追溯改写历史选型快照。保留全部五套split结果，披露跨划分验证选择及样本重叠，不把其测试解释为完全独立的泛化估计。结果和报告入口统一见 [HPO手册](../hpo-simple-runbook.md#当前选型与报告入口)。

## 验证

覆盖配置数量/唯一性、人工绑定、实际PRIVATE覆盖/零预算、seed与split分离、末轮预算、严格迁移/损坏拒载、缓存前后预测一致及单写入者、Dry Run无训练输出、设备与全局上限、失败和恢复、逐任务缺失/非有限与五折统计。正式Base默认序列化和既有产物边界保持；运行完整pytest、现役入口help、compileall、diff和文档链接检查，仅用临时数据，不启动正式GPU作业。
