# ADR-0122：Stage1 图大小准入与 H100 双卡 batch84

- 状态：已采纳
- 日期：2026-11-09
- 定向扩展：[0005](0005-exclude-invalid-pretraining-entities.md)、[0116](0116-stage1-dual-graph-transformer.md)

## 决定

Stage1 Base 与 H100 执行配置显式设置 `data.max_graph_nodes: 160`。在现有实体 QC 中，原子数或无向键数严格大于160时排除完整分子；等于160保留。两者分别对应 DGT 原子图与键线图的节点数，不是有向消息边数。规则在分词器拟合、描述符统计和分片发布之前执行，不截断原子、不改原始文件、不在训练中跳过 batch，也不补充替代样本。

共享配置默认 `null` 且序列化省略，历史配置不启用过滤。原有 BCUT、Ipc、SMILES 长度、泄漏排除及原始 split 规则不变。启用后在 `excluded_entities.csv` 记录 `graph_atoms_overlimit` / `graph_bonds_overlimit`、真实原子/键数及上限，元数据记录阈值与按原因、来源、角色、split统计。多个原因可重叠，不能将原因计数相加当作新增排除数。

## 阈值依据与保留边界

对当前已通过 QC 的5,179,473条语料逐条计数，超过160的14条均为 `augmentation/molecule` 训练样本，占0.0002703%。当前已入库原始样本最大145原子、156键，因此此次新增规则不排除原始训练或验证样本；不表示恢复此前已被其他 QC 排除的记录。统一160上限也适用于未来来源，源数据更新时需重新核查原始样本保留情况，不承诺任意未来原始分子都在范围内。

当前 Stage2/现役 Stage3 任务目录共15,075个唯一结构，最大121原子、124键，均未越界。原单卡 seed42/batch64 的第12,515步包含185原子、189键的扩增结构，与最后日志12,510步后的失败窗口吻合；没有重放完整 GPU 前向，不将其写成精确 OOM 复现。筛选是显存预算的适用范围决定，不判定这些结构化学无效，不使用验证/测试性质或误差选择阈值。

## Batch 与训练

仅将 `configs/v4/stage1/base-100h.yaml` 的全局 batch64改为84，双rank DDP每卡42。保持该配置5轮、LR1e-4、query32、attention checkpointing=false、局部compile=true及其余训练策略；标准 Base 仍为batch256/10轮。worker仍为每rank8个，预取窗口按现有公式随batch重算。没有承诺84及160上限一定不会OOM，不增加自动缩批、截断或失败样本跳过。

## 身份、准备与恢复

非默认图大小上限进入语料配置、corpus identity及prepare签名；训练显式校验请求上限与prepared身份一致，两个方向不匹配均要求重新prepare。训练通过corpus hash绑定该规则，batch84另改变训练预算和身份。默认关闭时原序列化及身份不增加字段，artifact格式不升级。

新配置须使用新prepared及匹配的teacher、电荷、DGT拓扑缓存，随后从头训练；不得重标旧缓存hash、跨合同resume或覆盖当前产物。原始数据与旧输出保持只读。已有进程不热更新；是否停止、归档或删除由用户另行决定。实施阶段仅运行临时小数据测试，不启动正式prepare、teacher或训练。

## 验证

覆盖串行及双worker prepare、等于上限保留、原子/键分别超限、过滤先于描述符计算、审计与原始样本保留、默认兼容、非法配置及新旧corpus训练互拒载。复用现有Stage1模型、DDP归一化、完整轮恢复和缓存完整性回归测试。
