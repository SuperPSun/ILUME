# ADR-0122：Stage1 训练索引大小过滤与 H100 双卡 batch84

- 状态：已采纳
- 日期：2026-11-09
- 定向扩展：[0005](0005-exclude-invalid-pretraining-entities.md)、[0116](0116-stage1-dual-graph-transformer.md)

## 决定

Stage1 Base 与 H100 执行配置显式设置 `training.max_graph_nodes: 160`。仅在训练启动前排除原子数或无向键数严格大于160的扩增训练分子；等于160保留。两者分别对应 DGT 原子图与键线图的节点数，不是有向消息边数。若新来源出现超限原始训练样本，明确拒绝启动，避免静默违背保留全部原始样本的决定。验证、梯度审计、独立回归头后训练及下游编码不应用此训练过滤。

本决定按用户后续确认，撤回最初的 `data.max_graph_nodes` prepare/QC 过滤实现；旧字段不再接受，不保留双重过滤入口。prepared、分词器、描述符/电子/电荷标准化统计、teacher向量及失败mask、原子标签和DGT拓扑全部保留原样，不修改任何来源hash、不重新prepare。统计仍包含这14条，这是显式保留原统计的训练子集方案，不等价于在所有统计拟合前删除分子。

共享训练配置默认 `null` 且序列化省略，历史配置、corpus/feature及缓存身份不增加字段。原有 BCUT、Ipc、SMILES 长度、泄漏排除及原始 split 规则不变。不截断原子、不改原始文件、不做运行时OOM回退或整batch跳过。

## 阈值依据与保留边界

对当前已通过 QC 的5,179,473条语料逐条计数，超过160的14条均为 `augmentation/molecule` 训练样本，占0.0002703%。当前已入库原始样本最大145原子、156键，因此此次新增规则不排除原始训练或验证样本；不表示恢复此前已被其他 QC 排除的记录。统一160上限也适用于未来来源，源数据更新时需重新核查原始样本保留情况，不承诺任意未来原始分子都在范围内。

当前 Stage2/现役 Stage3 任务目录共15,075个唯一结构，最大121原子、124键，均未越界。原单卡 seed42/batch64 的第12,515步包含185原子、189键的扩增结构，与最后日志12,510步后的失败窗口吻合；没有重放完整 GPU 前向，不将其写成精确 OOM 复现。筛选是显存预算的适用范围决定，不判定这些结构化学无效，不使用验证/测试性质或误差选择阈值。

## Batch 与训练

仅将 `configs/v4/stage1/base-100h.yaml` 的全局 batch64改为84，双rank DDP每卡42。保持该配置5轮、LR1e-4、query32、attention checkpointing=false、局部compile=true及其余训练策略；标准 Base 仍为batch256/10轮。worker仍为每rank8个，预取窗口按现有公式随batch重算。没有承诺84及160上限一定不会OOM，不增加自动缩批、截断或失败样本跳过。

## 身份、准备与恢复

非默认图大小规则只进入 training identity，不进入 corpus/feature/teacher/sidecar identity。启动时校验原manifest/index SHA，扫描manifest训练行确定排除样本。SMILES字符数同时为原子/键数上界，仅对字符数超过阈值的结构调用RDKit计数。按原prepare的seed/mix-key排序合同定位少量排除项，再读取并校验对应源分片SHA、sample ID及真实原子/键数；保留全部原有完整性检查。逐进程只缓存已验证来源对应的解析结果，不修改prepared或新增共享缓存。

过滤清单绑定源corpus、manifest/index SHA、阈值、逐项sample ID、结构、大小及dataset index，整体进入training identity并写入运行目录 `training_filter.json`。恢复检查该清单和checkpoint身份；缺失、修改、阈值变化或来源变化均拒绝。format4不变，无过滤/旧检查点身份保持兼容；新batch84及过滤方案必须从头运行，不能resume旧batch64或无过滤checkpoint。

DDP先由rank0解析并广播相同identity/清单，普通错误同步传播。sampler先保持原分片顺序和分片内shuffle，只移除清单中的索引，再在保留序列上做rank分配及既定最多world_size−1条的前缀补齐。禁止rank各自跳过batch；有效训练样本数驱动步数、warmup、cosine及checkpoint train_size。当前应为5,176,579条训练样本，双rank/global84每轮61,626步。

已有匹配prepared、teacher、电荷和拓扑无需重新生成或迁移；H100更新代码后即可使用现有来源，在未占用新训练目录启动。原始数据与旧输出保持只读。已有进程不热更新；是否停止、归档或删除由用户另行决定。实施阶段仅运行临时小数据测试和只读索引审计，不启动正式prepare、teacher或训练。

2026-11-13执行修复：szx的rank1在首个broadcast等待rank0校验时触发NCCL默认600秒超时；rank0尚未发布训练目录。Stage1入口将process group超时显式设为2小时（同样适用于后续collective），并由rank0输出索引/拓扑/teacher/电荷检查阶段与耗时；teacher分片首尾及约30秒间隔在完成分片时报告进度。保留完整来源和SHA检查、普通错误的全rank传播及固定末轮合同；执行日志不进入科研身份，不修改缓存或恢复配方。仍不能将所有启动超时都归因于NFS，需结合阶段日志判断。

## 验证

覆盖等于上限保留、原子/键分别超限、原始样本保护、缺失/坏SHA、实际分片位置校验、过滤前后缓存字节不变、默认兼容与新旧训练identity互拒载。验证原shuffle的保留子序列、多轮raw exposure、短尾、双rank一致性、有效更新预算、固定末轮、恢复及审计损坏拒绝；复用现有Stage1模型、DDP归一化和缓存完整性回归测试。
