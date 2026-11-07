# ADR-0116：Stage1 双路 DGT 图编码器

## 状态与边界

已接受（2026-11-06）。仅替代ADR-0090的现役Base图分支和图读出；SMILES十二层、轻量Fusion、learned1024、atom512、描述符211、loss、自然shuffle、2/2/1角色权重、80/10/10融合dropout及训练预算不变。Stage2/3永久冻结Stage1。历史D-MPNN及独立D-MPNN benchmark不变。

## 模型

使用[Nature Communications DGT](https://www.nature.com/articles/s41467-026-75005-9)的纯2D适配版，参考[作者实现](https://github.com/zhangsy-ryan/DGT/tree/2f5a4c3ef2e376d60238f481c36a87eeb5a581e8)，固定commit `2f5a4c3ef2e376d60238f481c36a87eeb5a581e8`。[MIT署名](../licenses/DGT-MIT.txt)保留。无需PyG、graph-tool或额外模型依赖，不导入作者训练框架。

Base设置`model.graph_encoder_type: dgt`、`graph_depth: 8`、`graph_feedforward_dim: 1024`。宽度512、8头，八个独立双路block；每路FFN为512→1024→512。沿用LayerNorm/GELU/dropout0.10，故不宣称严格复现作者完整配方。默认`dmpnn`省略新字段，原`shared/residual_blocks`参数名、初始化和前向保留；DGT不能混用非默认D-MPNN模式。

原子图与无向键线图保留原有索引。共享原子的两键在线图相连；入口按作者代码以入射特征求和融合。pair特征由mask后的原子/键投影、两图各自的最短路径与环关系形成。最短路径0表示自身、1–8为可达距离、9为更远或不可达；环关系0为无关系、1为同无弦环的相邻节点、2为同环非相邻节点。两图分别枚举长度≤18的无弦环，线图的分支三角形也依此定义。无3D坐标、RWSE或新训练目标。

每头QK标量加逐通道pair score后，沿key维逐通道softmax；加权对象为`V + pair value`。不同分子严格隔离，空键图使用安全dummy key并屏蔽输出。原子/键分别mean pooling，拼接后Linear1024→512，再与SMILES CLS512进入原Fusion。无键分子的键均值为0；读出层属于`graph_encoder`，随encoder-only导出。

2,048词表下实际参数量：图分支36,875,435、SMILES 39,010,304、Fusion 2,101,248，encoder合计**77,986,987**；当前211维监督及原子电荷头计入后训练模型为82,381,878。比较包含架构、容量和读出的共同变化，不能单独归因于attention。

## 数据与执行

`scripts/stage1/prepare.py --dgt-topology-only`从已有corpus生成独立sidecar；与`--partial-charge-only`互斥。`auxiliary.graph_topology_cache`指向其`artifacts`目录。按源分片/样本顺序保存uint8距离/环矩阵和int32线图索引，绑定corpus、sampler、源图结构、算法及SHA。完整分片可校验后继续，不改写corpus、teacher、电荷sidecar或源数据。缺片、错位、错误来源或损坏明确失败。下游新分子调用同一公开CPU构建器按需生成拓扑。

2026-11-07修复并行传输的文件句柄耗尽：worker输入/输出通过普通NumPy数组传输，父进程重建tensor，避免逐tensor共享storage消耗句柄。串行与并行拓扑值、dtype、排序、缓存格式及科学身份不变；不以提高系统上限或忽略失败作为修复。

`training.dgt_query_chunk_size: 32`，`dgt_attention_checkpointing: true`。按query分块重建pair特征，使用非重入checkpoint保留dropout RNG；共享线性pair映射先投影各加和项，避免常驻N×N×512矩阵。DGT最初实现保留全局batch512及原训练预算，不做length bucketing、OOM自动回退。执行设置不进入科研身份，但写入运行配置并在恢复时严格校验。

后续显存调整（2026-11-07）：按用户要求将现役Base的全局batch从512减半至256；单卡256，四rank DDP每卡64。10轮、LR、loss、自然shuffle和其他执行设置不变，更新数及warmup/cosine预算按新batch重新计算。batch属于训练身份，禁止从旧512检查点续训；匹配的corpus、teacher、电荷及拓扑缓存仍可复用。历史消融配置不批量修改。

## 身份与运行

DGT模型、双路读出和拓扑算法进入training/encoder及下游来源合同；training另绑定sidecar物化身份。Stage1 format4不变。旧D-MPNN按内嵌配置严格加载，不能与DGT交叉恢复或宽松加载。

同一描述符policy下，corpus、teacher、电荷sidecar身份不因DGT改变；仅须额外准备拓扑。当前本地旧prepared仍为0113的v1，而现役Base已启用0115的Gasteiger v2，因此必须先完成匹配prepare/teacher/电荷sidecar，不能重标旧身份。新Base从头训练Stage1，再重建Stage2/3表示并训练评估。历史输出只读，清理和覆盖不在本次授权内。

## 验证

覆盖逐通道attention参考数学、分块/重计算梯度及RNG、双图与mask、排列等变和批间隔离、无键离子、双路分子监督梯度、缓存来源与完整性、末轮导出、完整轮恢复、只读gradient audit及下游冻结。验收只运行临时小数据测试与性能检查，不运行正式prepare、teacher、训练或评估。
