> 历史说明：旧Stage3 v1实现地图，不作为现役入口；当前合同见 [README](../README.md) 和 [ADR-0095](adr/0095-v4-entity-home-without-object-encoder.md)，旧三项消融见 [ADR-0082](adr/0082-home-mainline-and-core-ablations.md)。

# Stage3 v1 实现地图

本表只记录历史实现 v1 的迁移历史，不是现役 v2 实现指南。现役合同见 [ADR 索引](adr/README.md)，早期设计理由见 [历史摘要](adr/history.md)。表中的精调、插件和验证最优不得套用到现役三阶段。

| 旧调用链/能力 | v1 处理 | 实现位置 |
|---|---|---|
| `reference.yaml`、`il21/aux6` 域注册表 | 删除，改为任务目录事实 + YAML 任务/组注册表 | `configs/v1/stage3/base.yaml`、`src/stage3/config.py`、`src/stage3/data.py` |
| 固定条件/phase、稠密任务假设 | 删除，改为任务局部可变宽度条件与稀疏观察数据载荷 | `src/stage3/data.py` |
| Stage 2 迁移拒载、旧冻结实体/离子对表 | 替换为公开冻结 Object v3 检查点加载器和内容寻址 object 缓存 | `src/stage2/frozen.py`、`src/stage3/prepare.py` |
| Stage1+2 表示消融 | 隔离的 RDKit 2D 后端使用折内预处理与两个 GLOBAL Linear→LayerNorm adapter；HoME 后半段不变 | `configs/ablations/stage1_stage2_rdkit_home.yaml`、`src/stage3/rdkit.py`、`src/stage3/model.py` |
| AdaTT、IndependentTaskHead、FeatureGate、SelfGate、BatchNorm 专家 | 删除，改为注册表驱动动态 HoME | `src/stage3/model.py` |
| 延后溶质引入特殊分支 | 替换为通用 primary/partner 槽位与组内共享交互 | `src/stage3/data.py`、`src/stage3/model.py` |
| 域 loss 聚合/反向 | 替换为任务梯度、样本加权微批累加与复合步 | `src/stage3/train.py` |
| 域隔离优化器 | 替换为显式 GLOBAL/GROUP/PRIVATE 参数归属；联合训练 phase 使用分层 PCGrad，精调使用逐任务 PRIVATE 优化器 | `src/stage3/model.py`、`src/stage3/pcgrad.py`、`src/stage3/train.py` |
| 早停、最优/域-最优、滚动/last 检查点 | 删除；保留定期完整轮检查点，并额外发布验证最优 PRIVATE 拼接产物 | `src/stage3/train.py` |
| 外部矩阵/折启动器 | 不恢复；唯一训练入口使用 spawn worker 和显式设备槽调度独立折 | `scripts/stage3/train.py` |
| 旧验证集/测试集最优检查点加载器 | 替换为默认逐任务精调后产物；显式 `--checkpoint-epoch N` 才严格加载普通检查点 | `src/stage3/evaluate.py` |
| 阶段/适配式分步扩展 | 替换为加载范围与适配范围分离的插件初始化 | `src/stage3/train.py` |

历史实现 v1 的训练、评估入口保持为 `scripts/stage3/prepare.py`、`train.py`、`evaluate.py`。Capacity v1 的 `scripts/stage3/capacity.py` 只负责探针、稳健性与比较报告；超参数搜索和配置物化已退役。`train.py --fold` 为必填参数，可接收一个或多个折；`--output` 始终是共同根目录，实际运行合同位于 `foldN/`。布尔 `--resume` 只恢复身份一致且检查点/metrics/诊断尾部严格对齐的折。Object 与 RDKit 产物/检查点不提供交叉兼容解析器。
