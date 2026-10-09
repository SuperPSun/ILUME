# ADR-0039：Global-RDKit v2 主线表示

- 状态：已接受
- 日期：2026-09-02
- 取代范围：ADR-0002 的主线 grouped 描述符 token 决定、ADR-0004 的主线指纹模态，以及 ADR-0013/0015 中五路 Stage 1 loss；历史实现 v1 与 Capacity v1 保留原合同。

> 2026-09-07：本文所称 Stage 3 采样不变已由 [ADR-0046](0046-stage3-ownership-clipping-raw-sampling.md) 修订为现役 v2 原始样本采样；历史实现 v1 与 Capacity v1 不变。
>
> 2026-09-07：本文所称 Stage 3 专家拓扑与精调不变已由
> [ADR-0047](history.md#adr-0047) 修订为 per-组/任务 capacity
> 与 deterministic 四阶段训练；准备产物表示合同不变。
>
> 2026-09-10：现役 Stage 3 schedule、PRIVATE 宽度与 final 产物已由
> [ADR-0048](0048-stage3-owner-lifetime-three-phase-training.md) 改为按 `unique_systems` 的
> owner-存续期三阶段合同；准备产物表示合同仍不变。

## 背景

旧主线把 SMILES、Graph、grouped RDKit 描述符与指纹融合后只导出 512 维 CLS。全局 physicochemical 信息必须经过单一 CLS 路径，且指纹与 Graph local environment 高度重叠。新主线需要显式保留 post-Fusion RDKit 表示，同时不改变 SMILES Transformer、D-MPNN、Fusion 主体和既有训练调度。

## 决定

正式主线迁移到 `configs/v2` 与 `outputs/v2`。Stage 1 v2 使用 SMILES、Graph、RDKit 三模态；历史实现 `configs/v1` 和 `configs/experiments_v1` 继续使用原五模态实现，不迁移产物或检查点。

RDKit 使用运行时固定顺序的全部 217 个描述符。训练集有限值拟合均值/std；非有限输入标准化为零并携带 validity/mask indicator，不删除零方差、重复或相关列。Whole-vector 编码器接收 217 个带mask value 与 217 个 indicator，执行 `Linear(434,1024)`、两个既有残差 MLP block、`Linear(1024,512)` 与输出 LayerNorm，只生成一个 RDKit token。

SMILES、Graph 原子/键与 RDKit 编码器各自在输出端归一化。Fusion 保持 512 维、8 注意力头、8 层、FFN 2048；CLS、SMILES、Graph、RDKit 使用四类 modality embedding，原子/键共享 Graph ID。curriculum modality dropout 与 asymmetric masking 在三个模态间等权选择；Graph 同时控制原子/键。Stage 1 只保留 SMILES、原子、键、RDKit 四项 reconstruction loss，lambda 均为 1，element-level 角色权重继续为 2/2/1。RDKit reconstruction 只从 post-Fusion RDKit token 经 `Linear(512,217)` 产生。

`encode()` 继续返回 512 维 CLS。新 `encode_entity()` 返回 post-Fusion CLS、post-Fusion RDKit、二者无投影拼接的 1024 维实体 embedding，以及 512 维原子状态与原子 batch。v2 Stage 1 语料/检查点使用格式 v3；v1 继续使用格式 v2，二者严格拒绝交叉加载。

Stage 2 teacher 与 live student 都使用 1024 维实体 embedding。ObjectEncoder 为 1024 维、8 注意力头、2 层、FFN 2048；teacher lambda 保持 0.10。Partial Charge 保留 512 维原子状态，仅在 AtomPropertyHead 内将 1024 维 object context 投影到 512。Stage 3 从冻结 Stage 2 产物读取 1024 维表示；HoME 的专家数、拓扑、参数归属、隐藏宽度比例、dropout、PCGrad、采样与精调不变。

## 身份与兼容性

v2 特征身份不含指纹，编码器身份使用 `encode-entity-v2` 并记录 token/原子/实体维度。teacher extraction 合同对 v2 使用版本 3。Stage 2/3 容器格式继续使用动态 shape 与 semantic/模型合同，不因宽度变化升级；新的编码器身份、状态 shape 与隔离输出路径禁止跨版本 reuse/恢复。

现有 v1、Capacity v1、基线、RDKit HoME ablation 和无Stage1消融均不改变。正式输出不覆盖、不迁移、不自动归档。

## 后果

v2 必须按 Stage 1 数据准备/训练、Stage 2 数据准备/teacher缓存生成/训练、Stage 3 数据准备/训练/评估从头生成。实体、ObjectEncoder 与 HoME 宽度翻倍会增加 Stage 2/3 参数量和显存，但本决定不自动调整 batch、learning rate 或其他训练参数，也不增加 OOM fallback。
