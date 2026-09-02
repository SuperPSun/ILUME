# ADR-0039：Global-RDKit v2 主线表示

- 状态：Accepted
- 日期：2026-09-02
- 取代范围：ADR-0002 的主线 grouped descriptor token 决定、ADR-0004 的主线 fingerprint 模态，以及 ADR-0013/0015 中五路 Stage 1 loss；legacy v1 与 Capacity v1 保留原合同。

## 背景

旧主线把 SMILES、Graph、grouped RDKit descriptor 与 fingerprint 融合后只导出 512 维 CLS。全局 physicochemical 信息必须经过单一 CLS 路径，且 fingerprint 与 Graph local environment 高度重叠。新主线需要显式保留 post-Fusion RDKit 表示，同时不改变 SMILES Transformer、D-MPNN、Fusion 主体和既有训练调度。

## 决定

正式主线迁移到 `configs/v2` 与 `outputs/v2`。Stage 1 v2 使用 SMILES、Graph、RDKit 三模态；legacy `configs/v1` 和 `configs/experiments_v1` 继续使用原五模态实现，不迁移 artifact 或 checkpoint。

RDKit 使用 runtime 固定顺序的全部 217 个 descriptor。训练集有限值拟合 mean/std；非有限输入标准化为零并携带 validity/mask indicator，不删除零方差、重复或相关列。Whole-vector encoder 接收 217 个 masked value 与 217 个 indicator，执行 `Linear(434,1024)`、两个既有 residual MLP block、`Linear(1024,512)` 与 output LayerNorm，只生成一个 RDKit token。

SMILES、Graph atom/bond 与 RDKit encoder 各自在输出端归一化。Fusion 保持 512 维、8 heads、8 layers、FFN 2048；CLS、SMILES、Graph、RDKit 使用四类 modality embedding，atom/bond 共享 Graph ID。curriculum modality dropout 与 asymmetric masking 在三个模态间等权选择；Graph 同时控制 atom/bond。Stage 1 只保留 SMILES、atom、bond、RDKit 四项 reconstruction loss，lambda 均为 1，element-level role weight 继续为 2/2/1。RDKit reconstruction 只从 post-Fusion RDKit token 经 `Linear(512,217)` 产生。

`encode()` 继续返回 512 维 CLS。新 `encode_entity()` 返回 post-Fusion CLS、post-Fusion RDKit、二者无投影拼接的 1024 维 entity embedding，以及 512 维 atom states 与 atom batch。v2 Stage 1 corpus/checkpoint 使用 format v3；v1 继续使用 format v2，二者严格拒绝交叉加载。

Stage 2 teacher 与 live student 都使用 1024 维 entity embedding。ObjectEncoder 为 1024 维、8 heads、2 layers、FFN 2048；teacher lambda 保持 0.10。Partial Charge 保留 512 维 atom states，仅在 AtomPropertyHead 内将 1024 维 object context 投影到 512。Stage 3 从 frozen Stage 2 artifact 读取 1024 维表示；HoME 的 expert 数、topology、ownership、hidden ratio、dropout、PCGrad、sampling 与 refinement 不变。

## Identity 与兼容性

v2 feature identity 不含 fingerprint，encoder identity 使用 `encode-entity-v2` 并记录 token/atom/entity 维度。teacher extraction contract 对 v2 使用版本 3。Stage 2/3 容器格式继续使用动态 shape 与 semantic/model contract，不因宽度变化升级；新的 encoder identity、state shape 与隔离输出路径禁止跨版本 reuse/resume。

现有 v1、Capacity v1、baseline、RDKit HoME ablation 和 No-Stage1 ablation 均不改变。正式输出不覆盖、不迁移、不自动归档。

## 后果

v2 必须按 Stage 1 prepare/train、Stage 2 prepare/teacher/train、Stage 3 prepare/train/evaluate 从头生成。entity、ObjectEncoder 与 HoME 宽度翻倍会增加 Stage 2/3 参数量和显存，但本决定不自动调整 batch、learning rate 或其他训练参数，也不增加 OOM fallback。
