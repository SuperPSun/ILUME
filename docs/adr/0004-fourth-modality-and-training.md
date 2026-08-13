# ADR-0004：指纹、role embedding 与单卡训练器

- 状态：Accepted
- 日期：2026-07-23

## 决定

分子指纹作为独立第四模态进入 Fusion，而不与连续 descriptor 值直接拼接。Morgan 和 MACCS 分块编码，使用 family/chunk identity；重建 loss 先按 family 归一化再等权平均。

可选共享 role embedding 加到 CLS 与全部非 padding fusion token，不进入各模态 encoder，也不建立 role 专属参数分支。正式参考配置启用 role embedding、MLP graph head、curriculum modality dropout 和 asymmetric masking。

提供单卡 `ilume-train`，包含 AMP、梯度累积/裁剪、warmup+cosine、validation、checkpoint/resume、RNG 与 sampler 状态恢复。暂不加入 DDP、TensorBoard 或自动实验矩阵。

## 理由

指纹是离散结构存在性信号，和连续 RDKit descriptor 的 mask、编码与损失语义不同，因此保留独立模态边界。role embedding 显式提供离子角色条件，同时保持三个 role 的主体参数共享。单卡训练器覆盖当前正式实验所需的可恢复性，又避免在没有多卡需求时引入分布式状态复杂度。

## 后果

Fusion layout 和总 loss 从四项升级为五项，旧 artifact/checkpoint 不兼容。验证必须关闭训练期 dropout/asymmetric 随机性，并按 role 单独报告指标。
