# ADR-0004：指纹、角色embedding 与单卡训练器

- 状态：部分被取代，替代ADR： ADR-0013
- 日期：2026-07-23

> 指纹与角色embedding 决定继续有效；单卡限定及旧恢复合同已由 [ADR-0013](0013-stage1-full-corpus-ddp.md) 取代。

## 决定

分子指纹作为独立第四模态进入 Fusion，而不与连续描述符值直接拼接。Morgan 和 MACCS 分块编码，使用 family/chunk 身份；重建 loss 先按 family 归一化再等权平均。

可选共享角色embedding 加到 CLS 与全部非 padding fusion token，不进入各模态编码器，也不建立角色专属参数分支。正式参考配置启用角色embedding、MLP graph 预测头、curriculum modality dropout 和 asymmetric masking。

提供单卡 `python scripts/stage1/train.py`，包含 AMP、梯度累积/裁剪、预热+余弦、验证、检查点/恢复、RNG 与 sampler 状态恢复。暂不加入 DDP、TensorBoard 或自动实验矩阵。

## 理由

指纹是离散结构存在性信号，和连续 RDKit 描述符的 mask、编码与损失语义不同，因此保留独立模态边界。角色embedding 显式提供离子角色条件，同时保持三个角色的主体参数共享。单卡训练器覆盖当前正式实验所需的可恢复性，又避免在没有多卡需求时引入分布式状态复杂度。

## 后果

Fusion layout 和总 loss 从四项升级为五项，旧产物/检查点不兼容。验证必须关闭训练期 dropout/asymmetric 随机性，并按角色单独报告指标。
