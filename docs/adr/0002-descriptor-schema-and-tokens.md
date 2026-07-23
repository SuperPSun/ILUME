# ADR-0002：描述符 schema 与分组 token

- 状态：Accepted
- 日期：2026-07-23

## 决定

描述符筛选支持 `full`、`clean` 和 `pruned`。所有删除与相关性决定只使用入选训练集，结果写入版本化 `DescriptorSchema`。标准化也只拟合训练集有限值。

描述符 token 数只允许 1、8、12。8/12-token 方案使用版本为 `rdkit-217-v1` 的固定语义映射和组专属 residual MLP；schema 为全部原始描述符显式保存 12 类映射，8 类再按固定合并表得到。reconstruction 从对应 fused group token 重建本组维度，再 scatter 回 schema 顺序。筛选后空组使用可学习 empty token，但不产生 loss。

## 理由

单 token 可能形成信息瓶颈，而为每个标量创建 token 会明显加长 fusion 序列。固定少量语义组在计算量、可解释性和消融可比性之间提供稳定折中。schema 显式保存删除原因和代表维度，避免不同数据划分之间发生不可见的特征漂移。

## 后果

更换 descriptor mode、token_count 或训练池后必须重新准备 artifact。旧 217 维 scaler 不能隐式复用。
