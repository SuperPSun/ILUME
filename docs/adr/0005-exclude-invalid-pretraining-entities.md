# ADR-0005：准备阶段直接排除异常实体

- 状态：Accepted
- 日期：2026-07-24

## 决定

在 train/valid 划分和扩增倍率选择之后、任何批量 RDKit 描述符计算之前，对原始和扩增实体执行同一组准入检查。含 BCUT2D 不支持键型、Ipc 非有限或绝对值超过 `sqrt(float64_max)`、以及当前 tokenizer 输入超过 256 token（含 `[CLS]`、`[SEP]`）的实体直接排除。一个实体可以保留多个排除原因，但只从语料计数中扣除一次。

被排除扩增实体不回填。tokenizer 只在最终保留训练集上拟合；数据驱动 tokenizer 若重新拟合后产生新的超长实体，则继续排除并重拟合到集合稳定。所有排除写入 `excluded_entities.csv` 和 metadata。`SmilesTokenizer.encode()` 仍拒绝超限输入，不截断。孤立氢警告不是排除条件。

corpus、index 和 shard 升级为 format v3，旧 v2 artifact 必须重新准备。descriptor schema、scaler 数学定义、模型接口和 checkpoint 格式不变。

## 理由

原始 57,605 个实体中不存在 BCUT2D 不支持键型或 Ipc 平方上溢；256-token 上限只排除 10 个极端长实体。实际失败由少量扩增异常触发。直接排除比修改 BCUT 化学语义、变换 Ipc 或扩大 Fusion 序列更简单，也能在昂贵描述符计算前提供明确的数据审计。

## 后果

QC 后的原始/扩增数量和实际倍率可能低于配置选择结果，训练覆盖保证以最终 artifact 为准。不同 tokenizer 后端可能得到不同的超长排除集合，因此 tokenizer 消融必须同时报告 `excluded_entities.csv` 和最终语料数量。任何有效标准化描述符若仍为非有限 float32，准备过程立即失败而不是生成可训练 artifact。

## 备选方案

- 保留配位键实体并仅屏蔽 BCUT2D：拒绝，因为当前需求优先简化训练语料。
- 对 Ipc 使用稳定缩放或 log 变换：拒绝，因为仅极少数扩增异常触发上溢。
- 保持 384 或扩大到 768：拒绝，因为 256 已保留 99.98% 原始实体，并限制极端 Fusion 显存峰值。
