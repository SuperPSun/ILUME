# ADR-0001：数据边界与 45/45/10 覆盖采样

- 状态：Partially Superseded by ADR-0013
- 日期：2026-07-23

> 2026-08-12：数据边界与 original split 继续有效；augmentation multiplier 和 45/45/10 采样已由 [ADR-0013](0013-stage1-full-corpus-ddp.md) 取代。以下内容作为历史决定保留。

## 决定

原始实体只来自 `data/stage1/{cation,anion,molecule}.csv`，扩增实体只来自 `data/stage1/augmentation/` 同名文件。原始实体按 role 分层形成 95%/5% train/valid；valid 不使用扩增，并通过 `seed_smiles_list` 排除其扩增后代。

正式训练采用全运行级 45% cation、45% anion、10% molecule 最大余数配额。每个 role 内无放回覆盖训练池后才进入下一轮；当显式 `max_steps` 不足以覆盖任一 role 时，在训练启动前失败。

## 理由

根目录数据现在代表整理后的三类原始单分子实体，旧 neutral 来源表不再是下游语料边界。45/45/10 保持离子实体占 90%，同时把 molecule 的覆盖预算由 5% 提升到 10%，降低完整遍历较大 neutral 池所需的总训练长度。

## 后果

扩增倍率和 role 比例是实验定义的一部分，必须写入配置和 metadata。验证集保留自然 role 分布，不使用训练 sampler 重平衡。

大语料准备使用磁盘 descriptor memmap、按 shard 第二遍 featurization、原子 shard 写入和 preparation signature。中断后只复用 signature、sample IDs 与格式都一致的 shard；成功后 metadata 最后提交并删除临时 memmap。
