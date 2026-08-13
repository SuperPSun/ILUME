# ADR-0014：Stage 1 Prepare 并行化与 Corpus v2

- 状态：Accepted
- 日期：2026-08-13
- 局部取代：ADR-0013 的 corpus v1 与串行 prepare 实现

## 决定

Stage 1 prepare 使用主进程顺序读写、worker 纯计算的有界进程池。augmentation canonicalization、Entity QC、AIS tokenization 和 RDKit descriptor 可并行；original canonical 排序、seed shuffle、split、leakage/audit 判定、SQLite/memmap 写入及 shard 发布顺序不变。`preparation` section 与 `--workers` 只定义执行资源，不进入实验 hash、checkpoint 恢复身份或 preparation signature。

AIS 在 Entity QC 后单遍产生长度和训练集 Counter，最终 vocabulary 真正执行 `count >= min_frequency`；正式 Base 使用 `min_frequency=1`，因此保留旧实现实际包含所有已出现 token 的科研语义。其他 tokenizer 后端保留依赖拟合结果的稳定迭代流程。

corpus artifact 固定为 `kind="ilume_stage1_corpus"`、`format_version=2`。shard 只保存训练消费的 tensor 与 sample/role id，fingerprint 以 `uint8` 保存并在 collate 时恢复为 float32；provenance 继续位于 manifest 和 audit。v1 及更早 corpus 一律拒绝并要求重新 prepare。Stage 1 checkpoint 的后续 v2 合同由 [ADR-0015](0015-stage1-high-throughput-epoch-resume.md) 定义。

输入 identity 的 SHA256、size 和 row count 在单次 invocation 中只计算一次。`performance.json` 记录本次执行各 phase 的数量、耗时、吞吐和复用状态，但不进入 artifact hash、preparation signature 或 checkpoint。

## 理由

约 530 万实体的主要 CPU 成本来自重复 AIS tokenization、RDKit QC、descriptor 和 augmentation canonicalization。进程池只承担确定性的逐实体计算，主进程保留所有有状态写入，可在提高 CPU 利用率的同时维持稳定排序、审计计数和简单恢复边界。紧凑 shard 删除重复字符串并以单字节保存二值 fingerprint，显著降低磁盘占用而不改变模型输入。

## 后果

workers 和 batch size 可以在不改变实验身份的情况下调整；Stage 1 prepare 复用目录只允许刷新 `preparation` section，其他配置仍严格一致。Entity QC、AIS 和 catalog 中断后整 phase 重做；descriptor 继续每 10000 个 durable row 保存进度，完整 shard 可复用。

现有 corpus v1 即使已完成也不能用于新训练，必须重新 prepare v2。该迁移不升级或兼容转换 Stage 1 checkpoint。
