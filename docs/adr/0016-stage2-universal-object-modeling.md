# ADR-0016：Stage 2 统一 ObjectEncoder 与全覆盖训练

- 状态：Partially superseded by ADR-0018
- 日期：2026-08-13
- 取代：ADR-0007 的任务调度、选模与 checkpoint 合同，以及 ADR-0009 的体系采样、双实体编码器和渐进解冻合同。

> 2026-08-14：ObjectEncoder、五任务监督、逐行全覆盖、task compensation、QM mask 和五 epoch 协议继续有效；artifact/checkpoint、optimizer、冻结快路径与执行合同由 [ADR-0018](0018-stage2-object-v2-throughput.md) 取代。本文其余内容作为历史决定保留。

## 决定

Stage 2 统一建模 molecule 和 ionic-liquid object。唯一 `ObjectEncoder` 接收 Stage 1 entity CLS：molecule 使用一个 neutral entity，IL 使用有序 cation/anion。输入序列为 learnable object CLS 与加入 Stage 2 role embedding 的 entity CLS；两层 pre-norm Transformer 不使用 positional embedding。其 CLS 经 zero-init projection 形成残差，分别加到 molecule CLS 或 cation/anion 均值后 LayerNorm。`d_model` 与 head 数只继承 Stage 1 checkpoint；Stage 2 配置仅定义层数、FFN 和 dropout。

QM、三项 IL property 和 transfer 保持五项独立监督。温度只进入 IL property head。TransferHead 分别编码 solute 与 solvent molecule，再在 head 内建模有序交互，不定义第三种 object topology。冻结 teacher 只约束 student entity CLS；不对 object CLS 添加 anchor、分布对齐或对比损失。

一个 epoch 对五个训练集各完成一次逐行无放回覆盖，保留最后的小 batch，再确定性打乱所有 batch 的任务顺序。任务 batch 的 loss 乘以 `task_weight * epoch_batch_count * batch_rows / task_rows`，使任务重要性不依赖数据规模或 batch 边界；不再进行 IL-system 或温度点重加权。

正式 Base 训练五个 epoch。第一 epoch 冻结 Stage 1 encoding backbone，第二至第五 epoch 解冻；reconstruction heads 永久冻结。一个 optimizer 和一个 LambdaLR 管理两个 param group：新模块覆盖全程 warmup+cosine，backbone 在冻结期为零并从解冻点以局部 step 独立 warmup+cosine。

每个 epoch 完成后运行五任务 full validation，再原子保存不可覆盖的 `checkpoint_epoch_XXXXX.pt`。不选 best、不 early stop、不生成 last 或 step checkpoint。恢复只接受完整 object v1 epoch checkpoint，并严格校验配置、Stage 1 model contract、data、teacher、任务规模、optimizer、scheduler、AMP 与 RNG。

## 理由

统一 object 机制保留 molecule 与 IL 的拓扑差异，同时避免把它们强制映射到同一分布。逐行全覆盖使 epoch 语义直接可审计，batch-size-aware task compensation 则把数据自然频率与任务重要性分离。单 scheduler 保留两条学习率曲线而不增加可恢复状态机。

## 后果

Stage 2 data、checkpoint 和 teacher cache 重新从 v1 开始，并使用独立 kind 防止误读历史同版本 payload；旧 Stage 2 不兼容且不迁移。Stage 3 对 object v1 的 topology 映射本轮延期，prepare 在写入前明确失败，既有 Stage 3 prepared-artifact train/evaluate 逻辑不变。
