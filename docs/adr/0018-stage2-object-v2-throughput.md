# ADR-0018：Stage 2 Object v2 Prepare/Train 高吞吐合同

- 状态：Accepted
- 日期：2026-08-14
- 取代：ADR-0016 的 artifact/checkpoint、optimizer、冻结快路径与执行合同；ObjectEncoder、五任务监督、逐行全覆盖、task compensation、QM mask 和五 epoch 协议继续有效。

## 决定

Stage 2 data artifact、teacher cache 和 checkpoint 升至 Object v2；v1 只明确拒绝，不提供迁移。正式 Base 第一 epoch 整 epoch冻结 Stage 1 encoding backbone，后四个 epoch 联合微调。冻结粒度只允许完整 epoch，train 只允许 entity shard 全量 preload；RAM 不足直接失败。

Prepare 对每个 CSV 只扫描一次，同时完成 canonicalization、数值解析、train-only scaler、overlap/duplicate/missing-target/source QC，并以紧凑数组暂存行数据。Entity key 按固定 role/canonical SMILES 排序。使用 `spawn` CPU worker 计算 QC、descriptor 和 Stage 1 sample；主进程按候选顺序过滤、重编号和发布 shard，因此 worker 数不改变 entity 顺序、task tensor、QC 或语义 artifact identity。worker 禁止 Torch/OpenMP/MKL/OpenBLAS 内部并行，异常携带候选 ID、role 和 canonical SMILES，不串行降级。

Task artifact 直接保存 normalized condition/target、mask、entity index 与 source row；validation 额外保存 raw target。缺失 target 的 normalized 值固定为零并只由 mask 决定监督。所有 payload SHA 和完整性始终验证，不提供关闭开关。

Teacher cache 使用内容寻址目录。Identity 包含 entity index 与全部 entity shard 的语义 hash、Stage 1 encoding-state hash、FP32 embedding dtype、提取合同和 PyTorch/CUDA 数学合同。Encoding-state 包含 encode 路径参数与相关配置，但排除 checkpoint 路径、optimizer、scheduler、RNG 和 reconstruction-only 参数。Metadata 最后发布，记录 embedding shape/dtype/hash 与实际 checkpoint 相对路径。训练 LR、task weight、epoch 和 packing 参数不参与 identity。

Train 启动时将 entity sample preload 到单一共享 RAM store，并将 teacher embedding、task index、normalized tensor、mask 和 validation 统计 tensor 一次放入 GPU。Unfrozen path 使用共享 RAM 的有序线程预取池：每个 accumulation window 跨 task 合并 entity ID、全局去重、只 pack 和 encode 一次，再按 inverse position 分发。Window loss 为 `sum(weighted_loss_i) / len(window)`，每个 window 只 backward、clip 和 optimizer step 一次；student embedding 不跨 optimizer step 缓存。

Frozen path 直接以 GPU FP32 teacher embedding 作为 student entity embedding，train/validation 均不读取 entity sample、不调用 packer、不执行 Stage 1 backbone；teacher loss 是设备上的精确 `0.0`。CUDA 固定使用 pinned memory、non-blocking H2D、PyTorch 2.9 TF32 precision API 和 fused AdamW；不支持即失败。CPU 测试使用 single-tensor AdamW。Validation 固定使用 `torch.inference_mode()`。

Optimizer 固定为 backbone、ObjectEncoder、task heads 三组 LR。Backbone 冻结期 `requires_grad=False` 且 LR 为零，解冻后从自身 warmup/cosine 起点运行；其余两组从训练开始 warmup/cosine。每个 epoch 固定 full validation，并原子保存不可覆盖的 `checkpoint_epoch_XXXXX.pt`；最终模型仍为 epoch 5，不产生 best、last 或 step checkpoint。

Resume 只接受完整 Object v2 epoch checkpoint，严格恢复 model、三组 optimizer、scheduler、AMP scaler、RNG 和 global optimizer step，并校验科研配置、data、teacher、任务规模、数学合同和 optimizer implementation。`preparation`、packing worker/prefetch 和日志频率属于执行参数，允许在恢复时改变。

## 理由

Stage 2 的热点来自重复 CSV 解析、重复 entity 反序列化/packing、window 内重复 backbone encode，以及冻结 epoch 仍执行学生 backbone。把静态工作前移并为冻结期建立等价快路径，可减少 train 的 CPU、I/O 和 GPU 计算，同时保持五任务覆盖、loss 与冻结边界等科研语义。TF32、fused AdamW 和合并后的运算顺序会产生新的浮点轨迹，因此 Object v2 不与 v1 做 bitwise 比较。

## 后果

正式运行前必须重新执行 Stage 2 prepare；旧 v1 artifact、teacher cache 和 checkpoint 均被明确拒绝。性能验收使用结构调用计数、CUDA smoke 与低频运行记录，不以 smoke 冒充正式 benchmark。Stage 3 到 Object v2 的表示迁移不在本决定范围内；Stage 3 prepare 继续在写 artifact 前明确拒绝。

## 参考

- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/2.9/notes/multiprocessing.html)
- [PyTorch data loading memory behavior](https://docs.pytorch.org/docs/2.9/data.html)
- [PyTorch AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.adamw.AdamW.html)
- [PyTorch numerical accuracy and TF32](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
