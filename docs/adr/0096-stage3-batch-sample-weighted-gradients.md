# ADR-0096：Stage3 按当前 batch 样本数加权共享梯度

- 状态：已接受
- 日期：2026-10-10
- 修订范围：ADR-0070 在现役 Entity-HoME Stage3 的共享梯度及 PRIVATE 缩放规定；历史非 Entity-HoME 合同不变。

## 决定

现役 Stage3 的每条实验训练记录具有相同名义权重，不按独立体系数、任务总行数或 GROUP 数加权。任务仍计算当前真实 batch 的平均 SmoothL1 Loss；microbatch 的 loss sum 除以该任务完整 batch size 后累积，保持原有数值定义。

Phase1/2 在每个 step 取得实际索引切片长度 `B_t = len(indices)`。GLOBAL 对当前参与的全部实验任务直接计算 `sum(B_t * g_t) / sum(B_t)`；GROUP 对当前参与的组内实验任务使用同一公式。GLOBAL 不先对 GROUP 均值等权聚合，也不再乘 `task_weight`、`group_weight` 或任务总数据量。

PRIVATE 使用本任务原始平均 Loss 梯度，不施加额外样本权重；之后仍执行原 owner 裁剪和优化器更新。Phase3 单任务更新不变。两项模拟辅助任务始终只更新自身 PRIVATE，不进入共享分子或分母；仅模拟参与的 step 不产生共享梯度。

保持 Raw Sampling，不重复、不补齐。尾 batch 按实际长度计权，已耗尽任务不进入该 step 的分母。只有一个实验任务时，共享梯度保持原值。参数缺失梯度按零贡献处理，不对该参数单独重归一化；完全没有梯度的参数不赋值。冻结机制、owner 存续期、学习率、轮数、loss、模型及实验/模拟评估协议均不变。

## 身份和兼容

训练计划使用 `math.gradient_aggregation: batch_sample_weighted_owner_raw_v1`，并记录：

```yaml
gradient_weighting:
  weight_source: actual_task_batch_size
  normalization: participating_experimental_samples
  global: cross_task_sample_mean
  group: within_group_sample_mean
  private: raw_task_mean_gradient
  simulation: private_only
```

以上字段纳入既有训练身份 hash。plan14/training18、checkpoint/final format4 及架构 kind 保持不变；版本相同不代表训练身份相同。旧 `weighted_owner_raw_v1` Entity-HoME 检查点不能恢复或加载到新合同，旧产物只读，复现须使用对应历史 Git 版本。新训练使用新输出目录。

准备产物、Stage1/Stage2 和数据身份不因本次Stage3梯度合同改变，可在原身份一致时复用。后续Stage2也采用样本加权的变更见 [ADR-0097](0097-stage2-batch-sample-weighted-gradients.md)，其旧训练产物不能用于新合同迁移。动态 batch 权重仅来自当前训练索引，不读取验证集/测试集，不新增 YAML 容量、预算或算法候选开关。

## 验证

验证跨组及组内样本加权、PRIVATE 原值、尾 batch、耗尽任务、缺失梯度、冻结 owner 和模拟隔离；共享梯度与当前实验记录 Loss 均值等价，microbatch 切分不改变任务平均梯度。使用临时 CPU 数据验证中断恢复与连续运行一致、新身份及旧合同拒载。完整 pytest、入口 help、compileall、diff 与文档链接检查；不执行正式 GPU 训练或改写历史产物。
