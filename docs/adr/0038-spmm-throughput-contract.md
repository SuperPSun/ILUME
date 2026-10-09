# ADR-0038：SPMM 吞吐合同

- 状态：已接受
- 日期：2026-09-02
- 覆盖：ADR-0035 的batch、CUDA matmul精度和训练行顺序条款

> Early-stopping patience、验证最优检查点与固定训练预算由 [ADR-0045](0045-fixed-budget-baseline-training.md) 取代；吞吐条款保持有效。

## 背景

ADR-0035固定batch 8、FP32且关闭CUDA matmul TF32。真实Stage 3任务因此每轮产生数千个优化器 steps，随机batch在较大batch候选上还会造成显著padding浪费。RTX 4090实测显示batch 128在双组分和三组分真实长度分布上分别达到约3034和2033 行/s；三组分99-token完整优化器步峰值reserved显存约15.2 GiB。batch 256峰值约37.8 GiB且吞吐收益很小。

## 决定

1. SPMM训练与评估 batch固定为128，末批保留；LR仍为`5e-5`，max 轮为10。预热保持一个完整轮，余弦总steps按`10 × ceil(train_rows/128)`重新计算。
2. 训练采用`sortish_length_bucketing_v1`：每轮以`seed+epoch`生成行随机排列，每`20×128`行形成窗口，按行内最大组分缓存token长度排序成batch，再确定性打乱batch顺序。每轮完整覆盖全部行且不drop last。
3. FP32参数、目标、loss和检查点不变；PyTorch 2.9的CUDA matmul与cuDNN TF32均开启，不启用AMP。OOM、NaN或CUDA错误仍硬失败。
4. batch、TF32、bucketing类型、窗口和训练-order 合同进入scientific 身份；DataLoader 运行时和GPU调度仍不进入。旧SPMM 检查点不迁移或恢复。
5. sweep推荐一张GPU一个作业；正式新合同使用独立输出根`outputs/benchmarks/v1/spmm-wp350-bs128`，不得与旧SPMM输出跨合同拼接。

## 后果

- 模型、数据、split、分词器、归一化、loss、优化器、LR、验证和报告合同保持不变，但batch membership、优化器步数量与TF32数值路径改变，因此必须形成新的训练身份并完整重跑。
- 训练集使用长度分桶；验证集/测试集保持原始行顺序，仅将评估 batch更新为128，不增加预测重排逻辑。
- 独立环境、依赖锁、官方检查点及既有outputs保持不变。
