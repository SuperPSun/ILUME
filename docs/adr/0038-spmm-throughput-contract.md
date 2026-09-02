# ADR-0038：SPMM 吞吐合同

- 状态：Accepted
- 日期：2026-09-02
- 覆盖：ADR-0035 的batch、CUDA matmul精度和训练row顺序条款

## 背景

ADR-0035固定batch 8、FP32且关闭CUDA matmul TF32。真实Stage 3任务因此每轮产生数千个optimizer steps，随机batch在较大batch候选上还会造成显著padding浪费。RTX 4090实测显示batch 128在双组分和三组分真实长度分布上分别达到约3034和2033 rows/s；三组分99-token完整optimizer step峰值reserved显存约15.2 GiB。batch 256峰值约37.8 GiB且吞吐收益很小。

## 决定

1. SPMM训练与evaluation batch固定为128，末批保留；LR仍为`5e-5`，max epochs仍为50，early-stopping patience仍为10。warmup保持一个完整epoch，cosine总steps按`50 × ceil(train_rows/128)`重新计算。
2. 训练采用`sortish_length_bucketing_v1`：每轮以`seed+epoch`生成row permutation，每`20×128`行形成窗口，按row内最大component缓存token长度排序成batch，再确定性打乱batch顺序。每轮完整覆盖全部row且不drop last。
3. FP32参数、target、loss和checkpoint不变；PyTorch 1.13的CUDA matmul与cuDNN TF32均开启，不启用AMP。OOM、NaN或CUDA错误仍硬失败。
4. batch、TF32、bucketing类型、窗口和training-order contract进入scientific identity；DataLoader runtime和GPU调度仍不进入。旧SPMM checkpoint不迁移或resume。
5. sweep推荐一张GPU一个job；正式新合同使用独立输出根`outputs/benchmarks/v1/spmm-wp350-bs128`，不得与旧SPMM输出跨合同拼接。

## 后果

- 模型、数据、split、tokenizer、normalization、loss、optimizer、LR、validation和reporting合同保持不变，但batch membership、optimizer step数量与TF32数值路径改变，因此必须形成新的训练身份并完整重跑。
- train使用长度分桶；valid/test保持原始row顺序，仅将evaluation batch更新为128，不增加预测重排逻辑。
- 独立环境、依赖锁、官方checkpoint及既有outputs保持不变。
