# 消融配置模板

`reference.yaml` 是可直接运行的 Base 参考模板。每次实验复制该文件，只改变一个轴，并同时修改 `data.artifacts_dir` 与 `training.output_dir`，避免不同 schema/tokenizer 的 artifact 或 checkpoint 混用。

## 实验轴

| 轴 | 配置字段 | 值 |
|---|---|---|
| 描述符筛选 | `descriptor.mode` | `full`, `clean`, `pruned` |
| 描述符 token | `descriptor.token_count` | `1`, `8`, `12` |
| SMILES tokenizer | `tokenizer.backend` | `ais`, `ape`, `bpe`, `spe` |
| 扩增倍率 | `data.augmentation` | 每个 role 为 `0`, `1`, `4`, `all` |
| 指纹 | `fingerprint.kind` | `none`, `morgan`, `maccs`, `both` |
| role 条件 | `model.role_embedding` | `false`, `true` |
| graph head | `model.graph_head` | `linear`, `mlp` |
| modality dropout | `masking.dropout_schedule` | `off`, `static`, `curriculum` |
| 模型 profile | 整组 model/training 字段 | `smoke`, `base`, `large` |

所有模板默认保留 `sampling.role_probabilities: [0.45, 0.45, 0.10]`。采样比例研究必须在单独命名的配置中显式修改，sampler 不会根据数据量自动推断比例。

当 `fingerprint.kind: none` 时将 `loss.lambda_fingerprint` 设为 0。`dropout_schedule: off` 会忽略四个配置概率；`static` 从训练开始使用完整配置概率；`curriculum` 按 0%–10% 为零、10%–60% 线性升至一半、60%–100% 线性升至完整概率。

准备和训练顺序：

```bash
ilume-prepare --config configs/ablations/my_experiment.yaml
ilume-train --config configs/ablations/my_experiment.yaml
```

本目录不提供自动矩阵 runner，以避免一次命令意外启动大量正式训练。
