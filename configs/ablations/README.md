# 核心消融配置

`reference.yaml` 是5个覆盖型 epoch 的 Base 参考配置。其余配置都只改变一个实验因素，保持45/45/10采样、模型规模、训练周期和优化参数一致。

| 配置 | 相对 reference 的变化 | 需要重新 prepare |
|---|---|---|
| `descriptor_full.yaml` | Desc-Clean → Desc-Full | 是 |
| `descriptor_pruned.yaml` | Desc-Clean → Desc-Pruned | 是 |
| `descriptor_tokens_1.yaml` | 8 → 1 descriptor token | 是 |
| `descriptor_tokens_12.yaml` | 8 → 12 descriptor tokens | 是 |
| `fingerprint_none.yaml` | 移除 fingerprint 模态及其 loss | 是 |
| `role_embedding_off.yaml` | 关闭共享 role embedding | 否 |
| `graph_head_linear.yaml` | MLP → linear graph head | 否 |
| `modality_dropout_off.yaml` | 关闭 modality dropout curriculum | 否 |
| `asymmetric_masking_off.yaml` | 关闭 asymmetric masking | 否 |

无需重新 prepare 的配置复用 `artifacts/pretrain_base`。其余配置写入各自的 `artifacts/ablations/<name>`；训练输出也使用独立目录，避免 checkpoint 和 `metrics.jsonl` 混用。

```bash
ilume-prepare --config configs/ablations/descriptor_full.yaml
ilume-train --config configs/ablations/descriptor_full.yaml

ilume-train --config configs/ablations/role_embedding_off.yaml
```

本目录不提供自动矩阵 runner。若要研究 tokenizer、扩增倍率或采样比例，应复制 `reference.yaml`、一次只改变一个轴，并使用新的 artifact/output 目录。
