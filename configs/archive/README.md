# 历史配置

本目录保留切换到覆盖型 epoch 训练器之前的 step-based 正式配置，仅用于核对历史实验参数。当前 `ilume-train` 会明确拒绝其中的 `max_steps`、`validation_interval` 和 `checkpoint_interval` 字段，v2 checkpoint 也不能由 epoch trainer 恢复。

不要直接修改这些文件来启动新实验；新训练应从 `configs/pretrain_base.yaml`、`configs/pretrain_large.yaml`、`configs/pretrain_xlarge.yaml` 或 `configs/ablations/` 中选择配置。
