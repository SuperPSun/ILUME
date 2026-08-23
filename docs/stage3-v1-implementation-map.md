# Stage 3 v1 implementation map

本表记录重构前调用链与 v1 替代位置，用于审计旧逻辑是否完整退出；它不定义独立于 ADR-0020 的科研合同。

| 旧调用链/能力 | v1 处理 | 实现位置 |
|---|---|---|
| `reference.yaml`、`il21/aux6` domain registry | 删除，改为 catalog fact + YAML task/group registry | `configs/v1/stage3/base.yaml`、`src/stage3/config.py`、`src/stage3/data.py` |
| fixed condition/phase、dense task assumptions | 删除，改为 task-local variable-width condition 与 sparse observation payload | `src/stage3/data.py` |
| Stage 2 migration rejection、旧 frozen entity/pair tables | 替换为公开 frozen Object v3 checkpoint loader 和内容寻址 object cache | `src/stage2/frozen.py`、`src/stage3/prepare.py` |
| AdaTT、IndependentTaskHead、FeatureGate、SelfGate、BatchNorm expert | 删除，改为 registry-driven dynamic HoME | `src/stage3/model.py` |
| late-solute special branch | 替换为通用 primary/partner slot 与 group-shared interaction | `src/stage3/data.py`、`src/stage3/model.py` |
| domain loss aggregation/backward | 替换为 task gradient、sample-weighted microbatch accumulation 与 composite step | `src/stage3/train.py` |
| domain isolation optimizer | 替换为显式 GLOBAL/GROUP/PRIVATE ownership 与 hierarchical PCGrad | `src/stage3/model.py`、`src/stage3/pcgrad.py` |
| early stopping、best/domain-best、rolling/last checkpoint | 删除，改为 interval full-epoch checkpoint 与固定 final epoch | `src/stage3/train.py` |
| 外部 matrix/fold launcher | 不恢复；唯一 train 入口使用 spawn worker 和显式设备槽调度独立 fold | `scripts/stage3/train.py` |
| 旧 valid/test best checkpoint loader | 替换为 full v1 checkpoint 严格加载与显式 epoch/task selector | `src/stage3/evaluate.py` |
| phase/adaptation-style staged expansion | 替换为 load scopes 与 adaptation scopes 分离的 plugin 初始化 | `src/stage3/train.py` |

三个公开入口保持为 `scripts/stage3/prepare.py`、`train.py`、`evaluate.py`。`train.py --fold` 接收一个或多个 fold，`--output` 始终是共同 root，实际 run contract 位于 `foldN/`；布尔 `--resume` 只恢复 identity 一致且 checkpoint/metrics/diagnostics 尾部严格对齐的 fold。旧 artifact/config/checkpoint 不提供兼容解析器。
