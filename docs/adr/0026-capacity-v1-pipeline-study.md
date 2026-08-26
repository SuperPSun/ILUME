# ADR-0026：ILUME Capacity v1 端到端容量研究

- 状态：Accepted
- 日期：2026-08-25
- 适用范围：`configs/experiments_v1/{stage1,stage2,stage3}` 与 `outputs/experiments_v1`

> 2026-08-26：Stage 2/3 late task-wise refinement、Capacity report schema v2 与 stitched validation 评分由 [ADR-0027](0027-late-taskwise-refinement.md) 修订；trial wave、搜索空间、人工决策与 seed 规则不变。

## 背景

本研究的目标是选择容量合适、下游迁移有效且训练稳定的 ILUME 主模型，并形成
S、Base、L、XL 四点 capacity trend；它不是严格 scaling-law 研究，也不隔离 Stage 1
encoder 的单独因果效应。Stage 2 ObjectEncoder 和 Stage 3 HoME 宽度均随 Stage 1
`d_model` 自然变化，因此结论只能表述为端到端 pipeline capacity trend。

本决定不取代 ADR-0013/0017、ADR-0019/0025 或 ADR-0020。现役 `configs/v1`、
既有 5-epoch Stage 1/2 和 100-epoch Stage 3 Base 继续保持原合同；capacity-v1 使用
独立配置、身份和输出，不恢复或覆盖现役 run。

## 决定

### Stage 1 与 Stage 2

1. Stage 1 固定四规模：
   - S：`d_model=384`、6 heads、SMILES/Fusion 6/6、FFN 1536、descriptor hidden 768；
   - Base：512、8 heads、8/8、2048、1024；
   - L：640、10 heads、10/10、2560、1280；
   - XL：768、12 heads、12/12、3072、1536。
2. 四规模均固定 graph depth 6、head dim 64、descriptor blocks 2，并保持现役数据、
   masking、loss、batch、optimizer、LR、dropout 和数值合同。每个 scale 从头训练
   15 epochs；所有 epoch checkpoint 保留，但 Stage 2 只消费 epoch 15。
3. 每个 scale 跑 Conservative、Default、Aggressive 三个 Stage 2 recipe，共 12 runs。
   `object_layers=2`、`object_ffn_dim=2*d_model`、dropout 0.1，固定训练 10 epochs，
   Stage 3 只消费 epoch 10 自动导出的 encoder。三档依次为：
   - Conservative：freeze 4，LR `3e-6/9e-6/3e-5`，teacher λ 0.30；
   - Default：freeze 2，LR `1e-5/3e-5/1e-4`，teacher λ 0.10；
   - Aggressive：freeze 0，LR `3e-5/9e-5/3e-4`，teacher λ 0.03。
4. 仅用 `configs/experiments_v1/stage1/base.yaml` prepare 一次 Stage 1 corpus，四规模共享
   `outputs/experiments_v1/stage1/base/prepare/artifacts`。Stage 2 在新的实验 artifact root
   发布一份 data artifact、每个 Stage 1 encoder 一份内容寻址 teacher cache，三个
   recipe 复用。

### Probe、HPO 与正式比较

5. 12 个 Stage 2 candidate 全部跑 Stage 3 folds 1/2 × 20 epochs。每 fold 的 proxy
   固定为 taskwise-refined stitched validation 的
   `macro_task_equal.normalized_mae.value`；candidate 再跨 fold 平均。每 scale 选择最低分
   recipe，精确并列时依次优先 Default、Conservative、Aggressive。
6. 四个 scale winner 之间的 anchor 由人工 Pareto 决策，证据限于 validation、曲线、
   参数量、显存、吞吐、wall time 与失败记录；必须先发布带理由的 anchor decision，
   HPO 入口才接受运行。test 不参与。
7. Anchor HPO 固定 40 attempted trials、Optuna seeded TPE、trial 0 Base、前 10 个
   startup trials、不 pruning。搜索 7 个变量：global/group/private expert 数、expert
   hidden ratio、dropout、LR 和 weight decay。两 trial 为同步 wave，每 trial folds 1/2，
   四张 GPU 各运行一个 fold；wave 完成后按 trial number 写回结果。
8. 每个失败 fold 只允许同配置再运行一次；第二次失败使 trial failed 并消耗预算。
   Top-5 加 Base 补 folds 3/4/5；Base 已在 Top-5 时不重复。五折均按 stitched validation
   评分，最终 recipe 由人工确认并冻结。
9. Stage 3 配置新增可选 `training.seed`。`null` 保持旧 `data.seed` RNG 和旧 training
   plan 形状；显式值只改变模型初始化、virtual sampler、task order 与 PCGrad RNG，
   并进入 training identity，不进入 prepared identity。robustness 固定 seeds
   `42/10042/20042/30042/40042`、folds 1/2 × 20 epochs，结果由人工复核；拒绝时停止，
   不自动换 recipe、加 seed 或重启 HPO。
10. 最终 recipe 原样迁移到四个 scale，expert 数固定、ratio 按各自 `d_model` 解析。
    四规模从头跑 5 folds × 50 epochs、seed 42，固定使用各 fold taskwise-refined artifact；
    20-epoch run 不恢复到 50 epochs。先按 validation/resource Pareto 冻结主 scale，再
    一次性评估四规模 refined 五折 test ensemble。test 不得反向改变 scale、recipe 或 artifact；本轮
    不追加 100-epoch 训练。

### 失败、身份与报告

11. 正式开始前冻结 clean commit、数据 identity、完整 YAML、trial manifest 和同类硬件。
    OOM、NaN、divergence 或不完整 validation 不触发 batch、LR、checkpointing 或 horizon
    自动调整。若要改变合同，必须形成新的预注册决定。
12. 报告包含 task/group 指标、完整曲线、fold/seed 波动、参数量、显存、吞吐、wall
    time、失败和人工理由。结论固定称为“经每 scale 独立 Stage 2 recipe 选择、共享
    Stage 3 recipe 后的端到端容量趋势”。

## 后果与取舍

- Final-only 将 Stage 2 candidate 从 24 减为 12，删除两层主观中间 checkpoint 选择，
  也无需增加任意 Stage 2 checkpoint 导出接口。
- 20-epoch probe/HPO 和 50-epoch formal 是不同 scheduler trajectory，不能 resume 或
  解释为同一训练的前后段；现役 100-epoch Base 也只作历史/健康参考。
- Anchor HPO 不为其他 scale 各自优化 Stage 3，因此四点回答的是受控、可负担的主模型
  选择问题，而不是每个 scale 的性能上界。
- 人工 Pareto、recipe 决策和 seed 复核保留科学判断，但每个判断的可见证据、时点和禁止
  使用的 test 信息均被冻结并写入 decision record。

## 备选方案

- 拒绝每 scale 两个 Stage 1 checkpoint 与 50-epoch 全 candidate probe：成本高且增加
  checkpoint 选择自由度。
- 拒绝 fixed-width Stage 3 adapter：它能更接近 encoder-only 因果比较，但改变本研究的
  端到端主模型目标和现役 HoME 接口。
- 拒绝以 20 epochs 作为正式终点或在选定 scale 后追加 100 epochs：前者证据不足，后者
  超出本轮已接受预算并形成第三条 scheduler 合同。

## 参考

- [ADR-0019：Stage 2 Catalog Object v3](0019-stage2-catalog-object-v3.md)
- [ADR-0020：Stage 3 sparse HoME/PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0021：Identity / Audit Contract v1](0021-identity-audit-contract-v1.md)
- [Capacity v1 操作手册](../capacity-v1-runbook.md)
