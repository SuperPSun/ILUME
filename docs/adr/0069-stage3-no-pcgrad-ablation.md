# ADR-0069：Stage 3 no-PCGrad 单变量消融

- 状态：Accepted
- 日期：2026-09-22
- 范围：现役 Base 的独立 Stage 3 三阶段消融

## 决定

1. `configs/ablations/stage3_no_pcgrad.yaml` 是现役
   `configs/v2/stage3/base.yaml` 的完整副本，唯一配置差异为
   `training.pcgrad_mode: "off"`。该字段只接受 `hierarchical` 和 `off`；省略时为
   `hierarchical`，legacy/Capacity 禁止 `off`。YAML 中的 `"off"` 必须带引号。
2. `off` 关闭 Phase 1 的组内 task GLOBAL/GROUP 投影及组间 GLOBAL 投影，关闭
   Phase 2 的组内 GROUP 投影。Phase 3 继续按原合同只训练 PRIVATE。
3. 逐 task 梯度仍由原 loss 与真实 microbatch 计算。每步只聚合当步有数据的 task；设
   当前组内任务集合为 `T_g`、task 权重为 `w_t`、当前有数据组的集合为 `G`、group
   权重为 `W_g`，关闭投影后的公式为：
   - GROUP：`sum(w_t * raw_GROUP_t) / sum(w_t)`。
   - GLOBAL：先在每组计算 `sum(w_t * raw_GLOBAL_t) / sum(w_t)`，再按
     `W_g / sum(W_g)` 聚合各组。
   - PRIVATE：`raw_PRIVATE_t * |T_g| * w_t / sum(w_t)`。
   Phase 2 只更新当前 GROUP 和仍可训练的 PRIVATE，不额外乘 group 权重。
4. 任务分组、模型结构、初始化种子、raw sampling、task 顺序、dropout RNG、owner
   LR/lifetime、冻结、loss、AdamW、scheduler、ownership clipping 和固定末轮 stitch
   全部沿用 Base。投影关闭后不消耗独立 PCGrad RNG，其 checkpoint 字段继续保留；
   其他 RNG 序列不受该开关影响。

## 身份与产物

- 复用 Base 的 20-task prepared artifact、Stage 2 Object encoder 与 normalization。
  不执行新的 prepare；新训练从相同初始化规则开始，不从 Base 的已训练状态分叉。
- 输出根为 `outputs/ablations/stage3_no_pcgrad`，训练、validation、test ensemble
  分别位于 `train`、`evaluate_valid`、`evaluate_test`，既有输出不覆盖。
- `resolved_training_plan.json` 的 phase recipe 与 `math.pcgrad` 均记录关闭状态，
  自动进入现有 training identity。默认模式的 plan、identity 和配置序列化保持原值；
  prepared identity、checkpoint format 与 final artifact kind 不升级。
- resume 严格匹配 training identity；evaluation 另校验配置与 checkpoint 的 PCGrad
  模式，禁止将 Base 和 no-PCGrad checkpoint 交叉使用。
- Phase 1/2 diagnostics 写 `pcgrad_applied=false`、`pcgrad_scope="off"`，保留
  raw task gradient norm、assembled owner norm 和裁剪统计。关闭模式不生成投影 pair
  diagnostics；gate diagnostics、prediction CSV 与 reporting schema 保持原合同。

## 比较与验收

- 固定 seed 42、system split、20-task catalog 与相同五折，比较最终 stitched state。
  主指标为每 task 五折 normalized MAE 均值的 task-equal macro；逐 task 报告
  `no-PCGrad - Base` 的配对差值，负值表示消融误差更低。
- test 按现有 evaluator 的 eligible task 集合执行五折 ensemble，独立报告结果，
  不据此改预算、选择 checkpoint 或调参。运行命令见 [README](../../README.md#stage-3-no-pcgrad-消融)。
- 必须检查 prepared 哈希、checkpoint/manifest 身份及对照 evaluation 完整性。仅有
  JSON 报告不能证明训练张量或 checkpoint 完整；缺失时先补齐对应产物。
- 临时小数据验收覆盖冲突梯度的原始加权聚合、raw 尾步、冻结 GLOBAL、默认路径、
  prepared/training identity 边界、跨模式 resume/evaluation 拒绝、零预算继承及
  Phase 1/2 epoch 边界恢复逐 bit 一致。正式五折训练由用户执行。

## 关联

- [ADR-0020：Stage 3 hierarchical PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0046：raw sampling 与 ownership clipping](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0048：三阶段 owner lifetime](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：owner recipe 与 identity](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0067：20-task catalog](0067-stage3-twenty-task-catalog.md)
