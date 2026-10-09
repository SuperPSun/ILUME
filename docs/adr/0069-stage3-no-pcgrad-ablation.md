# ADR-0069：Stage 3 no-PCGrad 单变量消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-22
- 范围：现役 Base 的独立 Stage 3 三阶段消融

## 决定

1. `configs/ablations/stage3_no_pcgrad.yaml` 是现役
   `configs/v2/stage3/base.yaml` 的完整副本，唯一配置差异为
   `training.pcgrad_mode: "off"`。该字段只接受 `hierarchical` 和 `off`；省略时为
   `hierarchical`，历史实现/Capacity 禁止 `off`。YAML 中的 `"off"` 必须带引号。
2. `off` 关闭 Phase 1 的组内任务 GLOBAL/GROUP 投影及组间 GLOBAL 投影，关闭
   Phase 2 的组内 GROUP 投影。Phase 3 继续按原合同只训练 PRIVATE。
3. 逐任务梯度仍由原 loss 与真实 microbatch 计算。每步只聚合当步有数据的任务；设
   当前组内任务集合为 `T_g`、任务权重为 `w_t`、当前有数据组的集合为 `G`、组
   权重为 `W_g`，关闭投影后的公式为：
   - GROUP：`sum(w_t * raw_GROUP_t) / sum(w_t)`。
   - GLOBAL：先在每组计算 `sum(w_t * raw_GLOBAL_t) / sum(w_t)`，再按
     `W_g / sum(W_g)` 聚合各组。
   - PRIVATE：`raw_PRIVATE_t * |T_g| * w_t / sum(w_t)`。
   Phase 2 只更新当前 GROUP 和仍可训练的 PRIVATE，不额外乘组权重。
4. 任务分组、模型结构、初始化种子、原始样本采样、任务顺序、dropout RNG、owner
   LR/存续期、冻结、loss、AdamW、调度器、按参数归属裁剪和固定末轮拼接
   全部沿用 Base。投影关闭后不消耗独立 PCGrad RNG，其检查点字段继续保留；
   其他 RNG 序列不受该开关影响。

## 身份与产物

- 复用 Base 的 20-任务准备产物、Stage 2 Object 编码器与归一化。
  不执行新的 prepare；新训练从相同初始化规则开始，不从 Base 的已训练状态分叉。
- 输出根为 `outputs/ablations/stage3_no_pcgrad`，训练、验证、测试集集成
  分别位于 `train`、`evaluate_valid`、`evaluate_test`，既有输出不覆盖。
- `resolved_training_plan.json` 的 phase 配方与 `math.pcgrad` 均记录关闭状态，
  自动进入现有训练身份。默认模式的 plan、身份和配置序列化保持原值；
  准备产物身份、检查点格式与 final 产物 kind 不升级。
- 恢复严格匹配训练身份；评估另校验配置与检查点的 PCGrad
  模式，禁止将 Base 和 no-PCGrad 检查点交叉使用。
- Phase 1/2 诊断写 `pcgrad_applied=false`、`pcgrad_scope="off"`，保留
  原始任务梯度 norm、assembled owner norm 和裁剪统计。关闭模式不生成投影 pair
  诊断；门控诊断、预测 CSV 与报告结构定义保持原合同。

## 比较与验收

- 固定种子 42、system split、二十任务目录与相同五折，比较最终拼接后的状态。
  主指标为每任务五折归一化MAE 均值的任务等权 macro；逐任务报告
  `no-PCGrad - Base` 的配对差值，负值表示消融误差更低。
- 测试集按现有 evaluator 的 eligible 任务集合执行五折集成，独立报告结果，
  不据此改预算、选择检查点或调参。历史运行命令须从对应Git版本的README读取；退役范围见 [ADR索引](README.md#冻结合同与历史)。
- 必须检查准备产物哈希、检查点/清单身份及对照评估完整性。仅有
  JSON 报告不能证明训练张量或检查点完整；缺失时先补齐对应产物。
- 临时小数据验收覆盖冲突梯度的原始加权聚合、原始尾步、冻结 GLOBAL、默认路径、
  准备产物/训练身份边界、跨模式恢复/评估拒绝、零预算继承及
  Phase 1/2 轮边界恢复逐 bit 一致。正式五折训练由用户执行。

## 关联

- [ADR-0020：Stage3分层PCGrad](0020-stage3-v1-sparse-home-pcgrad.md)
- [ADR-0046：原始样本采样与按参数归属裁剪](0046-stage3-ownership-clipping-raw-sampling.md)
- [ADR-0048：三阶段 owner 存续期](0048-stage3-owner-lifetime-three-phase-training.md)
- [ADR-0050：owner 配方与身份](0050-stage3-task-specific-owner-budget-and-private-capacity.md)
- [ADR-0067：二十任务目录](0067-stage3-twenty-task-catalog.md)
