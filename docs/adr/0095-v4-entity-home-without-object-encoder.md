# ADR-0095：v4 HoME 直接消费冻结实体

- 状态：已接受
- 日期：2026-10-08
- 替代范围：ADR-0094 的 Stage2/3 和旧 v4 ObjectEncoder 合同；Stage1 合同不变。

## 决策

完全移除独立 ObjectEncoder 的 Transformer、Unary、Pair、gamma 及其加载支持。历史产物不删除，复现须 checkout 对应历史 Git 版本。

每个对象保存有序 `[2,1241]` 实体、三类形式电荷角色和有效位。单实体第一槽有效；离子对必须阳离子、阴离子顺序。无参数拼接两槽输入、两槽角色 one-hot 和 mask，得到2490D。GLOBAL 和 GROUP L1 专家独立、并行读取相同2490D输入，各自的第一 Linear 投影属于对应 owner。专家沿用 Linear→SiLU→Dropout→Linear→SiLU，输出1024D；没有独立共享编码模块。

GLOBAL 为专家混合输出；GROUP对有效实体learned1024均值与专家混合输出之和进行LayerNorm。partner 使用同一 GROUP L1 独立编码，随后经过原 PartnerInteraction；GLOBAL 保持 primary 路径。L2/PRIVATE/flat 任务门控/ConditionFiLM/tower 不变。容量增长集中于每个 L1 的第一 Linear；取消旧4,962,305参数的 ObjectEncoder 不意味着总参数下降。

## 任务和训练

Stage2仅五项模拟：density、heat_capacity、thermal_expansion、heat_of_vaporization 属于 thermophysical，transfer_organic 属于 solvation。保留权重1/.8/.8/1/.5、SmoothL1、训练统计、完整行覆盖、256逻辑/微批及10轮末轮。全部 HoME 参数用原 HoME LR1e-4。Stage1永久冻结/eval。

Stage3为24实验+两模拟辅助。原22任务 PRIVATE 配方不变；anodic_potential_limit、cathodic_potential_limit 新建 electrochemical GROUP，继承 phase_stability 的专家数3、宽度1.5、Phase1 LR2e-4/15轮和Phase2 LR1e-4/20轮；两任务22体系、small PRIVATE、仅温度。电极及扫描条件为常量，保留来源但不输入。焓展示名为汽化焓（Enthalpy of vaporization），任务目录标识不变，仅温度，phase 不输入。

Phase1实验联合训练GLOBAL/GROUP/PRIVATE，模拟PRIVATE冻结。Phase2 GLOBAL冻结，GROUP/PRIVATE按原owner预算训练；Phase3仅PRIVATE。两个模拟任务Phase2/3仅更新自身PRIVATE，不参与共享梯度、聚合分母和共享调度器预算。保留原始样本采样、按owner裁剪、共同锚点、独立分支、末轮拼接和严格恢复。

## 数据、工件和兼容

原数据、split、目标、mask及归一化不改。Stage3从独立冻结 Stage1 检查点/特征生成实体槽位；准备产物数据身份与HoME权重分离。完整 Stage2 final 是迁移源，迁移GLOBAL、thermophysical/solvation GROUP和两项模拟PRIVATE，不再导出stage2_encoder.pt。

显式 architecture_kind=entity_home_v1、representation_contract=entity_home_v4、slot_contract=ordered_entity_slots_v1。Stage2 final kind=ilume_stage2_entity_home_final_v4；Stage3 final kind=ilume_stage3_entity_home_three_phase_final_v4，format4、plan14/training18。严格绑定架构/任务/owner/Stage1来源/条件/数据SHA/张量 hash，不依据参数名猜测版本。旧Object工件明确拒载。

只保留configs/v4的Stage1/2/3 Base，删除v5和v4消融配置。新输出根outputs/v4/stage2/entity_home及outputs/v4/stage3/entity_home，旧输出只读。基线历史训练配方不扩展，二模拟/四模拟汇总按明确协议隔离。

## 验证与取舍

验证实体顺序、角色、mask、共享专家并行输入与梯度；五任务导出/重载；24+2三阶段owner边界与仅PRIVATE更新；逐张量迁移、恢复、SHA拒载；临时端到端及五模型集成。完整pytest、入口help、compileall、diff与Markdown链接检查。正式训练由用户运行。

下游参数（不含冻结Stage1）：五任务Stage2旧98,453,543、新110,765,116；同24+2 Stage3旧222,699,189、新254,916,110。现22+2旧195,893,412；新增GROUP/任务增加26,805,777。实际工件记录total/可训练/owner参数量。拒绝另建共享投影：虽省参数，却重建了被取消的编码瓶颈；不引入Transformer或新的混合物机制。
