# ADR-0115：PEOE_VSA 的 Gasteiger 有效性联动屏蔽

## 状态与范围

已接受（2026-11-06）。在ADR-0113的Phi适用性和八项环计数log1p基础上，仅为现役Stage1 Base补充PEOE电荷来源校验。维度继续为211/1235/2478；六项排除、encoder、loss定义与系数、分子归一化、角色2/2/1、采样和训练预算不变。不采纳197维人工候选，不恢复独立描述符筛选工具。

## 原因

PEOE_VSA1～14按Gasteiger原子电荷区间汇总近似原子表面积。RDKit2026.09.1对某些环境缺少参数，例如PF6−中的六配位P。默认计算可能产生非有限电荷，但PEOE_VSA仍返回有限数值，单独检查描述符finite会把这些值当作有效监督。

PF6−的总形式电荷仍为−1，该问题只表示当前Gasteiger参数模型未给出可靠原子部分电荷，不构成删除分子的理由。不得把形式电荷或0强填为该方法的原子电荷，也不改原子杂化以绕过参数缺失。

## 决定

- 新增`descriptor.value_policy: phi_mask_ring_log1p_gasteiger_v2`，现役`configs/v4/stage1/base.yaml`显式启用。保留`raw`默认及ADR-0113的`phi_mask_ring_log1p_v1`原行为，旧配置不隐式升级。
- 在统一`mask_descriptor_values`内复制分子，执行`ComputeGasteigerCharges(nIter=12, throwOnParamFailure=True)`。参数/计算失败（ValueError或RuntimeError），或任何原子的`_GasteigerCharge`/`_GasteigerHCharge`非有限，均认定电荷来源无效。检查副本，不污染原分子上的其他描述符缓存。
- 来源无效时，仅把该分子的全部14项PEOE_VSA设为NaN。整组屏蔽是基于共同依赖的保守规则，不表示每个有限返回值都已分别证明错误；不按分子名称、元素或角色硬编码失败集合。
- 成功时保持原PEOE数值逐项不变，不重新计算或替换描述符结果。其他描述符继续原有Phi、环计数及逐项finite规则；本次不额外改变四项电荷极值或BCUT的定义/有效性规则。
- prepare在scaler拟合前应用该mask；NaN不进入train均值/标准差。样本构造使用同一函数，标准化后填0，Stage1描述符MSE由`descriptor_valid`排除这些维度。其余有效目标照常训练，分子不退出语料。
- Stage2/3实体构造也经同一schema/样本函数应用相同规则，下游显式PEOE输入在失败处标准化填0；不新增mask输入维度、不反传Stage1。独立原子电荷MOL2标签和其他辅助监督不受影响。

## 身份与运行

沿用schema format3，内嵌policy名称；加载支持旧v1和新v2，未知policy仍拒绝。新policy进入既有corpus/feature/training身份及来源SHA，因此相同211维也不得交叉恢复；不升级Stage checkpoint格式，不改SHA以兼容旧缓存。

新Base须在新目录重新prepare Stage1，准备匹配corpus身份的teacher/电荷sidecar并从头训练，再重新prepare/train Stage2和prepare/train/evaluate Stage3。当前teacher/sidecar绑定corpus身份，尽管其原始数学目标未改变，也不能重标旧身份复用。原始语料、split authority和Uni-Mol源权重保留；历史产物继续只读。实现不移动、删除、覆盖旧输出，不执行正式prepare、缓存生成或训练。

## 验证

- 真实PF6−的原PEOE有限但新policy整组无效；乙醇、BF4−和四配位膦鎓的有效PEOE保持原值，其他描述符与旧policy逐项一致。
- 严格参数检查也拒绝可能返回有限默认值的未知环境；模拟无异常但原子电荷或隐式氢电荷非有限，验证仍整组屏蔽且不修改输入数组/分子。
- scaler只拟合其余有效样本，失败位置填0/mask；描述符loss在这些位置的梯度为0，其他有效位置仍有梯度。
- 覆盖schema format2/3与新旧policy round-trip、训练来源不匹配拒绝、完整轮训练/恢复/导出，以及旧默认路径回归；不以本修正宣称下游性能已提升。
