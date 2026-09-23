# ADR-0071：退役等行数 Stage2→Stage3 transfer

- 状态：Accepted
- 日期：2026-09-23
- 范围：隔离的 Stage2→Stage3 transfer 消融
- 修订：[ADR-0065](0065-stage2-stage3-balanced-transfer-matrix.md)；保留
  [ADR-0062](0062-stage2-stage3-full-transfer-matrix.md) 和
  [ADR-0068](0068-stage2-stage3-joint-downstream-adaptation.md) 的 full-data 合同

## 决定

等行数 balanced transfer 不再作为现役实验。删除其 YAML、抽样计划、行选择接口和
专属身份路径；配置解析拒绝 `stage2.sampling_mode`，旧 balanced manifest 与
representation 不能用于 resume、prepare、train 或 summarize。

Full-data transfer 仍使用全部 source train rows、原有 seed、身份、联合更新
ObjectEncoder + MLP 的数值路径及产物格式。此退役不改变正式 Stage2/Stage3、
Single-task MLP 或 benchmark 合同。

## 产物边界

既有 balanced 输出及 ADR-0065 的实验记录保留为只读历史，不迁移、不覆盖、不删除。
需要运行迁移矩阵时只使用 full-data YAML 和隔离的新输出目录。当前代码不提供
balanced 复现或兼容加载入口；若未来重新研究等预算对照，应另立合同和身份。
