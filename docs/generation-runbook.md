# GFN 离子液体逆向设计（已退役）

2026-11-11，按用户决定淘汰 GFlowNet。`generation/` 的生成实现、`scripts/generation/` 的四个入口、`configs/generation/` 的两份配置和专属生成测试已移除，当前 checkout 不再提供 GFN 资源准备、训练、采样或评估功能。

公开冻结推理接口 [stage3/inference.py](../src/stage3/inference.py) 及其五折集成、单位换算、输入检查和来源验证保留；对应测试归入 Stage3 测试。Stage1–3 的训练与预测行为未改变。

[ADR-0105](adr/0105-universal-inverse-design.md) 和 [ADR-0106](adr/0106-dedicated-inverse-design.md) 保留为历史设计。历史配置、缓存及产物不作为现役运行入口；复现旧方案须切换到对应 Git 版本。清理历史 YAML、缓存和产物仍按仓库规则单独确认。
