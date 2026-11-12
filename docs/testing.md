# 测试与验证

`pytest -q`运行全部保留的测试，不默认排除慢测，不依赖并行测试插件。测试使用临时小数据，不启动正式prepare、训练或评估，不读写已有实验产物。

## 按改动选择验证

| 改动 | 必须执行 |
|---|---|
| 代码或测试 | 完整`pytest -q` |
| 仅文档 | Markdown链接与`git diff --check` |
| 仅配置 | 对应配置契约测试；同时修改实现则运行全量 |
| 影响CLI或模块导入 | 在上述检查外，检查受影响入口的`--help`及对应模块的compileall |

配置检查应覆盖实际修改的配置。例如Stage1正式配方由`tests/test_stage1.py::test_v4_residual_capacity_and_shared_compatibility`核对，当前消融配置由`tests/test_stage3_ablations.py::test_current_token_ablation_configs_preserve_base_and_retire_removed_experiments`核对；不能用无关配置测试替代。

## 2026-11-12精简覆盖对照

| 范围 | 精简前 | 保留或合并后 |
|---|---|---|
| Stage2→Stage3主链 | 历史MLP、Base、I1～I4和S1各跑完整链 | Base token/DGT与S1两条完整链；五模拟迁移、22实验＋两模拟、三阶段、最终重载和完整轮恢复均保留 |
| I1～I4 | 完整链之外另有小模型测试 | 在`test_stage3_token_attention.py`检查接入位置、梯度隔离、角色/mask、共享权重迁移、状态序列化和跨模式拒载 |
| 通用异常与诊断 | 随架构重复训练、损坏与恢复 | Base集中检查损坏拒载、split/训练seed分离、诊断开关不改变模型/优化器/调度器/RNG；S1另保留恢复一致性、类型分流及无样本分支零更新 |
| Stage2临时预算 | 每链10轮 | 2轮；256逻辑batch、257条边界样本、尾batch、任务耗尽和第二轮中断恢复不变；检查点名从预算推导 |
| Stage3临时预算 | Phase1两轮及最小后续阶段 | 保留；检查owner提前冻结、模拟隔离和零预算PRIVATE逐bit继承 |
| owner与单视图消融 | 在不同完整架构中重复 | no-global/no-group/no-private/no-stage2与no-smiles/no-graph各在Base参照下完整执行一次 |
| Stage1历史D-MPNN | shared/residual与DGT各跑teacher、导出、恢复、回归头链 | 完整链集中到DGT，保留电荷映射/重复观察、描述符policy、teacher完整性及回归头检查；历史D-MPNN保留配置、前向、状态重载、跨合同拒载 |
| 单任务实体MLP | 重复prepare，多任务多轮训练 | 复用Base准备产物；一次小数据训练、重载及验证；三类任务输入仍检查，正式10轮由配置契约检查 |
| 独有执行机制与基线 | DDP、跨world-size恢复、文件描述符、原子映射、D-MPNN基线 | 全部保留，不因使用历史小模型fixture删除 |

关键合同的落点：

- 身份、SHA与损坏拒载：`test_common.py`和Base主链；共享Stage1/特征/拓扑资源另有fixture结束时的SHA复核。
- 原始行覆盖：`test_stage2.py`的257条训练与round-robin检查，`test_stage3.py`的raw sampling和实际尾batch检查。
- 共享梯度与冻结：Stage2真实batch梯度测试、Stage3样本加权梯度测试、主链owner寿命检查及S1类型分流检查。
- 零预算继承与恢复：`test_three_phase_training_publishes_fixed_final_state`逐bit检查零更新PRIVATE继承；另保留主链、`test_three_phase_resume_histories_allow_zero_budget_branches`、`test_no_pcgrad_partial_resume_is_exact`及Stage1 Gloo跨world-size恢复。
- 原单位集成：Base主链两模拟的五折原单位均值检查、`test_five_fold_ensemble_averages_predictions_before_metrics`及Stage3公开推理集成检查。

共享fixture只提供磁盘上的Stage1来源、特征与拓扑。初始化使用局部固定RNG；每个用例独立创建模型、优化器、训练输出和待损坏文件，不使用硬链接。默认Base→S1全量通过；另按S1→Base逆序复跑，结果为`2 passed in 331.41s`，共享资源文件集合/SHA及完整轮恢复断言均通过。复核日志保存在本机`/tmp/ilume-tests-order.log`。

## 计时

同机、同一`ilume`环境，固定`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。计时命令为：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --durations=20
```

本轮基准采用起始版本`ea77db4`的测试快照，生产实现、科研配置和运行环境保持一致；先后串行运行，未同时执行其他验收测试。墙钟时间由`/usr/bin/time -p`记录，包含pytest启动和退出。

| 版本 | 墙钟时间 | pytest报告耗时 | 结果 |
|---|---:|---:|---|
| 精简前（起始版本`ea77db4`） | 1335.37秒 | 1330.45秒 | 407 passed, 1 skipped |
| 精简后 | 559.86秒 | 555.00秒 | 400 passed, 1 skipped |

墙钟耗时减少**58.1%**，约**2.39倍**速度，达到至少减半的目标。减少的7个完整用例来自5个重复/历史下游主链与2个历史Stage1训练链，其必要行为已按上表合并。唯一skip仍是缺少`chemprop`的D-MPNN基线环境规则，没有新增skip。

原始日志与计时保存在本机`/tmp/ilume-tests-before.log`、`/tmp/ilume-tests-after.log`及各自的`.time`文件；临时基准源码快照已移除。以下保留完整`--durations=20`明细，基准节点路径统一显示为`tests/`。

<details>
<summary>精简前的20项慢测</summary>

```text
280.59s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-l1-False]
277.76s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-group_l2-False]
152.92s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[mlp-l1-False]
99.13s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-group_l2_encoded-True]
89.60s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-group_l2_encoded-False]
88.75s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-all_l2-False]
87.58s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-group_private_l2-False]
33.67s call     tests/test_stage1.py::test_dgt_dataloader_batch256_with_low_file_descriptor_limit[True]
24.07s call     tests/test_stage1.py::test_dual_view_teacher_electronics_export_and_epoch_resume[dgt]
21.98s call     tests/test_stage1.py::test_dual_view_teacher_electronics_export_and_epoch_resume[residual_blocks]
16.19s call     tests/test_stage1.py::test_augmentation_leakage_and_worker_artifact_semantics
10.61s call     tests/test_hpo.py::test_all_static_rounds_and_manual_binding
10.22s call     tests/test_stage1.py::test_v4_residual_capacity_and_shared_compatibility
9.68s call     tests/test_stage1.py::test_dgt_dataloader_batch256_with_low_file_descriptor_limit[False]
8.18s call     tests/test_stage1.py::test_dgt_compiled_attention_gradients_and_rng
7.61s call     tests/test_stage1.py::test_two_rank_gloo_checkpoint_ownership_and_cross_world_resume[None]
6.52s call     tests/test_stage1.py::test_stage1_prefetch_prepares_next_shard_without_changing_batches
6.38s call     tests/test_stage1.py::test_two_rank_gloo_checkpoint_ownership_and_cross_world_resume[4]
6.07s call     tests/test_stage1.py::test_stage1_epoch_checkpoint_resume_and_attempt_log_preservation
5.63s call     tests/test_stage1.py::test_dual_view_teacher_electronics_export_and_epoch_resume[shared]
```

</details>

<details>
<summary>精简后的20项慢测</summary>

```text
267.90s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-l1-False]
72.00s call     tests/test_stage2.py::test_entity_home_prepare_train_transfer_three_phase_and_predictions[entity_token_attention_v1-group_l2_encoded-True]
32.97s call     tests/test_stage1.py::test_dgt_dataloader_batch256_with_low_file_descriptor_limit[True]
25.33s call     tests/test_stage1.py::test_dual_view_teacher_electronics_export_and_epoch_resume[dgt]
15.82s call     tests/test_stage1.py::test_augmentation_leakage_and_worker_artifact_semantics
10.32s call     tests/test_hpo.py::test_all_static_rounds_and_manual_binding
8.66s call     tests/test_stage1.py::test_two_rank_gloo_checkpoint_ownership_and_cross_world_resume[None]
8.50s call     tests/test_stage1.py::test_dgt_dataloader_batch256_with_low_file_descriptor_limit[False]
8.37s call     tests/test_stage1.py::test_two_rank_gloo_checkpoint_ownership_and_cross_world_resume[4]
8.35s call     tests/test_stage1.py::test_dgt_compiled_attention_gradients_and_rng
6.27s call     tests/test_stage1.py::test_stage1_prefetch_prepares_next_shard_without_changing_batches
5.43s call     tests/test_stage1.py::test_stage1_epoch_checkpoint_resume_and_attempt_log_preservation
4.76s call     tests/test_stage1.py::test_dgt_parallel_topology_with_low_file_descriptor_limit
3.38s call     tests/test_benchmark_catalog_v4.py::test_paper_summary_common_scales_complete_tasks_and_plots
3.17s call     tests/test_hpo.py::test_weakopt_frozen_recipes_limit_and_tamper_rejection
2.42s call     tests/test_stage1.py::test_v4_gradient_audit_reference_and_state_isolation[cuda-dgt]
2.39s call     tests/test_stage3.py::test_three_phase_test_reports_fold_and_pooled_gate_diagnostics
2.32s call     tests/test_architecture.py::test_stage2_registry_does_not_import_stage1_graph
2.23s call     tests/test_hpo.py::test_shared_owner_whitelist_sources_and_duplicate_budget
1.97s call     tests/test_hpo.py::test_weakopt_appends_new_configs_without_rebinding_folds
```

</details>

耗时受机器负载、磁盘和编译缓存影响，不作为科研模型速度结论。已有环境依赖的skip保持原规则，不新增skip满足提速目标。
