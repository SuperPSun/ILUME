# ADR-0062：Stage2→Stage3 全迁移矩阵消融

- 状态：Accepted
- 日期：2026-09-19
- 隔离范围：`configs/ablations/stage2_stage3_transfer.yaml`

## 背景

基于 signature overlap 的任务配对无法回答任意一个 Stage 2 physics task 是否改善任意
Stage 3 observation task。为直接测量这类迁移，本消融固定 Stage 1、数据划分、Stage 3
初始化和训练随机性，只改变冻结的 Stage 2 representation。

## 决定

1. 从同一个 v2 Stage 1 checkpoint 和 seed 构造十个完整 Stage 2 Object v3 模型。baseline
   不执行 optimizer step，直接导出初始化态 Stage1/ObjectEncoder；九个 source 每次只调度
   一个 catalog task 的完整训练数据，只把共享 backbone、ObjectEncoder 和该 source head
   放入 optimizer，`lambda_teacher=0`，固定训练 10 epochs并发布最后一轮 encoder。其他
   task head 保持冻结且不得产生梯度。第 1 epoch冻结 Stage 1 encoding backbone，第 2～10
   epoch按 v2 recipe解冻。
2. 十个 encoder 按现役 system-split Stage 3 prepared artifact 的同一 ObjectKey 顺序生成
   1024D representation bank。bank 绑定 Stage 3 prepared identity、encoder identity、artifact
   SHA、object-list hash、embedding hash和shape；不重建 split、normalization或task tensor。
3. 每个 representation 分别训练 20 tasks × 5 folds 的独立
   `input → 512 → 256 → 1` MLP。相同 `(task, fold)` 的 initialization seed、初始化状态、
   raw permutation、batch boundary、normalization和scheduler完全一致，seed不得包含
   representation variant。每个模型固定训练 10 epochs，validation只记录，最终始终发布
   epoch 10；不使用 test。
4. 正式误差为 validation raw MAE。每 fold 的迁移收益定义为
   `TG=(baseline_MAE-transfer_MAE)/baseline_MAE`。`TG_mean`是五个 fold-level TG 的算术
   平均，不能用平均 MAE 的比值替代；同时发布 median、逐 fold TG、positive folds、
   `9×20` CSV和零中心对称色阶 SVG。任务集合由
   [ADR-0067](0067-stage3-twenty-task-catalog.md) 修订。
5. checkpoint、encoder、representation、MLP和summary使用独立 kind/identity。不得与正式
   Stage 2/3、Single-task MLP、legacy或benchmark artifact交叉resume/load。失败的 Stage 3
   单 job不做中途恢复；完整 job可由`--resume`按identity和hash跳过。

## 后果

- 正式运行需要新生成10个encoder、10份representation bank和1000个MLP结果；现役
  Stage1 checkpoint、Stage2 prepared data/teacher cache和Stage3 prepared artifact可复用。
- baseline只训练一次，并由九个source共享引用。汇总前必须验证1000个job完整，以及同一
  task/fold的row/target顺序、MLP初始状态和permutation完全一致。
- 本实验只解释单一Stage2监督对冻结表示的迁移，不改变或替代正式Stage2/Stage3合同。
