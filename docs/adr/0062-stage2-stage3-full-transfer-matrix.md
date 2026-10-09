# ADR-0062：Stage2→Stage3 全迁移矩阵消融

- 状态：历史合同，已由以下ADR取代： [ADR-0082](0082-home-mainline-and-core-ablations.md)
- 日期：2026-09-19
- 隔离范围：`configs/ablations/stage2_stage3_transfer.yaml`

> 2026-09-22：本文的冻结1024D 表示与仅训练MLP的下游合同已由
> [ADR-0068](0068-stage2-stage3-joint-downstream-adaptation.md) 修订为冻结Stage1 slots、
> 同步更新ObjectEncoder与MLP；Stage2 来源生成、矩阵和TG定义不变。

## 背景

基于 signature overlap 的任务配对无法回答任意一个 Stage 2 physics 任务是否改善任意
Stage 3 observation 任务。为直接测量这类迁移，本消融固定 Stage 1、数据划分、Stage 3
初始化和训练随机性，只改变冻结的 Stage 2 表示。

## 决定

1. 从同一个 v2 Stage 1 检查点和种子构造十个完整 Stage 2 Object v3 模型。基线
   不执行优化器步，直接导出初始化态 Stage1/ObjectEncoder；九个来源每次只调度
   一个任务目录任务的完整训练数据，只把共享 backbone、ObjectEncoder 和该来源预测头
   放入优化器，`lambda_teacher=0`，固定训练 10 轮并发布最后一轮编码器。其他
   任务预测头保持冻结且不得产生梯度。第 1 轮冻结 Stage 1 encoding backbone，第 2～10
   轮按 v2 配方解冻。
2. 十个编码器按现役 system-split Stage 3 准备产物的同一 ObjectKey 顺序生成
   1024D 表示 bank。bank 绑定 Stage 3 准备产物身份、编码器身份、产物
   SHA、object-list hash、embedding hash和shape；不重建 split、归一化或任务张量。
3. 每个表示分别训练 20 任务 × 5折的独立
   `input → 512 → 256 → 1` MLP。相同 `(task, fold)` 的初始化种子、初始化状态、
   原始随机排列、batch 边界、归一化和调度器完全一致，种子不得包含
   表示 variant。每个模型固定训练 10 轮，验证只记录，最终始终发布
   轮 10；不使用测试集。
4. 正式误差为验证原始 MAE。每折的迁移收益定义为
   `TG=(baseline_MAE-transfer_MAE)/baseline_MAE`。`TG_mean`是五个折-level TG 的算术
   平均，不能用平均 MAE 的比值替代；同时发布 median、逐折 TG、positive folds、
   `9×20` CSV和零中心对称色阶 SVG。任务集合由
   [ADR-0067](0067-stage3-twenty-task-catalog.md) 修订。
5. 检查点、编码器、表示、MLP和汇总使用独立 kind/身份。不得与正式
   Stage 2/3、Single-任务 MLP、历史实现或基准比较产物交叉恢复/加载。失败的 Stage 3
   单作业不做中途恢复；完整作业可由`--resume`按身份和hash跳过。

## 后果

- 正式运行需要新生成10个编码器、10份表示 bank和1000个MLP结果；现役
  Stage1 检查点、Stage2 准备产物数据/teacher 缓存和Stage3 准备产物可复用。
- 基线只训练一次，并由九个来源共享引用。汇总前必须验证1000个作业完整，以及同一
  任务/折的行/目标顺序、MLP初始状态和随机排列完全一致。
- 本实验只解释单一Stage2监督对冻结表示的迁移，不改变或替代正式Stage2/Stage3合同。
