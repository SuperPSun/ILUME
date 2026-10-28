# ADR-0107：ChemBERTa-2 冻结表示替换 Stage1

- 状态：已接受
- 日期：2026-10-28
- 范围：C12/C13 独立消融；正式 Base 和既有产物不变。

## 决定

C12（`chemberta2_mtr`）使用 `DeepChem/ChemBERTa-77M-MTR`，固定 revision `66b895cab8adebea0cb59a8effa66b2020f204ca`；C13（`chemberta2_mlm`）使用 `DeepChem/ChemBERTa-77M-MLM`，固定 revision `ed8a5374f2024ec8da53760af91a33fb8f6a15ff`。三个实际消费文件 `config.json`、`tokenizer.json` 和 `pytorch_model.bin` 的官方 SHA-256 在实现中预注册；运行时不下载、不猜测版本、不回退模型。

两者直接替换整个 Stage1。只使用官方 RoBERTa 编码器最后一层首个 `[CLS]` 的384维隐藏状态，不使用 MTR/MLM预测头、原 Stage1、图编码器、融合层或 RDKit217。编码器始终冻结/eval，FP32/IEEE matmul/eager attention 提取，保存FP32实体缓存。官方预训练头按明确白名单剥离，编码器 strict 重载；仅允许验证过的历史 position/type buffer 转换。不随机补参数、不放宽 strict。

序列化 tokenizer.json 定义分词及特殊符号，当前 `[CLS]=12`、`[PAD]=0`，不能用模型配置中不同的默认编号重建分词器。右侧padding使用真实PAD编号和attention mask；保留模型自身position规则。最大总长度512，禁止截断、删分子或删行。固定 CLS 提取，不按验证结果挑 pooling。

每个实体输入为 `[CLS384, zero640, zero217]`，1241D不变；有序两槽＋角色one-hot＋mask仍为2490D。不存在可训练投影；HoME及partner、条件、残差路径和owner配方不变。补零区域始终为零。RDKit仅用于canonical化和形式电荷/角色检查，不计算或读取描述符。

## 数据与训练

配对 `optimized_3407` 的模型、data/3407 五折、任务、训练seed42、Stage3按42+fold派生、归一化、Raw Sampling、损失、预算及末轮发布。Stage2五项模拟共享预训练按ADR-0097执行；Stage3两模拟仍只更新PRIVATE，按ADR-0096隔离共享梯度。每个版本分别重做Stage2实体缓存、Stage2训练、Stage3实体缓存、五折训练和验证/测试；不复用Base迁移参数。总成本2次Stage2、10个Stage3折训练，无Stage1预训练。

Stage2只读取并校验Base准备产物中的实体标识、任务行、目标/条件张量、scaler和排除清单；不读取Base实体表示、Stage1统计或描述符文件。校验当前catalog/source身份后，保持原entity ID、行序、目标、条件、mask和归一化不变。新任务产物只更换表示族kind，独立物化ChemBERTa实体。Stage3复用原公开数据准备逻辑，并与Base的对象顺序、来源SHA、registry、五折归一化逐项核对。Base产物始终只读。

## 接口与身份

`common.frozen_encoder`提供显式注册的冻结编码源合同，Stage2/3配置新增可选`frozen_encoder`，缺省不进入原序列化配置，Base数值与身份不变。`preparation.reference_artifacts`定位配对Base数据，仅用于准备及覆盖校验。入口注册`ablations/entity_home/chemberta.py`，Stage包不导入消融实现，不伪装成`dual_view_v4`或恢复ObjectEncoder。

来源身份绑定模型、revision、三个文件SHA、tokenizer、CLS/FP32规则及无描述符补零策略。Stage2数据使用`ilume_stage2_external_frozen_data_v1`，周期检查点使用`ilume_stage2_external_frozen_home_checkpoint_v1`，完整final使用`ilume_stage2_external_frozen_home_final_v1`，内嵌编码器状态、原始配置/tokenizer字节和来源身份，允许离线严格重载。Stage3使用独立消融身份及kind；Base/MTR/MLM缓存、恢复和迁移不得交叉。实验注册名为`ilume_v4_chemberta2_mtr`、`ilume_v4_chemberta2_mlm`。

## 报告与边界

24实验macro normalized MAE、逐任务MAE/RMSE/R²、Base差值、六GROUP和规模分层沿用现役3407报告；两模拟独立，不进入24任务宏平均。记录实际参数量和训练成本。这是外部冻结表示的总体替换，包含预训练语料、监督、容量和显式描述符差异。MTR本身使用RDKit计算性质预训练，不能称完全不含RDKit知识。

原论文：[ChemBERTa-2](https://arxiv.org/abs/2209.01712)。官方检查点：[MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)、[MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)。

验收覆盖原生表示/补零、冻结、padding、角色/partner、超长拒绝、来源篡改、严格与离线重载、临时Stage2→Stage3→预测。正式资源准备、训练和评估只提供命令，由用户启动；保留原PyTorch环境，结果与完整权重按4090归档规则回传。运行见[消融手册](../v4-ablation-runbook.md#chemberta2-frozen-home)。
