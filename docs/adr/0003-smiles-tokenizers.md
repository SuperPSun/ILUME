# ADR-0003：SMILES tokenizer 后端

- 状态：Accepted
- 日期：2026-07-23

> 2026-08-13：AIS 的单遍拟合与 `min_frequency` 精确语义由 [ADR-0014](0014-stage1-prepare-performance-and-corpus-v2.md) 补充；多后端协议与版本固定继续有效。

## 决定

统一 `SmilesTokenizer` 协议支持 AIS、APE、BPE 和 SPE。后端只在相同的入选训练 SMILES 上拟合，共享五个特殊 token、词表预算、最低频率和最大长度约束。

AIS 固定 `atomInSmiles==1.0.2`；`min_frequency=1` 保留所有出现过的 AIS token，更高值真正删除出现次数不足阈值的 token。SPE 固定 `SmilesPE==0.0.3`；APE 固定上游 commit `ff1b3cc00476a8d017d7d54e925681a04475d47f`。APE 使用该 commit 的官方 pre-tokenization，再执行有界、确定性的相邻 pair merge 学习，避免上游小语料训练循环在词表预算不可达时无法终止。BPE 使用 Hugging Face `tokenizers` 的 `BpeTrainer`。

## 理由

tokenizer 对照必须隔离训练集并具有可重现的实现版本。统一 artifact 结构让下游 encoder 不依赖后端细节；显式超长错误避免不同方法因静默截断获得不可比较输入。

## 后果

非 AIS 后端需要安装 `tokenizers` optional extra。tokenizer 后端或训练数据变化都要求重新准备 artifact。
