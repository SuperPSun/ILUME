# ADR-0037：SPMM WordPiece 字符上限

- 状态：Accepted
- 日期：2026-09-02
- 覆盖：ADR-0035 的 WordPiece 单词字符上限

## 背景

SPMM官方downstream将`"[CLS]" + SMILES`作为一个无空格字符串交给关闭basic tokenization的WordPiece tokenizer，并将`max_input_chars_per_word`设为250。ILUME thermal decomposition数据包含一个268字符的合法non-isomeric SMILES；加前缀后为273字符，因此官方字符门槛会在100-token截断之前把整个输入压成`[UNK]`，随后首token切片又移除外围`[CLS]`。

## 决定

1. SPMM baseline将`max_input_chars_per_word`固定为350，并将该值纳入model config、input contract、scientific identity和环境快照。
2. 其余tokenizer行为不变：固定官方vocab、`do_lower_case=False`、`do_basic_tokenize=False`、字面量`"[CLS]" + SMILES`、自动special tokens、`max_length=100` token级截断和最外层首token切片。
3. 不按字符裁剪SMILES，不删除或替换该row，不修改ILUME split。超过100 tokens的输入继续按ADR-0035审计并截断。

## 后果

- 该273字符输入可编码为66 tokens，不产生`[UNK]`且保留手工`[CLS]`，因此无需触发100-token截断。
- 350字符合同与既有250字符SPMM运行身份不同；旧checkpoint不得作为350字符合同的训练或evaluation产物复用。
