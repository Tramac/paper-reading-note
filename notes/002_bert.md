# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[Paper](https://arxiv.org/pdf/1810.04805.pdf) | [Talk](https://www.bilibili.com/video/BV1PL411M7eQ)

## Part1. 标题&作者
Pre-training: 用来做预训练的

Bidirectional Transformers: 双向的transformer

Language Understanding: transformer主要用于机器翻译，bert将其扩展为语言理解任务，更加广泛

**BERT**：它是一个深的双向的transformer，是用来做预训练的，面向的是一般的语言理解任务

## Part2. 摘要
BERT名字来由：**B**idirectional **E**ncoder **R**epresentations from **T**ransformers
> ELMo, BERT, ERNIE均是美国著名儿童教育节目《芝麻街》中的人物

BERT与GPT和ELMo的区别：
- BERT: 使用未标注数据，联合左右的上下文信息，训练得到深的双向的表示；得益于此设计，针对不用的下游语言（问答、推理等）任务，只需要对预训练好的BERT模型另加一个额外的输出层即可
- GPT: 是一个单向过程，使用左边的上下文信息去预测未来，而BERT同时使用左侧和右侧信息（Bidirectional双向的由来）
- ELMo: 基于RNN架构，BERT是基于transformer；ELMo在用到一些下游任务上时通常需要对架构做一些调整，而BERT只需要改最上层即可

## Part3. 导言
sentence-level tasks & token-level tasks

## Part4. 结论

## Part5. 相关工作

## Part6. BERT模型

## Part7. 实验

## Part8. 评论
