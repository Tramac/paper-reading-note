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
NLP任务可分为两类: sentence-level tasks & token-level tasks
- sentence-level tasks: 句子层面的任务，主要用来建模句子之间的关系，比如句子情绪识别、两个句子之间的关系等
- token-level tasks: 词元层面的任务，比如实体命名识别（NER）等

在使用预训练模型做特征表示时，一般用两类策略：
- feature-based:对每一个下游任务，构造一个与该任务相关的神经网络（比如RNN结构），而预训练好的表示（可理解为预训练模型的输出）作为额外的特征和已有输入一起送给模型，代表作ELMo
- fine-tuning:在预训练模型用于下游任务时，只需要针对任务修改头部结构即可，模型预训练好的参数会在下游任务的数据上再进行微调一下（即所有的权重都会进行微调，对比feature-based来看由于输入的是特征相当于预训练模型的参数是固定的），代表作GPT

当前预训练模型技术的局限性：
- 标准的语言模型是一个单向的任务，导致选架构时有一些局限性，比如GPT使用的是一个从左到右的架构；但一些任务无论从左到右或者从右到左结构应该是一样的，比如句子情绪识别

BERT是如何突破以上局限性的？
- “masked language model”(MLM)：带掩码的语言模型，对句子随机掩盖部分词汇，然后预测这些词（完形填空），允许同时看左右的信息（与从左到右不同）
- “next sentence prediction”: 给定两个句子，判断这两个句子在原文中是否相邻，希望可以学习到句子层面的一些信息
> 我理解以上两种方式分别想针对token-level与sentence-level任务

## Part4. 结论
- BERT将之前基于单向结构的预训练模型拓展到双向结构，使得同样的一个预训练模型能够处理不同的NLP任务
- BERT是ELMo使用双向信息（但基于比较旧的RNN架构）与GPT使用新架构transformer（但只能处理单项信息）两者的结合

## Part5. 相关工作
- Unsupervised Feature-based Approaches: 代表作ELMo
- Unsupervised Fine-tuning Approaches: 代表作GPT
- Transfer Learning from Supervised Data: 在有标注的数据上，针对下游任务进行迁移学习

## Part6. BERT模型
BERT中的两个步骤：pre-training和fine-tuning

<img src="https://user-images.githubusercontent.com/22740819/145942921-7ed8cde2-d193-45bd-b169-dfed380de035.png" width=600>

- 首先pre-training阶段使用无标注数据进行训练得到预训练模型，然后fine-tuning阶段使用标注数据对模型进行微调
- fine-tuning时每一个下游任务都会有一个自己特定的bert模型，由预训练好的模型进行参数初始化（输出层除外），然后在自己任务的标注数据上进行微调，期间所有参数都会参与训练

## Part7. 实验

## Part8. 评论
