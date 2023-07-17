

# LLaMA: Open and Efficient Foundation Language Models

[paper](https://arxiv.org/pdf/2302.13971v1.pdf) | [code](https://github.com/facebookresearch/llama)

## 作者&标题

开放的、高效的基础语言模型。

作者团队来自Meta AI，一作 Hugo Touvron 同时也是 DeiT、DINO 等工作的作者。

## 摘要

本文提出了一个大语言模型 - LLaMA，参数量由 7B 到 65B（70亿～650亿），模型在数万亿的 tokens 上进行训练，展现出即便只使用公开数据集（不包含专有的、无法访问的数据集）也可以得到很好的效果。

LLaMA-13B 在大多数 benchmark 上都优于 GPT-3 (175B)，LLaMA65B 与最好的模型 Chinchilla-70B 和 PaLM-540B 具有同等的竞争力。

## 1.引言

在大量文本语料库上训练的大语言模型（LLM）已经显示出它们的涌现能力。大模型的一个假设是：更多的参数将带来更好的性能。但是，当计算资源有限时，通常是由在更多数据上训练较小模型来实现最佳性能的。

数据集大小和模型大小的选择不仅仅要考虑训练阶段的计算资源，更要关注推理时的预算。假设给定目标性能水平，首选模型不是训练速度最快的模型，而应该是推理速度最快的模型，尽管更大型模型更加容易达到指定性能水平，但较小的模型最终推理时会更便宜。 例如，尽管有工作建议在 200B tokens上训练 10B 模型，但我们发现即使在 1T tokens 之后，7B 模型的性能仍在继续提高。

基于以上的考虑，该工作聚焦于：

- 在训练语料固定的前提下，训练出一系列不同大小（7B、13B、33B、65B）的语言模型；
- LLaMA-13B 在大多数 benchmark 上都优于 GPT-3 (175B)，LLaMA65B 与最好的模型 Chinchilla-70B 和 PaLM-540B 具有同等的竞争力，由于其参数量少的优势，使得再单个 GPU 上也可以跑起来；
- 与 Chinchilla，PaLM 和 GPT-3 不同，LLaMA 训练只使用了公开数据集，并且是开源的；

最后，文章对 transformer 结构的修改、训练方法、效果对比做了一个介绍。

## 2.方法

LLaMA 的训练方法与 GPT-3 类似，并遵循了 Chinchilla 方法中的模型尺度规则，简而言之是使用标准的优化方法在大量的文本数据上训练得到一个大的 transformer。

### 2.1预训练数据

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/2104618d-c2f1-4764-8af5-8e37159398ec)

预训练数据混合了上面的多个来源，涵盖了多个领域，这些数据均是公开的。上表中给出了每个数据集的采样比例，每个数据集被训练的 epoch 数量（怎么 统计的？）以及所占硬盘的大小。

**English CommonCrawl**

整合了 2017-2020 的 CommonCrawl 数据，并做了以下处理：

- 进行 line level 的数据去重；
- 使用 fastText 去除非英文的界面；
- 用 ngram 语言模型去除低质量的内容；
- 使用线性模型进行references in Wikipedia v.s. randomly sampled pages，丢弃不在references中的页面；

**C4**

实验过程中发现，使用不同预处理的 CommonCrawl 数据可以提高性能，因此加入了 C4 数据集，并做了以下处理：

- 去重；
- 语言识别（language identification steps）：和CCNet进行高质量过滤的主要不同是，依赖启发式的方法，比如标点符号、token的个数、语句所在的webpage；

**Github**

使用了 Google BigQuery 上可用的公开 GitHub 数据，并做以下处理：

- 只保留了支持 Apache、BSD 以及 MIT 协议的 projects；
- 基于行长和字母数字的比例进行高质量代码筛选；
- 删除带有正则表达式的样板，比如头文件；
- file level使用精确匹配进行文件的去重；

**Wikipedia**

使用了维基百科 2022年6月 ～ 8月期间的数据，覆盖了 20 种语言，并删除超链接、注释等其他样板（boilerplate）。

**Gutenberg and Books3**

使用了两本书的预料：

- Gutenberg Project：包含开放域的数据；
-  Books3 section of ThePile：book level的去除，去掉了90%的重复样本；

**ArXiv**

使用了 arXiv 的数据，增强其科学数据的比例，并去除了文章的第一部分和参考文献，去除.tex file、各种注解。

**Stack Exchange**

问答网站的高质量问答数据，包含从计算机、化学等不同领域，并做以下处理：

- 保留来自最大28个网站的数据，去除掉HTML；
- 进行回答分数的排序；

### Tokenizer

使用 BPE（bytepair encoding）方法对上述语料进行 tokenizer，共有 1.4T tokens，除了 Wikipedia and Books domains训练了2个epoch之外，其他数据训练基本只过了一个 epoch。

### 2.2模型

和大多数的大模型一样，LLaMA 也是基于 transformer 结构，并且引入了后续的一些在其他大模型上的改进，以下是与原始架构的主要区别，以及我们在哪里找到了这种变化的灵感（括号中）：

**Pre-normalization [GPT3]**

为了提高训练稳定性，对于每一个 transformer sub-layer，对其输入做归一化，而不是对输出做，归一化方法使用的是 RMSNorm。

**SwiGLU activation function [PaLM]**

使用 PaLM 中的 SwiGLU 替代 ReLU，并且维度使用 2/3 4d，用于提高性能。

**Rotary Embeddings [GPTNeo]**

移除了绝对位置编码，取而代之的是 rotary positional embeddings（RoPE）。



不同尺度模型的具体参数如下：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/54c811e8-6caf-4613-9bf2-5352254ffd4a)

### 2.3 优化方法

优化方法为 AdamW，超参数：

- β1 = 0.9，β2 = 0.95
- cosine learning rate，终止学习率为最大学习率的 10%
- Weight decay = 0.1
- gradient clipping = 1.0
- Warmup steps = 2000

### 2.4 其他补充

为了提高训练速度，做了以下几个优化：

- 使用 causal multi-head attention 来减少内存使用和运行时间，该方法不存储 attention 的权重，不计算 key/query scores， 此实现可在 [xformers](https://github.com/facebookresearch/xformers) 库中找到；
- 为了进一步提高训练效率，减少了反向传播过程中重新计算的激活操作，更准确地说，保存了计算成本高昂的激活层，例如线性层的输出。 这是通过手动实现 Transformer 层的反向传播函数来实现的，而不是依赖 PyTorch autograd；
- 使用模型、序列并行进一步减少模型的内存使用量；
- 此外，尽可能地重叠激活计算和 GPU 之间通过网络进行的通信。

**LLaMA-65B 使用 2048 张 80G 显存的 A100，在 1.4T tokens 上训练需 21 天。**

## 3 结果

LLaMA 在 zero-shot 和 few-shot 任务上共评测了 20 个 benchmark。

- Zero-shot: 提供任务的文字描述（指令？）和测试示例， 模型生成答案或者对建议的答案进行排名；
- Few-shot：我们提供了一些任务示例（1 到 64 之间）和一个测试示例，该模型将此文本作为输入并生成答案或对不同的选项进行排名；

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/913319c9-298f-4c70-b8c7-d898d71a81c1)

其他 benchmark 上的结果对比详见论文。

## 4  Instruction Finetuning 指令微调

虽然 LLaMA-65B 已经可以遵循一些基础的指令了，但是发现少量的 finetune 在 MMLU 数据集上有明显的性能提升，

何谓指令指令精调（我的疑惑是它为什么叫 instruction finetuning 而不是直接叫 finetune，两者的区别是什么）？

**finetuning on instructions data** 或许比较好的解释了两者的区别和联系，由于语言大模型中微调数据一般都是带指令的，所以叫做指令微调，比如下面一组微调数据：

<img width="328" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/cd6ade17-9787-4cc2-8dd6-cdecb8adc6fa">

对传统的 finetune 而言，只需要 input 和 output 就可以了，而语言大模型中的微调通常会给出一个指令，比如上面的 instruction 信息。

指令微调数据通常不需要太多，大概使用1000条数据就可以取得不错的效果，太多反而有可能带偏基座模型，之所以少量数据可以驱动指令学习的两个原因：

- 数据质量高（一定要高才有效）；
- 基座模型强；

没指令微调和有指令微调的效果对比：

<img width="344" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/1a2c56a8-1578-49f9-8a7e-97f6451f7e65">

## 5 Bias, Toxicity and Misinformation

三者该如何理解？

LLM 已经被证明可以重现和放大训练数据中存在的 bias，并生成有害的或令人反感的内容。因为 LLaMA 的训练数据很大一部分来自于网络数据，因此可以确定的是 LLaMA 模型有可能生成此类内容，并在相关的 benchmark 上验证了 LLaMA-65B 确实存在此问题。几个 benchmark 如下：

- RealToxicityPrompts：有害的提示，比如侮辱、仇恨言论或威胁
- CrowS-Pairs：“众包刻板印象对”数据集,以衡量9种类型的偏见的程度-种族/肤色,性别/性别认同或表达,性取向等
- WinoGender：判断一个模型的共解析性能是否受到代词性别的影响
- TruthfulQA：用来衡量一个模型的真实性

## 6 Carbon footprint

很有意思的一个章节，以二氧化碳排放量的角度阐述了语言大模型训练一次需要效果多"耗资源"，

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/6795d680-d01c-4a62-9cdf-69b69acc2f24)

相比其它方法已经很节省资源了，但是 emm。。。

## 7 相关工作

略

## 8 结论

该文章开源了一系列语言模型，其中 LLaMA-13B 比 GPT-3 模型大小小10倍，但是效果优于 GPT-3，LLaMA-65B 比 Chinchilla-70B 和 PaLM-540B 更强。

和之前的一些工作不同，LLaMA 证明了，即便是只在公开的数据集上训练也可以得到 SOTA 效果。

并且想LLaMA开源，致力于促进语言大模型的发展。

未来将开源出更大规模的模型。