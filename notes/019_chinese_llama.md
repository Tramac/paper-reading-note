# Chinese-LLaMA

[paper](https://arxiv.org/pdf/2304.08177.pdf) | [code](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

## 1 题目&作者

针对中文 LLaMa 和 Alpaca 的高效且有效的文本编码器。

作者来自崔一鸣团队，作者[主页](https://ymcui.com/)。

## 2 摘要

随着 ChatGPT 和 GPT-4 的出现，大模型（LLMs）极大的改变了 NLP 领域的研究，并在通用人工智能（AGI）方面取得了可喜的进展。尽管如此，LLMs 高昂的训练及部署成本对学术研究构成了巨大阻碍。虽然社区已经开源了一些语言大模型，比如 LLaMA，但这些模型主要是针对英文语料，对于其它语言有一定的局限性。

本文提出了一种用来增强 LLaMA 中文理解能力，并且可以遵循一定指令的方法。在原生 LLaMA 的词表至上又扩展了 20000 个中文 tokens，从而提高了它的编码效率和中文理解能力。并且基于中文语料进一步做了二次预训练，然后使用中文指令数据集对模型进行微调，显著的增强了模型理解和遵循指令的能力。

实验结果表明，新提出的模型显着提高了原始 LLaMA 理解和生成中文内容的能力。 此外，在 C-Eval 数据集上的结果显示，本文提出的模型与更大模型相比也同样具有竞争力。

## 3 引言

**LLMs 面临的主要问题**

- 开源不友好：比如 GPT 系列不开源，限制了对模型的访问，不利于在其基础之上进一步研究；
- 高昂的训练、部署成本；

但是，也有诸如 LLaMA、Alpaca 等优秀的开源工作（Meta 内心OS：既然我打不败你 <GPT>，那我就发动全世界的人一起打败你！）。

**LLaMA、Alpaca的主要问题**

- 对中文的支持不友好：词表仅包含几百个中文 tokens，极大的限制了编码效率和中文理解能力；

**本文的主要贡献**

- 在原始 LLaMA 词表至上，扩充了 20000 个中文 tokens，提高了编解码效率以及中文理解能力；
- 使用 LoRA 方法来训练和微调，保证计算成本不会太高；
- 在指令遵循和其它 NLP 任务上的评估结果显示，本文提出的模型在中文理解任务上对原始 LLaMA 由实质的改进；
- 本文的范式可以迁移到其它的语言至上；

## 4 方法

#### 4.1 LLaMA

详细见：https://github.com/Tramac/paper-reading-note/blob/main/notes/018_llama.md

#### 4.2 Chinese Vocabulary Extension 中文词表扩充

LLaMA 的训练数据共涵盖了 1.4T 个 tokens，大部分来自英文语料和少部分的拉丁语、西里尔语。因此，LLaMA 的多语言和跨语言能力主要体现在欧洲语言中，其中文理解能力比较有限。

为了增强 LLaMA 的中文理解能生成能力，本文提出使用中文语料对 LLaMA 继续预训练（continue pre-training）。但是，直接使用中文语料进行预训练会有一些问题：

- 原始 LLaMA 词表不足一千个中文 tokens，对于中文文本的编码很有限，虽然通过 tokenizer 将 uft-8 字符转为 bytes，但是这种策略显然增加了序列长度，降低了中文文本的编解码效率，因为一个汉字被分割成了3-4个字节（一个 token 变 4 个 token）；
- byte token 并不是专门为表示汉字设计的。在其它语言中的 utf-8 编码也是用 byte token 来表示，所以对于中文字符语义的理解很有挑战；

为了解决以上问题，本文提出针对中文扩充原始 LLaMA 词表，并在扩充后的词表下对模型进行预训练。词表扩充过程：

- 为了增强 tokenizer 对中文的支持，本文使用 SentencePiece 在中文语料上训练得到了一个大小为 20000 的词表；
- 然后，将得到的词表与原始词表进行 merge，产出一个大小为 49953 的中文 LLaMA 词表；
- 为了使 LLaMA 模型适配新的词表，将模型的 word embedding 层和 head 部分从 shape VxH resize 到 V'xH，其中V=32000，V'=49953，新增的行附加到原始 embedding 层的末尾，确保原始词表中的 tokens 不受影响；

经实验表明，**使用 Chinese LLaMA tokenizer 产生的 tokens 数量大约是使用原始 LLaMA tokenizer 产生的 tokens 数量的一半**，显著的缩短了编码长度。

示例：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/3111310a-ec89-44f1-b4f4-615d650fca97)

**在固定文本长度的情况下，新的模型可以容纳大约两倍的信息，并且生成速度比原始 LLaMA tokenizer 快两倍**。这保证了所提出的中文 LLaMA 中文理解能力的有效性。

#### 4.3 使用LoRA 进行微调

训练 LLMs 时，如果更新其全部参数的成本是非常大的，对于大多数实验室或公司来讲，在时间和成本上都不能接受。LoRA 是一种有效的训练方法，它先保持预训练的模型权重，同时引入可训练的低秩分解矩阵，即冻住预训练模型的权重，每一层只学习一个低秩矩阵。这种方法大大减少了可学习参数，解决计算资源。

具体来说，对于一个维度为 d x k 的线性层 W0，其中 k 为输入维度，d 为输出维度，LoRA 添加了两个低秩矩阵，分别为 B（维度 d x r）和 A（维度为 r x k），其中 r 是一个提前指定的秩的大小。给定输入 x，输出为：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/1f6b77d1-eb01-4bbd-a211-08120830341b)

训练期间，W0 是冻住的，不进行梯度更新，只有 B 和 A 是更新的，如果 r 远远小于 d 和 k，那么内存消耗将会显著减少（其实就是 d x k 降低为 r x (d + k)）。

为了能在预算紧张的情况下实现高效的训练，本文在中文 LLaMA 和 Alpaca 的预训练和微调阶段均采用 LoRA 的方式。**主要是将 LoRA 用在 attention 模块和 MLP 层**。

#### 4.4 预训练目标

预训练中文 LLaMA 时采用了标准的因果语言模型（Casual Language Modeling, CLM）任务。给定一个输入的 token 序列 x=(x0, x1, x2, ...)，以自回归的方式预测其下一个 token xi，数学上，其目标为最小化以下似然函数：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/3bc4dfef-d75e-4b89-a68c-6dd40ccadb72)

其中 Θ 表示模型参数，D_PT 为预训练数据集，xi 为预测的 token，x1 ... x_i-1 为输入的 序列。

#### 4.5 有监督微调与中文 Alpaca

预训练得到的模型很难遵循用户指令，经常生成非预期的内容，这是因为预训练的目标函数是预测生成下一个 token，并不是"遵循指令或回答问题"。

为了使模型的行为与用户的意图保持一致，一种方式是可以通过进一步微调来使得模型可以遵循指令。Stanford Alpaca 是一个基于 LLaMA 进行指令微调的模型，它在由 Self-Instruct 技术生成的 52k 条的指令数据上训练生成。本文采用 Stanford Alpaca 的方法，在中文指令数据上对 Chinese LLaMA 进行 **self-instructed fine-tuning**，得到 Chinese Alpaca。

Chinese Alpaca 在一个组合的指令数据集上训练得到，每条数据包含一条指令和一个输出。微调时的任务与上面的 CLM 任务有些相似：模型输入指令提示，以自回归的方式生成输出。一般的，指令会以某种提示模板（**prompt** template）的方式送给模型。本文所使用的模板如下：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/db335cf9-b2fd-416f-90e9-7577c9e693a6)

损失函数：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/636879c3-d949-456c-8881-83ad918a7391)

式中，Θ 为模型参数，D_SFT 表示微调的数据集，x=(x0, ...) 为输入序列。

本文的方法与 Stanford Alpaca 方法主要的区别是：本文只对训练数据中没有显式的 "input" 的数据使用 prompt template，而 Stanford Alpaca 对含有 “input” 和不含 “input” 的数据分别采用两种 prompt template。如果一条数据包含一个非空的 "input"，那么将会把指令 “instruction” 和 "input" 用 \n 字符连接起来形成一条新的 "instruction"。所以，在 Chinese Alpaca 模型里有一个新的 padding token，所以词表大小为**49954**。

## 5 实验

#### 5.1 预训练实验

Chinese LLaMA 的 7B 与 13B 在预训练是均是 fp16 精度，而 33B 模型为了使训练更高效，采用了 8-bit 精度。

对于 Chinese LLaMA-7B，采用了 two-stage 的预训练方法。在阶段一，首先冻住 transformer encoder，在中文数据集上只训练 embedding 层，为了最大限度的减少对原始模型的干扰；在阶段二，将 LoRA 加入到 attention 中，同时训练 embedding 和 LM heads 以及新增的 LoRA 参数。需要注意的是，**two-stage 的训练方式只在 7B 模型中使用，对 13B 与 33B 模型来说作用不大**。

另外，本文提供了基础版本与 plus 版本，其中基础版本实在 20GB 的通用中文语料上（来自 Chinese BERT-wwm、MacBERT、LERT 等的训练数据）预训练得到的，而 pluse 版本是在 120GB 的中文语料上（新增了来自 CommonCrawl 数据和百科全书数据）训练得到的。

模型在 A40 GPU（48G 显存）上训练了一个 epoch，根据模型大小的不同，最多使用 48 块 GPU，LoRA 的实现采用了 PEFT 开源库。同时使用了 DeepSpeed 来优化内存使用，

训练参数：

- A40 GPU（48G 显存），最多使用 48 块卡；

- 训练一个 epoch；

- LoRA 采用 PEFT 开源库的实现；

- AdamW 优化方法；

- 2e-4 初始学习率，5% 的 warm-up，cosion scheduler 衰减方式；

- 梯度裁剪，最大值为 1.0；

  ![image](https://github.com/Tramac/paper-reading-note/assets/22740819/a3c1403a-194c-4d4f-873c-6725a988157a)

#### 5.2 指令微调实验

在 基础版本的 Chinese LLaMA 基础之上，使用 LoRA（所有的线性层上）进一步微调，共使用了 2M~3M 的指令数据，组合如下：

- 翻译数据 550K 条；
- pCLUE 250K 条；
- Stanford Alpaca 50K+50K 条;
- crawled SFT 数据

对于 Plus 版本，指令数据扩展到 4M ~ 4.3M，主要是整合了 STEM（科学、技术、工程和数学）数据以及物理、化学、生物学、医学和地球科学等多个学科的数据。对于 Alpaca 33B 模型，还加入了 OASST1 数据集，是由 gpt-3.5-turbo API 产出并翻译的数据，共 20K。

训练参数：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/b8952d58-c11f-41a8-bfaa-00fe1eaed4c0)

相比基础版本，plus 版本采用了更大的 LoRA rank，除了调整 lr 和 batch size 之外，其它参数都保持和预训练阶段的设置一致。

## 6 指令遵循任务上的结果

#### 6.1 任务设计与评估方法

由于文本生成任务的形式存在差异，评估文本生成任务的性能具有挑战性，这使得其与文本分类、阅读理解等自然语言处理任务不太一样。

和一些前序工作一样，使用 GPT-4 作为评分工具，为每个样本提供总体评分，这比人工评估更加高效。然而，GPT-4 并不总是能提供准确的得分，所以通常还会人工检查一下，人工检查确保分数的一致性以及评估模型的有效性。本文使用以下的 prompt 来对比两个模型的输出：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/0ca33b3a-39e0-4303-8b47-c016ec30590a)

即给定两个模型的输出到 GPT-4，让它评判哪个结果更好。

通过采用 GPT-4 初评+人工检查修正的方式，本文建立了可靠的评估框架，可以有效的衡量 Chinese Alpaca 模型在一系列自然语言处理任务上的表现。

本文提供的评估集旨在全面评估 Chinese Alpaca 模型在各个自然语言理解和生成任务中的表现，改评估集包含 200 个样本，涵盖十个不同的任务，包括问答、推理、文学、娱乐、翻译、多轮对话、编码和道德等等。**特定任务的总分是用过对该任务中所有样本的分数求和并将总分归一化到百分制来计算的**。这种方法确保评估集能够反映出模型在各种任务上的能力，有效评估其平衡性与鲁棒性。

#### 6.2 针对 decoding 的实验设置

LLMs 的 decoding 过程对文本生成的质量和多样性起着至关重要的作用。在本文的实验中，decoding 过程使用以下参数：

- Context size：2048，决定了模型生成文本是最多可以考虑的 token 数；
- Maximum sequence length：生成的最大序列长度设置为 512，确保输出能够保持重点并与输入的 prompt 的相关性；
- Temperature：0.2，控制采样过程的随机性，值越小，则模型生成内容更有针对性和确定性，值越大输出更有多样性，在多轮对话任务中，改值调整至 0.5，以增加输出的多样性；
- Top-k sampling：40，意味着模型每一步都会从 top 40 个最可能的 tokens 中选择一个作为下一个生成的 token，增加输出的随机性和多样性；
- Top-p sampling：0.9，通过考虑一组动态的 token 来进一步增强多样性，这些 token 占90% 的概率；
- Repetition penalty：为了阻止模型生成重复文本，将重复惩罚项设置为 1.1，对已经选择的 token 进行惩罚；

#### 6.3 实验结果

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/6c982789-2b98-4f3a-b59f-d6633fbd758c)

Chinese Alpaca 7B、13B、33B 之间的横向对比，整体 33B 更好。

- 多轮对话：ChatGPT 最重要的是它流畅的多轮对话能力，上面的结果可以看到，Plus 版本的 7B、13B 模型的多轮对话能力是要强于基础版的 33B 模型的；这表明了**针对对话体验而言，增多训练数据比增大模型参数更有效**，特别是该模型是从原始 LLaMA 训练得到的，其语言知识无法直接迁移；

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/044c7c54-dd0b-49ea-9e36-f5d417a6a1c8)

- 文本生成：LLMs 最重要的是其文本生成能力，从上面的示例可以看出，Alpaca-Plus-7B 和 13B 均生成了正确的信件格式，满足 prompt 需求，相比而言 13B 的结果更加全面，质量最好。而基础版的 33B 的生成结果没有遵循信件格式，内容有些过于简化，显然不如两外两者。这也表明，**使用更多的数据在较小模型上训练可能比使用更少数据在更大模型上训练的性能更好**。

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/950dc63f-b030-4d7a-85cd-019a13d8070f)

- 数值计算与推理：数值推理一直被认为是检验大型语言模型推理能力的最重要的任务之一。从上面的示例来看基础版的 33B 模型明显优于另外两者，结果表明**模型的大小在数值推理任务中更加重要**。

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/8d4e9d2d-6c65-4de7-9fbf-1d87c684d784)

- Coding 能力：**较大的模型往往在代码生成等复杂任务中表现更好，这可能是因为它们能够捕获训练数据中更复杂的模式**。

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/bff6c153-4d25-4a49-8fd6-c2fb0ddc963e)

- Ethics：针对一些非法输入，33B 模型表现更好，不仅可以拒绝非法提示，还可以给出适当的建议。

整体而言，基础版的 33B 模型在多方面都比 Plus 版的 7B和13B模型效果更好，包括推理、编码、道德等。推测是因为较大的模型比较小的模型能够更好的处理这些能力，尽管基础版的 33B 模型 的训练数据更少；另一个可能的原因是继承了原始 LLaMA 的能力（原始肯定 33B 更强），编码和推理能力相对与语言无关；同时也能注意到 Alpaca-33B 在文本生成、多轮对话等方面的效果较差，那是因为 Plus 系列模型使用了更多训练数据，因此能够提供更加多样化和丰富的内容，预计 Alpaca-Plus-33B 在这些方面将会更好。

## 7 自然语言理解任务上的结果

#### 7.1 任务描述

C-Eval 是一个多项选择问答数据集，主要涵盖了 4 类：STEM（科学、技术、工程和数学）、社会、任务以及其它，共包含 52 个学科，总计 14K 个样本。评估方法是给定问题，要求模型生成正确的选项标签。本文主要在验证集（1346个样本）和测试集（12342个样本）上对模型进行了评估，其中测试得分是通过将模型的结果提交至官方排行榜获得的。

#### 7.2 decoding 策略

评估 LLaMA 模型时，是直接将样本送给模型，而评估 Alpaca 模型时，将样本使用 4.5 章节的 prompt 后送给模型。然后要求模型进行一步预测并给出下一个 token 的概率分布，再将概率分布映射到有效标签。

#### 7.3 与原始 LLaMA 的结果对比

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/9b5d2f3d-6e2f-4c10-a32a-1d14c03cc222)

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/0258d91d-64e4-4cce-98c3-79451586297e)

上图和上表展示了 Chinese-LLaMA/Alpaca 模型相比原始 LLaMA 在 C-Eval 数据集上的结果对比。

- Chinese LLaMA 相比原始 LLaMA 的提升：有适度的提升，表明了中文数据的预训练对 C-Eval 上对评估有一定的影响，但也有个别异常。而 Chinese LLaMA Plus 相比 Chinese LLaMA 而言并没有明显的提升，设置在 13B 模型上表现更差。**这可能表明了原生的语言模型在 C-Eval 或类似任务上并不擅长**，增加预训练数据量并没有带来太大的收益。
- Alpaca 相比 LLaMA 提升明显：在不同的实验设置中（zero-shot 或 5-shot），Alpaca 模型系列比 LLaMA 模型系列有明显的改进，**表明指令遵循模型比纯语言模型更有能力处理 NLU （nature language understanding，自然语言理解）类任务**。与 LLaMA 不同，可以看到**Chinese Alpaca Plus 模型比基础的 Chinese Alpaca 模型有明显的提升**，这进一步表明指令遵循模型更有能力处理 NLU 类任务，并且可以“释放”更多的预训练数据的能力。
- LLaMA 通常在 few-shot 任务上性能更好，而 Alpaca 更擅长 zero-shot 任务：简单来说，**LLaMA 在 5-shot 任务上比 zero-shot 上表现更好，而 Alpaca 在 zero-shot 上的表现比 5-shot 更好**。因为 LLaMA 不是为指令遵循设计的，因此 few-shot 可能提供有关如何遵循 C-Eval 中的问答结构的有价值的信息；而 Alpaca 微调时早已见过大量的指令数据，再给几条指令数据收益很小。

#### 7.4 与其它模型对比

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/e324a65e-4fa2-45b7-9c63-3d3127a39ea5)

- 没开源的模型的效果还是更好，CPT-4 NB！
- 在开源的模型中，Chinese Alpaca 还是比较有竞争力的，即便是其他的模型尺度更大；

## 8 不同量化方法的效果

由于大量的计算资源需求，在个人电脑上（尤其是只有 CPU）部署 LLMs 充满了挑战，然而随着众多开源的工作，比如 llama.cpp，用户也可以比较容易的部署 LLMs 了。

量化 LLMs 并部署在个人电脑上有以下几个优点：

- 确保敏感信息不被传输到外部服务器上，而是保留在本地，保护数据隐私；
- 是计算资源有限的用户降低学习 LLMs 的门槛；
- 有利于促进本地部署 LLMs 的新应用和研究方向的发展；

使用 llama.cpp 将 Alpaca 系列模型进行不同程度的量化，效果对比如下：

![image](https://github.com/Tramac/paper-reading-note/assets/22740819/2599142b-5634-4609-a058-d208cb80c210)

- 量化水平与内存使用和推理速度强相关，所以在选择合适的量化水平时必须进行权衡；
- 与原始 FP16 模型相比，8-bit 量化方法的困惑度（？）几乎相同，内存要求只是 FP16 的一半，表明在个人电脑部署时 8-bit 是一个更好的选择；
- 6-bit 与 8-bit 相比效果也差不多，在性能和速度之间也有比较好的权衡，但如果量化水平更高时，效果衰减就比较明显了；
- 模型越小，对量化程度越敏感；

## 9 结论

本文提出了一个方法用来增强原始 LLaMA 的中文理解能力，已知原生 LLaMA 的词表中包含的中文 tokens 很有限，再次基础之上扩充了 20000 个中文 tokens，极大的提高了中文的编解码效率和理解能力。

在中文 LLaMA 的基础上，使用中文指令数据集进一步 fine-tuning，得到的中文 Alpaca 模型提高了指令遵循能力。

为了评估模型的有效性，在 10 种不同任务上标注了 200 个样本，并且使用 GPT-4 进行评估，实验结果显示，相比于原始 LLaMA，本文提出的模型对于中文的理解和生成能力有明显的提升。同时在 C-Eval 数据集上的结果显示，本文提出的模型相比于更大的模型也很有竞争力。

未来，计划引入 RLHF 或者 RLAIF 方法，进一步提高模型输出和人类意图的一致性。另外，探索采用更有效的量化方法，比如 GPTQ，研究 LoRA 的替代方法，使训练、微调效率更高。

## 10 局限性

- 有害且不可预测的内容：即使模型已经针对一些不道德的查询做了处理，但还是有可能产生违反人类价值观的内容。该问题通常是因为训练数据的偏差或模型在某些内容上缺少辨别能力导致的。
- 训练不充分：受计算资源和数据的限制，模型的训练其实不是很充分，中文理解能力仍然有提升空间；
- 缺乏鲁棒性：当面对一些对抗性输入或者很少见的输入时，模型可能会有输出不一致或无意义的内容；
- 综合评价：LLMs 的评估也是当今一个重要课题，虽然目前已经有较多的评估 benchmark，但是它们对模型的全面性和适用性还未得到充分的检验。更多样化、更全面的评估数据集和基准将对 LLMs 的研究有巨大的帮助；
- 可扩展性和效率：即使本文使用了 LoRA 和量化来保证模型的使用性更友好，但是和原始 LLaMA 结合时，模型的尺寸和复杂性仍然会导致部署困难，尤其对于计算资源有限的用户；