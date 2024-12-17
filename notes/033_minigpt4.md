# MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models

[paper](https://arxiv.org/pdf/2304.10592) | [project page](https://minigpt-4.github.io/) | [code](https://github.com/Vision-CAIR/MiniGPT-4) | [huggingface](https://huggingface.co/Vision-CAIR/MiniGPT-4)

团队来自阿卜杜拉国王科技大学。

## 摘要

最近的 GPT-4 已经证明了非凡的多模态能力，我们认为 GPT-4 的多模态生成能力的主要是利用了更强的 LLM。为了验证该结论，本文提出 MiniGPT-4，仅通过一个 projection layer 将 visual encoder 和 LLM 进行对齐。实验结果表明，MiniGPT-4 展示出了很多类似于 GPT-4 的功能，同时还观察到其它新的能力。

实验中发现仅对原始 image-text pair 进行预训练可能会产生不自然的语言输出，这些输出缺乏连贯性，包括重复和碎片化的句子。为了解决这个问题，在第二阶段构造了一个高质量、对齐的数据集，使用对话模板进一步微调模型。这一步被证明对于增强模型的生成可靠性和整体可用性至关重要，另外模型计算效率很高，因为只使用大约 500 万个对齐的 image-text pair 训练投影层。

## 1 引言

本文主要贡献：

- 通过将 visual feature 和 LLM 对齐，可以实现 vision-language 的涌现能力，实验证明 MiniGPT-4 拥有 GPT-4 类似的能力；
- MiniGPT-4 是 computational efficiency 的，仅通过训练一个 projection layer 就可以实现 align；
- 本文发现仅通过公开的数据集进行预训练，输出缺乏连贯性，需要 high-quality、well-aligned 数据继续进行 fine-tune。

## 2 相关工作

**LLM**

早期的语言模型 BERT、GPT-2、T5，大语言模型 GPT-3、Megatron-Turing NLG、Chinchilla、PaLM、OPT、BLOOM、LLaMA，预训练+后训练范式的形成 InstructGPT、ChatGPT、Alpaca、Vicuna 等。

**Leveraging Pre-trained LLMs in Vision-Language Tasks**

连接 visual encoder 与 LLM 的主要方式有：

- Flamingo 中的 gated cross-attention
- BLIP-2 中的 Q-Former
- MLP

## 3 方法

<img src="https://github.com/user-attachments/assets/7d76b13c-2121-40fe-9e6b-393e377ac592" style="zoom:80%;" />

MiniGPT-4 旨在将预训练的 vision encoder 与 LLM 之间进行对齐。其中选择 Vicuna 作为 language decoder，与 BLIP-2中 一样使用 ViT-G/14 + Q-Former 作为 visual encoder，使用一个 linear projection layer 进行两者之间的对齐。

MiniGPT-4 采用了两阶段的训练，pre-training 阶段使用收集的大量对齐的 image-text 对来获取视觉语言知识，fine-tune 阶段在一个小的高质量的 image-text 数据集上训练。

### 3.1 pre-training stage

数据：Conceptual Caption，SBU，LAION 数据集结合得到预训练阶段的数据，使用 batch size 256 共训练 20000 steps，覆盖约 5 million image-text pairs。

参数：预训练时仅训练 linear projection layer。

预训练阶段的问题：很难产生连贯的语言输出，例如生成重复的单词或句子、碎片化的句子或不相关的内容。

### 3.2 high-quality aligment dataset

为了解决上述预训练模型的问题，需要构造一批高质量、对齐的数据进行后续的 fine-tune。

**Initial aligned image-text generation**

给定一张图片，首先使用第一阶段的预训练模型生成描述，为了使模型能够产生更详细的描述，我们设计了一个符合 Vicuna 语言模型 conversation 格式的提示，如下所示：

```python
###Human: <Img><ImageFeature></Img> Describe this image in detail. Give as many details as possible. Say everything you see. ###Assistant:
```

其中 `<ImageFeature>` 表示 linear projection 后的 visual features（一堆浮点数？应该就是类似 visual token embedding）。

为了识别不完整的句子，然后检查生成的句子是否超过 80 个 token。如果没有，继续使用 prompt：`###Human: Continue ###Assistant:` 令模型继续生成，最终使模型生成更加全面的描述。从 CC 数据集中随机选取 5000 张图使用上述方式生成了一份微调数据集。

（使用预训练模型生成数据继续微调，左脚蹬右脚上天？不太能理解。）

**Data post-processing**

以上生成的图像描述仍然有很多噪声并且包含错误，例如单词或句子的重复以及不连贯陈述的存在。为了解决这些问题，使用 ChatGPT 对描述信息进行 refine，prompt 如下：

```python
Fix the error in the given paragraph. Remove any repeating sentences, meaningless characters, not English sentences, and so on. Remove unnecessary repetition. Rewrite any incomplete sentences. Return directly the results without explanation. Return directly the input paragraph if it is already correct without explanation.
```

后处理之后，再手动验证每个图像描述的正确性，保证其高质量。具体来说，检查每个生成的图像描述是否遵循我们想要的格式，并通过消除 ChatGPT 无法检测到的冗余词或句子来手动细化生成的描述。最终 5000 个样本清洗出 3500 条作为第二阶段的训练。

### 3.3 Fine-tune

在第二阶段，训练数据格式如下：

```python
###Human: <Img><ImageFeature></Img> <Instruction> ###Assistant:
```

（这和第一阶段有什么区别，为什么要单独说？）

其中，`<Instruction>` 随机从类似下面的指令中进行选取：

```
"Describe this image in detail"
"Could you describe the contents of this image for me"
```

NOTE：It is important to note that we do not calculate the regression loss for this specific text-image prompt.

（也很奇怪，哪些方法会对 `<Instruction>` 这写 prompt 计算 loss？）

## 4 结果展示

- 生成图像描述
- 识别图像中的有趣信息
- 发现图像中不寻常的内容
- 根据手写文本生成网站
- 识别图像中的问题并提供解决方案的能力
- 根据图像作诗或歌词
- 根据图像写故事
- 生成图像评论
- 检索与图像相关的信息
- 给定图像教人烹饪

论文展示了上述的一些能力，具体例子直接看 paper。

## 5 局限性

- Language hallucination：幻觉
- Inadequate perception capacities：感知能力不足

Paper 读下来整体感觉像是一个报告，很多东西都没有交代清楚，但是 MiniGPT-4 这个工作很火，可能是是做多模态比较早？
