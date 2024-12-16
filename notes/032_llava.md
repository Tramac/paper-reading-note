# Visual Instruction Tuning

## 摘要

使用机器生成（machine-generated）的指令遵循数据来微调 LLMs，可以提高模型的 zero-shot 能力，但是在多模态领域该方式的有效性还未得到验证。本文首次尝试仅使用 GPT-4 来生成多模态 language-image 指令遵循数据，并且通过在这些生成的数据上进行指令精调，提出了一个端到端训练的多模态模型 LLaVA。LLaVA 连接了一个 vision encoder 和一个 LLM，用于通用的视觉-语言理解。

为了促进未来对视觉指令遵循的研究，本工作构建了两个 benchmark，包含多样性（diverse）和挑战性（challenging）的应用导向（application-oriented）任务。实验表明，LLaVA 展现出了令人印象深刻的多模态聊天能力，在合成的多模态指令遵循数据集上与 GPT-4 相比获得了85.1%的相对分数。在科学问答（Science QA）上进行微调后，LLaVA 和 GPT-4 的协同作用达到了92.53%的准确率。

## 1 引言

本文主要贡献：

- Multimodal instruction-following data：提出了一个数据重构的流程，使用ChatGPT/GPT-4将图像-文本对转换为适当的指令遵循格式
- Large multimodal models：提出了一个多模态大模型，通过将 CLIP 的 visual encoder 与 Vicuna 的 language decoder 连接，并在生成的指令性视觉-语言数据上进行端到端的微调
- Multimodal instruction-following benchmark：开源了两个多模态 benchmark
- Open-source：开源，包括多模态指令数据生成方式、codebase、checkpoints、visual chat demo

## 2 相关工作

Multimodal Instruction-following Agents 多模态指令遵循范式主要分为两大类：

- end-to-end 训练：比如图像编辑领域的工作 InstructPix2Pix，输入图像和文本指令，实现图像编辑
- 通过 LangChain/LLMs 协调各种模型系统：如 Visual ChatGPT、X-GPT、MM-REACT、VisProg、ViperGPT 等

Instruction Tuning 指令精调：在 NLP 领域，指令精调已经成为了训练范式的一部分，如 InstructGPT、ChatGPT  等都是指令精调后的对话模型；在多模态领域，有一些利用蒸馏的方法，如 Flamingo 等

## 3 视觉指令数据生成

已经有一些开源的多模态数据 image-text 对，比如 CC、LAION 等。但是无法直接作为指令遵循数据使用，如果进行人工处理是非常耗时且标准难以统一的。受 NLP 领域利用 GPT 实现文本标注的启发，本文提出了一种使用 ChatGPT/GPT-4 实现多模态指令遵循数据生成的方法。

对于一张图像 $X_v$ ，其 caption 表示为 $X_c$ ，自然地可以构造一组问题 $X_q$ ，期望模型以 caption 回答这些问题。本文利用 GPT-4 生成了一些 question list，如下：

- 生成简短的图像描述指令 list

  ![brief image desc](https://github.com/user-attachments/assets/cb253fce-ff77-4cc4-b931-d320cd05aff0)

- 生成详细的图像描述指令 list

  ![detail image desc](https://github.com/user-attachments/assets/c42f9c3b-4613-4b23-bd92-ae10f577ca2b)

LLaVA 中单条指令数据格式可以构造为：$Human: X_q X_v<STOP> Assistant: X_c <STOP>$。

数据格式已经确定，但是近通过上述简单的 question 构造的数据缺乏多样性以及推理难度。为了解决该问题，使用 GPT-4 来丰富数据，但是 GPT 只能接收文本 prompt，所以需要先将图像编码为视觉特征，让视觉特征作为 prompt 来请求 GPT，那么视觉特征的表示形式（symbolic representations）有以下两种类型：

- Caption：通常从不同的角度描述视觉场景
- Bounding boxes：通常定位场景中的目标，每个 box 对目标及其空间位置进行编码

利用以上两种信息，可以生成三类指令遵循数据：

- Conversation：问题涉及图像的视觉内容，包括对象类型、数量、对象动作、对象位置、对象之间的相对位置等，只有那些有明确答案的问题才被考虑在内。
- Detailed description：为了包含图像的丰富和全面的描述，首先使用 GPT-4 生成了一个问题列表；对于每张图像，然后从列表中随机抽取一个问题，让 GPT-4 生成详细的描述。
- Complex reasoning：以上两种类型侧重于视觉内容本身，在此基础上进一步创建了深入的推理问题，答案通常需要通过遵循严格的逻辑逐步推理过程。

对于每种数据类型，首先手动设计一些示例（这是数据收集阶段唯一的人工标注），然后被用作 GPT-4 上下文学习的种子示例。以下是每种数据类型的示例：

![](https://github.com/user-attachments/assets/4368a6dd-930a-4fca-8f0b-93499dcacd4a)

（我觉得这里应该给出 prompt 示例，举例说明是如何从 caption 和 boxes 信息请求 GPT-4 得到以下三种对话类型的）

通过以上方式，LLaVA 共收集了 158K 样本，其中 58K conversations，23K detailed description，77K complex reasoning。同时做了 ChatGPT 和 GPT-4 生成质量的对比，结果是 GPT-4 更好。

## 4 视觉指令微调（Visual Instruction Tuning）

### 4.1 模型结构

![](https://github.com/user-attachments/assets/20ca1402-f781-4cb4-9849-2c4142868254)

主要目标是有效地利用预训练的 LLM 和视觉模型的能力。模型结构如上，LLM 选择的是 Vicuna，Vision Encoder 选择的为 CLIP。

整体结构比较好理解，主要就是图像通过 vision encoder 编码之后，需要一个 **Projection W** 将其对齐到语言空间，然后将 visual tokens 和 language tokens 拼接之后送入 LLM，最终得到 response。

其中，projection W 是非常轻量的，LLaVA 中使用的是 MLP，另外也可以考虑用 gated cross-attention 和 Q-former 代替。

### 4.2 训练

模型训练的输入序列如下（示例为两轮对话）：

![](https://github.com/user-attachments/assets/05be3b2c-0f6b-4949-8fe7-53c5b76e886e)

对于每一张图 $X_v$，生成多轮对话数据 $(X^1_q, X^1_a, ..., X^T_q, X^T_a)$，其中 T 是对话轮数。第 t 轮的指令为：

![](https://github.com/user-attachments/assets/514e0e0c-cc1d-430f-8e36-9afd7e4b24a2)

或者说，只有在第一轮对话时，指令输入包含图像，后续的对话指令只有问题。

指令微调本质还是 LLM 预测下一个 token，训练目标仍然是一个自回归过程。具体来说，对于长度为 L 的序列，目标回答 $X_a$ 的概率为：

![](https://github.com/user-attachments/assets/6e860f70-1bf2-4361-9986-b67f53bbadfb)

其中，$/theta$ 是可训练参数，模型预测答案和停止的位置，因此只有绿色的序列/标记用于计算自回归模型中的损失。

对于 LLaVA 的训练，包含了两个阶段的指令调优过程：

- **Stage 1: Pre-training for Feature Alignment**

  首先对 CC3M 数据集清洗得到 595K 的 image-text pair 对（paper 附录中有清洗流程，数据处理挺重要的，建议看一下细节），按照最简单的方式将其转换为指令遵循数据格式，转换之后每个样本都是一个单轮对话。

  在 pre-training 阶段，只训练 projected W，vision encoder 和 LLM 都是 frozen 的，主要目的就是将图像特征 $H_v$ align 到 LLM 的语言空间中。这一阶段可以理解为**训练一个兼容于 LLM 的 visual tokenizer**。

- **Stage 2: Fine-tuning End-to-End**

  该阶段仍然保持 visual encoder frozen，继续训练 projection layer 和 LLM。主要在两个场景上进行了微调：

  - Multimodal Chatbot：多模态对话，训练数据就是上面清洗得到的 158K 样本
  - Science QA：科学问答，训练数据就是将 ScienceQA 处理成了单轮对话的形式

训练细节：

- 8 x A100
- pre-training on CC-595K for 1 epoch，lr=2e-03，batch_size=128
- finetune on LLaVA-Instruct-158K for 3 epoch，lr=2e-05，batch_size=32
- Adam with no weight decay，cosine learning rate，warmup ratio of 3%
- Pretraining on CC-595K 耗时 4h，finetune on Instruct-158K 耗时 10h，finetune on ScienceQA 耗时 4h

## 5 实验

### 5.1 Multimodel Chatbot

对话效果展示直接看 paper。

**Quantitative Evaluation**

量化评估主要衡量模型的指令遵循能力，利用 GPT-4 来判断生成结果的质量（感觉这里有点不公平，LLaVA 的训练数据就是 GPT-4 生成的，然后用 GPT-4 作为标准肯定是更友好的，但是应该也没有更好的办法了），具体方式为：

> After obtaining the responses from both models, we feed the question, visual information (in the format of textual descriptions), and the generated responses from both assistants, to the judge (i.e., text-only GPT-4).

benchmark 有两个：

- **LLaVA-Bench (COCO)**

  从 COCO-Val-2014 中随机选取 30 张图，对于每张图片，使用上面提到的方法生成三种类型（(conversation, detailed description, complex reasoning）的 question，共 90 条数据。该 benchmark 反映了模型对具有一致视觉输入的对齐行为和能力。对于三种类型的训练数据做了消融实验，结果是同时使用三种类型数据，性能更好，指标如下：

  ![](https://github.com/user-attachments/assets/f275b3a0-98bb-4bd9-b871-8b4b1c5391c8)

- **LLaVA-Bench (In-the-Wild)**

  为了评估模型在更具挑战性的任务和对新领域的泛化能力，本文收集了24张多样化的图片，总共60个问题，包括室内和室外场景、梗图、绘画、素描等，并为每张图片提供了高度详细且手动策划的描述和适当的问题选择。与 Flamingo 和 BLIP-2 的对比结果：

  ![](https://github.com/user-attachments/assets/71c618a8-60c8-4ccb-8ccf-f4d79767ed80)

### 5.2 ScienceQA

ScienceQA 数据集包含 21k 多模态多项选择题，覆盖 3 个 projects、26 个 topics、127 个 categories 和 379 个 skills，其中训练集 12726，验证集 4241，测试集 4241。

和其它方法之间的对比：

![](https://github.com/user-attachments/assets/8b40e854-48ca-426c-b63c-b309933c6862)

另外，在 ScienceQA 数据上进行了一系列的消融实验，主要对比了以下几个方面：

- visual features

  对比了使用 CLIP vision encoder 最后一层之前和之后的特征，结果是使用最后一层之前的特征性能更好，原因可能是与之前的层相比，CLIP的最后一层特征可能更多地关注全局和抽象的图像属性，而之前的层可以更多地关注有助于理解特定图像细节的局部属性。

- Chain-of-thought

  对比了 reasoning-first 和 reasoning-first，结论是 reasoning-first 可以大大提高收敛性，但对最终性能的贡献相对较小。

- Pre-training

  跳过 pre-training，直接在 ScienceQA 上 train from scratch，性能下降 5.11%，说明了 pre-training 的有效性。

- Model size

  对比了 13B 和 7B 模型，显然 13B 更好。

具体指标如下：

![](https://github.com/user-attachments/assets/a018dc6a-b9c1-45ee-863e-dbca79abfd1b)

## 6 数据生成时的一些细节

- GPT-4 生成 conversation 数据的 prompt

  ![](https://github.com/user-attachments/assets/54073b8d-9bd0-4d86-8107-c60a20573626)
