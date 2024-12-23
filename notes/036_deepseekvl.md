# DeepSeek-VL

## TL;DR

- hybrid vision encoder + vision adaptor + LLM
- 更加注重预训练阶段，多模态-语言之间平衡训练，避免语言能力退化；
- 引入 hybrid vision encoder 支持 1024x1024 高分辨输入，提高对图像的细节理解能力；
- 使用 case taxonomy（分类），构造了一份高质量的 SFT 数据；

## Abstract

DeepSeek-VL 主要在以下三个方面进行了设计：

- Data Construction

  数据构造确保了多样性、scalable 并且广泛涵盖了现实世界的场景数据，如网络截图、PDF、OCR、图表以及专家知识等。另外，通过一个 case 分类的方法区分各种任务和场景，构造出一个指令微调数据集。

- Model Architecture

  考虑到性能以及实际使用场景的需要，DeepSeek-VL 使用了一个 hybrid vision encoder，可以高效处理 1024x1024 分辨率的图像输入，同时保持相对较低的计算开销。高分辨率输入确保了模型能够捕捉到各种视觉任务相关的语义和细节信息。

- Training Strategy

  首先，一个强的视觉-语言模型应该具有很强的语言能力。为了确保在 pre-training 阶段能够保留 LLM 能力，DeepSeek-VL 从一开始训练就对 LLM 和视觉能力之间进行了动态的平衡，从文本开始，逐渐调整比率以促进两种模态的平衡，最终实现加入视觉能力的同时不损失 LLM 本身的能力。

DeepSeek-VL 提供了 1.3B 和 7B 两个大小的模型，从实验结果看均取得了不错的效果。

## 1 Introduction

在现实世界场景中，大多数开源模型和专有模型的性能差距在很大程度上很明显，这主要是由于以下原因：

- 预训练不充分，大多都在指令微调阶段做工作；
- 为了在 benchmark 上刷分，很多方法在指令微调阶段合并各种学术数据集，但在真实场景使用体验方面往往不是很好；
- 在模型架构方面，之前的工作大多采用一个 pre-trained vision encoder，通过一个 adapter 与 LLM 进行对齐。这种方式通常仅支持较低分辨率的图像输入，如 336x336 或 448x44，对于小目标的识别不是很友好；
- 多模态训练后，语言能力会退化，视觉和语言之间没有做好 trade-off。

## 2 Data Construction

训练数据分为两部分：Vision-Language pre-training Data 和 Vision-Language Supervised Fine-Tuning Data。

其中，VL pre-traning data 用来增强模型的基础多模态理解能力，用在 stage1 和 stage2 的训练；VL supervised fine-tuning data 用来教模型完成特定的下游任务，用于 stage3 的训练。

### 2.1 Vision-Language pretraining Data

预训练数据包含一系列开源数据以及一部分 DeepSeek 私有数据。然后对数据集做了详细的分类：

- Interleaved image-text：图像-文本交错输入的数据，有利于模型的 in-context learning 能力；
- Image caption
- Table and chart：图、表数据
- Web Code
- Document OCR：文档中提取 OCR
- Scene text OCR：场景图片中提取 OCR
- Text-only corpus：DeepSeek-LLM 的纯文本训练数据，主要为了避免语言能力退化

各个部分数据来源以及占比：

<center>
    <img src="https://github.com/user-attachments/assets/de495ec2-f8de-429c-a438-f1a36508498c">
</center>

需要注意的是，纯文本数据占了 70%，说明在预训练阶段语言能力的保留非常重要。

### 2.2 Supervised Fine-tuning Data

SFT 数据同样包含多模数据和纯文本数据，也有一部分是私有数据，私有数据主要针对实际使用场景进行了精心设计。每个部分的数据来源和占比如下：

<center>
    <img src="https://github.com/user-attachments/assets/aef2df66-4fec-439e-81fc-b65d21fcff83">
</center>

纯文本数据仍然占有很高的比例，47.9%，精调的数据 占有 10.5%。

具体的 case taxnonomy 方法可以看后面的 Bonus 部分。

## 3 Approach

### 3.1 Architecture

**Hybrid Vision Encoder**

hybrid vision encoder 用的是 SigLIP + SAM-B 的组合。

SigLIP 这类的 CLIP family 模型主要提取的是 high-level 的语义特征（这和 CLIP 的训练数据有关，都是图像级的描述），它对于一些 low-level 特征并不敏感，比如 OCR 或者 grounding 等。为了解决该问题，引入了另一个 pre-trained vision encoder，即 SAM-B，可以接受 1024x1024 的高分辨率输入，用来针对图像中的细节信息。

具体来说，给定 1024x1024 的图像，SAM-B 会生成 64x64x256 的 feature map，然后 VL adaptor 将其插值为 96x96x256，然后使用两层 stride=2 的卷积层对 feature map 进行下采样，得到 24x24x1024 的特征图，最终 reshape 成 576x1024 的 feature map。另外，低分辨的分支 SigLIP 也会生成 576x1024 大小的特征图，将两者 concat 之后最终得到 576x2048 的 feature map，也就是说图像最终会分为了 576 个 token，每个 token 的特征维度为 2048，这也是前文提到的 **fixed token budget**。

**Vision-Language Adaptor**

adaptor 用的是两层 hybrid MLP。首先分别使用一个单层 MLP 处理 SigLIP 和 SAM-B 两个分支的特征输出，然后将特征 concat 之后得到上面的 576x2048 的 feature map，再使用一个 MLP 将其映射至 LLM 空间。

**Language Model**

LLM 用的是 DeepSeek-LLM，这里不做展开了。

### 3.2 Training Pipelines

<center>
    <img src="https://github.com/user-attachments/assets/8eda54c2-c65d-4d56-9ee6-247ace551a46">
</center>

DeepSeek-VL 的训练分为三个阶段：1）vision-language adaptor warmup；2）joint vision-language pre-training；3）supervised fine-tunining。

#### 3.2.1 Stage 1: Training Vision-Language Adaptor

第一阶段的主要目标就是对齐视觉和语言特征空间。和 LLaVA 一样，该阶段将 vision encoder 和 LLM freeze 住，只训练 adaptor 的参数，训练数据为来自 ShareGPT4V 的 1.25 million 的 caption 数据和 2.5 million 的 Document OCR 数据。

实际上，LLM 相比，adaptor 的参数量要小得多，有可能限制模型能力的学习。因此有个问题是：**在这个阶段，是否需要这么多的数据训练？**本文做了实验验证：

<center>
    <img src="https://github.com/user-attachments/assets/2623f012-4b58-482c-ad47-9acb2c9f0026">
</center>

结果表明，在这个阶段扩展数据规模并不能带来收益，甚至可能导致性能较差。所以在第二阶段会选择 unfreeze LLM。

#### 3.2.2 Stage 2: Joint Vision-Language pretraining

该阶段保持 vision encoder freeze，同时训练 adaptor 和 LLM。

最初，DeepSeek-VL 尝试用多模态数据直接训练LLM，然而，发现虽然多模态性能的指标逐步提高，但语言指标存在明显且严重下降。这种现象可能有两个因素引起：1）大多数多模态训练语料过于简单，并且与语言数据的复杂性和分布存在显着差异；2）训练过程中在多模态和纯语言之间似乎存在竞争的关系，有可能导致 LLM 语言能力的灾难性遗忘。

为了解决该问题，DeepSeek-VL 提出 joint language-multimodel training strategy。在训练期间，不仅使用多模态数据，同时还用了大量的纯语言数据，在两者之间做一个 trade-off。经过实验分析，有以下结论：

1. 引入语言数据显著减轻了语言能力的下降；
2. 引入引言数据并不会导致多模态性能的显著损失；
3. 不同模态的性能与训练数据集中各自的比例密切相关，证实了两种模态之间的竞争关系。

最终，纯文本数据和多模态数据的比例大约为 7:3。

然而，模型的预训练阶段会产生大量的计算成本，并且在 7B 模型上执行迭代需要大量的计算能力和时间。一种策略是先较小的模型（1.3B 模型）上进行实验验证，然后将其迁到 7B 模型。但是，在第二阶段的训练中，1.3B 模型的表现不是特别稳定，随后的实验使我们能够确定这个问题的根本原因：1.3B模型参数量有限以及训练数据集中 SFT 数据的缺失，这两者都阻碍了模型准确遵循指令的能力。即使模型拥有正确选项的知识，也很难精确地生成它们。

#### 3.2.3 Stage 3: Supervised Fine-tuning

该阶段同时优化 LLM、adaptor 以及 vision encoder（但是 SAM-B 部分 frozen，显存扛不住），基本和其它多模方法一致的设置。

关于训练阶段的消融实验：

<center>
    <img src="https://github.com/user-attachments/assets/d2000566-6749-4471-9394-65c404e8f109">
</center>

看起来 adaptor warm up 带来的提升有限。

### 3.3 Hyperparameters and Infrastructures

训练超参数如下，使用的训练框架为 high-flyer（幻方自己家的训练平台）

<center>
    <img src="https://github.com/user-attachments/assets/f1fb02f5-3eec-4e57-8e62-1f8a02df92e4">
</center>

## 4 Evaluation

评测部分包括公开的多模 benchmark、LLM benchmark、人工评估，结果在这里不详细展开了，直接看 paper 吧。

## Bonus

1. case taxonomy 具体是如何做的？

   首先收集一些线上的 GPT-4v 和 Gemimi 测试用例，然后对这些测试用例进行分类，如 recognition, conversion, analysis, reasoning, evaluation, and safety，具体细节如下：

   <center>
       <img src="https://github.com/user-attachments/assets/d001766e-28c4-4c30-9245-8563bc4b8a43">
   </center>

   结构化的分类可以作为 representative prompts（？）选取时的 guideline，并且可以通过调整每个类别数据的占比来优化模型性能。

2. modality warm-up？

   其实就是第一个阶段只训 adaptor，用来先把模态空间对齐。
