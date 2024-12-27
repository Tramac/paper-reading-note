# InternVL 1.5

[paper](https://arxiv.org/pdf/2404.16821) | [code](https://github.com/OpenGVLab/InternVL) | [zhihu](https://zhuanlan.zhihu.com/p/699439759) | [model](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) | [demo](https://internvl.opengvlab.com/)

## TL;DR

- 更强的 vision encoder，结合 continuous learning 训练策略
- 动态分辨率，将图像分割成 448×448的小块，最高可支持4k分辨率的图
- 多语言数据集，主要是中英文

## Abstract

InternVL 1.5 主要做了以下改进：

- **更强的 vision encoder**：对 vision encoder InternViT-6B 实施了 continuous learning 的策略，使用高质量的图文数据对其进行提炼。这一过程不仅增强了模型理解视觉内容的能力，还提高了其在各种 LLMs 中的适应性。此外，使用 InternLM2-20B 作为语言基础模型也提供了强大的初始语言处理能力

- **动态分辨率**：采用动态高分辨率策略，将图像分割成448×448的小块，根据图像的纵横比和分辨率，小块的数量从1到40不等（即4K分辨率）。为了捕捉全局上下文，还额外包括了一个全图的缩略图输入

- 多语言数据集：收集了多样化的公共数据集，涵盖了高质量的自然场景、图表、文档以及英文和中文对话。此外，还使用开源 LLMs 开发了一个数据翻译流水线，可以轻松扩展到更多语言

## 1 Introduction

开源模型和专有商业模型如（GPT-4V、Gemini系列等）之间的能力仍存在明显差距。本文]认为主要原因有以下几点：

- Parameter Scale：最近的专有商业 MLLMs 通常不少于1000亿参数，而开源模型通常使用3亿参数的视觉基础模型（VFM），并将其与7亿或13亿参数的LLMs集成；
- Image Resolution：专有商业模型通常采用动态分辨率方法，保持原始纵横比以促进详细场景和文档理解。相比之下，开源模型通常使用固定分辨率进行训练，如336×336和448×448，导致与商业模型相比能力上有相当大的差距；
- Multilingual Capability：专有模型通常利用广泛的多语言数据集进行训练，提高不同语言的性能。然而，开源模型主要使用英语数据，依靠LLMs的零样本能力来处理其他语言。

## 2 Related Work

**Proprietary Commercial MLLMs**

GPT-4v、Gemini 1.5、QwenVL-Plus/Max、Claude-3V、Step-1V、Grok-1.5V

**Open-Source MLLMs**

LLaVA、MiniGPT-4、VisionLLM、Qwen-VL、CogVLM

**Vision Foundation Models for MLLMs**

CLIP-ViT、SIGLIP、CLIP-ViT+CLIP-ConvNext、SigLIP-L+SAM-B

## 3 InternVL 1.5

### 3.1 Overall Architecture

<center>
    <img src="https://github.com/user-attachments/assets/d258ab18-a8ca-467b-802c-994b8fd408ea">
</center>

InternVL 1.5 的整体结构如上，遵循 ViT-MLP-LLM 的架构，将预训练的 InternViT-6B 与预训练的 InternLM2-20B 通过一个随机初始化的 MLP 投影层进行集成。

在训练期间，使用了一种动态分辨率策略，根据输入图像的纵横比和分辨率，将图像划分为448×448像素大小的小块，数量从1到12不等。在测试期间，可以直接扩展到40块（即4K分辨率）。为了增强高分辨率的可扩展性，我们简单地采用了 Pixel Shuffle 操作，将视觉token的数量减少到原始数量的四分之一。因此，在模型中，一个 448×448 的图像由256个 visual token 表示。

> patch size 为 14，448/14=32，32×32/4=256

### 3.2 Strong Vision Encoder

已有的多模态模型通常都是使用一个 pre-trained ViT 作为 vision encoder，这些 ViT 一般是在固定的低分辨率 224×224 上训练得到的，因此，当任务处理来自互联网以外的来源的高分辨率图像或图像时，它们的性能会降低，例如文档图像。

**InternViT-6B-448px-V1.2**

在 InternViT-6B 的基础之上做了以下改进。首先，发现倒数第四层的特征对多模态任务更好，所以，因此直接丢弃了最后三层的权重，将 InternViT-6B 从 48 层减少到 45 层。然后，我们将 InternViT-6B 的分辨率从 224 提高到 448，并将其与 Nous-Hermes-2-Yi-34B（LLM）连接。

> 这里没说明白，为什么 LLM 突然从 InternLM 换成了 Yi-34B

为了使模型具有处理高分辨率和 OCR 的能力，vision encoder 和 MLP 均参与训练，该部分的训练数据为 image caption 和 OCR 数据的结合，最终得到 InternViT-6B-448px-V1.2。

**InternViT-6B-448px-V1.5**

继续对 v1.2 训练，该阶段图像的分辨率从固定的448×448扩展到动态448×448，即将图像分割为448×448的patch，数量从1到12不等，此外，还扩充了预训练数据集的数据规模、质量和多样性，使得 1.5 版本模型强大的鲁棒性、OCR能力和高分辨率处理能力。

值得一提的是，尽管 LLM 从 Nous-Hermes-2-Yi-34B 更改为 InternLM2-20B，InternViT 与新的 LLM 保持了出色的兼容性和可移植性，这表明 InternViT-6B 在 MLLM 预训练阶段学习到的视觉特征广泛适用，并且不限于特定的 LLM。

### 3.3 Dynamic High-Resolution

动态分辨率的处理主要包括两个过程：

**Dynamic Aspect Ratio Matching**

<center>
    <img src="https://github.com/user-attachments/assets/c62b04d2-2e9c-441e-850a-925d5275bb6b">
</center>

如上图所示，为了在处理过程中保持自然纵横比，我们从预定义的纵横比集合中动态匹配最优的纵横比。由于计算资源有限，我们在训练中允许最多12块。因此，这个集合包括了由1到12块形成的所有35种可能的纵横比组合，例如{1:1, 1:2, 2:1, 3:1, ..., 2:6}。在匹配过程中，对于每个输入图像，我们计算其纵横比，并与35个预定义的纵横比进行比较，通过测量绝对差值。如果多个预定义的纵横比匹配（例如，1:1和2:2），我们优先选择不超过输入图像面积两倍的纵横比，从而防止低分辨率图像的过度放大。

**Image Division & Thumbnail**

确定了合适的纵横比后，将图像调整到相应的分辨率。例如，一个 800×1300 的图像将被调整到 896×1344。然后，将调整大小的图像分割成 448×448 像素的小块。除了这些小块，我们还包括了整个图像的缩略图（Thumbnail）以捕捉全局上下文。这个缩略图被缩小到448×448，帮助模型理解整体场景。因此，在训练期间，visual tokens 的数量范围从 256 到 3328。在测试期间，小块的数量可以增加到最多40块，从而产生10496个 visual tokens。

### 3.4 High-Quality Bilingual Dataset

**Pre-training Dataset**

<center>
    <img src="https://github.com/user-attachments/assets/05456055-4a7b-4966-81b6-99bc2ed342c2">
</center>

预训练阶段，多样化的数据集组合确保了 InternVL 的鲁棒性，迎合了跨任务的各种语言和视觉元素。

**Fine-tuning Dataset**

<center>
    <img src="https://github.com/user-attachments/assets/8467d911-6aec-4d7e-b03d-7d4a23019315">
</center>

finetune 阶段使用精心挑选的数据集，以提高模型在广泛的多模态任务上的性能。

**Data Translation Pipeline**

<center>
    <img src="https://github.com/user-attachments/assets/af122cf4-d3dc-491a-9e2a-09c5166703ed">
</center>

为了增强模型的多语言能力，使用了一个数据翻译 pipeline。利用最先进的开源 LLMs 或 GPT-3.5 将英语数据集转换为另一种语言（例如中文），在双语标记中保持一致性和准确性。此外，它可以通过调整语言提示轻松扩展到包含更多语言，而不依赖于手动注释过程。在 finetune 部分的数据表格中，注释了每个数据集的语言。对于最初是英文的数据集，"zh"的注释表明我们已经使用翻译流水线将其翻译成中文。例如，COYO 和GRIT最初是英文数据集，我们已经将它们翻译成中文。通过利用这个翻译流水线，InternVL 1.5的中文能力得到了极大的增强。

# 4 Experiments

InternVL 1.5 的训练分为两个阶段：1）pre-training 阶段训练 vision encoder 和 MLP projector；2）fine-tune 阶段训练所有参数。

**Comparison with State-of-the-Art MLLMs**

<center>
    <img src="https://github.com/user-attachments/assets/316ecc62-36e7-43fd-8186-26c7f73b36f6">
</center>

<center>
    <img src="https://github.com/user-attachments/assets/14db7896-0d53-4e38-824c-24e3cc444671">
</center>

**Ablation Study**

Larger LLMs need Larger VFMs：更大的 LLM 需要搭配更大的 vision encoder

