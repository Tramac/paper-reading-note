# InternVL

[paper](https://arxiv.org/pdf/2312.14238) | [github](https://github.com/OpenGVLab/InternVL) |[zhihu](https://zhuanlan.zhihu.com/p/702946079)

## TL;DR

- 将 vision encoder scale up 到 6B；
- 相比其它的 VLM 方法还多了一个 QLLaMA；
- 提出一种 progressively aligns 渐进式对齐策略将视觉与LLM空间对齐（contrastive training + generative training + SFT）；
- InternVL 可以做 visual perception、image/video-text retrival、image caption、VQA、multi-model dialogue 多种任务；
- 虽然效果很好，但还是感觉 pipeline 太复杂了；

## Abstract

InternVL 将 vision encoder 扩展至 6B，然后使用一种渐进式对齐策略逐步与 LLM 对齐。该模型可以把 vision encoder 当做一个 foundation model 来做一些常规的视觉任务，如像素级图像识别等，也可以和其它多模态模型一样应用于视觉-语言任务，如 zero-shot 图像/视频分类、检索，也可以做对话系统。

## 1 Introduction

已有的 VLLMs 通常使用一个轻量的 adaptor 将 vision 与 language 特征空间对齐，这种方式有以下问题：

- Disparity in parameter scales：不同模块之间的参数量存在明显差异，LLM 通常在 10B 量级，而 vision encoder 一般在 1B 左右，这种 gap 可能导致对 LLM 的能力利用不充分；
- Inconsistent representation：vision encoder 的特征空间与 LLM 的特征空间通常是不一致的（这不就是 adaptor 要解决的问题吗，列在这里感觉不合适）；
- Inefficient connection：轻量、随机初始化的的 adaptor 可能并不高效。

InternVL 的出发点就是弥补 vision encoder 与 LLM 之间 parameter scale 和 feature representation ability 的 gap。有三个主要设计：

1. Parameter-balanced vision and language components：vision encoder scaled up 至 6B，选择一个 8B 的 LLM middleware（讨厌死这些强行的新概念了，不如直接 adaptor）实现 reorganize visual features based on user commands（真的不知道在说些什么，我理解就是做 align）；
2. Consistent representations：使用一个 pre-trained multilingual LLaMA 作为 middleware，实现 vision encoder 与 LLM 的对齐；
3. Progressive image-text alignment：渐进式对齐策略，先在在大规模噪声 image-text 数据上进行对比学习，逐渐过渡到在 fine-grained 数据上进行生成学习。

这些设计使得 InternVL 有以下优势：

1. Versatile：多功能的，vision encoder 可以单独作为一个视觉基础模型来处理视觉感知任务；
2. Strong：强大的性能
3. LLM-friendly：由于与 LLM 对齐特征空间，模型可以顺利地与现有的 LLM 集成，比如 LLaMA 系列、Vicuna 以及 InternLM。

## 2 Related Work

Vision Foundation Models：CNN、ViT 等

Large Language Models：GPT 系列、LLaMA、Vinuna、InternLM、MOSS、ChatGLM、Qwen 等

Vision Large Language Models： GPT-4v、MiniGPT-4、VisionLLM、KOSMMOS-2、Qwen-VL 等

## 3 Method

### 3.1 Architecture

**Large-Scale Vision Encoder: InternViT-6B**

InternViT-6B 就是一个 vanilla ViT，只是将模型参数量扩展至 6B，为了在精度、速度和稳定性之间做好 trade-off，对 InternViT-6B 进行超参搜索，最终模型参数为：

<center>
    <img src="https://github.com/user-attachments/assets/1ee7344e-d59c-4439-98be-f5b2a1480c86">
</center>

**Language Middleware: QLLaMA**

QLLaMA 是 LLaMA 的多语言版本，在这里用来做 visual feature 和 linguistic feature 之间的对齐。在 LLaMA 权重的基础之上，额外添加了随机初始化的96个可学习 query 以及 cross-attention（共有10亿参数）。通过这种方式，QLLaMA可以将视觉特征平滑地整合到语言模型中，从而进一步增强了视觉特征与语言特征的对齐程度。

和其它常规的 adaptor （如 Q-Former、MLP）相比，QLLaMA 有三个优势：

1. 使用 LLaMA pre-trained weight 初始化，相比 Q-Former、MLP 的随机初始化具有更多的先验（但是 query 和 cross-attention 不还是随机初始化的么？这部分参数也很大）；
2. QLLaMA 也是 8B 的参数规模，比 Q-Former 大 42 倍，大模型可能带来更高的性能；
3. 对齐阶段的训练可以使用对比学习，这种方式可以模型做一些 zero-shot image classification 和 image-text retrieval 任务（其实就是 CLIP 的训练方式）

**"Swiss Army Knife" Model: InternVL**

<center>
    <img src="https://github.com/user-attachments/assets/59532d9d-99bd-4d42-b632-aadbade2b8de">
</center>

通过 vision encoder 和 language middleware 之间的灵活结合，InternVL 可以支持多种纯视觉或 vision-language 任务（前面说过太多次了:)）：

1. visual perception tasks：vision encoder 单独抽出来就是一个特征提取器，和传统视觉任务一样，接个 global average pooling 或 MLP 可以做图像分类；
2. contrastive tasks：得益于对比学习的对齐方式，InternVL 还可以用来做 image-text retrieval 任务，如上图的 (a)(b)；
3. generative tasks：和 QFormer 不同，得益于 scaled-up 参数量，QLLaMA 固有 image caption 能力；QLLaMA 的 queries 可以 reorganize the visual representations，然后作为 text 输入给 QLLaMA，随后生成文本；
4. multi-modal dialogue：多模态对话，这里有两种选择，其一是只利用 vision encoder 的特征作为 LLM 的输入，图(c)，另外一种是同时用 vision encoder 和 QLLaMA 的输出，图(d)。

### 3.2 Alignment Strategy

InternVL 将整个训练过程叫做 alignment，而其它 VLM 方法大多把训练 adaptor 的阶段叫做 align，注意这里的区别。所以，stage2 的图示可以看到，从 InternViT-6B 到 QLLaMA 是通过一个 cross-attention 连接的，这部分在其它 VLM 中叫 adaptor（emm...）。

<center>
    <img src="https://github.com/user-attachments/assets/60efad7c-3f33-4b92-8411-03f825b91f62">
</center>

InternVL 的训练包含三个渐进式阶段：

**Vision-Language Contrastive Training**

该阶段是在 web-scale、noisy image-text pairs 数据上，通过 contrastive learning 将 InternViT-6B 与 multilingual LLaMA-7B 对齐。数据集都是公开的，包括 LAION-en、LAION-multi、LAION-COCO、COYO、Wukong 等，将这些数据集整合之后过滤掉一些质量极低的数据，得到该阶段的训练数据，共 4.98B，详细信息如下：

<center>
    <img src="https://github.com/user-attachments/assets/2a695b47-e879-4f97-89cd-cb0f108310df">
</center>

该阶段之后，vision encoder 就可以作为一个视觉基础模型使用了。

**Vision-Language Generative Training**

该阶段，通过一个 cross-attention 将 vision encoder 与 QLLaMA 进行连接，其中 QLLaMA 是用第一阶段的 LLaMA 权重做初始化。

> 这里有点不懂，为什么要做第一个阶段将 vision encoder 与 LLaMA 对齐，难道只是为了可以做一些视觉任务？即便是先对齐 LLaMA，这里的 QLLaMA 应该也不可以任意替换其它模型吧，毕竟这里是用第一阶段的 LLaMA 来做初始化的

训练时，保持 vision encoder 和 QLLaMA frozen，仅训练额外新增的 learnable queries 和 cross-attention 层。

> 这里的 learnable queries 和 cross-attention 应该对应于其它 VLM 方法中的 adaptor（就是 QFormer？），所以，InternVL 本质并不是把 adaptor 给 scale up 了，而是在常规的 adaptor 后接了一个 LLaMA，这个 LLaMA 是在第一阶段训好的

这一阶段的训练数据是在第一阶段的数据基础之上进一步过滤，最终只有 1.03B。

和 BLIP-2 中的训练目标一致，该阶段的 loss 包含三个部分：image-text contrastive (ITC) loss、image-text matching (ITM) loss 和 image-grounded text generation (ITG) loss。

**Supervised Fine-tuning**

为了验证 InternVL 在多模态对话任务中的性能，通过一个 MLP 层将其与现成的 LLM decoder 进行连接，继续进行 SFT 训练。数据构成如下，大约共 4 million 数据：

<center>
    <img src="https://github.com/user-attachments/assets/6d6fe4cf-2d85-43d8-893a-78dd4ce3b153">
</center>

由于 QLLaMA 和 LLM 的特征空间相似（QLLaMA 的特征空间能和 Qwen 的相似吗？），即使冻结 LLM decoder，也可以实现稳健的性能，可以只训练 MLP 或者训练 MLP 和 QLLaMA，这种方法不仅加快了SFT过程，而且保持了LLM 的原始语言能力。

## 4 Experiments

实验部分给出了在 Visual Perception Benchmarks（image classification、semantic segmentation）、Vision-Language Benchmarks（image/video classification、image-text retrieval、image caption）、Multi-Modal Dialogue Benchmarks 多个任务上的效果。

下面给出一个和其它 VLM 方法对比：

<center>
    <img src="https://github.com/user-attachments/assets/1c4c647a-03aa-4fd8-9a5a-f034f8b87955">
</center>

一个注意的点是，用 MLP 替换 QLLaMA 貌似掉点并不多。

## Bonus

1. learnable query 具体是什么概念？如何实现？

   概念应该来自 Q-Former

2. The queries of QLLaMA reorganize the visual representations. 具体是什么操作？
