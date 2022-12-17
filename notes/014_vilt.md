# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision

[paper](https://arxiv.org/pdf/2102.03334.pdf) | [talk](https://www.bilibili.com/video/BV14r4y1j74y/?spm_id_from=333.999.0.0)

## 前言

ViLT提出了一个极其简单的做多模态学习的框架结构，它把模态的特征抽取做到了极小化，把主要的计算量都放在了后面的模态融合部分，大大的提高了模型的推理速度，而且让整个方法的建模变得非常简单，在很大的程度上推动了过去一年多模态学习的进展。

它主要就是通过**移除多模态学习框架中的目标检测**（region feature）部分，极大的简化了整个学习框架。

<img width="530" alt="image" src="https://user-images.githubusercontent.com/22740819/207489981-1a984494-07ec-40e9-9f34-65be3f904f64.png">

如上图所示，通常的多模态学习框架中，对于text部分都是用一个linear embedding来得到word embedding，但是对视觉部分的处理非常不同，可主要分为三类：

- Region Feature：给定一张图像，通过一个CNN，然后再用一些ROI操作抽取得到区域性特征，相当于做了一个目标检测任务，得到的bbox可以想象成文本侧的token，组成序列；比如，图像中检测出100个物体，相当于100个token，然后于文本的token做后续的融合。其优点是能和文本侧一样有一个离散型的token（也就是bbox），缺点就是由于使用了目标检测，视觉处理耗时占比很大，如上图中整个pipeline耗时900ms，其中视觉的处理占了885ms。
- Grid Feature：使用CNN对图像进行特征提取，用最后一层的特征图作为融合部分的输入，比如7x7大小的特征图拉直之后就是49个token的特征序列。
- Patch Projection：对图像先分patch，然后通过linear projection得到图像的embedding。ViLT视觉部分耗时只有0.4ms，但是其训练耗时并没有缩短，需要64张32g的V100，训练3天才能完成训练；而且效果和之前的region feature方法相比也比较一般，其最大优势就是推理耗时的大大缩减。

## 标题&作者

Vision-and-language Transformer：使用transformer做多模态任务；

without convolution and region feature：并不需要使用一个CNN backbone抽取特征，也不需要通过目标检测提取区域特征；

## 摘要

Vision-and-Language Pre-training(VLP)视频文本预训练任务，非常依赖于视觉特征的抽取过程，比如region feature based方法先用目标检测提取区域特征，grid feature based方法也是使用了CNN来提取特征，总之都是需要一个额外的网络来抽取视觉特征。这两类方法主要有两个缺点：1）效率/速度比较低，抽取视觉特征的时间甚至比做多模态融合的时间还要多；2）受限于预训练模型抽取局部特征，表达能力有限，还是端到端的训练更好。

ViLT中视觉部分采用和文本部分同样的处理方式，只用了一个linear embedding，速度相比之前的多模态方法快了很多，而且性能在某些下游任务上也有提升。

## 引言

先pre-training再finetuning的范式也被拓展到了多模态学习领域，由于预训练的重要性，也催生了多模态预训练的研究工作。

一般的，多模态方法都是采用图像-文本对作为预训练数据，以图像-文本匹配和masked language modeling作为学习目标，然后再在一些下游任务上进行微调。

具体的，文本通常就是采用transformer结构，而对于图像来说，如何将图像pixels转换成具有语义的离散型的特征（类似于文本token）送给transformer是至关重要的。

在ViLT之前，对于图像的处理通常是使用一个目标检测器进行目标检测，优势是：1）得到明确的离散化高语义特征(bounding box)；2）VQA、image caption、image-text retrieval等这些下游任务往往都是和物体有非常直接的联系。大多数的VLP模型都是采用的在**Visual Genome**数据集上预训练好的目标检测器，该数据集包含1600类物体、400个属性，主要是类别够丰富。这类方法的缺点就是**抽取特征成本太大**。

为了降低视觉特征抽取的计算成本，Pixel Bert方法是使用了一个在ImageNet上预训练好的ResNet对图像进行特征提取，然后把最后得到的特征图当作是一个离散的序列，和**vit hybrid**类似，这样计算量就只剩下backbone，没有了检测相关的部分（roi，nms）。

大部分的VLP方法都还是聚焦于如何提高视觉编码器的性能来使提升最后表现，不太关注运行效率的问题。而且对于region feature方法来说都是提前把目标特征抽好存在本地硬盘，抽视觉特征的过程是离线的，训练时直接拿来用，还是比较轻量的，但是做推理时就无法避免。

ViLT借鉴了ViT对图像的处理方式，通过将图像切分patch，然后利用一个linear projection得到图像embedding，从而取消掉繁琐的图像特征提取过程。在一定程度上，ViLT算是ViT在多模态领域里的扩展。

ViLT主要贡献：

- ViLT是迄今为止最简单的用来做vision-language的模型，这种设计显著降低了运行时间和参数量；
- 在减少计算复杂度的同时，还能保证性能不掉；
- 训练时用了更多的数据增强方式；

## 背景知识

**Vision-and-Language Models分类**：

1）图像和文本的表达力度是否平衡，比如参数量与计算量是否相当；

2）图像和文本两个模态如何做融合；

根据上面的方式，众多vision-and-language模型可分为以下四类：

![image](https://user-images.githubusercontent.com/22740819/207830401-145ea70f-91e6-4a25-b9bf-37414fef88f4.png)

（a）text embedding与模态融合都比较轻量，visual embedding很重，代表工作为VSE；

（b）text和visual embedding都比较重，只有模态融合部分比较轻量，代表作为CLIP，特别适合做抽特征、retrieval 的任务，不太适合做VQA等任务，因为融合做的不够充分；

（c）text比较轻量，visual embedding比较重，通常使用目标检测，而且做模态融合时计算量也比较大，又使用了一个transformer，性能也是最好；

（d）ViL指出，多模态任务中，特征融合部分一定要充分；

**模态融合方法**

1）single-stream approaches：只用一个模型处理两个输入，可以先把两个输入concat；

2）dual-stream approaches：用两个模型先各自处理各自的输入，后续再做融合；

由于dual-stream参数更多，ViLT还是选择了single-stream的融合方法。

**视觉特征抽取方法**

对于大多数的VLP方法，文本侧的处理一般都是使用预训练好的bert里的tokenizer，不同的是视觉特征抽取：

1）region feature：backbone抽取特征 -- RPN获得候选框 -- NMS过滤得到最终bounding boxes -- RoI head将bbox映射得到一维向量，最终得到了region feature；

2）grid feature：使用一个CNN backbone提取特征，将最后一层特征图拉直后作为embedding；

3）patch projection：先分patch，再通过linear projection映射得到embedding；

## 模型方法

<img width="700" alt="image" src="https://user-images.githubusercontent.com/22740819/208019759-05b7e9b4-826a-4f88-a8cf-7b395f512f9c.png">

**模型结构：**

ViLT是一个single-stream方法，只用一个transformer encoder做模态融合。

文本侧是通过一个预训练好的bert tokenizer得到word embedding，假设文本序列长度为L（token数量），embedding的维度为H（一般为768），那么text embedding就是一个LxH的矩阵。

图像侧是先切分为patch，通过linear projection后得到visual embedding，假设有N个patch，每个patch embedding维度为H，那么visual embedding就是一个NxH的矩阵。

Modal-type embedding指定了当前token是哪种模态（0为文本，1为图像），维度为1xH；对于single-stream方法来说，两个模态的输入是拼接后送入一个transformer的，所以需要告知encoder对于拼接的输入哪部分分别是什么模态。

Token position embedding与Patch position embedding都是位置的编码，从1～N。

实际操作时，Modal-type embedding、Token/Patch position embedding与Word/Patch embedding三者是直接相加起来的，而不是concat；而text embedding与visual embedding两者作为整体是concat起来的，然后送给transformer，序列维度为 (N + L + 2) x H。

**loss:**

- Image Text Matching：成对的text-image为正样本，随机配对的为负样本，这里只用CLS token的输出做计算；
- Masked Language Modeling：专门针对文本的loss，和bert中的相同；
- Word Patch Alignment：使用了optimal transports最优传输理论，简单来讲就是将文本的输出和图像的输出当作是一个概率分布，计算两个分布的距离，对于成对的输入来说距离越近越好；

**两个细节：**

- Whole Word Masking：整个词做mask。以词汇“giraffe”（长颈鹿）为例，使用tokenizer会将其分为词根 ["gi", "##raf", "##fe"] ，如果只把中间的token做mask，["gi", "[MASK]", "##fe"]，其实以gi为开头fe为结尾的词汇并不多，这就会导致模型很容易重建出giraffe，那么在做多模态任务时，可能不需要借助图像的信息就可以重建出来，这个loss就失去了意义，可以理解为学到的是short-cut解。为了避免这种情况，ViLT是把整个词汇做mask，这样句子中就没有giraffe，如果想重建出来肯定需要图像的信息，以此来加强文本于图像的联系。
- Image Augmentation：之前的一些借助于目标检测的多模态学习方法，由于视觉特征离线提取的，没办法做数据增强。而ViLT既然做到了端到端的学习，是可以做一些数据增强的。但是，对于比如color jitter增强方式，对于颜色的改变可能导致与文本不匹配，比如文本为“绿色的草地”，如果图像变成了红色则会引入噪声。所以，ViLT所采用的RandAug中舍弃了color jitter与cutout，最大限度保证了图像和文本可以匹配。

## 实验

**数据集**

<img width="392" alt="image" src="https://user-images.githubusercontent.com/22740819/208236145-963b65f5-e76b-4ce0-94c0-21cf91c06b49.png">

ViLT在预训练阶段采用了以上4个数据集：

- MS COCO：每个图片对应5个caption，虽然只有10w张图，但是有50w image-text pair对，文本长度平均有11个词；
- VG：10w张图片，500w image-text pair对，文本比较短平均5个词；
- GCC：300w张图片，每张图片只对应一个caption；
- SBU：80w张图片，每张图片只对应一个caption；

以上4个数据集合起来通常被成为**4 Million**（4个数据集的图片数量）。

**实验对比**

- 分类任务

<img width="392" alt="image" src="https://user-images.githubusercontent.com/22740819/208237828-ec866727-13b3-4861-a212-fc03c6811ba8.png">

从推理耗时来看，ViLT有明显的优势，大概需要～15ms，从性能来看ViLT还是不如region feature方法，只是速度-性能比较平衡。

- Retrieval 任务

<img width="804" alt="image" src="https://user-images.githubusercontent.com/22740819/208237944-d5811131-b17c-4cc2-8de8-89a3da8fd20e.png">

表3为one-shot结果，即没有在下游任务的数据集上进行finetune，表示进行了finetune。在检索任务上的结论与分类任务类似，ViLT速度快，但是性能不如region feature方法。

- 消融实验

<img width="803" alt="image" src="https://user-images.githubusercontent.com/22740819/208238061-3bf99d8d-b27c-4c78-8892-7634c0f0597e.png">

从消融实验结果来看，训练步数越高，性能也会提升；whole word masking有提升但比较小；图像做mask并重建效果并不好；图像侧数据增强很有通，提升明显；

## 结论与展望

本文提出了一个非常轻量的多模态预训练方法-- ViLT，没有使用任何的CNN backbone提取视觉特征。性能虽然不是最好，但是速度快，可作为一个不错的baseline。

几个扩展方向：

- Scalability：模型尺度，越大越好；
- Masked Modeling for Visual Inputs：视觉部分的mask方式，图像重建辅助任务；
- Augmentation Strategies：数据增强策略；