# CLIP

[Paper](https://arxiv.org/pdf/2103.00020.pdf) | [Talk](https://www.bilibili.com/video/BV1SL4y1s7LQ)

CLIP自提出以来就引起很大关注，其方法简单但又效果很好。CLIP的迁移学习能力非常强，预训练好的模型能够在任意一个视觉分类的数据集上取得不错的效果，而且是zero-shot的，即它完全没有在这些训练集上做过训练，就能得到很好的效果。文章在30多个数据集上做了测试，包括了OCR，视频动作检测，坐标定位以及一些细分类任务，涵盖面非常广，尤其是在ImageNet上的结果，CLIP在不使用ImageNet训练集的情况下，就可以取得和之前ResNet50在有监督情况下的结果。

## Part1.标题&作者

Learning Transferable Visual Models From Natural Language Supervision：利用自然语言处理来的一些监督信号，去学习一个可以迁移效果很好的视觉模型。

Transferable: 迁移性能好

## Part2.摘要

- 目前的视觉模型训练通常是使用提前定义好的一些物体类别的集合，然后模型通过去预测这些定义好的类别，从而完成模型的训练。提前定义好的类别集合：比如ImageNet 1000类，cifar10 10类，coco 80个类，cityscapes 19个类，kinetics 400个类等。这种提前定义好的样本集合可以大大简化模型学习的难度，但是也限制了模型的泛化性
- 直接从自然语言里获取一些监督信号，只要是语言描述过的物体，就有可能让视觉模型去识别到这个物体，而不仅仅是提前定义好的那些固定的类别。
- 本文证实了用一个简单的预训练任务，就可以高效的、可扩展的去学习到最好的图像表征。具体的，给定一些图片和一些文本，模型去判断图片和哪些文本是配对的。既然要做的是配对的任务，那么所需要的训练数据就应该是配对的，本文爬取了一个4亿大小的图片、文本对，然后用自监督的方式去预训练模型。
- CLIP是利用多模态的对比学习去完成模型预训练，然后利用自然语言去引导视觉模型去做物体分类。分类也不再局限于已经学到的视觉概念，还能扩展到新的类别，也就是这个预训练模型可以直接在新的下游任务上做zero-shot推理的。

## Part3.引言

- 直接从文本数据里去预训练一个模型已经在NLP里取得了革命性的成功，比如BERT、GPT等，不论是使用自回归预测的方式还是使用掩码完形填空的方式，都是一种自监督的训练方式。所以，它的目标函数是和下游任务无关的，只是想通过预训练得到一个泛化能力好的特征，这套框架基本都是"text-to-text"，即文字进文字出，其模型架构也是跟下游任务无关的，所以在处理下游任务时，无需再针对下游任务去设计head或者其它特殊处理。比如GPT3，可以做分类、翻译、生成等。
- VirTex、ICMLM、ConVIRT均是基于transformer的工作，不同的是VirTex是用自回归预测的方式做预训练，ICMLM是利用完形填空的方式，ConvVIRT与CLIP非常相似，但是只在医疗图像上做的实验。三者在模型或数据上都没有取得很大的规模。
- 之前的一些引入自然语言信号的弱监督预训练方法效果不好，基本都是因为训练数据量级小，而CLIP收集了400M（4亿）对图像/文本对作为训练样本。模型上从ResNet50、EfficientNet到Vit-L，在大数据大模型的加持下，CLIP在多个数据集上取得很好的效果。
- 迁移学习的效果基本是和模型的大小呈正相关的，模型逐步增大，迁移学习的效果也在逐步增高，是一个比较平滑的过程。
- 作者为了进一步验证CLIP模型学到的特征的有效性，做了类似对比学习中的linear-probe验证，效果也比之前在ImageNet上有监督的结果好，而且泛化性能更好。

## Part4.模型方法

**4.1 利用自然语言信号训练视觉模型的优点**

- 不需要再去标注数据，直接抓去图片/文本对即可，很容易把数据规模做大
- 监督信号是一个文本，不再是N选1的标签，模型的输入输出的自由度就会大很多
- 训练时，将图片与文本绑定在一起，学得的特征不再是单单的视觉特征，而是一个多模态特征，很容易去做zero-shot迁移学习

**4.2 常见的一些图片/文本数据集**

- MS-COCO
- Visual Genome
- YFCC100M

前两者数据标注质量高，但是规模小，约10w数据；YFCC100M虽有1亿数据，但是标注质量差，清洗后的规模只有1500w。

CLIP收集了一个4亿规模的数据集--WebImageText。

**4.3 预训练方法的选择**

***将预测性任务改为对比型任务***

视觉模型的训练代价一般都比较高，模型大，计算量高。如果引入自然语言信号，而且数据集更大，任务更难，代价将会更大。

CLIP首先尝试类似VirTex的方式，对于图像使用一个CNN，对于文本使用一个Transformer，均是从头开始训练，任务是给定一张图片，然后预测图片所对应的文本，此为预测性任务。

> 预测性任务的缺点：对于给定图片，去预测其文本描述，需要逐字逐句的去预测该文本，这个任务相对来说就太难了。因为，对于一张图片可以有很多不同的描述，而且文本描述之间可能差距很大，如果用预测性任务去预训练模型，就会有太多的可能性，导致模型的训练非常困难。
> 
> 对比学习的优点：对比学习任务只需要判断图片和文本是否配对，任务简单的很多，就不需要逐字逐句的预测文本了，约束放宽了很多，该监督信号也更加的合理。

<image src="https://user-images.githubusercontent.com/22740819/153564599-f9aabcd8-ba05-4c5c-a830-99977ba740a0.png" width=400>

CLIP将预测性任务替换为对比型的任务，训练效率提高了4倍。

***伪代码***
  
<image src="https://user-images.githubusercontent.com/22740819/153567259-8f7d45d3-15f2-4c7f-a622-ecf1ccd6bb3e.png" width=400>

- 给定一个mini-match的图片输入I与文本输入T，分别通过一个编码器得到对应的特征I_f与T_f；
- 对于两个特征分别通过一个投射层，然后再做L2归一化，得到特征I_e与T_e；
- 对两个特征计算余弦相似度，最终计算loss。

***一些细节***

- CLIP由于使用的数据集特别大，所以在训练时不存在over-fitting的问题，图片编码器与文本编码器都不需要提前进行预训练；
- 最后对特征进行投射时，并没有使用非线性投射层，使用的而是线形投射层，两者对效果的影响不大。猜测非线性投射层只对纯图片的单模态任务比较重要；
- CLIP使用的数据增强只有随机裁剪；
- 由于模型、数据规模很大，调参成本太大，CLIP在训练时将对比学习中的temperature参数设置成可学习的标量，该参数在训练过程中直接被优化了。

**4.4 训练**

- 对于视觉侧共训练了8个模型，包括5个ResNet与3个Vision Transformer，所有这些模型使用Adam训练32个epoch；
- 调参时，只使用ResNet50训练1个epoch进行对比；
- 训练时，batch size为32768，非常大；
  
## Part5.实验

**5.1 Zero-Shot Transfer**

之前自监督或无监督的方法，它们主要是研究的是特征学习能力，目标是学得一种泛化能力比较好的特征，比如MoCo，SimCLR等。即便有了好的特征，当应用到下游任务时，还是需要有标签的数据进行finetune，