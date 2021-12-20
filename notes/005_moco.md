# Momentum Contrast for Unsupervised Visual Representation Learning

[Paper](https://arxiv.org/pdf/1911.05722.pdf) | [Talk](https://www.bilibili.com/video/BV1C3411s7t9/)

## Contrast Learning基本概念

- Pretext task: 代理任务，可以是人为设定的一些规则，这些规则定义了哪些样本是相似的哪些样本是不相似的，从而可以提供一个监督信号去训练模型，也就是所谓的自监督训练
- Instance discrimination: 一种pretext task，假设从数据集中随机选取一张图片`X_i`，然后做两次数据增强得到`X_i1`、`X_i2`，因为两者来自同一张图片，语义信息不应该发生变化，就可以被当作正样本，而数据集中其它图片都可以被当作负样本。从这个例子可以看出instance discrimination的含义，个体判别，因为在这个代理任务中，每个图片自身就是一个类别
- NCE loss: 对比学习中一种常用的损失函数

## Part1. 标题&作者

使用动量的对比学习方法去做无监督的视觉表征学习

- Momentum Contrast: 动量对比学习的方法；动量在数学上可以理解为一种加权移动平均，`y_t = m * y_t-1 + (1 - m) * x_t`，可以理解为*不想让当前时刻的输出完全依赖于当前时刻的输入*，所以也上之前的输出参与其中
> 其中，m为动量的超参数，y_t-1是上一个时刻的输出，y_t是当前时刻想要改变的输出，x_t是当前时刻的输入

## Part2. 摘要

- 本文提出MoCo方法去做无监督的视觉表征学习
- 本文将对比学习看作是一个字典查询（dictionary look-up）的任务，具体来说是做了一个动态的字典，字典由一个队列queue和移动平均编码器moving-averaged encoder
- 队列里的样本不需要做梯度回传，可以放很多负样本；移动平均编码器是想让字典里的特征尽可能的保持一致
- MoCo自监督预训练可以获得比同模型有监督预训练更好的结果

## Part3. 引言

- GPT和BERT已经证明了无监督学习在NLP领域已经获得了成功，在CV领域有监督预训练还是占主导地位
> 两者的差异可能是它们所输入的信号空间不同，NLP中是离散的信号空间（比如词、词根词缀），从而可以建立一个字典空间将每个词映射成对应的特征，容易建模；CV中的输入是连续的、高维空间的信号，不像单词一样有很强的语义信息
- 目前基于对比学习的无监督方法基本都可以归纳为一种方法：动态字典dynamic dictionaries（关于动态字典的解释建议直接看视频，13:40处）
- 动态字典的方式做对比学习需要做到两点：字典尽可能大（字典越大，样本越多，含有的视觉信息越多，越能学到可区分的特征）；一致性（字典中的特征都应该用相同或相似的编码器获得，这样才能保证它们和query对比时尽可能的保持一致，否则如果特征是通过不同编码器得到的，query很有可能找到的是和它是同一个编码器得到key，而不是包含相同语义信息的key）
- MoCo与一般对比学习的框架的异同，队列queue和momentum encoder（建议直接看视频，19:15处）
- MoCo只是针对对比学习建议了一种动态字典的训练机制，可以任意选pretext task（代理任务），本文选的是instance discrimination

## Part4. 结论

- MoCo将数据集由ImageNet-1k换成Instgram-1B时，提升比较小，大规模的数据集并没有很好的被利用起来，可能需要更换代理任务解决
- MoCo是否可以采用masked auto-encoding的代理任务（MAE，大佬两年前就已经开始挖坑了）

## Part5. 相关工作

无监督/自监督学习一般有两个方向可以做：1.代理任务(pretext task)；2.损失函数

***1.Loss Function***

- 生成式目标函数：衡量模型的输出和target之间的差距（新建图与原图之间的差异），比如L1、L2 loss
- 判别式目标函数：假如将图像分为九宫格，给定中间位置信息和任意一块信息，预测这两个块的方位（转换成了分类问题），这就是一种判别式网络的做法
- Contrastive losses: 在一个特征空间里，衡量各个样本之间的相似性，相似特征的目标尽量近，否则尽量远
> 与生成式和判别式网络的目标函数的区别：生成式和判别式的目标都是固定的，而对比学习的目标函数是在训练过程中不断改变的（在训练过程中，目标是由一个编码器提取出来的特征而决定的）
- Adversarial losses: 主要衡量的是两个概率分布之间的差异，主要用来做无监督的数据生成的任务

***2.Pretext Tasks***

- denoising auto-encoders: 重建整张图
- context autoencoders: 重建某个patch
- colorization: 给图片上色做监督信号
- form pseudo-labels: 给数据伪标签

不同的代理任务是可以和某些Contrastive losses配对使用的，比如MoCo使用的instance discrimination和NCE

## Part6. MoCo方法

***1.Contrastive Learning as Dictionary Look-up***

目前的对比学习都可以看作是训练一个编码器，从而去做一个字典查找的任务

假设，给定一个编码后的query特征`q`，和一系列编码好的样本`{k_0, k_1, k_2, ...}`，这些可以看作是一个字典里的key，而且在这个字典里只有一个key和query是配对的（互为正样本对），记为`k_+`

- 对比损失应该所具有的性质：当`q`和唯一正样本`k_+`相似时，loss值应该比较低，当`q`与其它不相似的样本距离比较远时，loss值也应该比较低
- InfoNCE loss: noise contrastive estimation，把多分类问题简化成一系列二分类问题，其中一个是数据类别data sample，另一个是噪声类别noise sample，每次去拿数据样本和噪声样本做对比（**建议反复看视频，33:58～41:23**）

<img src="https://user-images.githubusercontent.com/22740819/146709578-4cd42c01-bf49-4e1b-ab0e-a02861d9a893.png" width=200>


## Part7. 实验

## Part8. 总结
