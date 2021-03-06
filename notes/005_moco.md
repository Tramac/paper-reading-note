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
> 在对比学习任务中，每个图片自成一类，对ImageNet来说就不再是1000分类，而是128w个类别，如果使用交叉熵函数作为监督信号，其中的K值为128w，而softmax操作在K很大时是无法工作的，同时还有exp操作，当向量的维度是几百万时，计算复杂度是相当高的，所以这里采用的是InfoNCE作为监督函数

<img src="https://user-images.githubusercontent.com/22740819/146709578-4cd42c01-bf49-4e1b-ab0e-a02861d9a893.png" width=200>

> 式中，K为队列中负样本的数量；τ为温度超参数，用来控制分布的形状，τ值大会使得分布更平滑，否则分布会更集中，如果τ值设置的比较大，对比损失会对所有的负样本一视同仁，导致模型的学习没有轻重，但如果τ值过小，又会让模型只关注那些特别困难的样本负样本，而这些负样本很有可能是潜在的正样本，会导致模型很难收敛，或者学好的特征很难去泛化

- query编码器与key编码器可以是相同的网络（参数也可以共享或部分共享），也可以是不同的网络

***2.Momentum Contrast***

如果想学习一个好的特征，MoCo中的字典必须具有两个特征：large（大）和consistent（一致性），因为字典大会包含很多语义丰富的负样本，有助于学到更有判别性的特征，一致性主要是为了避免模型学到一些捷径解

- Dictionary as a queue 如何把字典看作队列：MoCo方法的核心就是把字典用队列的方式表现出来，队列中的元素为key的表示，训练过程中，每一个mini-batch会送进来一批新的key，同时有一批老的key移出去。字典中的元素一直是整个数据集的子集，维护字典的开销也非常小
- Momentum update 动量更新：队列的方式使得字典可以非常大，但字典非常大时导致没办法对队列里所有的元素进行梯度回传，就是说key的编码器没办法通过反向传播更新参数了。假如把query编码器的参数拿过来直接给key的编码器（参数共享）以此来更新的话，结果会不好，因为query编码器的每个iteration都是更新的，如果key编码器每个iteration也更新会导致队列里的元素不具备一致性。所以提出了动量更新的方式：

<img src="https://user-images.githubusercontent.com/22740819/147742571-488c7331-4134-4353-852a-011b1f2e9c58.png" width=200>

> 其中，`θ_k`为key编码器的参数，`θ_q`为query编码器的参数，除了第一次更新时`θ_k`用`θ_q`来做初始化，之后都主要根据自己的参数来更新，因为`m`只是一个比较大的值（0.999），只有0.001的权重依赖于`θ_q`，所以`θ_k`更新的非常缓慢。这样就导致队列中的key虽然不同，但是由于各自的编码器参数区别比较小，所以产生的这些key的一致性还是比较强。

***3.伪代码***

<img src="https://user-images.githubusercontent.com/22740819/148006086-c26bf9db-487f-4c7d-ae89-f804f5392058.png" width=300>

***4.shuffling BN***

用过BN之后，很有可能造成batch内样本之间的信息会泄漏，因为BN要算这些样本的running mean和running vairance，而模型会通过这些泄漏的信息很容易找到那个正样本，导致最后学得的并不是一个真正好的模型。
> BN操作大部分时候都是在当前的GPU上计算的，在进行多卡训练之前，先把样本顺序打乱，再送到所有的GPU上，计算完特征之后，再把顺序恢复，然后去计算最后的loss。最终对loss的计算没有影响，但在每个GPU上的BN计算变了，就不会存在信息泄露的问题（？不太理解）

## Part7. 实验

***1.数据集***

预训练数据集ImageNet-1M、Instagram-1B

***2.Linear Classification Protocol***

冻结backbone的网络参数，只微调最后用于分类的fc层

<img src="https://user-images.githubusercontent.com/22740819/148007725-d566d912-728b-4740-817d-66abec431a26.png" width=400>

***3.动量参数的消融实验***

<img src="https://user-images.githubusercontent.com/22740819/148007774-0dbd774e-5595-41b9-9103-4048b7be2957.png" width=400>

***4.特征迁移效果***

无监督学习最主要的功能就是去学习一个可以迁移的特征

<img src="https://user-images.githubusercontent.com/22740819/148008336-870cee2c-6a32-4f5f-8d0d-8448363d8c1e.png" width=400>


## Part8. 总结

略
