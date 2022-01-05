# 对比学习论文串烧

[Talk](https://www.bilibili.com/video/BV19S4y1M7hm?spm_id_from=333.999.0.0)

四个阶段：百花齐放 -- CV双雄 -- 不用负样本 -- Transformer

## 1.InstDisc

[Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/pdf/1805.01978.pdf)

<img src="https://user-images.githubusercontent.com/22740819/148052604-3be86864-4886-4172-9b43-c8a6660c3a37.png" width=400>

InstDisc借鉴了有监督方法。假设把一张豹子的图片送给一个已经用有监督方式训练好的分类器，得分较高的全都是和豹子相关的（猎豹、雪豹等），让这些相似图片聚在一起的原因并不是它们有相似的语义标签，而是因为这些图片本身就比较相似。基于此，作者提出了个体判别任务，将每一个instance都看作一个类别，目标是学习一种特征，可以把每一个图片都可区分开来。

<img src="https://user-images.githubusercontent.com/22740819/148043206-82ec4399-944e-4b9e-ab02-1e75d3de5e9c.png" width=600>

- 通过CNN将图片编码成一个特征，希望该特征在特征空间中能够尽可能的分开
- 以对比学习的方式训练该模型，正样本为图片本身或者增强后的图片，负样本为数据集里其它所有的图片
- 将样本所有的特征存到memory bank中，假如imagenet有128w图片，那么memory bank中就存128w个特征，考虑到存储成本，特征维度选择为128
- 前向过程：batch_size=256的图片通过cnn计算得到特征，正样本数量和batch_size大小一致，从memory bank中选取4096个样本作为负样本，计算infoNCE loss，更新参数。然后将batch中的样本替换掉memory bank中对应的样本。

InstDisc也是一个里程碑式的工作，它不仅提出了个体判别的代理任务，而且用这个代理任务和NCE loss去做对比学习，从而取得了不错的无监督表征学习的结果，同时它还提出了用memory bank这种结构存储大量的负样本，以及如何对特征进行动量的更新，对后续的对比学习工作起到了至关重要的推进作用。

## 2.InvaSpread

[Unsupervised Embedding Learning via Invariant and Spreading Instance Feature](https://arxiv.org/pdf/1904.03436.pdf)

该论文可以看作为SimCLR的前身，没有使用额外的数据结构存储大量的负样本，相应地是使用mini-batch中的样本作为负样本，而且只使用一个编码器进行端到端的学习。

<img src="https://user-images.githubusercontent.com/22740819/148056255-0c7ca030-75c5-4824-bbb8-56c067aeba96.png" width=400>

该论文的方法就是基于最基本的对比学习。同样的图片经过编码器之后得到的特征应该很类似，不同图片的特征应该不相似。也就是说对于相似的图片、物体，它们的特征应该保持不变型（Invariant），对于不相似的图片它们的特征应该分散开（Spreading）。

<img src="https://user-images.githubusercontent.com/22740819/148180616-06ecbd10-d8b0-4501-9b33-4a09317f6094.png" width=600>

- 代理任务：个体判别
- 正负样本的选取：batch内图片经过数据增强得到其对应的正样本，而负样本为batch内剩余的其它所有图片（包括原始图片以及数据增强后的图片）。假如batch_size=256，那么batch内正样本的数量为256，负样本的数量为(256 - 1) * 2。这种batch内选取负样本的方式可以使得用一个编码器进行端到端的训练。

该文章的思路和SimCLR的思路差不多，但效果不如SimCLR，主要还是字典不够大，batch_size仅为256，同时缺少SimCLR中多样的数据增强方式以及MLP模块。

## 3.CPC

[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)

以上两篇工作都是使用个体判别作为代理任务，而这篇文章是利用其他代理任务进行对比学习的。

机器学习中，一般可分为判别式模型和生成式模型。个体判别显然是属于判别式范畴，而本文中的预测性任务则属于生成式模型。

<img src="https://user-images.githubusercontent.com/22740819/148183946-7acb2447-2fe5-4478-8010-1e921fc3991c.png" width=600>

CPC中提出的模型是一个通用的结构，不仅可以处理音频，而且还可以处理图片、文本以及在强化学习中使用。

假设给定一个时序的序列X，其中`X_t`为当前时刻，`X_t-n`为之前时刻，`X_t+n`为未来时刻。把之前时刻的输入送给一个编码器`genc`，得到对应的特征`z_t`，然后将这些特征送给一个自回归模型`gar`（如lstm、rnn等），得到一个context representation `c_t`（代表上下文的一个特征表示）。如果这个上下文特征表示`cx`足够好（包含了当前和之前的所有信息），它应该可以做出一些合理的预测，比如未来时刻的特征输出`z_t+n`。其中对比学习体现在：

正样本为未来时刻的输入`x_t+n`通过编码器后得到的未来时刻的特征`z_t+n`。相当于上面预测得到的作为query，而真正未来时刻的输出是由这些输入`x_t+n`得到的。负样本的定义可以很广泛，比如可以任意选取的输入通过编码器得到的输出，都应该和你的预测是不相似的。

CPC思想比较普适，假如输入序列为text，任务可以是用前面的单词去预测后面单词的特征输出；如果输入序列是图像patch，从左上到右下形成一个序列，任务可以是用上半部分的图像特征去预测后半部分的图像特征。

## 4.CMC

[Contrastive Multiview Coding](https://arxiv.org/pdf/1906.05849.pdf)
