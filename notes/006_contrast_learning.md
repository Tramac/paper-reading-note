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

该论文的方法就是基于最基本的对比学习。
