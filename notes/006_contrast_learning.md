# 对比学习论文串烧

[Talk](https://www.bilibili.com/video/BV19S4y1M7hm?spm_id_from=333.999.0.0)

四个阶段：百花齐放 -- CV双雄 -- 不用负样本 -- Transformer

## 第一阶段：百花齐放
------------

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

> 正样本为未来时刻的输入`x_t+n`通过编码器后得到的未来时刻的特征`z_t+n`。相当于上面预测得到的作为query，而真正未来时刻的输出是由这些输入`x_t+n`得到的。负样本的定义可以很广泛，比如可以任意选取的输入通过编码器得到的输出，都应该和你的预测是不相似的。

CPC思想比较普适，假如输入序列为text，任务可以是用前面的单词去预测后面单词的特征输出；如果输入序列是图像patch，从左上到右下形成一个序列，任务可以是用上半部分的图像特征去预测后半部分的图像特征。

## 4.CMC

[Contrastive Multiview Coding](https://arxiv.org/pdf/1906.05849.pdf)

CMC中定义正样本的方式更为广泛，它认为一个物体的很多个视角都可以被当作是正样本。

人是通过不同的视角来感知环境的（比如看、听、触等），而且每一个视角都是有噪声、不完整的，但是最重要的那些信息其实是在所有的视角中共享的。基于此，CMC是想学习的到一个具有视角不变性的特征，增大视角之间的互信息，可以抓到所有视角下的关键因素。

<img src="https://user-images.githubusercontent.com/22740819/148215374-a9befd0a-72a6-4877-8354-5ab835e7e7f8.png" width=300>

CMC基于NYU RGBD数据集，该数据集同时有4个视角，分别为原始图像、图像深度图、surface normal以及分割图像。这4个不同模态的视角所对应的都是同一张图片，它们可以互为正样本，在特征空间中四者应该比较近，而其它图片的任意视角都应该与之比较远。

CMC是第一个做多视角的对比学习，不仅证明了对比学习的灵活性，而且证明了多视角、多模态的可行性。

CMC的局限性：当处理不同的视角时，需要使用不同的编码器，因为每个模态的输入之间差异很大，导致有几个视角就需要几个编码器，使得训练阶段计算量很高

第一阶段介绍的四篇paper，它们的工作代理任务不尽相同，有个体判别，有预测未来，还有多视角多模态。使用的目标函数也不尽相同，有NCE，infoNCE以及其它变体。使用的模型也可以是不同的，比如InvaSpread使用的是相同的编码器，CMC使用多个编码器。

## 第二阶段：CV双雄

---------------

## 5.MOCO v1

<img src="https://user-images.githubusercontent.com/22740819/149867385-94b2687a-3757-470a-8e83-d2420d8cf3c1.png" width=400>

MOCO的主要贡献就是把之前对比学习的方法，都归纳总结为一种字典查询问题，它提出了队列与动量编码器两个东西，从而去形成一个又大又一致的字典，能帮助更好的对比学习。

## 6.SimCLR v1

<img src="https://user-images.githubusercontent.com/22740819/149864558-14dd4d7b-62ed-4c04-9886-abfbce8c2434.png" width=600>

主要流程：

- 对于一个mini-batch的图片`x`，经过两次不同的数据增强得到`x_i`和`x_j`，那么同一张图片增强得到的两个样本互为正样本。如果batch_size为n，那么正样本个数就为n，负样本个数为batch内剩下所有的样本以及它们增强后的样本，为2(n-1)个；
- 得到样本后，通过两个编码器`f`对样本进行特征提取，得到特征表示`h`，而这里的两个`f`是共享权重的，也就是说其实只有一个编码器；
- 得到特征表示`h`之后，接一个特征映射层`g`，也就是MLP，最终得到特征`z`，这里的MLP使得在ImageNet分类任务上提升了10个点；
- 得到特征表示`z`之后，利用一个normalized temperature-scaled交叉熵函数衡量正样本的相似度，normalized是说对特征做一个L2归一化；

> 其中的MLP `g`函数只有在训练的时候用，在接下游任务时仍然是使用`f`编码器得到的特征`h`，和其它对比学习方法对齐，加上投影函数的目的也仅仅是想让模型训练的更好

SimCLR和Inva Spread非常相似，主要区别是：

- SimCLR用了更多的数据增强
- SimCLR加了一个`g`函数(MLP)
- SimCLR用了更大的batch size，训练的时间更久

> 这些单一的技术之前就已被提出过，而SimCLR的成功之处就是能够将这些技术有效的结合，其使用的数据增强方式、MLP层、lars优化器用于大batch_size任务在后续的对比学习方法中也被广泛应用

消融实验：

- 对于数据增强，其中最有效的为crop以及color变换，其余的增强方式只是锦上添花，可有可无的
- 对于MLP，linear+relu优于只用linear，优于不使用mlp，使用mlp比不使用提了10个点，至今无法解释

## 7.MOCO v2

MOCO v2是一个只有两页的技术报告，它在MOCO的基础上借鉴了SimCLR中的MLP和数据增强。

<img src="https://user-images.githubusercontent.com/22740819/149939378-476b78de-2ce5-4616-a54c-ba3a3123e750.png" width=400>

MOCO v2相对于MOCO的改进点：

- 添加一个MLP层；
- 利用更多的数据增强；
- 使用cosine的lr衰减方式；
- 更多的epoch；

## 8.SimCLR v2

<img src="https://user-images.githubusercontent.com/22740819/149940841-269fc890-1806-436b-bde8-9e725ee4e548.png" width=600>

SimCLR v2提出了一套用自监督网络作半监督训练的流程。该流程是用大的网络（SimCLR v2）作自监督的预训练，预训练部分是没有特定下游任务的，因此不具备下游任务知识；之后使用少量有标注数据对模型进行微调，从而让模型学习下游任务的特定知识；然后让微调的模型作为teacher模型，为更多的数据打伪标签，从而实现自学习。

SimCLR v2相对于SimCLR的改进点：

- 使用更大的模型，resnet50 -> resnet152，并且使用selective kernels；
- 将一层MLP变为两层MLP；
- 使用动量编码器，只提升了一个百分点，原因是SimCLR v2已经有很大的batchsize了，负样本已经足够多，使用队列或动量所带来的收益不大；

## 9.SwAV

<img src="https://user-images.githubusercontent.com/22740819/149943488-d4c6a380-f421-4802-a3c1-2fa1eb9e9a8d.png" width=600>

之前的对比学习的方法都是将一个图片`x`通过两次数据增强变为`x1`和`x2`，然后通过编码器得到特征表示，然后通过度量特征的相似度来更新编码器。

而SwAV认为这种特征与特征作对比的方式比较原始，而且有些资源浪费，因为所有图片都自成一类。对MoCo来说，假设队列大小为65536，可理解成将整个数据集的负样本（128w）近似成了6w多个负样本，而SwAV则考虑是否可以利用一些先验信息不做近似呢？也就是说不去跟大量的负样本作对比，而是跟一些更简洁的东西比呢？SwAV的做法是去和图片的聚类中心比（右图中的`c`），聚类中心`c`实际是一个矩阵，维度为`d x k`，其中`d`为特征的维度（128），k为聚类中心的数量（3000）。

SwAV整体过程：给定一个mini-batch `x`，做两次数据增强得到`x1`和`x2`，分别通过编码器得到各自的特征`z1`和`z2`，得到特征之后并不是直接在特征上计算loss，而是通过一个clustering的方法，令特征`z`与prototypes `c`计算各自生成一个目标`q1`和`q2`，其中`q1`和`q2`相当于ground truth一样的东西。SwAV认为，如果`x1`与`x2`互为正样本，那么得到的特征表示`z1`和`z2`也应该很相似，那么两者理论上是可以互为做预测的，也就是拿`z1`与`c`做点乘是可以预测`q2`的，反之亦然，也就是说`z1`与`c`点乘结果可以去预测`q2`，其ground truth为`q1`，通过这种换位预测（swapped prediction）的方式更新模型参数，形成新的代理任务。

使用聚类的好处：

- 如果是和负例进行对比的话，需要和成千上万个负例进行对比，即使是MOCO中6w个负例，也只是一个近似的值，但是聚类的话，就仅仅需要和3000个聚类核心即可；
- 这些聚类中心是有明确的语义含义的，而如果像之前一样用负样本进行对比学习的话，有的还可能是正样本，有的负样本不均衡，不如聚类中心有效。

**Multi-crop: Augmenting views with smaller images**

SwAV之所以有如此好的效果，不仅仅是因为它和聚类方法融合到了一起，另外一个主要提升性能的方法来自于一个trick：multi-crop。

<img src="https://user-images.githubusercontent.com/22740819/150052992-5cb004cd-7bb7-43d8-bf63-7085e73d2b12.png" width=500>

之前的对比方法都是用的两个crop，给定一张图片`x`，首先resize至`256x256`，然后随机crop出两个`224x224`大小的图片`x1`与`x2`互为正样本，因为这两张图相对原图比较大，所以它们重叠的区域也非常多。SwAV认为，用如此大的crop，明显抓住的是整个场景的特征，如果更想学习这些局部物体的特征该怎么办呢？所以说最好可以使用多个crop。

但是，增加crop数量或者增加view会导致模型计算复杂度变高，相当于使用了更多的正样本。如何使用多个正样本，而又不增加太多的计算成本呢？首先取两个160大小的crop去学习全局特征，然后随机选取4个96大小的crop去学习局部特征，现在共有6个视角，而之前只有2个视角。这样在计算量变化不大的情况下，可以获取更多的正样本。

到了第二阶段，对比学习中很多细节都基本趋于统一了，比如目标函数都是使用infoNCE，模型都用一个encoder+projection head了，均采用一个更强的数据增强，都想用一个动量编码器，也都尝试训练更久。


## 第三阶段：不用负样本

---------------

## 10.BYOL

[paper](https://arxiv.org/pdf/2006.07733.pdf)

bootstrap: 你已经有什么东西了，然后在其上面进行改造
latent: 其实就是特征的意思（类似于hidden，feature，embedding）

**模型坍塌**

在对比学习中，负样本是一种不可或缺的约束。假如在计算loss时只有正样本，这时你的目的就只有一个，就是让所有相似物体的特征也尽可能的相似，那么就有一个明显的捷径解，就是无论你给这个模型什么输入，它都给你返回同样的输出，这样得到的特征就是一模一样的，然后计算loss时就都是0，模型直接用这个捷径解就可以完美解决你的问题。而只有加上负样本这个约束，不仅使相似的物体要有相似的特征，而且不相似的物体要有不相似的特征，这样模型才有动力去继续学，因为如果它输出的特征都一样，那么在负样本上的loss会趋向无穷大，只有在正负样本上的loss都往下降才能得到最优解。所以，负样本在对比学习里是一个必须的东西，防止模型学到捷径解，论文中经常将其称为model collapes（**模型坍塌**），说白了就是什么也没学到。

BYOL的神奇之处就是它没有使用任何形式的负样本，也得到了不错的效果。

<img src="https://user-images.githubusercontent.com/22740819/150060005-7cc25920-ad4b-46dd-b41f-30678a51e684.png" width=600>

**BYOL流程**

给定一个mini-batch的输入`x`，经过两次数据增强后得到`v`和`v'`，然后各自通过一个编码器`f_θ`和`f_ξ`（这两个编码器使用的是同样的网络架构，但是参数不同，`f_θ`是随梯度更新参数的，而`f_ξ`是使用moco中的方式动量更新的），分别得到特征`y_θ`和`y'_ξ`，然后各自再通过一个MLP `g_θ`和`g_ξ`（结构相同，参数不同，`g_ξ`为动量更新）得到特征`z_θ`和`z_ξ`，接下来就到了与之前对比学习不同的地方，BYOL中上面分支之后又接了一个`q_θ`（其结构和`g_θ`完全一样），得到一个新的特征`q_θ(z_θ)`，目标是想让该预测结果`q_θ(z_θ)`与`z_ξ`尽可能的一致。其实是把之前的匹配问题变成了预测问题。

**BN的必要性以及影响**

[Blog](https://generallyintelligent.ai/blog/2020-08-24-understanding-self-supervised-contrastive-learning/)

详细解释见talk 58:23处。

## 11.SimSiam

[Paper](https://arxiv.org/pdf/2011.10566.pdf)

- 不需要负样本；
- 不需要大的batchsize；
- 不需要动量编码器；

SimSiam不需要以上三个对比中常用的条件，不仅没有模型坍塌，而且取得了不错的效果。

<img src="https://user-images.githubusercontent.com/22740819/150077394-42f3b27b-e0c9-440f-8b84-af27d12903e5.png" width=500>

之所以叫simsiam network（孪生网络）是因为两个编码器`f`的结构一般是一样的，并且是共享参数的。

<img src="https://user-images.githubusercontent.com/22740819/150078411-d8b9271a-e01c-4859-805e-ec28562520d8.png" width=500>

整体架构与BYOL非常类似，前向过程是给定一个mini-batch图像`x`，经过两次数据增强变为`x1`和`x2`，之后经过孪生的encoder `f`，得到特征表示`z1`和`z2`，然后经过predictor得到`p1`和`p2`，之后用`p1`预测特征`z2`，用`p2`预测`z1`，这里与BYOL唯一的不同是它没有用动量编码器

之所以SimSiam能够成功训练，不会有模型坍塌，主要是因为有stop gradient这个操作存在，由于这个操作，SimSiam结构可以想象成一个EM算法。

<img src="https://user-images.githubusercontent.com/22740819/150078901-05a7b7aa-3fe1-408a-8878-c9954fd5284f.png" width=500>

SimSiam与其它基于孪生网络的对比：SimCLR使用的是端到端的训练，两个encoder；SwAV是和聚类中心进行对比；BYOL是一个预测任务，同时使用使用的是动量编码器；SimSiam也是预测任务，但是没有使用动量编码器，使用的是stop gradiant的方式进行预测的。

## 第四阶段：Transformer

---------------

## 12.MOCO v3

[Paper](https://arxiv.org/pdf/2104.02057.pdf)

MOCO v3其实就是做了一个很直接、很小的一个改动，让自监督的ViT训练变得更稳定。

<img src="https://user-images.githubusercontent.com/22740819/150080097-172ff404-9e67-4b9e-9e86-03b9b1c10934.png" width=500>

从伪代码看，MOCO v2相当于是MOCO v2与SimSiam的一个合体。MOCO v3还是有两个编码器`f_q`与`f_k`，其中`f_k`使用动量更新，目标函数使用的是对比学习的loss，从该角度看与MOCO v2类似，但从细节看，`f_q`是由backbone + project mlp + predictor mlp组成，这里其实就和BYOL或者SimSiam类似，而且loss的计算是一个对称项，很像SimSiam。

<img src="https://user-images.githubusercontent.com/22740819/150080902-6ebc2b1f-0419-4e26-9d51-f64c3cb9e5df.png" width=500>

从上图来看，把resnet50替换成ViT，当batchsize比较小时1024～4096，效果还可以，但是当batchsize比较大时（6144），训练将很不稳定，并且最终结果不如小batchsize。

针对以上问题，作者观察了一下模型梯度回传时候的梯度变化情况。当每次loss有大幅的震动，导致准确度大幅下降的时候，梯度也会有一个波峰，而且波峰发生在第一层，也就是在做patch projection时（也就是一个可训练的全连接层），由于这一层经常出现问题，所以作者尝试将这一层随机初始化然后冻住，不参与训练，得到了很平滑的loss曲线。该trick不仅对MOCO v3有用，对BYOL也有不错的效果。

