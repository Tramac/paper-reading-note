[Paper](https://arxiv.org/pdf/2012.06567.pdf) | [Talk 上](https://www.bilibili.com/video/BV1fL4y157yA/?spm_id_from=333.788)

<img width="800" alt="image" src="https://user-images.githubusercontent.com/22740819/166135532-db529b5f-96ab-4646-8d7d-baf0ebf10d22.png">

该Paper是一篇视频理解领域的综述论文，大概提及了超过200篇视频领域的论文，除video transformer之外，基本包含了之前用deep learning去做video action recognition的大部分工作。
Talk大概以上图中的时间线结构，从最开始的用卷积申请网络做DeepVideo的工作，然后到双流网络(Two-Stream Network)以及双流网络的各种变体(TDD、LRCN、Beyond-Short-Snippet、Fusion、TSN、TLE等)，
还有3D网络(C3D)以及3D网络的一些变体(LTC、I3D、P3D、Non-Local ～ ECO等)，以及最新的一些基于Video Transformer的工作。

大概分为以下四个部分：从Hand-crafted到CNN -- Two-Stream -- 3D CNN -- Video Transformer

## Part1.从Hand-crafted到CNN

[DeepVideo]()

DeepVideo是AlexNet出现之后第一个用比较深的卷积神经网络做大规模视频分类的工作。作者团队对视频理解领域做出了很大的贡献，不仅包括文章中提出的Sport-1M数据集，在两年之后还提出了一个更大的视频数据集Yotube-8M数据集，
还有最近为了action detection提出的AVA数据集，全都来自于该团队，在很大程度上都推动了视频理解领域的发展。

<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/166136129-07d173c8-1c33-4666-945d-e7b20cc951e8.png">

论文中的方法比较直接，想法就是如何把卷积神经网络从图片理解领域应用到视频理解领域。视频与图片唯一的不同就是它多了一个时间轴，有更多的视频帧而不是单个的图片。所以，自然的有上图中后三种融合方式。

- Single Frame: 就是单帧图像方式，其实就是图片分类任务，在视频中随机选取一帧，经过卷积神经网络得到分类结果；
- Late Fusion: 在网络输出层面做融合，例如在视频中随机选取几帧，然后每帧都单独通过一个CNN（权值共享），将得到的特征融合，再通过一个FC得到最终输出；
- Early Fusion: 在输入层做融合，比如将5帧视频帧在RGB的维度上直接合并，原来一帧RGB图像3个通道，现在5帧图像融合后就是15个通道，这也就意味着网络结构也要进行一定的改变，尤其是第一层，第一个卷积层接收的输入的通道数就由原来的3变成15，之后的结构都可以与之前保持一致；这种做法从刚开始就能从输入的层面感受到时序的改变，希望能够学到全局的运动和时序信息；
- Slow Fusion: 该方式是Late和Early的结合，也就是说Late合并的太晚而Early又合并的太早；每次选一个有10个视频帧的一小段视频，然后每4个视频帧通过一个卷积神经网络提取特征，刚开始的层都是权值共享，抽出特征之后将4组特征合并成两组，然后再经过一些卷积操作提取更深层特征，再将两组特征合并成一个特征，再通过卷积和FC得到最终输出；

这几种方式效果都差不多，而且在大数据集上预训练然后在UFC101上迁移，其效果还不如手工特征。

由于以上结果并不是很好，作者做了一些其它尝试：

<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/166246776-62d87fac-85d6-4da3-a080-869a3d9aa71b.png">

以上发现2D卷积神经网络无法很好的学得时序信息，干脆就先不学了，然后把图像领域使用卷积神经网络一些好用的trick拿过来，看能不能在视频领域也可以work，于是有了上面这种多分辨率网络。
简单来说，就是将输入分为两个部分，一个是原图，另一个是从原图的正中间抠出来一部分作为输入，因为不论是对图片还是视频来说，一般最有用的物体都会出现在中间部分。上面的分支叫做fovea（人眼视网膜中最中心的东西） stream，下面的分支叫做context（图片整体信息） stream，想通过此操作既能够学得图片中最中心的最有用的信息，又能够学得图片的整体的理解。该结构也是一种双流结构，并且两个分支是权值共享的，也可以理解成早期的一种使用注意力的方式，它强制性的想让网络关注中心的区域。

<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/166403459-7bb84875-5c98-4009-8a98-8a522d8dfed7.png">

从结果来看，相比于Single Frame，多分辨网络有一些提升但是不明显，另外Early Fusion和Late Fusion效果都不好，只有Slow Fusion的效果相对好一点。

<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/166403644-aeff1a2d-6a43-444e-88e9-4f4a2e50bbcd.png">

模型在UCF101上的效果也不好，最好的才65.4%，而手工特征的准确可以达到87%。虽然该论文的方法效果并不是很好，但是对后续视频理解的发展做了很好的铺垫。

## Part2.Two-Stream

从DeepVideo方法可以看出，2D网络不能很好的学得时序信息，而视频理解相对于图片理解最关键的区别就是多了一个时间维度，如何更好的处理时序信息，是效果提升的关键。

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/166419450-68eddd82-c0a2-4985-9123-15d5728f5314.png">

双流网络的意义是说，当用一个卷积神经网络无法很好的处理时序信息时，只需要再加一个卷积神经网络去专门处理这个时序信息。从原始视频中抽取光流图像，光流图像中包含了丰富的物体运动信息和时序信息，空间流网络学习从RGB图像到分类的映射，时间流网络学习从光流图像到分类的映射。这两个分支网络都不用去学习时序信息，只需要去学习对应的映射就可以了，而且两个网络各司其职，互不干扰，相对而言也比较好优化。

双流网络的四个改进方向：

- Fusion方式：双流网络是在softmax之后做的fusion，属于Late Fusion，其中一个研究方向是如何做Early Fusion；
- Backbone的使用：文中使用的是AlexNet，后续还有工作使用InceptionNet，VGG，ResNet等，尝试更深的网络对视频理解也是一个研究方向；
- 双流网络是直接把RGB图像和光流图像提取特征之后去做分类，事实上RNN和LSTM是专门用来处理时序信息的，现在视频帧上抽特征，然后在特征上加一个LSTM将帧之间的时序信息模拟出来，最后得到的特征应该会更强；
- 无论使用单帧图片还是多帧光流图像，其实都非常短，所覆盖到的视频片段也比较有限，对于一个帧率为25的视频来说，即使是10帧图像也才对应不到0.5s的片段，但是一般一个动作或者事件，它的持续时间至少也是2～3秒，如何做长时间的视频理解也是一个研究方向。

## Part3.Beyond-Short-Snippets （使用LSTM做特征融合）

[Paper](https://arxiv.org/pdf/1503.08909v2.pdf)

Beyond Short Snippets: 不想再做特别短的视频理解了，short snippets是指2～3的视频段。

<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/167277944-d749971c-fada-4871-ab5c-cbeb2ff92abe.png">

如果按照双流网络的思想去做，原始图像只有一帧或几帧，光流图只有10帧，那它能处理的视频其实是非常短的。如果现在有特别多的视频帧应该如何处理呢？

一般来说，对于这些所有的视频帧，我们都先用一种方法去抽取特征，比如原始的手工特征SIFT特征，或者CNN特征，关键是抽取特征之后如何去做pooling操作。

<img width="323" alt="image" src="https://user-images.githubusercontent.com/22740819/167278090-f554d0a3-51d4-4e55-be6a-236460aa5dfe.png">

该论文尝试了max pooling，avg pooling，conv pooling以及late pooling，early pooling等，最后发现效果其实都差不多，conv pooling相对好一些。

<img width="325" alt="image" src="https://user-images.githubusercontent.com/22740819/167278116-359427ee-ecfb-4c41-9394-c7b3305b9d28.png">

另外还尝试了使用lstm去做特征融合，理论上lstm能够掌握时间上的变化，应该是一种更好的融合方式，但是从结果来看提升也比较有限。图中的C代表每一帧的特征，这些特征在时序上是有顺序的，所以才会想到用LSTM去做特征融合，文中使用了5层lstm去处理特征，图中橘黄色部分代表softmax操作，之后再做分类。

该改动相对于双流网络是比较直接的，前面的操作都没有改变，只是在最后一个卷积层之后，不再是简单的做一层FC操作去做分类，而是把所有的视频帧特征通过一些LSTM层去做一些时序上的处理和融合，最后再做分类。

<img width="666" alt="image" src="https://user-images.githubusercontent.com/22740819/167278296-3bd39817-3c06-4624-859f-f52ee0d8cfcd.png">

从Sport-1M数据上的结果来看，本文提出的Conv Pooling和LSTM能够接收更多的视频帧，而且效果也相对更好（主要是使用光流和使用更多的帧带来的收益），其中Slow Fusion就是DeepVideo方法。只是对比Conv Pooling和LSTM融合方式其实差别不大，说明LSTM对时序的利用上没有想象的好。

<img width="324" alt="image" src="https://user-images.githubusercontent.com/22740819/167278373-d002f3ca-7c2c-4b32-ae4f-809eaf0b331a.png">

在UCF101数据上，手工特征准确为87.9%，DeepVideo只有65.4%（因为该方法没有很好的利用时序信息），Two-Stream因为使用光流效果达到88.0%，该论文方法因为完全是在双流网络的基础上做的，其效果要比双流网络的好才行，事实上，当使用120帧的Conv Pooling是准确也只有88.2%，只比88.0%高一点，使用LSTM时用了30帧，效果也只有88.6%，所以在UCF101这样的只有6～7秒的短视频上，LSTM的提升是比较有限的。

**为什么擅长时序信息提取的LSTM并没有带来明显的提升？**

> 一种解释是，像LSTM这种操作，它学的是一个更high level，更有语义信息的特征，给LSTM的输入必须得有一定的变化，LSTM才能起到它应有的作用，它才能学到时序上的改变，但是如果视频比较短比如只有5～7秒，这个语义信息很可能根本就没有改变，你从视频中抽出很多帧，然后在通过CNN提取到更具语义信息的特征后，这些特征可能都差不多，把这些特征传给LSTM，相当于你把很多一模一样的东西传给了LSTM，其实什么也学不到，这也是为什么在短视频上LSTM不能发挥其威力的原因，如果是在长视频或者视频帧变化比较激烈时，LSTM还是有用武之地的。

## Convolutional Two-Stream Network Fusion (Early Fusion)

[Paper](https://arxiv.org/pdf/1604.06573v1.pdf)

本篇论文非常细致的讲了到底如何去做时间流与空间流的合并。作者从spatial fusion，temporal fusion以及在哪一层做fusion这三个方面分析，最终得到一个非常好的做Early Fusion的网络，比之前直接做Late Fusion的双流网络效果好不少。

### Spatial Fusion

<img width="322" alt="image" src="https://user-images.githubusercontent.com/22740819/167290878-619842a4-80de-493b-99ee-65b0144ede11.png">

当有了时间流和空间流两个网络之后，如何保证两者的特征图在同样的位置上的response是联系起来，这种在特征图层面的Fusion才算做是Early Fusion。文章中做了以下几种尝试：

- Max fusion：对于a和b两个特征图，在同样的位置上只取两者中的最大值；
- Concatenation fusion：直接将两个特征图合并起来，尽可能的把信息保留下来；
- Conv fusion：先把两个特征图堆叠起来，然后再做一层卷积操作；
- Sum fusion：在两个特征图对应的位置上做加法；
- Bilinear fusion：给定两个特征图，在两个特征图上做一次outer product，然后在所有维度上做一次加权平均；

### 在网络的哪个部分做Fusion操作

<img width="320" alt="image" src="https://user-images.githubusercontent.com/22740819/167290546-1a0b6219-7ee9-4a59-93d6-b662880f73aa.png">

上面给出了在空间维度上几种做fusion的方式，接下来的问题就是应该在网络的哪个部分去做合并。作者通过大量的消融实验，得出上面两种比较好的融合方式：

- 空间流与时间流先分别提取特征，在conv4操作之后进行fusion，合并之后就从两个网络变成了一个网络，这也是传统意义上的early fusion；
- 空间流与时间流先分别提取特征，在conv5层之后，将空间流上的conv5特征和时间流上的conv5特征做一次合并，合并之后的特征成为spatial temporal特征，其既有空间信息又有时间信息；另外空间流的特征继续做FC操作，保证了空间流的完整性，在FC8层之后两者再做一次合并。这样操作可以简单的理解为，网络在之前的特征层还没有学到特别high level的语义特征之前先合并一下，去帮助时间流进行学习，然后等到FC8时已经提到比较高的语义信息之后再去做一次合并（Late fusion），从而得到最终的分类结果；

### Temporal fusion

<img width="331" alt="image" src="https://user-images.githubusercontent.com/22740819/167291707-b21e378d-dbbe-4908-8226-4ec0c9149fe0.png">

Temporal fusion是说，让你有很多视频帧，每一帧抽取特征之后，如何在时间轴维度上将其合并起来。作者采用了以上两种方式：

- 3D Pooling
- 3D Conv + 3D  Pooling

### 网络总体框架

在结合上面三种分析结果之后，作者提出了整体的网络框架：

<img width="694" alt="image" src="https://user-images.githubusercontent.com/22740819/167291780-bf1bf24b-36ee-4cd5-86d7-dae402e248ff.png">

- 输入为每一时刻的RGB图像及其对应的光流图像；
- 图中蓝色代表的是空间流操作，绿色代表的是时间流操作，针对每个时刻的RGB图像和光流图像，先提取特征，然后在按照之前提到的对空间流和时间流进行spatial fusion；因为其中牵扯到空间信息和时间信息，想让其交互的更好一些，所以采用了3D Conv + 3D Pooling的操作，在pooling之后再进行FC就可以计算Spatiotemporal Loss（因为同时包含了时间和空间信息）；
- 因为在视频理解领域，时间信息非常关键，所以作者又将时间流单独抽出来，做pooling+fc操作，单独计算一个Temporal Loss损失。

综上，整体的网络架构包含两个分支，一个是时空信息的学习分支，一个是时间信息的学习分支，训练时也是用了上面提到的两个损失函数，推理时类似之前提到的双流网络，将两个分支的结果做Late Fusion之后得到最终结果。

<img width="453" alt="image" src="https://user-images.githubusercontent.com/22740819/167298496-84fbcf68-b629-4932-9788-70e3597b5169.png">

- 双流网络backbone由AlexNet换成VGG16，在UCF101数据集上效果从88.0%提升至91.7%，但是在HMDB51数据集上略有下降由59.4%至58.7%；
- 基于Two-Stream，将Late fusion换成Early Fusion，在UCF101上由91.7%提升至92.5%，在HMDB51上提升很明显，由58.7%至65.4%；

Early fusion的方式可能算是一种变相的对网络的约束，让模型在早期的训练中，就能够从时间流和空间流这两支网络中去互相学习，所以说在一定程度上可能弥补了数据不足的问题，能够让Early fusion比Late fusion学得更好。

本篇文章的贡献不仅仅是比较了early fusion和late fusion哪种方式更好，它主要的贡献有以下两点：

- 做了大量的消融实验，从spatial fusion到temporal fusion，再到在哪里做fusion，彻底的对网络结构分析了一遍，为后续的工作带来了很大的便利；
- 尝试了3D Conv和3D Pooling，增加了研究者使用3D网络的信心，变相的推动了3D-CNN的发展；

