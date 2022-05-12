[Paper](https://arxiv.org/pdf/2012.06567.pdf) | [Talk 上](https://www.bilibili.com/video/BV1fL4y157yA/?spm_id_from=333.788)

<img width="800" alt="image" src="https://user-images.githubusercontent.com/22740819/166135532-db529b5f-96ab-4646-8d7d-baf0ebf10d22.png">

该Paper是一篇视频理解领域的综述论文，大概提及了超过200篇视频领域的论文，除video transformer之外，基本包含了之前用deep learning去做video action recognition的大部分工作。
Talk大概以上图中的时间线结构，从最开始的用卷积神经网络做DeepVideo的工作，然后到双流网络(Two-Stream Network)以及双流网络的各种变体(TDD、LRCN、Beyond-Short-Snippet、Fusion、TSN、TLE等)，
还有3D网络(C3D)以及3D网络的一些变体(LTC、I3D、P3D、Non-Local ～ ECO等)，以及最新的一些基于Video Transformer的工作。

大概分为以下四个部分：从Hand-crafted到CNN -- Two-Stream -- 3D CNN -- Video Transformer

## Part1.从Hand-crafted到CNN

### 1.1 DeepVideo

DeepVideo是AlexNet出现之后第一个用比较深的卷积神经网络做大规模视频分类的工作。作者团队对视频理解领域做出了很大的贡献，不仅包括文章中提出的Sport-1M数据集，在两年之后还提出了一个更大的视频数据集Yotube-8M数据集，
还有最近为了action detection提出的AVA数据集，全都来自于该团队，在很大程度上都推动了视频理解领域的发展。

#### 1.1.1 视频帧特征的融合方式

论文中的方法比较直接，想法就是如何把卷积神经网络从图片理解领域应用到视频理解领域。视频与图片唯一的不同就是它多了一个时间轴，有更多的视频帧而不是单个的图片。所以，自然的有图中后三种融合方式：

<div align=center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/166136129-07d173c8-1c33-4666-945d-e7b20cc951e8.png">
</div>

- Single Frame: 就是单帧图像方式，其实就是图片分类任务，在视频中随机选取一帧，经过卷积神经网络得到分类结果；
- Late Fusion: 在网络输出层面做融合，例如在视频中随机选取几帧，然后每帧都单独通过一个CNN（权值共享），将得到的特征融合，再通过一个FC得到最终输出；
- Early Fusion: 在输入层做融合，比如将5帧视频帧在RGB的维度上直接合并，原来一帧RGB图像3个通道，现在5帧图像融合后就是15个通道，这也就意味着网络结构也要进行一定的改变，尤其是第一层，第一个卷积层接收的输入的通道数就由原来的3变成15，之后的结构都可以与之前保持一致；这种做法从刚开始就能从输入的层面感受到时序的改变，希望能够学到全局的运动和时序信息；
- Slow Fusion: 该方式是Late和Early的结合，也就是说Late合并的太晚而Early又合并的太早；每次选一个有10个视频帧的一小段视频，然后每4个视频帧通过一个卷积神经网络提取特征，刚开始的层都是权值共享，抽出特征之后将4组特征合并成两组，然后再经过一些卷积操作提取更深层特征，再将两组特征合并成一个特征，再通过卷积和FC得到最终输出；

这几种方式效果都差不多，而且在大数据集上预训练然后在UFC101上迁移，其**效果还不如手工特征**。

#### 1.1.2 多分辨网络

动机：根据以上结果发现，2D CNN无法很好的学得时序信息，干脆就先不学了，然后把图像领域使用卷积神经网络一些好用的trick拿过来，看能不能在视频领域也可以work，于是尝试了下面的多分辨率网络。

<div align=center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/166246776-62d87fac-85d6-4da3-a080-869a3d9aa71b.png">
</div>

简单来说，就是将输入分为两个部分，一个是原图，另一个是从原图的正中间抠出来一部分作为输入。因为不论是对图片还是视频来说，一般最有用的物体都会出现在中间部分。上面的分支叫做fovea（人眼视网膜中最中心的东西） stream，下面的分支叫做context（图片整体信息） stream，想通过此操作既能够学得图片中最中心的最有用的信息，又能够学得图片的整体的理解。该结构也是一种双流结构，并且两个分支是权值共享的，也可以理解成早期的一种使用注意力的方式，它强制性的想让网络关注中心的区域。

#### 1.1.3 实验结果

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/166403459-7bb84875-5c98-4009-8a98-8a522d8dfed7.png">
</div>

从结果来看，相比于Single Frame，多分辨网络有一些提升但是不明显，另外Early Fusion和Late Fusion效果都不好，只有Slow Fusion的效果相对好一点。

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/166403644-aeff1a2d-6a43-444e-88e9-4f4a2e50bbcd.png">
</div>

模型在UCF101上的效果也不好，最好的才65.4%，而手工特征的准确可以达到87%。虽然该论文的方法效果并不是很好，但是对后续视频理解的发展做了很好的铺垫。

## Part2.双流网络

### 2.1 Two-Stream

从DeepVideo方法可以看出，2D网络不能很好的学得时序信息，而视频理解相对于图片理解最关键的区别就是多了一个时间维度，如何更好的处理时序信息，是效果提升的关键。

#### 2.1.1 网络结构

<div align=center>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/166419450-68eddd82-c0a2-4985-9123-15d5728f5314.png">
</div>

双流网络的意义是说，当用一个卷积神经网络无法很好的处理时序信息时，只需要再加一个卷积神经网络去专门处理这个时序信息。

从原始视频中抽取光流图像，光流图像中包含了丰富的物体运动信息和时序信息，空间流网络学习从RGB图像到分类的映射，时间流网络学习从光流图像到分类的映射。这两个分支网络都不用去学习时序信息，只需要去学习对应的映射就可以了，而且两个网络各司其职，互不干扰，相对而言也比较好优化。

#### 2.1.2 双流网络的四个改进方向

- Fusion方式：双流网络是在softmax之后做的fusion，属于Late Fusion，其中一个研究方向是如何做Early Fusion；
- Backbone的使用：文中使用的是AlexNet，后续还有工作使用InceptionNet，VGG，ResNet等，尝试更深的网络对视频理解也是一个研究方向；
- 双流网络是直接把RGB图像和光流图像提取特征之后去做分类，事实上RNN和LSTM是专门用来处理时序信息的，现在视频帧上抽特征，然后在特征上加一个LSTM将帧之间的时序信息模拟出来，最后得到的特征应该会更强；
- 无论使用单帧图片还是多帧光流图像，其实都非常短，所覆盖到的视频片段也比较有限，对于一个帧率为25的视频来说，即使是10帧图像也才对应不到0.5s的片段，但是一般一个动作或者事件，它的持续时间至少也是2～3秒，如何做长时间的视频理解也是一个研究方向。

### 2.2.Beyond-Short-Snippets （使用LSTM做特征融合）

[Paper](https://arxiv.org/pdf/1503.08909v2.pdf)

Beyond Short Snippets: 不想再做特别短的视频理解了，short snippets是指2～3的视频段。

如果按照双流网络的思想去做，原始图像只有一帧或几帧，光流图只有10帧，那它能处理的视频其实是非常短的。如果现在有特别多的视频帧应该如何处理呢？

#### 2.2.1 网络结构

<div align=center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/167277944-d749971c-fada-4871-ab5c-cbeb2ff92abe.png">
</div>

一般来说，对于这些所有的视频帧，我们都先用一种方法去抽取特征，比如原始的手工特征SIFT特征，或者CNN特征，关键是抽取特征之后如何去做pooling操作。

- max pooling，avg pooling，conv pooling，late pooling，early pooling

<div align=center>
<img width="323" alt="image" src="https://user-images.githubusercontent.com/22740819/167278090-f554d0a3-51d4-4e55-be6a-236460aa5dfe.png">
</div>

该论文尝试了max pooling，avg pooling，conv pooling以及late pooling，early pooling等，最后发现效果其实都差不多，conv pooling相对好一些。

- lstm

<div align=center>
<img width="325" alt="image" src="https://user-images.githubusercontent.com/22740819/167278116-359427ee-ecfb-4c41-9394-c7b3305b9d28.png">
</div>

理论上lstm能够掌握时间上的变化，使用lstm去做特征融合应该是一种更好的融合方式，但是从结果来看提升也比较有限。图中的C代表每一帧的特征，这些特征在时序上是有顺序的，所以才会想到用LSTM去做特征融合，文中使用了5层lstm去处理特征，图中橘黄色部分代表softmax操作，之后再做分类。

该改动相对于双流网络是比较直接的，前面的操作都没有改变，只是在最后一个卷积层之后，不再是简单的做一层FC操作去做分类，而是把所有的视频帧特征通过一些LSTM层去做一些时序上的处理和融合，最后再做分类。

#### 2.2.2 实验结果

<div align=center>
<img width="666" alt="image" src="https://user-images.githubusercontent.com/22740819/167278296-3bd39817-3c06-4624-859f-f52ee0d8cfcd.png">
</div>

从Sport-1M数据上的结果来看，本文提出的Conv Pooling和LSTM能够接收更多的视频帧，而且效果也相对更好（主要是使用光流和使用更多的帧带来的收益），其中Slow Fusion就是DeepVideo方法。只是对比Conv Pooling和LSTM融合方式其实差别不大，说明LSTM对时序的利用上没有想象的好。

<div align=center>
<img width="324" alt="image" src="https://user-images.githubusercontent.com/22740819/167278373-d002f3ca-7c2c-4b32-ae4f-809eaf0b331a.png">
</div>

在UCF101数据上，手工特征准确为87.9%，DeepVideo只有65.4%（因为该方法没有很好的利用时序信息），Two-Stream因为使用光流效果达到88.0%，该论文方法因为完全是在双流网络的基础上做的，其效果要比双流网络的好才行，事实上，当使用120帧的Conv Pooling是准确也只有88.2%，只比88.0%高一点，使用LSTM时用了30帧，效果也只有88.6%，所以在UCF101这样的只有6～7秒的短视频上，LSTM的提升是比较有限的。

**为什么擅长时序信息提取的LSTM并没有带来明显的提升？**

> 一种解释是，像LSTM这种操作，它学的是一个更high level，更有语义信息的特征，给LSTM的输入必须得有一定的变化，LSTM才能起到它应有的作用，它才能学到时序上的改变，但是如果视频比较短比如只有5～7秒，这个语义信息很可能根本就没有改变，你从视频中抽出很多帧，然后在通过CNN提取到更具语义信息的特征后，这些特征可能都差不多，把这些特征传给LSTM，相当于你把很多一模一样的东西传给了LSTM，其实什么也学不到，这也是为什么在短视频上LSTM不能发挥其威力的原因，如果是在长视频或者视频帧变化比较激烈时，LSTM还是有用武之地的。

### 2.3.Convolutional Two-Stream Network Fusion (Early Fusion)

[Paper](https://arxiv.org/pdf/1604.06573v1.pdf)

本篇论文非常细致的讲了到底如何去做时间流与空间流的合并。作者从spatial fusion，temporal fusion以及在哪一层做fusion这三个方面分析，最终得到一个非常好的做Early Fusion的网络，比之前直接做Late Fusion的双流网络效果好不少。

#### 2.3.1 Spatial Fusion

<div align=center>
<img width="322" alt="image" src="https://user-images.githubusercontent.com/22740819/167290878-619842a4-80de-493b-99ee-65b0144ede11.png">
</div>

当有了时间流和空间流两个网络之后，如何保证两者的特征图在同样的位置上的response是联系起来，这种在特征图层面的Fusion才算做是Early Fusion。文章中做了以下几种尝试：

- Max fusion：对于a和b两个特征图，在同样的位置上只取两者中的最大值；
- Concatenation fusion：直接将两个特征图合并起来，尽可能的把信息保留下来；
- Conv fusion：先把两个特征图堆叠起来，然后再做一层卷积操作；
- Sum fusion：在两个特征图对应的位置上做加法；
- Bilinear fusion：给定两个特征图，在两个特征图上做一次outer product，然后在所有维度上做一次加权平均；

#### 2.3.2 在网络的哪个部分做Fusion操作

<div align=center>
<img width="320" alt="image" src="https://user-images.githubusercontent.com/22740819/167290546-1a0b6219-7ee9-4a59-93d6-b662880f73aa.png">
</div>

上面给出了在空间维度上几种做fusion的方式，接下来的问题就是应该在网络的哪个部分去做合并。作者通过大量的消融实验，得出上面两种比较好的融合方式：

- 空间流与时间流先分别提取特征，在conv4操作之后进行fusion，合并之后就从两个网络变成了一个网络，这也是传统意义上的early fusion；
- 空间流与时间流先分别提取特征，在conv5层之后，将空间流上的conv5特征和时间流上的conv5特征做一次合并，合并之后的特征成为spatial temporal特征，其既有空间信息又有时间信息；另外空间流的特征继续做FC操作，保证了空间流的完整性，在FC8层之后两者再做一次合并。这样操作可以简单的理解为，网络在之前的特征层还没有学到特别high level的语义特征之前先合并一下，去帮助时间流进行学习，然后等到FC8时已经提到比较高的语义信息之后再去做一次合并（Late fusion），从而得到最终的分类结果；

#### 2.3.3 Temporal fusion

<div align=center>
<img width="331" alt="image" src="https://user-images.githubusercontent.com/22740819/167291707-b21e378d-dbbe-4908-8226-4ec0c9149fe0.png">
</div>

Temporal fusion是说，让你有很多视频帧，每一帧抽取特征之后，如何在时间轴维度上将其合并起来。作者采用了以上两种方式：

- 3D Pooling
- 3D Conv + 3D  Pooling

#### 2.3.4 网络总体框架

在结合上面三种分析结果之后，作者提出了整体的网络框架：

<div align=center>
<img width="694" alt="image" src="https://user-images.githubusercontent.com/22740819/167291780-bf1bf24b-36ee-4cd5-86d7-dae402e248ff.png">
</div>

- 输入为每一时刻的RGB图像及其对应的光流图像；
- 图中蓝色代表的是空间流操作，绿色代表的是时间流操作，针对每个时刻的RGB图像和光流图像，先提取特征，然后在按照之前提到的对空间流和时间流进行spatial fusion；因为其中牵扯到空间信息和时间信息，想让其交互的更好一些，所以采用了3D Conv + 3D Pooling的操作，在pooling之后再进行FC就可以计算Spatiotemporal Loss（因为同时包含了时间和空间信息）；
- 因为在视频理解领域，时间信息非常关键，所以作者又将时间流单独抽出来，做pooling+fc操作，单独计算一个Temporal Loss损失。

综上，整体的网络架构包含两个分支，一个是时空信息的学习分支，一个是时间信息的学习分支，训练时也是用了上面提到的两个损失函数，推理时类似之前提到的双流网络，将两个分支的结果做Late Fusion之后得到最终结果。

#### 2.3.5 实验结果

<div align=center>
<img width="453" alt="image" src="https://user-images.githubusercontent.com/22740819/167298496-84fbcf68-b629-4932-9788-70e3597b5169.png">
</div>

- 双流网络backbone由AlexNet换成VGG16，在UCF101数据集上效果从88.0%提升至91.7%，但是在HMDB51数据集上略有下降由59.4%至58.7%；
- 基于Two-Stream，将Late fusion换成Early Fusion，在UCF101上由91.7%提升至92.5%，在HMDB51上提升很明显，由58.7%至65.4%；

Early fusion的方式可能算是一种变相的对网络的约束，让模型在早期的训练中，就能够从时间流和空间流这两支网络中去互相学习，所以说在一定程度上可能弥补了数据不足的问题，能够让Early fusion比Late fusion学得更好。

本篇文章的贡献不仅仅是比较了early fusion和late fusion哪种方式更好，它主要的贡献有以下两点：

- 做了大量的消融实验，从spatial fusion到temporal fusion，再到在哪里做fusion，彻底的对网络结构分析了一遍，为后续的工作带来了很大的便利；
- 尝试了3D Conv和3D Pooling，增加了研究者使用3D网络的信心，变相的推动了3D-CNN的发展；

### 2.4.TSN（如何处理更长视频）

该论文不仅通过一种特别简单的方式能够处理更长的视频并且效果特别好，更大的贡献是它确定了很多好用的技巧，比如如何去做数据增强，如何做模型初始化，该怎么使用光流，该使用哪个网络，如何防止过拟合等。

#### 2.4.1 网络结构

<div align=center>
<img width="669" alt="image" src="https://user-images.githubusercontent.com/22740819/167600485-d45d8ece-d486-4849-8040-d9975884847b.png">
</div>

论文的研究动机：对于一个双流网络，一帧视频帧通过空间流，几帧光流图通过时间流，覆盖到的视频长度最多10帧，也就不到半秒时间。那么如何去处理一个更长的视频呢？TSN的做法：

- 将一个视频切成n段，在每一段中随机抽取1帧图像，然后以该帧为起点再选取几帧计算光流图像，然后送入双流网络计算各自的logits；
- 所有段中抽取的RGB图像和光流图像所送入的双流网络是共享权重的，事实是只有一组双流网络；
- TSN中认为，如果输入的视频不算太长只包含一个事件内容或动作，虽然每一段的视频帧和光流不一样，但它其实最高层的语义信息应该都描述的是一个东西，所以可以把每一段的空间流输出的logit做一个Segmental Consensus（达成共识、融合，可以是加法、avg、max等，总之就是融合各个段的结果），时间流同理；
- 最后将空间的Segmental Consensus和时间的Segmental Consensus做一个late fusion（加权平均），得到最后的预测。

以UCF101数据集为例，有101个类，上述每个双流网络输出的logits都是一个(1, 101)维的向量，得到的Segmental Consensus也是(1, 101)维的向量。

TSN中这种给视频做temporal segment的思想，不光是可以作用于短一点的视频（只包含一个动作），可以工作与更长一些的视频，因为即使视频很长也可以分成更多的段落，每个段落里的含义也不需要一样，如果不一样后面的logits不做avg就可以了，可以做LSTM去模拟时间的走势也是可以的。

#### 2.4.2 训练技巧

- Cross Modality Pre-training：modality指的就是图像和光流两种模态，对于空间流可以直接使用ImageNet上的预训练模型，然而光流并没有预训练模型；事实上光流分支也可以用ImageNet上的预训练模型，效果也不错，但是ImageNet上的预训练模型接收的输入是3通道，而每张光流图是两通道（x和y两个方向），10张光流图就是20通道，解决办法就是把预训练模型的第一个卷积层3个通道先做一次平均变成一个通道，然后再复制20份。
- Regularization Techniques：对于BN层，如果用的好效果就很好，如果用不好就会出现各种问题。在视频领域初期阶段，BN层的应用不是太好，因为数据集太小，虽然BN能够让训练加速，但是同样也带来了很严重的过拟合问题；TSN中提出了一个partial BN，动机是如果原始的BN层是可以微调的，因为数据集比较小，一调就会过拟合，但是如果把所有的BN全冻住又会导致迁移效果不太好，因为在图像数据集上估计出来的统计量可能不太适用于视频数据集；partial BN的做法是只把第一层BN放开，后面的BN全冻住，因为输入变了，第一层的BN必须要在调整一下才能重新适应新数据集。
- Data Augmentation：TSN中提出了corner cropping和scale-jittering两种数据增强方式。random crop的方式经常会crop出正中间附近，很少裁出边角部分，corner cropping是强制性的在边角处做出一些裁剪；scale-jittering是想通过改变图像的长宽比来增加输入的多样性，实现方式也很高效，就是把视频先resize到256x340，然后从{256, 224, 192, 168}随机选取出两个数作为长和宽去裁剪得到新图。

#### 2.4.3 实验结果

<div align=center>
<img width="618" alt="image" src="https://user-images.githubusercontent.com/22740819/167636271-533114d3-e43e-4bfc-a37b-c07eb3d44066.png">
</div>

- 双流网络在UCF101上准确维88.0%，C3D效果相对并不好为85.2%，Two stream +LSTM对应的是Beyond-Short-Snippets文章，效果为88.6%，提升也不是很明显；
- TSN用两个模态（RGB+光流）准确到94.0%；

## 视频理解串讲前半部分的总结

<div align=center>
<img width="631" alt="image" src="https://user-images.githubusercontent.com/22740819/167637616-7d936d16-90f7-443b-a05c-3926faba9eea.png">
</div>

- DeepVideo第一次将神经网络迁移到视频领域；
- 由于视频中时序信息的重要性，Two-Stream提出了空间流和时间流的双流思想，大大提升了效果；
- TDD是以沿着轨迹的方式对光流进行堆叠，LSTM的引入是为了更好的对时序信息建模，另外还有一些Early Fusion的工作；
- 为了加强对长视频的理解，出现了TSN，对长视频进行分段，在融合每段的思想；
- 在TSN的基础上，DVOF和TLE又引入了全局编码，去学得一个更加全局的video level的特征，其中DVOF不是端到端的，而TLE是端到端的。

以上就是2014年～2017年视频领域中双流网络的一个发展历程，从UCF101数据集上的效果来看，从DeepVideo的65%提升到了TLE的96%。

## Part3. 3D CNN

Part2中介绍的双流网络效果不错，但是其局限性是光流的抽取成本太高。

**抽取计算、时间成本**

以tvl one算法为例（一种抽取光流的方式），在GPU上处理一对视频帧耗时0.06s（一对视频帧抽取出一张光流图）。常见数据集上抽取光流的耗时（只使用一个GPU）：

- UCF101数据集，其包含1w个视频，平均时长为6s，帧率为30fps，整个数据集就包含10000 * 6 * 30 = 180w视频帧，需要耗时180w * 0.06s ~ 1.5天；
- Kinetics400数据集，包含24w视频，平均时长为10s，帧率为30fps，整个数据集就包含240000 * 10 * 30=7.2kw视频帧，需耗时7.2kw * 0.06s ～ 50天；

**存储成本、读取io限制**

抽取光流不仅耗时而且占空间，在双流网络中，会将光流图像提前存成图片，对于UCF101数据集大概需要20G的存储空间，K400数据集需要500G存储空间，虽然存储成本较低，但是考虑到运算时的io，也会带来一定的限制。

**推理阶段的限制**

由于光流的运用，在推理阶段无论是用在哪个场景下，都需要先抽取出光流。对于tvl one算法来说，处理一对的耗时为0.06s，那么单卡处理图像级上限为1/0.06s ~ 15fps，是低于实时要求的，实时的处理一般要求是25fps+。所以，即便不考虑模型的耗时，仅抽取光流的耗时就已经无法达到实时要求了。

视频理解的应用场景大多都是要求实时的，所以后续的一些工作很多都在尽量避开双流网络，直接从视频中进行学习。3D网络理论上是可以学习到时空特征，它既有spatial又有temporal上的信息，没有必要再对时序信息单独建模了。

### 3.1 C3D

[Paper](https://arxiv.org/pdf/1412.0767.pdf)

题目就是用3D卷积神经网络去学习时空特征，作者团队来自Facebook团队。

之前也有一些用3D CNN的工作，而C3D的主要贡献就是提出了更深的3D ConvNets做时空特征的学习，并且在更大的数据集上sport-1M（100w个视频）进行了训练。

#### 3.1.1 模型结构

<div align=center>
<img width="631" alt="image" src="https://user-images.githubusercontent.com/22740819/167767203-d7987d2e-1254-4ec0-b3d9-923cf1e7b603.png">
</div>

网络结构整体包含11层，卷积核大小均为3x3x3，有些类似3D版的VGG16网路，只是每个Block里都减少了一个卷积层。以输入16帧图像，尺寸大小为112x112为例，其特征图尺寸变化如上图所示。

#### 3.1.2 结果对比

<div align=center>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/167766380-16960214-aedc-49f7-9230-6deeb103ab39.png">
</div>

C3D相对于其他方法在ASLAN、YUPENN、UMD和Object四个数据集上都取得了更好的结果，而Sport1M和UCF101上相对差一些。

<div align=center>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/167796720-ffabda7e-ae77-4881-bc71-15aec5ea00e6.png">
</div>

C3D相对于DeepVideo(2D)在Sport-1M上的结果好一些；在更大的数据集上I380K预训练可以进一步提升；

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/167797398-d88a0016-6001-4e7e-ae95-4fb7b133f308.png">
</div>

在UCF101上的结果对比：

- 一个C3D网络准确为82.3%左右，训练三个C3D然后在集成，准确能到85.2%；
- C3D低于同期的其它工作，比如手工特征，双流网络等；

C3D的卖点并不是网络结构和训练方式等，而是它将C3D视作一个提取特征的工具，然后可以在后面做svm分类等，另外在当时提供了python和matlab两个版本的提取特征工具。和现在transformer的运用相似，因为transformer训练成本太高，往往是只作为特征提取，然后拿到特征后继续做下游任务。

### 3.2 I3D

I3D的主要意义是降低了3D网络的训练难度，提出了一个很好很大的K400数据集，并且在标准的Benchmark数据集（UCF101，HMDB51等）上的结果都比之前的2D和双流网络效果好，从而奠定了3D网络的地位。

#### 3.2.1 网络结构

<div align=center>
<img width="500" alt="image" src="https://user-images.githubusercontent.com/22740819/167800302-7943dc60-961d-4f36-bec8-db6a4a422984.png">
</div>

I3D的核心就是inflated操作，其出发点是C3D网络在大规模的数据集上进行预训练之后其效果提升并不明显。作者就坚信3D网络中也一定要有一个类似于ImageNet的预训练模型，要有一个好的初始化从而能够让接下来的任务训练更简单。但是之前ImageNet的预训练模型都是2D，如何将2D变成3D也就是inflated的主要目的，就是将一个2D网络扩充成3D网络，而网络的整体结构是不变的。

以图中的Inception-V1为例，基本就是将conv和pool由2D变成了3D，其整个网络架构是不变的，这样就可以通过一种方式将2D上的预训练模型移植到3D网络上。

**I3D的最大亮点就是通过inflated操作将2D网络扩充为3D，不仅不用设计网络架构，还可以移植2D网络上的预训练模型，从而简化训练过程。**

#### 3.2.2 结果对比

<div align=center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/167802464-c0137eec-df10-4937-9108-d9e336629223.png">
</div>

- I3D的结果远高于之前C3D的结果；
- 对比了利用光流之后效果仍会提升，说明光流不是没用，只是计算代价太高；
- 将UCF101和HMDB51刷爆了，后续工作也基本不在这两个数据集上做对比了。

### 3.3 Non-Local

[Paper](https://arxiv.org/pdf/1711.07971.pdf)

在视频领域一个重要的问题是如何对时序建模以及如何处理更长的视频。时序建模常见的方法是用一个LSTM，受NLP领域的影响，随着transformer的出现，自注意力操作被提出来，其本身就是用来学习远距离信息的，和LSTM的作用不谋而合，所以Non-local中就尝试了将self-attention用在了视觉领域中，效果也不错。

无论是卷积操作还是递归操作，每次处理都是在一个局部区域进行处理的，如果能够看到更长或者更多的上下文信息，肯定是对各种任务都是有帮助的。所以作者提出了一个可以泛化的、即插即用的Non-local模块，用来建模长距离的信息。

#### 3.3.1 模块结构

<div align=center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/22740819/167820523-8c22716f-276e-45f8-8850-26735c643270.png">
</div>

上图给出的是一个spacetime non-local block，用来做视频理解的时空的算子。从结构来看就是一个自注意力操作：

- 给定一个输入X，通过三个不同的映射得到q，k，v；
- q和k之间计算一个attention操作，得到一个自注意力；
- 拿得到的attention与v做加权平均，得到新的v之后通过映射升维，然后与输入做残差连接，得到最终的输出。

#### 3.3.2 消融实验

<div align=center>
<img width="500" alt="image" src="https://user-images.githubusercontent.com/22740819/167836576-4b79110a-94a6-4991-91d5-18760d88141d.png">
</div>

- 表(a)给出了自注意力计算方式的对比，其中用dot-product的效果是最好的，transformer中的自注意力默认使用的也是dot-product；
- 表(b)给出了Non-local应该加在哪一个res block里面，结果显示是加载2～4内是最好的，加在res5中效果不好，解释是res5后的特征图已经比较小了，没有远距离的信息，non-local起不到作用；另一方面由于non-local的计算复杂度高，最终只在res3和res4阶段用了non-local；
- 表(c)给出了non-local使用的层数的影响，确实是越多的non-local block，效果越好，证明了non-local的有效性；
- 表(d)给出了同时在时间和空间做non-local的必要性；
- 表(g)证明了，越长的输入，效果越好，说明了non-local对于长时序输入的有效性；

#### 3.3.3 对比结果

<div align=center>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/167838479-eb1dcaf7-b3be-448f-a5c8-58a6b039318d.png">
</div>

- I3D的backbone由Inception替换成ResNet50之后会带来一个点的提升；
- Non-local I3D相对于I3D提升明显，由72.1% -> 76.5%，甚至高于双流的I3D；

Non-local的主要贡献就是将自注意力操作引入到了视觉领域，并且针对视频理解任务，把space的自注意力操作扩展到spacetime的自注意力操作，在该方法之后很少再有视频模型使用lstm操作了。

### 3.4 R(2+1)D

[Paper](https://arxiv.org/pdf/1711.11248.pdf)

题目的意思就是对于动作视频任务，时空卷积到底应该如何用，2D or 3D或是2D+3D，是一篇非常具有实验性的论文。

研究动机：作者发现，如果用2D网络对视频帧抽取特征然后预测在动作识别上的表现也很好，比3D网络差不了太多。既然2D网络的效果也不错，如果部分用一下2D卷积可能也不错，毕竟计算量比较低。经过实验之后发现，把一个3D的卷积改成一个空间上2D+时间上1D的网络效果更好。

#### 3.4.1 网络结构对比


<div align=center>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/167981761-5917bd28-a43e-458d-bf64-770eb281022f.png">
</div>

- R2D：纯2D网络，视频抽帧然后过2D网络，预测；
- MCx：先通过一些3D网络，在底层学一下时空特征，在上层再换成2D网络，降低计算复杂度；
- rMCx：开始先把clip抽帧，每帧先经过2D网络，得到特征之后再用3D网络去做融合；
- R3D：纯3D网络，也就是C3D、I3D的做法；
- R(2+1)D：把3D卷积拆分成两个卷积，先做一个2D的spatial上的卷积，再做一个1D的temporal上的卷积；

#### 3.4.2 参数量及效果对比

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/167982544-21a56952-06b5-4b07-8771-0f0912c0067e.png">
</div>

- 纯2D的网络参数量肯定是最少的，其它用到3D的参数量明显变高；
- 结果上，纯用2D和纯用3D的效果都不太好，2D和3D混合用的相对会好一些，R(2+1)D是最好的。

#### 3.4.3 (2+1)D的拆分方式

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/167983070-1f386c75-8770-4b2e-beac-62854e00e952.png">
</div>

由图所示，对于原始的txdxd的3D卷积核，(2+1)D的做法是先在空间上做dxd的卷积，然后做一次特征映射Mi（做映射是想降维，让参数量和纯3D网络保持一致），之后再在时间上做tx1x1的卷积。

**(2+1)D这种拆分的方式比之前3D网络效果好的原因：**

- 增强了网络的非线性：之前只有一个3D Conv，后面就只接了一个ReLU激活层，也就是只做了一次非线性操作；现在是做了两次卷积，两个ReLU，两个非线性变换，学习能力也就变强了；
- 从优化角度看，直接学习一个3D Conv相对是不好学的，分成一个2D+1D会好优化一些

#### 3.4.4 效果对比

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/167985779-0c094300-7d76-4c6d-83bc-8ccbb6a8afca.png">
</div>

与I3D对比，在K400上，无论是单独的RGB还是单独的Flow，R(2+1)D的效果都是更好一些，但是Two-Stream时，反而是I3D的效果更好。

### 3.5 SlowFast

[Paper](https://arxiv.org/pdf/1812.03982.pdf)

#### 3.5.1 网络结构

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/167987605-850651f9-d498-4f4b-ae09-7c6fd35044e6.png">
</div>

SlowFast网络分为两个分支，Slow和Fast，假如给定一个包含64帧的视频（2s左右）：

- Slow：以较低的帧率抽帧，比如每隔16帧取1帧，共取出4帧作为Slow分支的输入，该分支主要用于学习静态、场景信息；由于静态信息比较难描绘，所以大部分的网络参数都在Slow分支，该分支的网络结构比较大，其实就是一个I3D网络，但是输入只有4帧，相对而言计算复杂度并不是很高；
- Fast：以较高的帧率抽帧，比如每隔4帧取1帧，共取出16帧作为Fast分支的输入，该分支主要用来描述运动信息；由于输入变多，为了维持整个模型的计算复杂度低一些，让fast分支的网络尽可能的小一些。

整体来看，SlowFast也是一种双流的网络，只是没有用光流，分为慢分支与快分支，慢分支用小输入大网络，快分支用大输入小网络，两个分支之间再做一些later connection，对信息做一些交互，从而能够更好的学得时空特征。

#### 3.5.2 前向计算过程

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/167989229-426ca32a-97fa-4a45-aa79-9fc2b4d5ccb0.png">
</div>

- Slow pathway其实就是一个以resnet50为架构的I3D网络；
- Fast pathway相对slow而言主要是通道数做了一些衰减（图中橙色部分），意味是比较轻量的，但是整体架构也是一个resnet50结构的I3D版本；
- SlowFast和之前的I3D，R(2+1)D相同，在时序上都没有进行下采样，可以看到特征图的尺寸，慢分支一直是4，快分支一直是32，因为帧本来就少，想多维持时序的信息，下采样只在空间上做；

#### 3.5.3 结果对比

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/167990037-4f8c5768-ceb9-48bb-9613-8b49045a30a1.png">
</div>

SlowFast相对于其它的3D网络无论是在精度上还是计算复杂度上都有优势，也是3D网络范畴内的最好表现了。

## Part4.Video Transformer

随着vision transformer的火热，如何把图像领域的transformer运用到视频中也是一个重要的研究方向。

### 4.1 TimeSformer

[Paper](https://arxiv.org/pdf/2102.05095.pdf)

题目：在视频理解领域，时空注意力是不是all you need。

TimeSformer是一篇偏实验性的论文，探究了如何将图像领域的vision transformer迁移到视频领域。

#### 4.1.1 结构尝试

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/168003966-f48a59ec-1ac8-4bbc-ac07-fd12d527dd6e.png">
</div>

- Space Attention: 只在空间特征图上做自注意力，其实就是图像中vision transformer所用的自注意力；
- Joint Space-Time Attention: 在视频的三个维度上都做自注意力，这种方式很占显存；
- Divided Space-Time Attention: 考虑到显存限制，将3D自注意力拆分成两部分，先在时间上做自注意力，再在空间上做自注意力，大大降低了复杂度，因为每个序列长度就变得很小；
- Sparse Local Global Attention: 由于全局的序列长度太长，就现在局部计算，然后再全局计算，也会降低复杂度；
- Axial Attention: 只沿着特征的轴做attention，先按时间轴做attention，再沿着横轴做，最后沿着纵轴做，把一个三维的问题拆成三个一维的东西；


**自注意力计算方式的可视化解释**

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/168006392-4083ce20-a202-4e05-b263-a6bc311a5696.png">
</div>

每一列都是从t-1 -> t -> t+1的连续三帧图像：

- Space Attention: 假设拿frame t中蓝色的patch为基准点，它只会跟它当前的这一帧去计算自注意力（frame t中红色区域），因为该方式就是图像上的自注意力，不牵扯时间轴；
- Joint Space-Time Attention: 复杂度最高，对于frame t中的蓝色patch，会与前后帧以及自身的所有patch计算自注意力；
- Divided Space-Time Attention: frame t中蓝色patch首先会和前后帧对应位置的patch(绿色)进行自注意力计算，然后在当前帧所有patch(红色)计算自注意力；
- Sparse Local Global Attention: 对于蓝色patch，先计算局部小窗口内的自注意力(黄色区域)，再计算全局的自注意力(为了减少计算量做了稀疏，洋红色区域)；
- Axial Attention: 对于蓝色patch，时间轴上先和绿色的patch计算自注意力，再在横轴上与黄色区域计算自注意力，最后在纵轴上与洋红色区域计算自注意力；

结合效果和效率的考虑，文章中以Divided Space-Time Attention的方式提出了TimeSformer的架构。

#### 4.1.2 消融实验

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/168008857-561b2a80-c9e2-4bba-ba93-fde58ccef56e.png">
</div>

- Divided Space-Time的方式相对而言，参数量适中，效果最好；
- 只用2D的自注意力在K400上的效果也不错，因为K400是有些偏静态的，如果换成SSv2数据集效果下降很多；

<div align=center>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/22740819/168009284-1ef5a6c0-d148-4a1d-b312-109512efd362.png">
</div>

- 从计算复杂度来看，Divided Space-Time的复杂度基本是和输入尺寸成线性增长的，而Joint Space-Time近似指数增长，容易造成OOM；

#### 4.1.3 结果对比

<div align=center>
<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/168009666-94abfac9-c09e-4b2f-a932-80065a80d06c.png">
</div>

TimeSformer对比了最经典的I3D和最新的SlowFast两种3D网络：

- 从训练时间和做推理时的FLOPs来看，TimeSformer都是最低的，从效率上是优于之前的3D网络的；
- 在效果上，TimeSformer-L用ImageNet-21K预训练，在K400上效果也能到80.7%；

TimeSformer方法之后，vidtr、mvit、vivit等一系列video transformer的工作思想都差不多，基本都是考虑如何对时空自注意力做拆分。

## 视频理解工作总结

<div align=center>
<img width="656" alt="image" src="https://user-images.githubusercontent.com/22740819/168011303-7a17e6c1-5fcb-4fee-be19-9e17358d15ff.png">
</div>

- DeepVideo：AlexNet之后，最早的将CNN用到视频理解领域中来，但是没有用到运动信息，效果不是很好，不如手工特征IDT的效果；
- Two-Stream：加入光流图像，利用了运动信息，证明了时序信息的重要性；
- 为了提高对更长视频的理解能力以及双流方法的融合方式，Beyond short snippts用lstm学习时序特征，Early Fusion考虑了空间和时间的其它融合方式，TDD根据光流的轨迹堆叠光流图；为了处理更长视频，TSN将视频分成多段，然后每段去做分析；在TSN基础上，结合全局建模，又有DOVF、TLE、ActionVLAD等工作提出；
- 从C3D到I3D，开始用3D网络处理视频；在I3D的基础上，通过替换**2D backbone**有R3D、MFNet、STC等工作；考虑到3D的复杂度，出现了S3D、R(2+1)D、ECO、P3D等工作，基本都是把3D拆成了**2D+1D**；为了处理更**长序列输入**，提出了LTC、T3D、Non-local、V4D等工作；为了提**高效率**，出现了CSN、SlowFast、X3D等工作
- Video Transformer：TimeSformer、VidTr、Vivit、Mvit都是考虑到自注意力的复杂度对video transformer进行改进。
