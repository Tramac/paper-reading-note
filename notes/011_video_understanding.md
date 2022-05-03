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

## Part3.Beyond-Short-Snippets



