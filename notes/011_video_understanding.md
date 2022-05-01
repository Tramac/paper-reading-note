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

<img width="485" alt="image" src="https://user-images.githubusercontent.com/22740819/166136129-07d173c8-1c33-4666-945d-e7b20cc951e8.png">

论文中的方法比较直接，想法就是如何把卷积神经网络从图片理解领域应用到视频理解领域。视频与图片唯一的不同就是它多了一个时间轴，有更多的视频帧而不是单个的图片。所以，自然的有上图中后三种融合方式。

- Single Frame: 就是单帧图像方式，其实就是图片分类任务，在视频中随机选取一帧，经过卷积神经网络得到分类结果；
- Late Fusion: 在网络输出层面做融合，例如在视频中随机选取几帧，然后每帧都单独通过一个CNN（权值共享），将得到的特征融合，再通过一个FC得到最终输出；
- Early Fusion: 在输入层做融合，比如将5帧视频帧在RGB的维度上直接合并，原来一帧RGB图像3个通道，现在5帧图像融合后就是15个通道，这也就意味着网络结构也要进行一定的改变，尤其是第一层，第一个卷积层接收的输入的通道数就由原来的3变成15，之后的结构都可以与之前保持一致；这种做法从刚开始就能从输入的层面感受到时序的改变，希望能够学到全局的运动和时序信息；
- Slow Fusion: 该方式是Late和Early的结合，也就是说Late合并的太晚而Early又合并的太早；每次选一个有10个视频帧的一小段视频，然后每4个视频帧通过一个卷积神经网络提取特征，刚开始的层都是权值共享，抽出特征之后将4组特征合并成两组，然后再经过一些卷积操作提取更深层特征，再将两组特征合并成一个特征，再通过卷积和FC得到最终输出；

这几种方式效果都差不多，而且在大数据集上预训练然后在UFC101上迁移，其效果还不如手工特征。
