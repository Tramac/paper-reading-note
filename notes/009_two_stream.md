# Two-Stream Convolutional Networks for Action Recognition in Videos

[Paper](https://arxiv.org/pdf/1406.2199.pdf) | [Talk](https://www.bilibili.com/video/BV1mq4y1x7RU?spm_id_from=333.999.0.0)

视频本身是一个很好的数据来源，比2D的单个图像能够包含更多的信息，比如物体之间移动的信息、长期的时序信息、音频信号，非常适合做多模态学习。

双流网络是视频理解的一篇开山之作，它是第一个在视频领域能够让卷积神经网络可以和最好的手工特征打成平手。

## Part1.标题&作者

双流卷积神经网络用来做视频中的动作识别。

Two-Stream: 顾名思义，就是使用了两个卷积神经网络。

<img src="https://user-images.githubusercontent.com/22740819/155945646-ac96ce19-f01f-43d4-b565-9498dfa0e6cb.png" width=600> 

