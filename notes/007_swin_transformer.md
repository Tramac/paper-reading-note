# Swin Transformer

[Paper](https://arxiv.org/pdf/2103.14030.pdf) | [Talk](https://www.bilibili.com/video/BV13L4y1475U?spm_id_from=333.999.0.0)

## Part1.标题&作者

Swin Transformer是一个用了移动窗口（shifted windows）的层级式（hierarchical）的vision transformer。

Swin Transformer想让vision transformer像卷积神经网络一样，也能够分成几个block，也能够做层级式的特征提取，使得提取到的特征具有多尺度的概念

## Part2.摘要

Swin Transformer可以被用来作为CV领域的一个通用的骨干网络，而ViT当时只做了分类任务，虽然当时可以看到Transformer在视觉领域的潜力，但是并不确定是不是在所有的视觉任务上都可以做好，而Swin Transformer证明了是可以的。

直接将Transformer从NLP迁移至CV领域主要面临两个挑战：

- 尺度问题：假如一张街景图片，其中代表同样语义词的物体尺寸大小不一，比如汽车，而NLP中不存在此问题；
- 图像分辨率太大：如果以像素点作为基本单位，序列长度将会很长；

基于以上两个问题，本文提出一个hierarchical Transformer，它的特征是通过一种滑动窗口（Shifted windows）的方式计算所得。移动窗口操作不仅提高了自注意力的计算效率，而且通过shifting操作能够令相邻的两个窗口之间有了交互，从而变相的达到了一种全局建模的能力。

层级式结构的好处：

- 可以提供各个尺度的特征信息；
- 自注意力在小的窗口内计算，它的复杂度是随着图像大小线性增长的，而不是平方级增长。

## Part3.引言

本文的研究动机就是想要证明Transoformer是可以作为通用的骨干网络，在所有的视觉任务上都可以取得很好的效果。

<img src="https://user-images.githubusercontent.com/22740819/150341323-b038a942-7d4e-490f-ba0b-5354c96f1ebc.png" width=400>

**ViT的缺点：**

- ViT是把一张图分成patch，因为是patch大小为16 * 16，也就是下采样率为16x，而每一个patch自始至终所包含的信息都是16 * 16大小，每一层的transformer block所看到的都是16x的下采样率，虽然可以通过全局的自注意力操作达到全局建模的能力，但是对多尺度特征的把握就会弱一些，不适合处理密集预测性任务（检测、分割）。在一些下游任务中，多尺度特征是至关重要的，比如检测中的FPN，分割种的skip-connection都是为了结合不同尺度的特征，来提升效果。
- ViT中的自注意力始终都是在整图上进行的，是一个全局建模，其复杂度是跟图像的尺寸进行平方倍的增长，检测、分割领域一般常用的尺寸都很大。

**Swin Transformer的优点：**

- Swin采取了在小窗口之内进行自注意力的计算，而不是在整图上计算，这样一来只要是窗口大小是固定的，那么自注意力的计算复杂度就是固定的，整张图的计算复杂度就会跟这张图片的大小形成线性增长的关系，类似于卷及神经网络中的locality局部相关性的先验知识。

## Part4.结论

## Part5.模型整体架构

## Part6.窗口自注意力

## Part7.移动窗口自注意力

## Part8.实验
