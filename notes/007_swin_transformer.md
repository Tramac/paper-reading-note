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

## Part4.结论

## Part5.模型整体架构

## Part6.窗口自注意力

## Part7.移动窗口自注意力

## Part8.实验
