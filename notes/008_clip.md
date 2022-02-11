# CLIP

[Paper](https://arxiv.org/pdf/2103.00020.pdf) | [Talk](https://www.bilibili.com/video/BV1SL4y1s7LQ)

CLIP自提出以来就引起很大关注，其方法简单但又效果很好。CLIP的迁移学习能力非常强，预训练好的模型能够在任意一个视觉分类的数据集上取得不错的效果，而且是zero-shot的，即它完全没有在这些训练集上做过训练，就能得到很好的效果。文章在30多个数据集上做了测试，包括了OCR，视频动作检测，坐标定位以及一些细分类任务，涵盖面非常广，尤其是在ImageNet上的结果，CLIP在不使用ImageNet训练集的情况下，就可以取得和之前ResNet50在有监督情况下的结果。

## Part1.标题&作者

Learning Transferable Visual Models From Natural Language Supervision：利用自然语言处理来的一些监督信号，去学习一个可以迁移效果很好的视觉模型。

Transferable: 迁移性能好

## Part2.摘要

- 目前的视觉模型训练通常是使用提前定义好的一些物体类别的集合，然后模型通过去预测这些定义好的类别，从而完成模型的训练。
- 提前定义好的类别集合，比如ImageNet 1000类，cifar10 10类，coco 80个类，cityscapes 19个类，kinetics 400个类等。这种提前定义好的样本集合可以大大简化模型学习的难度，但是也限制了模型的泛化性
- 直接从自然语言里获取一些监督信号，
