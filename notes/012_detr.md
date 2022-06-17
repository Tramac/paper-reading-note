[Paper](https://arxiv.org/pdf/2005.12872.pdf) | [Talk](https://www.bilibili.com/video/BV1GB4y1X72R?spm_id_from=333.999.0.0)

## Part1.标题&作者

End-to-End Object Detection with Transformers: 使用transformer来做端到端的目标检测。

End-to-End对目标检测任务意义很大，事实上从深度学习开始火一直到DETR，目标检测领域都很少有End-to-End的方法，大部分方法至少都还需要一个nms（非极大值抑制）后处理操作，无论是proposal based方法
还是anchor based方法，亦或是non-anchor based的方法，最后都会生成很多的预测框，需要利用nms来去除冗余的框。由于nms的存在，模型在调参上就变得很复杂，而且就算训练好一个模型，部署起来也非常困难，
因为nms这个操作不是所有硬件都支持的。所以，一个简单的End-to-End的目标检测系统是大家一直梦寐以求的。

DETR解决了上面的痛点，它既不需要proposal也不需要anchor，直接利用transformer能全局建模的能力，把目标检测看作是一个集合预测的问题。而且，也因为有了这种全局建模的能力，DETR不会输出那么多冗余的框，
最后出什么结果就是什么结果，不需要再用nms去做后处理。如此一来，让模型的训练和部署都变得简单不少。

作者在官方代码里也有提到，他们的目的就是不想让大家都觉得目标检测是一个比分类难很多的任务，它们其实都可以用一种简单优雅的框架来做，而不像之前的很多目标检测框架一样需要很多的人工干预、先验知识，而且还需要
很多复杂的库，或者普通的硬件不支持的一些算子。

作者团队全部来自Facebook AI。

## Part2.摘要

