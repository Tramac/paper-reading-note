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

DETR将目标检测任务直接看成是一个集合预测的问题。本来的任务是给定一张图片，然后去预测一堆框，得到框的位置坐标和框内的物体类别，而这些框其实就是一个集合，对于不同的图片来说，它里面包含的框也是不一样的，每个图片所对应的集合也是不一样的，我们的任务就是给丁一张图片，然后去把这个集合预测出来。

DETR的主要贡献：

- 将目标检测做成一个端到端的框架，把之前特别依赖于人的先验知识的部分都给删掉了，最突出的就是nms部分以及生成anchor的部分；一旦将这两部分移掉之后，就不用费劲心思的去设计anchor，最后也不会出很多框，也不会用到nms，也不会有那么多的超参需要去调，整个网络就会变得非常简单；

DETR主要提出的两个东西：

- 新的目标函数：它通过二分图匹配的方式，强制模型去输出一组独一无二的预测，也就是说没有冗余的框了，每个物体理想状态下就会生成一个框；
- 使用了transformer encoder-decoder架构：一个细节是transformer decoder中还有另外一个输入learned object queries（有些类似anchor的意思）；

DETR的优势：

- 简单：不依赖其它的库，只要硬件支持cnn和transformer，就一定能支持detr；
- 性能：在COCO上与Faster RCNN基线网络取得差不多的效果，而且模型内存、速度和Faster RCNN也差不多；
- 拓展性：detr能够非常简单的拓展到其它任务上，比如全景分割。

## Part3.引言

DETR的引言其实就是摘要的加长版本，然后加了很多细节。

目标检测任务就是对于每一个感兴趣的物体去预测一些框和物体的类别，其实就是一个集合预测的问题。但是现在大多数框架都是用一种间接的方式去处理集合预测的问题：

- proposal based:：RCNN系列工作，Faster R-CNN、Mask R-CNN、Cascade R-CNN；
- anchor based：YOLO、Focal Loss；
- non-anchor based：CenterNet、FCOS；

以上方法都没有直接的去做集合预测的问题，而是设计了一个替代任务，要么是回归要么是分类去解决目标检测问题。但是这些方法的性能很大程度上都受限于后处理操作(nms)，因为这些方法都会生成大量冗余的框。

之前也有一些简化后处理的尝试工作，比如learnable nms、soft nms，它们一定程度上简化了目标检测流程，但是要么是融入了更多的先验知识，要么就是在比较难的benchmark数据集上效果不好。

**DETR的主要流程：**

<img width="678" alt="image" src="https://user-images.githubusercontent.com/22740819/174287527-1aa870a3-4d5f-4051-b586-e4043b4475eb.png">

- 给定一张图片，先用一个CNN去抽取特征；
- 将特征拉直后送入一个transformer的encoder-decoder，其中encoder的作用就是去进一步地学习全局信息，实验显示这种全局特征有助于去移除冗余的框；
- 经encoder全局建模之后，用decoder生成框的输出。
