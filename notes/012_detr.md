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
- 经encoder全局建模之后，用decoder生成很多预测框；
- 把预测框和ground truth框做匹配，最后在匹配上的框里面去计算loss。

图中未画出的object query: 当有了图像特征之后，还会有一个object query，query其实就是限定了你要出多少个框，然后通过query和特征不停地去做交互（在decoder里做自注意力操作），从而得到最后的输出的框。论文中选择的框数是100，意味着无论是什么图片，最终预测得到的都是100个框。

接下来的问题就是：所预测出的100个框，如何去和ground truth去做匹配计算loss呢？这也就是本文最重要的一个贡献，把该问题看做是一个集合预测的问题，最后就可以利用二分图匹配的方法去计算loss。
> 以上图为例，ground truth只有两个框，在训练时，通过计算预测得到的100个框和这两个ground truth框之间的matching loss，从而得到在这100个预测中哪两个框是独一无二的对应到图中红色和黄色ground truth框。一旦决定好这个匹配关系之后，然后才会想普通的目标检测一样，去算一个分类的loss，再算一个bounding box的loss，至于那些没有匹配到ground truth的框，其实就会被标记为"no object"，也就是所谓的背景类。

推理阶段：推理时前三步和训练时相同，只有第四步不同，由于推理时不需要计算loss，只需要在第三步得到预测框之后，通过卡定一个阈值，比如置信度>0.7的结果作为最终输出。

实验结果：DETR在COCO数据集上，取得了和fast rcnn相当的结果，无论在ap指标还是在模型大小和速度上都差不多。值得一提的是，DETR在大物体的上的ap结果会好很多，这得益于transformer中优异的全局建模能力，而基于anchor的方法就会受限于anchor的大小；但同时DETR也有缺陷，比如在小物体上的效果就不太好（deformble detr通过多尺度的特征解决了小物体的问题，同时也解决了detr训练慢的问题）。

## Part4.相关工作

该节大概讲了三个部分：一是先讲了一下集合预测这个问题，以及之前大家都是用什么方法去解决集合预测问题；二是介绍了一下transformer以及parallel decoding，就是如何不像之前的transformer decoder一样去做那种自回归的预测，而是一次性的将预测结果返回；三是介绍了目标检测之前的相关工作。

**目标检测**

现在大多数的检测方法都是根据一些已有的相关猜测然后去做预测。比如，对于two-stage的目标检测方法来说，它们的猜测就是中间的proposal，对于single-stage的目标检测方法来说，它们的初始猜测就是anchor，或者说是物体的中心点。之前这些方法的性能和刚开始的初始猜测非常相关，所以如何做后处理得到最后的预测对最后的性能的影响是至关重要的。作者从以下两个方面阐述了这件事情：

- Set-based loss：用集合的思想来做，之前也有类似的用集合思想来做目标检测的方法，比如learnable nms和relation network，它们都可以利用类似自注意力的方法去处理物体之间的联系，从而得出独一无二的预测结果，不需要任何的后处理步骤，但是这些方法的性能往往比较低，为了和最好的方法对齐，这些方法经常需要用一些人工干预，比如用手工设计的场景特征去帮助模型学习。而DETR的目标是将目标检测做的尽可能的简单，不希望过程特别复杂，也不希望用到过多的人工先验知识。
- Recurrent detectors：之前也有用encoder-decoder结构做目标检测的工作，当时的recurrent detector都是用的rnn结构，属于自回归模型，所以时效性和性能会很差。而DETR用的是transformer的encoder-decoder结构，encoder使模型能够得到全局的信息，而且不带掩码的decoder可以使目标框一次性的输出，从而达到了parallel decoding。

## Part5.模型方法

本章主要讲两个东西：（1）基于集合的目标函数；（2）DETR网络结构。

### 5.1 Set prediction loss

DETR的输出是一个固定的集合，无论输入的图片是什么，最终都会输出n个框（论文中n=100）。一般的，n=100应该是要比图中所包含的物体数多很多的，普通的一张图片也就包含几个或十几个物体，对COCO数据集而言，包含物体最多的也没有超过100，所以设置成100是足够用的。

DETR每次出100个输出框，但实际上一个图片的GT box可能只有几个，如何去做匹配计算loss呢？如何知道哪个预测框对应哪个GT box呢？作者将该问题转化成了一个二分图匹配的问题。

#### 二分图匹配

<img width="176" alt="image" src="https://user-images.githubusercontent.com/22740819/182006137-acdd45b7-9bfd-4412-971c-6c615b81b454.png">

一个例子：如何分配一些工人去做一些工作，使得总支出最少。如上图，有三个工人abc，去完成三个工作xyz，因为每个工人个体的差异，完成每项工作所需的时间是不同的，于是有一个cost matrix，每个元素代表完成时间，二分图匹配就是寻找一个最优解使得cost最小。

由上可知，目标检测也可以转换成二分图匹配问题，abc看作是100个预测的框，xyz看作是GT_box，cost matrix中元素放的就是loss值，然后通过scipy.linear_sum_assignment包（匈牙利算法）就可以得到最优解。

寻找最优解和之前利用先验知识去把预测框和proposal或者anchor做匹配的方式是差不多的，只不过这里的约束更强，就是一定要得到一个一对一的匹配关系，不像之前是一对多，也就是说现在只有一个框和Ground Truth框是对应的，这样后面才不需要去做nms后处理。**一旦确定100个框中有哪几个框是和Ground Truth是对应的，就可以计算loss了**。

<img width="572" alt="image" src="https://user-images.githubusercontent.com/22740819/182007696-2feffa27-da07-4040-8924-8c2653a58dc8.png">

损失函数如上，也是包含分类损失与回归损失两个部分，DETR中有两个与普通的目标检测不同的地方：

- 为了使分类损失与回归损失在相同的取值空间，去掉了log计算；
- 回归损失中不再使用L1 loss，因为L1 loss和出框的大小有关系，框越大，计算得到的loss就容易大。因为DETR用了transformer，对大物体比较友好，经常会出大框，就会导致loss大，不利于优化，所以DETR中不仅用了L1 loss，还使用了一个generalized iou loss，该损失与框的大小没有关系。

### 5.2 网络框架

<img width="715" alt="image" src="https://user-images.githubusercontent.com/22740819/182016059-629a0224-f70f-4c1b-aa6b-7b3c78ca387c.png">

DETR前向过程：

- 假设输入尺寸为3x800x1066，首先通过一个CNN得到特征，维度为2048x25x34（1/32），由于后续需送入transformer，通过1x1卷积降维至256x25x34；
- 通过Positional encoding给transformer加入位置编码，这里用的是一个固定的位置编码，维度为256x25x34（因为需要和上面的特征相加，维度必须一致），相加之后就是transformer的输入；
- 将特征拉直（25x34 -> 850）为850x256，其中850就是输入transformer的序列长度，256为transformer的head dimension；
- 通过6个transformer encoder进行特征提取，输出的特征维度不变，仍然为850x256；
