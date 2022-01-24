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

- Swin采取了在小窗口之内进行自注意力的计算，而不是在整图上计算，这样一来只要是窗口大小是固定的，那么自注意力的计算复杂度就是固定的，整张图的计算复杂度就会跟这张图片的大小形成线性增长的关系，类似于卷及神经网络中的locality局部相关性的先验知识。就是说语义相近的物体大概率会出现在相邻的区域，所以即使是在一个小范围的窗口里去计算自注意力也是够用的，全局的计算自注意力对于视觉任务有点浪费资源。
- 多尺度特征：卷积神经网络中通过池化操作，增大卷积核的感受野，从而使得每次池化后得到的特征可以抓住物体的不同尺寸；类似的，Swin Transformer中也提出了一个类似于池化的操作patch merging，就是把相邻的小patch合成一个大patch，这样一来，合成得到的大patch就可以看到之前4个小patch看到的内容，它的感受野就增大了，同时它也能抓住多尺度的特征。一旦有了多尺度信息，4x、8x、16x的特征图，很自然的就可以把这些多尺寸的特征图输给一个FPN，就可以做检测了。

**移动窗口操作：**

<img src="https://user-images.githubusercontent.com/22740819/150480563-2ba49171-8fc0-4a47-954d-29f5d321f554.png" width=400>

假设，在Transformer第L层，将输入或特征图分成4个小窗口，就会有效的降低序列长度，从而减少计算复杂度。图中，灰色的小patch为最基本的单元（4x4像素大小的patch），每一个红色的框是一个中型的计算单元（窗口），在Swin Transformer中，一个窗口中默认有7x7=49个小patch。shift操作就是将整个特征图（示意图中的4个窗口）往右下角整体移动两个patch，变成了右图中的格式。然后在新的特征图中，再次把它分成4个窗口，最终得到了9个窗口。

shift的好处是窗口与窗口之间可以进行互动了。如果没有shift操作，各个窗口之间都是不重叠的，如果每次自注意力操作都在窗口内进行，那一个窗口里的patch永远无法注意到其它窗口中patch的信息，这就无法达到使用transformer的初衷了（更好的理解上下文）。如果这些窗口都不重叠，那么自注意力就变成了孤立自注意力了，就没有了全局建模的能力。shift之后，一个窗口的中patch可能会跑到另外一个窗口中，与新的patch进行交互，而这些新窗口中的patch可能是上一层来自其它窗口的patch，最终就达到了cross-window connection，窗口与窗口之间可以交互了，再配合上后面的patch merging，合并到transformer最后几层时，每一个patch本身的感受野已经很大了。再加上shift window操作，所谓的窗口内的局部自注意力也就变相的等于是一个全局自注意力操作了，即省内存效果也好。

## Part4.结论

Swin Transformer是一个层级式的Transformer，而且它的计算复杂度是跟输入图像的大小呈线性增长。

Shifted Window操作，它对很多视觉的任务，尤其是对下游密集预测型的任务是非常有帮助的。

## Part5.模型整体架构

<img src="https://user-images.githubusercontent.com/22740819/150487310-377ba080-060e-4fd0-b32b-b41a4f85ae30.png" width=800>


**前向过程**

第一阶段：patch projection + transformer block * 2

给定一张224x224x3大小的图片，首先将图片分成patch，swin transformer中patch大小为4x4（ViT中为16x16），分块之后大小为56x56x48，然后经过linear embedding，将输入向量化得到张量大小为56x56x96。

> 在transformer计算中，前面的56x56会拉直变成3136（序列长度），输入特征则变成3136x96（ViT中，patch为16x16，序列长度为196），而3136长的序列对Transformer还是太长，所以因为了窗口计算，每个窗口里只有7x7=49个patch，相当于序列长度只有49，解决了计算复杂度问题。

Transformer本身不改变输入尺寸，在经过两层transformer block之后，输出大小还是56x56x96。

第二阶段：Patch merging

<img src="https://user-images.githubusercontent.com/22740819/150746039-f5629a7f-6f2c-4eb7-a9dc-9bc5b9d22f5a.png" width=800>

给定一个张量，大小为HxWxC，Patch Merging把相邻的小patch合并成一个大patch，这样就可以起到下采样特征图的效果。假如下采样两倍，选点时就是每隔一个点选一个，图中标号为1的点为选的点，标号为2/3/4的点是要和1合并的点，合并完之后得到4个尺寸均为H/2 * W/2的张量，尺寸缩小了一倍，然后把4个张量在C维度拼接起来，特征尺寸则变成H/2 * W/2 * 4C，相当于用空间的维度换了通道数，为了和CNN中池化操作保持一致（池化后通道数变为2倍），用1x1的卷积对C维度进行降维（4C -> 2C），最终patch merging操作将原始HxWxC大小的张量变成了H/2 * W/2 * 2C的张量，空间大小减半，通道数翻倍。

对于第一阶段的结果，大小为56x56x96，经Patch Merging之后尺寸变为28x28x192，然后经过两个transformer block，尺寸仍为28x28x192。

第三阶段和第四阶段同理，最终尺寸为7x7x768。为了和CNN保持一致，Swin Transformer并没有用ViT中所使用的CLS token，而是直接通过一个global avg pooling将7x7x768拉直成一个768维的向量，然后做下游任务。

## Part6.窗口自注意力

**动机** 

全局的自注意力会导致平方倍的复杂度，尤其对视觉领域中密集型预测任务或者对大尺寸输入的图像，全局计算自注意力的复杂度会非常高。

**窗口如何划分**

<img src="https://user-images.githubusercontent.com/22740819/150756350-c2afd5a0-352e-4d41-8a19-187e365d594a.png" width=100>

以第一阶段后的输出为例。首先，对于尺寸为56x56x96的输入会被平均的分为一些没有重叠的窗口，每一个窗口中含有m * m个patch，Swin Transformer中m默认为7，然后所有的自注意力计算都在这些小窗口中完成（序列长度永远是7x7=49），而输入特征图会被分为8x8=64个窗口（56/7=8）。

**计算复杂度对比**

<img src="https://user-images.githubusercontent.com/22740819/150757586-9ec78849-46d4-466c-ac34-fe595db6fe2d.png" width=500>


其中，h、w代表一个输入有h * w个patch，C为patch的特征维度，M为窗口中每行的patch数量。

公式的具体推倒建议看视频：31:00处。

## Part7.移动窗口自注意力

**动机**

窗口自注意力虽然很好的解决了内存和计算量的问题，但是窗口与窗口之间没有通信，就无法达不到全局建模，会限制模型的能力。

<img src="https://user-images.githubusercontent.com/22740819/150758446-a913a372-caf5-4649-b041-41d50bc00e36.png" width=400>

<img src="https://user-images.githubusercontent.com/22740819/150758959-ae65a641-79e0-442d-ad0d-f1b508ddc014.png" width=300>

<img src="https://user-images.githubusercontent.com/22740819/150759278-33f9ddf5-1f5e-43f8-bdf2-a84802e1884d.png" width=200>


在Swin中，transformer block中每次都是先做一次基于窗口的多头自注意力计算，然后再做一次基于移动窗口的多头自注意力，根据Part3中的解释，这样就可以起到窗口与窗口之间交互的作用。

**基于mask计算的移动窗口自注意力**

<img src="https://user-images.githubusercontent.com/22740819/150761604-d539027c-cf38-4bec-9512-adaecef5f405.png" width=300>

基础版本的移动窗口会由原来的4个窗口变为9个窗口，而且每个窗口里的patch数量大小不一，如果想做快速运算，把所有窗口压成一个batch就无法做到了。一种解决方式是在小窗口周围padding 0操作可以是长度一致，但是之前是指用算4个窗口，现在要算9个窗口，计算复杂度还是变高的。

<img src="https://user-images.githubusercontent.com/22740819/150762439-901787d9-579d-4762-aa2e-358d85027661.png" width=600>

Swin给出的解法是，对于移动后得到的9个窗口，做一次循环移位，然后再把移位后的窗口重新划分为4个窗口。这样最终得到的窗口还是4个，窗口数固定，计算复杂度也就固定了。但是循环移位的窗口中的patch有一些原本并不是相邻的，理论上不应该进行自注意力计算，Swin中通过一些masked多头自注意力使得即使同一窗口中的一些patch也互不干扰，整体计算完自注意力之后再通过循环回去将窗口位置还原回去。

<img src="https://user-images.githubusercontent.com/22740819/150772755-be5f5498-6023-480c-b1ff-c54110cca2a9.png" width=600>

掩码的解释详细看视频：39:10

issue: https://github.com/microsoft/Swin-Transformer/issues/38

**Swin Transformer的几个变体**

<img src="https://user-images.githubusercontent.com/22740819/150773201-569b5f1d-947d-4642-9600-dd0d07d64feb.png" width=500>

其中Swin-Tiny的计算复杂度与resnet50差不多，Swin-Small与resnet101差不多。

## Part8.实验

**分类**

分别在Imagenet-1k和ImageNet-22k上做预训练，在ImageNet-1k上做测试。

**检测**

**分割**

**消融实验**

移动窗口、相对编码的影响
