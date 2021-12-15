# Attention Is All You Need

[Paper](https://arxiv.org/pdf/1706.03762.pdf) | [Talk](https://www.bilibili.com/video/BV1pu411o7BE/)

## Part1. 标题&作者

Transformer开创了继MLP、CNN和RNN之后的第四大类模型。

xxx is all you need

## Part2. 摘要

**sequence transduction models: 序列转录模型**

- 序列转录模型是说给定一个序列，生成另外一个序列。以机器翻译为例，给定一句英文，生成一句中文。

- 序列转录模型通常是基于RNN或CNN，包含encoder-decoder的架构，encoder与decoder之间通过attention连接。

**Transformer**: 仅依赖attention机制，不包含RNN和CNN结构。

## Part3. 结论

- Transformer是第一个仅使用attention来做序列转录任务的模型，同样基于encoder-decoder架构，但是将其中的RNN或CNN替换成了multi-head self-attention
- 在机器翻译任务上，Transformer的训练速度相对于其它结构更快、效果更好
- 纯attention模型可以用于其它领域，图片、音频、视频等

## Part4. 导言

**基于RNN的序列模型的缺点**

- 在RNN中，给定一个序列，必须从左往右的计算，即计算第t个词的输出`h_t`时，是由第t-1个词的输出`h_t-1`和当前词t共同计算的，导致很难并行
- 由于时序信息是逐步向后传递，假如序列很长，容易导致早期的时序信息在后面被丢失，如果不想丢掉需要`h_t`比较大，内存开销就会变大

**attention在RNN中的应用**

- attention在RNN中的应用主要体现在它如何把encoder的信息有效的传递给decoder

## Part5. 相关工作

**基于CNN的序列模型的缺点**

- 卷积神经网络对比较长的序列难以建模，受限于局部窗口操作，想提升感受野必须增加卷积层数
- Transformer中的自注意力机制可以使一层就可以看到整个序列的信息

**multi-head self-attention的由来**

- 卷积操作的一个优点是可以多通道输出，一个输出通道其实可以它可以去识别不一样的模式，为了模拟此效果，transformer引入了multi-head-attention概念，每一个head可以当作一个channel

## Part6. 模型

### 基于encoder-decoder结构的序列转录模型

- 对于encoder而言，假设输入为`(x_1, ..., x_n)`, `x_n`为第n个输入的向量表示，编码器会将其映射为新的表示`(z_1, ..., z_n)`，序列长度仍然为n
- 对于decoder而言，拿到encoder的输出`(z_1, ..., z_n)`，然后生成一个长度为m的序列`(y_1, ..., y_m)`，这里输出序列长度可能与输入序列长度可能不同, n->m
- decoder与encoder最大的不同是，解码器里的词是一个一个生成的，此过程被称为**自回归auto-regressive**，比如说要得到`y_t`时，前面已经得到的`y_1`~`y_t-1`可作为当前的输入，这个过程就是自回归。

### Transformer结构

Transformer也是基于encoder-decoder架构，encoder与decoder均是由self-attention、point-wise、fully-connected layers组成

<img src="https://user-images.githubusercontent.com/22740819/146134838-6eb2864f-9667-487e-8df3-4a705029f7eb.png" width = "300">

图上，左侧部分为encoder，右侧部分为解码器。以中译英为例，encoder的输入inputs就是中文句子，在做预测的时候，decoder是没有外部输入的，实际上是之前encoder的输出来作为decoder的输入。

***1.嵌入层Input Embedding***

- input embedding: 模型输入inputs为一个一个的词，首先需要将其表示为一个向量
- position encoding: 

***2.Encoder***

- Multi-Head self-attention: 多头注意力机制
- Add & Norm: 残差连接 + LayerNorm归一化
- Feed Forward: 其实就是MLP

以上三个操作组成transformer的block，然后N个block堆叠组成encoder，encoder的输出作为decoder的输入使用

在encoder中，由于残差连接要求特征维度一致，简单起见，固定每一层的输出维度均为512

> **LayerNorm与BatchNorm的对比**

<img src="https://user-images.githubusercontent.com/22740819/146143314-5d6127c4-7992-47c8-ac41-3097242137a9.png" width=300>

- BatchNorm：是在batch内在样本间做归一化，均值为0，方差为1；假如n本书，每本书的第一页拿出来，根据n本书的第一页的字数均值做Norm
- LayerNorm：是对每一个样本内做归一化；假如n本书，针对某一本书，这本书的每一页拿出来，根据次数每页的字数均值，自己做Norm

一般的，如果模型的输入是变长的，不使用BatchNorm

（建议这部分直接看视频，视频25:40处）

***3.Decoder***

- Masked Multi-Head Self-Attention：在multi-head self-attention上加了mask
- Add & Norm
- Encoder Block: 该部分和encoder相同

以上三种操作堆叠重复N次，组成decoder

> **Decoder中masked multi-head self-attention的由来**

- 因为decoder做的是自回归任务，也就是说当前输出的input是上面一些时刻的输出，意味着当你做预测的时候你不能看到之后那些时刻的输出，但是在自注意力机制里，每一次都能看到整个完整的输入，为了避免这种情况发生，在decoder训练时，在预测第t个时刻的输出时，不应该看到t时刻以后的那些输入，所以引入了mask

### Attention

<img src="https://user-images.githubusercontent.com/22740819/146146897-0b84ee49-f30f-4a2b-a043-98032e069396.png" width=100>

- attention模块的输入为query, key, value，其中key-value成对，三者均为向量
- query与key计算相似度得到权重，然后value通过得到的权重加权得到新的输出output

***1.Scaled Dot-Product Attention***

<img src="https://user-images.githubusercontent.com/22740819/146154573-005a7d8f-b2b9-4b1b-b6a3-4b4ba47e50a0.png" width=200>

<img src="https://user-images.githubusercontent.com/22740819/146150223-c5ad5d6e-b7ae-454b-8f45-96371924bc7a.png" width=250>

其中，query和key维度相同`d_k`，value维度为'd_v'，query与key计算内积，作为相似度，然后除以`sqrt(d_k)`，然后再通过一个`softmax`得到权重

> **为什么要做scaled(除以sqrt(d_k))?**

- 当d_k不是很大时，其实做不做scaled影响不大；当d_k比较大时，所得到的点积结果可能会比较大，当值的范围比较大时，值与值之间的相对差距就会比较明显，经softmax处理之后，较大的值会趋向于1，较小的的值趋向于0，导致有点两极分化，进而会导致在计算梯度时梯度会比较小，收敛困难，paper中的d_k=512，算比较大

> **attention中的mask如何做？**

- mask主要是为了避免你在第t时间看到以后时间的信息。假设query与key等长都为n，对于第t时间的`q_t`，在做计算时，应该只能看到`k_1`~`k_t-1`，但是注意力机制的性质导致`q_t`会和所有的`k`做计算。解决方法：在query与key计算之后，将`q_t`与`k_t`～`k_n`的计算结果置为一个极小值（如-1e-11），这样在softmax之后这部分权重基本趋向于0，这样在和value计算加权时相当于只用了`v_1`~`v_t-1`的信息（这部分建议看视频，42:00处）

***2.Multi-Head Attention***

<img src="https://user-images.githubusercontent.com/22740819/146154695-92091fa7-cdf5-456a-8b10-18a1100723bb.png" width=200>

- multi-head attention：先通过线性层分别把q, k, v投影到低维，然后把低维空间的q、k、v分别送入h个attention模块计算得到h个输入，然后将H个输出concat起来，再投影回来得到最终输出
- **意义**：基于dot-product的attention中其实是没有可学习参数的，但是为了识别不同的模式，希望有一些不一样的计算相似度的方法；每一次计算attention之前都先投影到低维空间，其中投影矩阵w是可学的，也就是说给你h次机会，希望可以学到不同的投影方法，使得在那个投影进去的度量空间能够去匹配不同的模式它需要的一些相似函数，类似于CNN中的channle

> **注意**：由于残差连接的存在，multi-head attention的输入与输出维度需要是一致的。所以，h个低维向量concat起来需要和输入维度一致，一般都会将低维空间的维度设置为`d_in // h`

***3.multi-head attention在transformer中的应用***

Transformer中分别有三处用到了multi-head attention，其作用分别如下:

- Encoder中的multi-head attention: 这里的q、k、v是相同的值copy三份，所以是self-attention
- Decoder中的masked multi-head attention: 这里的输入同样q、k、v是相同的，也是self-attention，与encoder中不同的是用了mask
- Decoder中的multi-head attention: 这里的key-value来自于encoder的输出，query来自于decoder的masked multi-head attention，可以理解为通过这里的query将encoder的输出中感兴趣的信息提取出来

（建议看视频理解，48:00处）

### Position-wise Feed-Forward Networks

其实就是一个单隐层的MLP，维度从512->2048->512

- point-wise: 指对输入的每一个词过一次MLP，或者说MLP只作用在最后一个维度

***1.Transformer与RNN传递时序信息方式的对比***

<img src="https://user-images.githubusercontent.com/22740819/146162754-12e7588e-7ec3-477e-99cf-454937648684.png" width=250>

- transformer中的attention已经汇聚了序列信息，所以MLP只针对单点计算即可（每个词单独过MLP）
- RNN中的序列信息是逐步由前向后传递，将上一时刻的信息作为下一个时刻的输入

### Embeddings and Softmax

- embedding: 将输入的词token映射为向量表示，向量维度为`d_model`，paper中为512
- 编码器、解码器、最后softmax之前的3个embedding共享权重，使得训练更简单

### Positional Encoding

attention本身是不具备时序信息的，query和key计算相似度后与value加权所得到的结果和value在序列中的位置没有关系，也就是说给定一个序列，如果将序列的顺序打乱，所得到的结果是一致的，这显然是不合理的。一个句子把词的顺序打乱可能语义会发生变化，所以需要把时序信息加进来

- RNN中，当前时刻的输入是上一时刻的输出，是按时刻逐步计算的，本身含有时序信息
- Transformer的做法是在输入里加入时序信息，假设某个词所在序列中的位置是i，把i映射为向量加到输入里（position encoding）

### self-attention与rnn, cnn复杂度等的对比

***1.复杂度对比***

<img src="https://user-images.githubusercontent.com/22740819/145567402-7799b78f-756b-452b-9dd0-30527c237bd2.png" width=500>

- Complexity per Layer: 模型计算复杂度，越小越好
- Sequential Operations: 顺序的计算，或者说你下一步的计算需要等前面多少步计算完成，影响并行度
- Maximum Path Length: 一个信息点走到另一个信息点要走多远（可以理解为时序信息的传递效率）

表中，n为序列长度，d为向量长度；当序列比较长时，self-attention计算复杂度就会变高

***2.归纳偏置***

- CNN中的inductive bias: locality和spatial invariance，即空间相近的像素有联系而远的没有，和空间不变性（kernel权重共享）
- RNN中的inductive bias: sequentiality和time invariance，即序列顺序上的timesteps有联系，和时间变换的不变性（rnn权重共享）
- attention缺少归纳偏置，对整个模型的先验假设更少，导致需要更多的数据和更大的模型才能训练出来和RNN、CNN同样的效果。所以，现在基于transformer的模型一般比较大，而且对数据量要求高

## Part7. 实验

**WMT 2014数据集**：英语翻德语数据集，包含4.5w个句子对

## Part8. 评论

略。
