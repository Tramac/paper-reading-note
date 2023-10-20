# LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

[paper](https://arxiv.org/pdf/2106.09685.pdf) | [code](https://github.com/microsoft/LoRA) | [LoRA-PEFT](https://github.com/huggingface/peft)

## 1 题目&作者

LoRA：语言大模型的低秩适配。目前除了 NLP 领域，LoRA 在 stable diffusion 中的应用也异常广泛，所以这里叫**大模型的低秩适配**更合适。

作者来自微软团队，Edward Hu [主页](https://edwardjhu.com/)。

## 2 摘要

自然语言处理的一个重要范式是先在一般领域的数据上进行大规模训练，然后迁移至特定任务或领域上使用。当预训练更大的模型时，重新训练所有的模型参数基本变得不太可行。以 GPT-3 175B 为例，如果将其部署在各个实例上，每个实例都有 175B 个参数，这个成本时非常昂贵的。本文提出了低秩适配（Low-Rank Adaptation），它将模型的预训练权重冻结，然后在 Transformer 的每一层使用可训练的低秩分解矩阵，从而大大减少了下游任务的可训练参数量。与使用 Adam 全量微调的 GPT-175B 相比，LoRA 可以将可训练参数数量减少 10000倍，GPU 内存需求减少 3 倍。就表现而言，LoRA 在 RoBERTa、DeBERTa、GPT-2 和 GPT-3 等多个模型上与全量微调相比性能相当或更优，尽管它有更少的可训练参数，更高的训练吞吐量，另外不像 adapter，没有额外的推理延迟（？）。除此之外，本文还对语言模型适应中的等级缺陷进行了实证研究，证实了 LoRA 的有效性。

## 3 引言

自然语言处理中的许多应用都依赖于一个大的预训练模型，然后再迁移到下游任务，该过程通常是通过微调的方式来实现，一般会对预训练模型的所有参数都进行更新。微调的主要缺点是新模型包含和预训练模型同样多的参数量，当模型逐渐变得很大时，计算和时间成本将变得越来越高。

为了缓解这种情况，许多工作尝试在迁移到新的任务时仅调整模型的一部分参数或者学习一个外部模块。这种方式，除了每个任务的预训练模型之外，只需要存储和加载一小部分针对该任务的参数，大大提高了部署效率。然而，目前通过扩展模型深度或减少模型可用序列的方法会引入**推理延迟（inference latency）**，更重要的是，这些方法通常不如 baseline 效果好，只是在速度和性能上做了权衡。

一些工作指出，模型通常是**过参数化**的，它们的有效性可能只依赖更小的内在维度（low intrinsic dimension），**模型主要依赖这个低的内在维度去做任务适配**。所以，假设**模型在适配任务时参数的改变量是低秩的**，由此引出低秩自适应方法lora，**通过低秩分解来模拟参数的改变量**，从而以极小的参数量来实现大模型的间接训练。

<div align="center">
<img width="287" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/654a1e76-c89a-48fd-a626-09000d1df96e">
</div>

如上图，以 GPT-3 175B 模型为例，左侧 W 为预训练模型权重，维度为 dxd，微调过程中只推理不参与训练，右侧 A、B为两个低秩的矩阵，维度分别为 dxr 和 rxd，一般 r 会远远小于 d，微调过程中只更新 A、B两个低秩权重，输出为两个分支的和。可学习参数量由 dxd 降低为 dxr + rxd=dx2r。

LoRA 的优势：

- 使用 LoRA，针对下游任务微调时，预训练模型权重可以冻住，只更新若干个 LoRA 模块（即A、B矩阵），显著减少存储需求和下游任务之间的切换开销；
- LoRA 使训练更高效以及对硬件要求更低，因为对大多数参数来说不需要再计算梯度和维护优化器状态；
- LoRA 简单的线性设计使得可训练矩阵和冻住的权重在 merge 时很容易，方便部署，且没有引入推理延迟；

## 4 问题描述

给定一个预训练好的自回归语言模型 PΦ(y|x)，其中 Φ 表示模型参数。PΦ(y|x) 可以是一个基于 Transformer 结构的通用的多任务模型，比如 GPT，当将该预训练模型迁移至下游任务时，比如文本摘要生成，其训练数据为 context-target 文本对 Z = {(xi , yi), ...}，xi 为文章内容，yi为它的摘要。

全量微调时，模型先用预训练模型权重Φ0做初始化，然后以下面的目标函数逐步更新权重Φ0 + ∆Φ：

<div align="center">
<img width="307" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/8379caac-83cf-4383-a833-61270fa66f83">
</div>

全量微调主要的缺点是，对于每一个下游任务，学习一组不同的参数 ∆Φ，而这组 ∆Φ 的维度和 Φ0 的维度相同，因此，如果预训练模型很大，比如 GPT-3，那么 Φ0 的大小为 175B，这么多的参数在存储和部署多个实例时将变得很难。

本文提出了一个更高效的参数更新方法，模型的参数更新量 ∆Φ=∆Φ(Θ) 被进一步编码成一组维度更小的参数 Θ，其中 Θ 的大小远远小于 Φ0，因此，计算 ∆Φ 的任务变为只优化 Θ：

<div align="center">
<img width="368" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/3ede575d-132d-4f6e-a573-c8cb079417b2">
</div>

因此，本文提出了一个使用低秩表示来编码 ∆Φ 的方法，既计算高效且存储高效。以 GPT-3 175B 为例，可训练参数量 Θ 只有全量权重 Φ0 的 0.01%。

## 5 相关工作

已有的工作主要有两个问题：

- **Adapter Layers 会引入推理延迟**：？
- 直接优化 prompt 比较难

## 6 方法

#### 6.1 低秩分解

一个神经网络一般包括多个密集连接层，这些层通过矩阵乘法来实现，而这些矩阵通常是满秩（full-rank）的。一些工作指出，预训练好的**语言模型具有较低的内部维度（instrisic dimension）**，在任务适配（迁移下游任务）过程中，即使随机投影到较小的子空间，仍然可以学习到相关的信息。受此启发，本文假设在更新权重时，∆Φ 应该也符合低秩的特性。

对于一个维度为 dxk 的预训练权重 W0，在对其更新时我们将其增量  ∆W 低秩分解为两个矩阵 B 和 A，维度分别为 dxr 和 rxk，并且 r 远远小于 d 和 k，即 W0 + ∆W = W0 + BA。在训练期间，W0是冻住不更新的，只有 A 和 B 包含可训练参数，需要指出的是 W0 和 ∆W 均是和相同的输入做乘法运算，然后将其各自的输出相加得到最终的输出。对于模型的其中一层运算，可用下面公式表示：

<div align="center">
<img width="297" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/ef91d0b6-16e7-4e75-b465-e437a89d2da6">
</div>

其中，A 使用高斯分布初始化，B初始化为0，所以在训练开始时 ∆W = BA = 0，这也比较符合常理。需要注意：在实际计算时，对 ∆Wx 做了尺度变换，乘以 α / r，这个操作**可以减少调整超参数的需要**。

#### 6.2 Transformer 中使用 LoRA

原则上，可以将 LoRA 应用于神经网络中的任意权重。在 Transformer 结构中，self-attention 模块有 4 个矩阵，分别为 Wq，Wk，Wv和 Wo，以及两个 MLP模块中的矩阵。本文仅在 attention 模块的矩阵中使用 LoRA，MLP 模块在训练时冻住不训练。

**优点和限制**

- 最大的优点是减少内存和存储的需求，相比全量微调可降低 3 倍。以 GPT-3 175B 为例，显存消耗从 1.2T 减少到 350G；当 r=4，且仅在Wq 和 Wv 上使用 LoRA 时，checkpoint 大小减少了 10000x 倍，从 350G 到 35M，可以用更少的 GPU 进行训练，避免 I/O 瓶颈。
- 可以在部署时以更低的成本切换任务，只替换 LoRA 权重，而不是所有参数。
- 与全量微调相比，速度提高了 25%。

## 7 实验

本文在多个模型上（RoBERTa、DeBERTa、GPT-2、 GPT-3 175B），多个任务上（NLU、NLG），多个数据集上（GLUE、WikiSQL、SAMSum）做了实验。

几种对比方法：

- Fine-Tuning (FT) ：全量微调，或微调某几层的全部参数；
- Bias-only or BitFit：只训练 bias，其余参数冻住；
- Prefix-embedding tuning (PreEmbed)：在输入的 tokens 中插入特殊的 tokens，这些特殊 tokens 一般不在词表中并且已经训练过了，我理解应该是只学习词嵌入层；
- Prefix-layer tuning (PreLayer)：是上面方法的扩展，只学习 transformer 层后的激活层；
- Adapter tuning：在 self-attention 模块和 MLP 模块与残差连接之间插入 adapter（前面频繁提到的会引入推理延迟的方法）；
- LoRA：只在  self-attention 的 Wq 和 Wv 中使用 LoRA，可训练参数量由 r 决定，|Θ| = 2 × L × d × r，其中L为使用 Lora 的层数；

**文本理解任务对比**

<div align="center">
<img width="700" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/9171076b-4344-4b62-ada3-9e2f7b203581">
</div>

**文本生成任务对比**

<div align="center">
<img width="700" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/a8a9ea4c-e862-4b48-8291-cc720cb71525">
</div>

<div align="center">
<img width="700" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/95a3db97-fd55-4c60-b29e-f7dcd7b95777">
</div>

从上表的结果看，LoRA 在参数量和效果上的 trade-off 做的最好。

## 8 深入 Low-Rank 的应用

#### 8.1 哪些 weight 应该使用 LoRA

Transformer中的权重矩阵比较多，比如 self-attention 部分：Wq、Wk、Wv、Wo；MLP部分：W1、W2。论文只研究了LoRA应用于transformer Attention层上的效果，发现**同时对Wq和Wv应用Lora效果较好**。

<div align="center">
<img width="700" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/ded26ef3-42e2-4adf-ace1-9e587baba01a">
</div>

#### 8.2 r 参数的影响

对于一般的任务，rank=1,2,4,8 即可，而对于一些领域差距比较大的任务可能需要更大的rank。

<div align="center">
<img width="700" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/795e808b-1e8a-4ecb-99aa-15088fc8b0d6">
</div>

## 9 结论

微调大语言模型是非常昂贵的，本文提出 LoRA ，一个有效的低秩适配方法，既不引入推理延迟，也不减少输入序列的长度，同时保持模型的高质量，重要的是，当部署成服务时，它可以通过共享大多数模型参数来快速切换下游任务。虽然本文只专注于 Transformer 语言模型，但其原理也适用于任何具有密集型层的网络。

未来工作：

- LoRA 可以和其它有效的适配方法相结合，有可能带来改进；
- 微调和 LoRA 背后的机制尚不清楚，如何将预训练期间学到的特征更有效的转化到下游任务中，或许使用 LoRA 比全亮微调更容易回答这个问题；
- 目前主要依靠启发式方法来选择应用 LoRA 的权重矩阵，有没有更有指示性的方法来做？
- ΔW 是低秩不足（？）的，表明 W 可能也是低秩不足的，这或许有一定的启发作用；

## Bonus: 几个关键点

- 什么是low-rank？rank-deficiency？ full-rank？intrinsic dimension？intrinsic rank？
  - Low-rank：低秩，通常指的是矩阵的秩（rank）比较小，即该矩阵包含的信息量比较少，可以被较少的特征向量或奇艺值所表示。在很多应用中，低秩矩阵可以用来表示数据中的共性信息和规律，从而实现数据的压缩、降维和分析等目的。
  - Rank-deficiency：秩亏，指的是一个矩阵的秩比它的行数或列数小，即矩阵中存在线性相关的行或列，从而导致矩阵的行列式为零，不能被逆运算。通俗地说，秩亏矩阵的行或列中存在冗余信息，可以由其它行或列表示出来，因此该矩阵包含的信息量比较少，不能完全描述数据的特征。秩亏矩阵在实际应用中很常见，例如在图像处理和机器学习中，数据矩阵中可能存在噪声、缺失值或冗余特征，导致矩阵秩降低。在这种情况下，我们需要采用一些方法来处理秩亏矩阵，以提高数据的表现力和预测准确率，常用的方法包括：PCA、SVD、核矩阵分解（Kernel Matrix Factorization）等，这些方法可以用来降维、去噪、填补缺失值等。
  - Full-rank：满秩，指的是一个矩阵的秩等于它的行数或列数中的较小值，即矩阵中的所有的行或列都是线性独立的，可以唯一的表示出矩阵的所有元素。满秩矩阵通常具有很好的性质和应用，如可逆、有唯一解、唯一表示等。满秩矩阵在实际应用中也很常见，例如在线性回归、最小二乘法、矩阵分解等问题中，通常需要矩阵满秩的条件来保证算法的有效性和稳定性。
  - Instrisic dimension：内在维度，指数据集中所包含的最少信息所需要的维数，或者说数据集中实际独立变化的方向的数量。它是指数据集固有的本质维度，不受数据在高维空间中的嵌入方式或表示方式的影响。具体来说，如果一个数据集可以被嵌入到一个 d 维空间，而且在这个空间中，数据集的每个样本点只需要 k 个坐标来描述，那么数据集的内在维度就是 k。内在维度对于数据分析和机器学习来说是非常重要的，因为它可以帮助我们更好的理解数据集的本质结构，并且可以减少计算和存储的开销。例如，当处理高位数据时，内在维度可以帮助我们找到数据中真正有用的特征，从而提高模型的性能和准确度。另外，内在维度还可以用于数据压缩、可视化和降维等领域。
  - Intrinsic rank：内在秩，指一个数据集中最重要的线性无关的特征的数量，它可以被看作是数据集的内在维度的一个度量。内在秩的概念通常用于描述低秩矩阵近似问题中，其中我们希望通过一个低秩矩阵来近似一个高秩矩阵，以减少存储和计算的开销。具体来说，如果一个数据集可以表示为一个秩为 k 的矩阵的线性组合，而且这个 k 是数据集中最重要的线性无关特征的数量，那么这个数据集的内在秩就是 k。换句话说，**内在秩是指能够用来描述数据集最重要信息的最少的线性无关特征向量**。内在秩和内在维度的概念有些类似，但内在秩更多的关注于矩阵的结构和线性无关的特征的数量，而内在维度则更多的关注于数据集的实际维度和独立变化方向的数量。在某种情况下，内在秩和内在维度可能是相通的，但在其他情况下，它们可能会有所不同。
- 什么是推理延迟 inference latency？为什么说 LoRA 是 No Additional Inference Latency 的？
  - **Latency**：延时，主要从用户的视角来看，也就是用户提交一个prompt，然后得到response的时间。特殊情况batch_size=1只给一个用户进行服务，Latency是最低的。计算方法为生成一个token所需要的单位时间数，如16 ms/token。
  - **Throughput**：吞吐率，主要是从系统的角度来看，单位时间内能处理的tokens数，如16 tokens/sec。扩大Throughput的方法一般就是提升Batch_size，也就是将一个一个用户的请求由之前的串行改为并行。
  - LLM inference 主要关注两个指标：Latency 和 Throughput。Latency 影响体验，也就是返回结果要快，用户体验好；Throughput 影响系统成本，高 Throughput 则系统的单位时间处理量就到，利用率高，但是会影响到 Latency，这两个指标一般情况下要做 trade-off。
  - LoRA 可以明确的计算并存储 W （W = W0 + BA），而且 W0 和 BA 的维度一致均为 dxk，当切换到其它下游任务时，可以通过 W - BA 得到 W0，从而复用初始权重 W0，只需要加上新的 B'A'即可。我猜是因为 W0 + BA 这个 + 操作太简单了，所以说没有引入推理延迟。
- LoRA 训练参数相比全量微调参数节约多少的计算方式？
  - 使用LoRA 的参数量：|Θ| = 2 × L × d × r，其中 L 为层数；不使用 LoRA 的参数量：|Θ| = L × d × d；节约比例：d / (2r)。
- Adaptation 怎么理解？
  - 文中出现了多次 adaptation 这个词，个人理解，将预训练模型迁移至下游任务的这个过程叫做 adaptation，这也是 LoRA 名字的由来 Low-Rank Adaptation，一种低秩的迁移下游任务的方法。
