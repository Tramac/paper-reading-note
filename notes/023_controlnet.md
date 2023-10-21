# Adding Conditional Control to Text-to-Image Diffusion Models

[paper](https://arxiv.org/pdf/2302.05543.pdf) | [code](https://github.com/lllyasviel/ControlNet)

### 标题&作者

向 text-to-image 模型添加条件控制。顾名思义，即使你的生成模型（如 stable diffusion）的可控性更高。

作者团队来自斯坦福大学。一作 Lvmin Zhang 有很多生成方面的工作，比如 Style2Paints，详细内容可看其[主页](https://lllyasviel.github.io/Style2PaintsResearch/lvmin)。

### 摘要

本文提出一个神经网络架构 -- ControlNet，用来给 text-to-image diffusion model 添加空间条件控制。

我们可以把 diffusion model 的参数划分为 locked 和 trainable，locked block 是想保留预训练模型已有的能力，trainable block 用来学习条件控制。

ControlNet 是一个「zero convolutions」与 trainable 参数相串联起来的结构，其输出与 locked block 的输出进行相加作为最终输出。
> zero convolutions 是指初始化参数全为 0 的卷积层，在未进行训练时，它的输出为 0，确保不会引入噪声而影响到原始模型。

本文在 stable diffusion 模型上测试了多种控制条件，比如边缘、深度、分割、人体姿态等，也测试了单控制条件和多控制条件，用 prompt 与不用 prompt，训练集规模小到 < 50k，大到 > 1M，结果证明模型鲁棒性非常不错。

### 1.引言

原生的 stable diffusion model 若想生成高质量的图像，依赖很多精巧的 prompt 设计，而这些 prompt 通常是不太好掌握的，比如我们常见的关键词有“美女”、“漂亮女孩”等，但是这些词往往不能得到高质量的图像，反而是“4K高清”、“高>质量”、“阴影效果”这些词汇会明显提升画质。

另外，stable diffusion 模型对图像的空间控制性不好，生成想要的图像需要反复的测试 prompt，即便 prompt 再怎么精确，也经常不太符合预期。

那么，我们能否先提供一张图片，作为我们预期得到图像的模板，来指导生成过程呢？

这在 image-to-image 模型中比较常见，这类模型通过学习到从条件图像到目标图像之间的映射关系来达到目的。同时，学术界也有一些通过空间掩模、图像编辑指令、定制化微调来控制 text-to-image 模型的方法。虽然有些问题（如生成图>像修复）可以通过 「training-free」的方式来解决，例如限制 denoising diffusion 过程、编辑 attention 层等，但更广泛的问题，比如 depth-to-image、pose-to-image 等还是需要以数据驱动来重新 end-to-end 的训练。

但是，通过 end-to-end 的方式学会对 text-to-image 模型进行条件控制是极具挑战的。因为，与用于训练 text-to-image 基础模型的数据量相比，用来训练某个控制条件的数据量是很少的。例如，相关数据集中（如 object shape/normal、
human pose extraction等）最大规模的也才 100K，比用来训练 stable diffusion 的数据集 LAION-5B 小了 50000 倍。**直接在大模型上用一个小数据进行微调很容易造成过拟合或灾难性遗忘**。一些研究通过限制可训练参数的规律来减轻>这种遗忘现象，而在我们的问题中，可能需要设计更深或更定制化的网络结构来处理生成过程中的复杂形状、更高语义等问题。

本文提出的 ControlNet 是一个 end-to-end 的网络结构，用来学习对 text-to-image 模型的条件控制。ControlNet 通过锁住模型本身的参数来保证其原本的质量和功能，然后 copy 一份模型的 encoding layers 作为 trainable 参数，以此
来学习条件控制。

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/558877c8-c132-42bb-83a2-372e62cbc8b1">
</div>

实验证明，ControlNet 可以通过多种输入条件完成对 stable diffusion 的控制，比如 Canny edges、Hough lines、user scribbles、human keypoints、segmentation maps、shape normals、depths 等等，如上图。
并且，在单卡 NVIDIA 3090ti 上就可以完成训练。

### 2.相关工作

#### 2.1 微调技术

“传统”的微调方法都是在预训练模型的基础上用新的数据上继续训练，但是随着模型尺度越来越大，预训练数据越来越多，在少量数据上直接进行全量微调往往会导致**过拟合**、**模型坍塌**、**灾难性遗忘**等问题。所以，现在有很多工作
在研究避免以上问题的微调策略：

- HyperNetwork：该方法源于 NLP 领域，通过训练一个小的 RNN 来影响更大的模型；
- Adapter：广泛应用于 NLP 领域，通过在 transformer 中嵌入新的模块来进行任务迁移；
- Additive Learning：先冻结原始模型权重，通过使用剪枝、hard attention、添加少量新参数训练来避免遗忘；
- LoRA：不展开了；
- Zero-Initialized Layers：ControlNet 中所使用的，在方法部分展开解释；

#### 2.2 图像扩散

- Image Diffusion Models：Latent Diffusion Models、Disco Diffusion、Imagen、DALL-E2、Midjourney
- Controlling Image Diffusion Models：控制扩散模型，典型的工作有 Textual Inversion、DreamBooth

#### 2.3 Image-to-Image Translation

从一个图像域到另一个图像域的迁移都叫做 Image-to-Image，典型的方法有：

- Conditional GANs
- StyleGANs

### 3.方法

ControlNet 是一个可以给 text-to-image 模型添加控制的网络结构。

#### 3.1 ControlNet

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/4fd6b556-8928-4889-bc80-114b2d393d6a">
</div>

ControlNet 将附加条件注入到神经网络 block 中。在这里，我们用「network block」表示神经网络中通常被组在一起的层，比如 resnet block、conv-bn-relu block、multi-head attention block、transformer block 等。

假设用 $\mathcal{F}(·;\Theta )$ 表示一个 block，其参数为 $\Theta$，那么，将一个输入特征图 $x$ 转换为另一个特征图 $y$ 可以表示为：

$$
y = \mathcal{F}(x;\Theta)
$$

在本文中，$x$ 和 $y$ 通常是 2D 特征图，如上图中的（a），对一个「network block」添加 ControlNet，首先将该 block 的参数 $\Theta$ 冻住，同时将该 block 复制一份作为「trainable copy」，其参数为 $\Theta_c$，如上图中的（b），「trainable copy」以一个外部条件向量 $c$ 作为输入。

当该结构应用于大模型比如 Stable Diffusion 中时，那些冻住的参数保留了用数十亿张图像训练得到的模型能力，而「trainable copy」模块重新利用该模型去建立一个深的、鲁棒的、强大的 backbone 来处理各种各样的输入条件。

「trainable copy」通过「zero convolution」与「locked model」连接，我们用 $\mathcal{Z}(·;·)$ 来表示「zero convolution」，具体来说，「zero convolution」 是一个 1 × 1 的卷积层，其 weight 和 bias 都被初始化为 0。对于一个 ControlNet 结构，通常有两个「zero convolution」，我们分别用 $\Theta_{z1}$ 和 $\Theta_{z2}$ 表示它们的参数，那么一个完整的 ControlNet 结构可以用以下公式表示：

$$
y_c = \mathcal{F}(x;\Theta) + \mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c;\Theta_{z1});\Theta_{c});\Theta_{z1})
$$

其中，$y_c$ 为 ControlNet block 的输出。在训练的第一步，因为「zero convolution」的 weight 和 bias 均是 0，在上面的公式中，第二项为 0 ，所以有：

$$
y_c = y
$$

这样一来，当训练开始时，「trainable copy」中的有害噪声将不会对网络产生影响。而且，因为 $\mathcal{Z}(c;\Theta_{z1}) = 0$ 并且「trainable copy」也接受输入 $x$，所以它保留了预训练模型的能力并可以进一步学习得到更强的 backbone。「zero convolution」在初始的训练步骤中，通过消除随机噪声作为梯度（？）来保护 backbone 不被训崩，其梯度计算方式在补充材料中有说明。

#### 3.2 ControlNet for Text-to-Image Diffusion

本章使用 Stable Diffusion 作为例子来展示如何使用 ControlNet。Stable Diffusion 本质上是一个拥有 encoder、middle block、skip-connected decoder 的 U-Net 结构，encoder 与 decoder 均包含 12 个 block，加上 middle block 整个模型共有 25 个 blocks，其中 8 个 block 为 down-sampling 或 up-sampling 卷积层，其它 17 个 blocks 均是由 resent layers 和 ViTs 组成的结构，每个 ViTs 包括几个 cross-attention 和 self-attention 层。

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/c1c49fe7-d2de-4c89-a399-33443a58bd6b">
</div>

上图给出了结构示意图。其中Stable Diffusion 的「SD Encoder Block」包括 4 个 reset layers 和 2 个 ViTs，「Text Encoder」使用的为 CLIP 的 text encoder，「Time Encoder」实际为 positional encoding 操作。ControlNet 结构应用在 U-Net 的每一个 encoder 上和 middle block 上。

ControlNet 的这种连接方式非常具有计算效率，因为那些 locked 的参数是不用计算梯度的，实际测试，在单块 NVIDIA A100 PCIE（40G）上，同时优化 Stable Diffusion + ControlNet 比仅优化 Stable Diffusion 多了 23% 的显存占用和 34% 的训练时长。

Image diffusion model 实际是学习一个逐渐去燥的过程，这个过程可以直接从「pixel space」开始，也可以从「latent space」开始。Stable Diffusion 的「training domain」为 「latent images」，因为从「latent images」训练时比较稳定的。

> 我们把 512 x 512 大小的原始图像称为「pixel-space images」，使用 VQ-GAN 中的预处理方法将其处理为 64 x 64 的「latent images」。

对于 ControlNet 的输入，我们也是先将「conditioning image」（edge、pose、depth 等）从 512 x 512 转为 64 x 64 大小的「feature space vector」$c_i$，具体操作是，使用一个包含 4 层卷积层的小网络 $\varepsilon (·)$ 将 「image-space」condition $c_i$ 转为 「feature space」conditioning vector $c_f$：

$$
c_f = \varepsilon(c_i)
$$

最终将 $c_f$ 送给 ControlNet 作为输入。

#### 3.3 训练

给定一张输入图像 $z_0$，diffusion 过程是逐渐对该图像添加噪声，最终生成一张噪声图 $z_t$，其中 $t$ 为添加噪声的次数。

给定一组「conditions」，包含添加次数 $t$，text prompts $c_t$ 以及 condition $c_f$，diffusion 算法的目标是学习一个网络 $\epsilon_\theta $ 去预测添加在噪声图像 $z_t$ 上的噪声，损失函数为：

$$
\mathcal{L} = \mathbb{E}_{z_0,t,c_t,c_f,\epsilon \sim \mathcal{N}(0,1)}[||\epsilon - \epsilon_\theta (z_t,t,c_t,c_f)||_{2}^{2}] 
$$

在训练过程中，我们随机替换 50% 的 text prompts $c_t$ 为空字符串，该操作增强了 ControlNet 识别输入的 condition 的语义能力，从而不用显式的给一个 prompt 专门提示 condition 的类别。

因为「zero convolutions」并没有将噪声引入到网络中，所以模型总是可以生成高质量的图像。经过观察，模型并不是逐渐学习到条件控制的能力，而是突然地拥有遵循「conditioning image」的能力。

#### 3.4 推理

完成训练之后，我们可以通过多种方式来使用 ControlNet 去影响扩散过程。

**Classifier-free guidance resolution weighting**

控制输入条件的权重。

Stable Diffusion 生成高质量图像依赖一种叫「Classifier-Free Guidance」的技术，CFG 的公式表示为：

$$
\epsilon_{prd} = \epsilon_{uc} + \beta_{cfg}(\epsilon_c - \epsilon_{uc})
$$

其中，$\epsilon_{prd}$、$\epsilon_{uc}$、$\epsilon_c$ 和 $\beta_{cfg}$ 分别为模型的最终输出、没有添加条件控制时的输出、条件控制以及权重参数。（该公式主要是想说明条件控制的强度对模型最终输出效果的影响）

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/26e0c48a-8da2-491f-b3ad-f4adcdf09ac2">
</div>

如上图，（a）为输入的控制条件，即$\epsilon_c$，（b）为不使用控制条件的结果 $\epsilon_{uc}$，（c）为控制条件权重很大时的结果，（d）为权重比较均衡时的结果。

**Composing multiple ControlNets**

同时使用多个控制条件。

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/01dbbe2a-3b26-4366-b3ea-3d4e01ca3513">
</div>

上图结果同时使用了 pose 和 depth 两个控制条件，从结果来看可以同时对结果产生影响。

### 4.实验

本文实验了在 Stable Diffusion 模型中使用 ControlNet 时的效果，控制条件包括  Canny Edge、Depth Map、Normal Map、M-LSD lines、HED soft edge、ADE20K segmentation、Openpose、user sketches 等等。

#### 4.1 定性结果

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/33169ee1-845a-4234-b0e4-9835367002e0">
</div>

上图给出了仅使用控制条件，不使用 text prompt 时的结果，可以看到仅根据 conditions 也可以得到不错的效果。

#### 4.2 消融实验

ControlNet 的结构设计：

- 「zero convolutions」vs 标准卷积
- 用整个 block 作为「trainable copy」vs 仅把 block 中的单个卷积层作为「trainable copy」

text prompt 设计：

- 不使用 prompt
- 不精确的 prompt
- 有冲突的 prompt
- 完美的 prompt

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/844cb587-0741-4965-87dd-e2beeb0faede">
</div>

从上图来看，肯定是使用整个 block 作为「trainable copy」+「zero convolutions」+ 完美的 prompt 时的结果最好。

#### 4.3 定量评估

- 用户质量排序

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/7769e379-ef07-4836-8793-661547a47b8a">
</div>

- 与其它方法直观结果对比

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/44dc78e5-4b49-49a3-abdb-0e418201598f">
</div>

- 训练集大小的影响

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/c30a2470-54cd-4e51-9916-9cd92ca7a4c9">
</div>

- ControlNet 使用在其它生成模型上时

<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/46cad5f2-0108-4454-8974-54fdf2aaaf78">
</div>
