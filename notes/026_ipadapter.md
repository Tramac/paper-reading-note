# IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/5dbaf207-438f-4975-97c0-3cf6d910ec44">
</div>
一句话：IP-Adapter 提供了一种 image prompt 能力，它不修改已有的 text-to-image 模型结构，可以和 ControlNet、text prompt 共存使用，轻量且高效。它主要引入的是图像的整体信息，如风格、内容，可以理解为“垫图”功能。

## 题目&作者
Text Compatible Image Prompt Adapter：文本 prompt 与图像 prompt 可兼容
Text-to-Image Diffusion Models：针对于文生图任务
作者均来自腾讯AI实验室。

## 摘要
近年来，文生图模型因其出色的生成能力而备受关注，能够创建高保真度的图像。然而，仅使用文本 prompt 来生成所需图像是非常具有挑战性的，因为这通常涉及复杂的 prompt 工程。与文本 prompt 相比，图像 prompt 是另一种选择，正如俗话所说：“一图胜千言”。尽管现有的直接从预训练模型进行微调（主要就是搞一批数据，给定一些较为明确、相对简单的 prompt，在生成时也可以使用这些简单的 prompt 生成想要的图像）的方法一定程度上缓解了此问题，但 finetune 需要大量的计算资源，并且与其他基础模型、文本 prompt 和涉及到网络结构的控制条件不兼容。在本文中，我们提出了IP-Adapter，这是一种有效且轻量级的适配器，能够为预训练的文生图扩散模型实现图像 prompt 能力。其关键设计是解耦的 cross-attention 机制，它将文本特征和图像特征的 cross-attention 层分离。尽管方法简单，但仅有 22M 参数的IP-Adapter可以实现与完全微调的图像 prompt 模型相当甚至更好的性能。由于冻结了预训练的 SD 模型，所提出的IP-Adapter不仅可以推广到从同一基础模型微调的其他定制模型，还可以与现有的可控工具一起使用，实现可控生成。得益于解耦的 cross-attention 策略，图像提示还可以与文本提示很好地配合使用，实现多模态图像生成。

**关键词**：decoupled cross-attention

## 1 引言
图像生成技术随着最近文生图扩散模型的成功而取得了显著进展，例如GLIDE、DALL-E 2、Imagen、Stable Diffusion (SD)、eDiff-I和RAPHAEL等。用户可以通过编写文本 prompt 令文生图模型来生成图像。但是，编写良好的文本 prompt 以生成所需内容并不容易，因为通常需要复杂的提示工程。此外，文本在表达复杂场景或概念时通常无法提供充足的信息量，这可能成为内容创作的障碍。考虑到 text prompt 的上述限制，我们可能会问是否有其他类型的 prompt 可以用来生成图像。一个自然的选择是使用图像 prompt，因为与文本相比，图像可以表达更多的内容和细节，正如常说的：“一图胜千言”。DALL-E 2首次尝试支持图像 prompt，其扩散模型是基于图像 embedding 而不是文本 embedding 进行条件化的，并且需要一个先前的模型来实现文生图能力。然而，大多数现有的文生图模型是基于文本条件生成图像的，例如，流行的SD模型是基于从冻结的CLIP text encoder 提取的文本特征进行条件化的。图像提示是否也可以在这些文生图模型上得到支持？本文工作尝试以简单的方式为这些文生图模型启用图像 prompt 的生成能力。
一些先验工作如 SD Image Variations 和 Stable unCLIP 等工作的缺陷：

 - **计算资源密集**：直接对预训练的文本到图像扩散模型进行微调以支持图像提示，通常需要大量的计算资源；
 -  **模型兼容性问题**：微调后的模型通常不能直接转移到其他从相同基础模型微调的的模型。这意味着为了在不同的应用中使用图像提示能力，可能需要对每个模型进行单独的微调，这增加了工作量和复杂性；
 - **与现有工具的不兼容性**：微调后的模型可能与现有的结构控制工具（如ControlNet）不兼容，这对下游应用使用不太友好。

还有些工作将 text encoder 替换成了 image encoder，虽然缓解了上述微调带来的缺点，但也引入新的问题：

- 无法使用 text prompt 了:)
- 仅仅微调 image encoder 通常不足以保证图像质量，并且可能导致泛化问题

一些工作致力于在不修改原有 text-to-image 模型的前提下，引入 image prompt 能力，如 ControlNet、T2I-adapter 等，通过一个额外的网络结构，可以即插即用的对已有的 text-to-image 模型实现生成控制。其中 ControlNet 聚焦于引入图像结构相关的控制条件，如草图、深度图、语义分割图等，T2I-adapter 主要是图的风格和内容作为控制条件，具体实现是先通过 CLIP image encoder 提取图像特征之后，再通过一些可训练层得到新特征，然后与文本特征融合，以此替换原有的文本特征，后续引入到 UNet 中。这种 image prompt 使用方式，使生成的图像仅部分区域接近于 image prompt， 结果往往也比较差。

本文认为上述方法的主要问题在于文生图模型中的 cross-attention 模块。**预训练模型中 cross-attention 模块的 key projection weights 和 value projection weights 都是被训练来适配文本特征的，因此，将图像特征和文本特征融合之后送入 cross-attention，只能完成图像特征与文本特征的对齐，但这可能会丢失一些图像特定的信息，最终导致仅产生粗粒度的可控生成（比如图像风格）**。

本文所提出的 IP-Adapter 通过解耦的 cross-attention 机制克服了以上一系列缺点，对于 UNet 中的每一个 cross-attention 层，引入一个额外的 cross-attention 层，单独处理图像特征，在训练阶段，只有新增的 cross-attention 层参与训练，原始的 UNet 模型参数被 frozen 住。

总结：

- 提出 IP-Adapter，通过 decoupled cross-attention 机制引入 image prompt，轻量且有效，仅有 22M 参数；
- IP-Adapter 具有可重用性和灵活性，可以应用于从同一基础 SD 模型微调的其他模型，有优秀的泛化能力。此外，IP-Adapter与其他可控适配器（如ControlNet）兼容，使得图像提示能够轻松地与结构控制相结合；
- 基于 decoupled cross-attention 机制，图像提示与文本提示也兼容，可实现多模态图像生成；

## 2 相关工作

### 2.1 Text-to-Image Diffusion Models
text-to-image 模型主要被划分为两类：autoregressive models 和 diffusion models。
- **自回归模型 (Autoregressive Models)**: 早期的文生图模型，如DALLE、CogView和Make-A-Scene，采用自回归方法。这些模型使用一个 image tokenizer （如 VQ-VAE）将图像转换为 tokens，然后通过文 text tokens 作为控制条件，训练一个 transformer 来预测 image tokens。然而，这些模型通常需要大量的参数和计算资源来生成高质量的图像。
-  **扩散模型 (Diffusion Models)**: 近期，扩散模型成为了文本到图像生成的新标准。GLIDE使用级联扩散架构，DALL-E 2采用基于图像嵌入的扩散模型，Imagen使用T5作为文本编码器，SD基于 latent 扩散模型，eDiff-I设计为使用多个条件的集合的文生图扩散模型，Versatile Diffusion提供了一个统一的多流扩散框架，Composer通过联合微调策略实现可控图像合成，RAPHAEL引入了混合专家策略来增强图像质量和审美吸引力。

### 2.2 Adapters for Large Models
因为 finetune 一个大的 pre-trained model 是比较低效的，一个解决方法是使用 adapter，即冻住原始模型参数，新增少部分可训练参数，这些具有少量新增参数的网络结构被称作是 adapter。
在 text-to-image 模型中，adapter 通常被作为引入额外的控制条件来使用。如 ControlNet 和 T2I-adapter 都是引入的 structure 控制条件，另外还有一些类 ControlNet 方法引入的是风格控制条件。

## 3 方法

### 3.1 stable diffusion
老生常谈了，还是简单说一下。
Diffusion models 可以看作由两个过程组成：扩散过程和去噪过程。扩散过程是指逐步往图像中添加随机噪声，使之变成一个符合正态分布的噪声；去噪过程是指逐步预测噪声，通过减去噪声恢复原图的过程。Diffusion Models 也可以通过引入其它输入实现条件控制，比如 text，于是就有了 text-to-image diffusion model。
训练过程的目标函数如下：

$$L_{simple}=\mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0, I),c,t } \left \| \epsilon - \epsilon _{\theta }(x_{t},c,t )  \right \| ^{2}$$

其中，$x_0$ 代表原图像，$c$ 为控制条件，$t$ 代表步数，$\epsilon _{\theta }$ 代表模型参数，$x_t$ 由原图逐步添加 t 步随机噪声得到，具体实现时通常由原图和不同强度的噪声一次生成，即 $x_t=\alpha _{t}x_{0}+\sigma _{t}\epsilon$，其中 t 越大，$sigma _{t}$ 也越大，$x_t$ 的噪声强度越大。噪声图 $x_t$、控制条件 $c$，步数 $t$ 输入到模型 $\epsilon _{\theta }$ 中，预测得到的噪声要尽可能的逼近实际添加的噪声 $\epsilon$ 。

### 3.2 Image Prompt Adapter

目前的一些 adapter 工作效果都不是很好，原因是这些方法都是把 image 特征和 text 特征 concat 之后送入 cross-attention 层中，而预训练模型中的 cross-attention 是适配 text 特征的（因为预训练时只有 text-image 对，没有额外的 image 特征作为控制条件），这就阻碍了模型从 image prompt 中提取特征的能力。
而 IP-Adapter 通过解耦的 cross-attention 来实现 text 与 image 特征的隔离，新增一个 cross-attention 单独负责 image 特征。
IP-Adapter 由两部分组成：1）image encoder 用来从 image prompt 中提取图像特征；2）额外新增的 cross-attention 实现图像特征到 diffusion model 中的集成。

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/98af2434-7dee-450d-88ac-a63601608fa6">
</div>

#### 3.2.1 Image Encoder

和大多数方法类似，IP-Adapter 也是使用 pretrained CLIP image encoder 作为图像特征提取器。在训练阶段，image encoder 也是 freeze 的。

#### 3.2.2 Decoupled Cross-Attention

在原始的 SD 模型中，text feature 是通过 cross-attention 融合到 UNet 中。公式如下：
$$Z^{'} = Attention(Q,K,V) = Softmax(\frac{QK^{T} }{\sqrt{d} } )V$$
其中 $$Q=ZW_{q}$$，Z 为噪声图像特征，$K=c_{t}W_{k}$，$V=c_{t}W_{v}$，$c_{t} 为 text feature。$W_{q}, W_{k}, W_{v}$ 为 weight matrices。

image feature 的引入和 text 一致：
$$Z^{'} = Attention(Q,K^{'},V^{'}) = Softmax(\frac{Q(K^{'})^{T} }{\sqrt{d} } )V^{'}$$
其中$K=c_{i}W^{'}_{k}$，$V=c_{i}W^{'}_{v}$，$c_{i}$ 为 image feature。$W^{'}_{k}, W^{'}_{v}$ 为新增的 weight matrices（新增的参数就这么多），训练时用 $W_{k}, W_{v}$ 的值做初始化。

### 3.3 Training and Inference

训练时，只训 IP-Adapter 的参数，diffusion model 的值不参与训练。
IP-Adapter 的训练数据也是用的 image-text 对（文中说仅使用 image 数据也可以，因为它只涉及到对图像的处理，我想这里想表达的是用 unconditional diffusion model 也可以完成对 IP-Adapter 的训练），其目标函数和原始 SD 的一致：
$$L_{simple}=\mathbb{E}_{x_0, \epsilon,c_t,c_i,t } \left \| \epsilon - \epsilon _{\theta }(x_{t},c_t,c_i,t )  \right \| ^{2}$$

在训练阶段，对 image conditions 做了随机 drop 处理，以此在推理阶段能达到 classifier-free （即使不给定 prompt 也能达到不错的效果）的目的：
$$\hat{\epsilon }_{\theta }(x_t,c_t, c_i,t)=w\epsilon _{\theta }(x_t,c_t,c_i,t) + (1-w)\epsilon _{\theta }(x_t,t)$$
具体实现是，对 image condition 进行 drop 时，把其 CLIP image embedding 置零即可。

text prompt 和 image prompt 之间融合时也有一个权衡，通过参数 $\lambda $：
$$Z^{new} = Attention(Q,K,V) + \lambda \cdot Attention(Q,K^{'},V^{'})$$
当 $\lambda=0$ 时，就退化为了原始的 text-to-image 模型。

## 4 实验

### 4.1 实验设置

#### 4.1.1 训练数据

训练数据 ~ 1kw text-image 对，来自两部分 LAION-2B 和 COYO-700M。

#### 4.1.2 训练细节

基模为 SD-v1.5，image encoder 使用的是 CLIP ViT-H/14，SD 模型中共 16 个 cross-attention 层，所以 IP-Adapter 中也是新增了16 个 cross-attention。
IP-Adapter 所有的可训练参数只有 projection network（从 CLIP image encoder 得到特征到送入 cross-attention 之前做的进一步特征提取）和 adapted modules（即新增的 cross-attention 层），大概 22M。实现基于 HuggingFace 的 diffusers 库，同时用 DeepSpeed ZeRO-2 做训练加速。
训练时使用 8 张 V100，每张卡上 batch = 8 共训了 1000W 步，lr=0.0001，weight decay=0.01，AdamW 优化方法，图像尺寸通过 crop resize 到 512x512。推理时使用 DDIM 采样器，50 steps，CFG为 7.5。

### 4.2 结果对比

为了验证 IP-Adapter 的有效性，首先和其它使用 image prompt 的方法作对比，这些方法主要包括三种类型：training from scratch，fine-tuning from text-to-image model 以及 adapters。
trained from scratch 方法包括：open unCLIP、Kandinsky-2-1 和 Versatile Diffusion；
fine-tuning from text-to-image model 方法包括：SD Image Variations 和 SD unCLIP；
adapter 方法包括：style-adapter of T2I-Adapter、global controller of Uni-ControlNet、ControlNet Shuffle, ControlNet Reference-only 和 SeeCoder。

#### 4.2.2 定量对比
 
 验证集使用 COCO2017，包含 5000张图像，为了公平对比，每个样本生成 4 张图像，每个方法共生成 20000 张图，使用以下两个评价指标：

- CLIP-I：生成图像的 CLIP image embedding与 image prompt 的CLIP image embedding 之间的相似度；
- CLIP-T：生成图像与其 image prompt 对应的 caption 之间的 CLIPScore；

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/6dd2ff3f-f82a-4f00-85ab-a6a1112ea6a4">
</div>

对比结果如上图，从结果看 IP-Adapterde 效果明显优于其它 adapter 方法，甚至比一些 fine-tune 模型也好。

#### 4.2.3 定性对比

每个方法生成 4 张图像，选择最好的一张作为结果。

![enter image description here](https://github.com/Tramac/paper-reading-note/assets/22740819/2e5f5ddc-dca8-46d3-9398-1527d7e2bb57)

从结果看，IP-Adapter 明显优于其它 adapter 方法，比 fine-tune 模型的结果略好，与 trained-from-scratch 方法也有可比性。

### 4.3 更多实验结果

IP-Adapter 可以搭配其它同类型（SD-v1.5、SD-v2.1、SDXL）的底模以及其它类型的 adapter。

#### 4.3.1 不同底模

这里选用了三个不同的基于 SD-v1.5 训练的底模：Realistic Vision V4.0、Anything v4 和 ReV Animated。
![enter image description here](https://github.com/Tramac/paper-reading-note/assets/22740819/bd61beac-df53-4cc3-ad79-e52c19924900)

#### 4.3.2 和其它结构控制条件搭配的玩法

ControlNet 和 T2I-Adapter 的搭配：
![enter image description here](https://github.com/Tramac/paper-reading-note/assets/22740819/fb99ab60-9f42-4f19-bdf2-12824debd892)

#### 4.3.3 图生图和重绘

![enter image description here](https://github.com/Tramac/paper-reading-note/assets/22740819/c527528b-422e-44f3-bbb3-4ab46560bbd3)

#### 4.3.4 Multimodel Prompts

这里指的主要是同时使用 text prompt 和 image prompt，和一些全量 finetune image prompt 模型不同，IP-Adapter 保留了较好的 text prompt 能力。
![enter image description here](https://github.com/Tramac/paper-reading-note/assets/22740819/43237da7-7cdc-4611-90b9-7fd12fe9c7bd)

### 4.4 消融实验

#### 4.4.1 Decoupled Cross-Attention 的重要性

这里主要对比 decoupled cross-attention 和直接将 text feature 和 image feature concat 后送入 cross-attention 的效果对比：
![enter image description here](https://github.com/Tramac/paper-reading-note/assets/22740819/54b67e94-7f0d-4138-b910-b81b9d4e30d5)

## 5 总结与展望
略。
