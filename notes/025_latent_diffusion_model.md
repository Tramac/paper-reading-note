## 1 题目&作者
利用 latent diffusion model 实现高分辨图像生成。
作者团队来自 Runway。

## 2 摘要
Stable Diffusion 通过将图像生成过程分解为逐步去噪自编码过程实现了SOTA的生成结果，并且该方式允许通过额外的控制条件来指导生成。然而，由于这些模型通常直接在像素空间中操作，导致 diffusion model 的训练成本和推理成本都很高。
为了在有限的计算资源上训练 diffusion model，同时保留生成图像的质量和灵活性，本文提出，首先通过一个 pretrained autoencoder 将图像从像素空间转到 latent 空间，然后在 latent 空间上进行 diffusion 训练，通过找到一个近似最优点，实现模型复杂度和细节保真度之间的平衡。
另外，通过在模型结构中引入 cross-attention layer，使得 diffusion model 生成图像时更加灵活，可以处理文本、边界框等一般条件的输入，并且可以以卷积的方式实现高分辨率图像的合成。

## 3 引言
图像生成在计算机视觉领域取得了显著的成就，但同时也伴随着巨大的计算资源需求。特别是复杂自然场景下的高分辨率图像生成，目前主要是基于可扩展的自回归（AR）transformers 模型，可能包含数十亿参数。另一方面，生成对抗网络（GANs）受限于数据，对于复杂、多模态数据分布拟合的并不是很好。
最近，以 denoising autoencoders 层级结构为基础的 diffusion models 在图像生成任务上取得了很好的结果，并且在条件控制以及高分辨生成上也有 SOTA 的结果。另外，无条件控制的 diffusion models 也可以应用于图像修复、着色等任务。diffusion models 是一个 likelihood-based 模型，不会像GANs那样出现模式崩溃（mode-collapse）和训练不稳定，并且通过大量参数共享，能够在不涉及数十亿参数的情况下学习自然图像的复杂数据分布。

**Democratizing High-Resolution Image Synthesis**
Diffusion Models 属于是 likelihood-based 方法，可以容易的通过扩大模型规模来实现对数据中不易察觉的细节建模。但是直接在 RGB 图像这种高维空间进行训练和推理是非常耗时的。所以，要想使得高分辨率图像生成变得可用，降低复杂度是非常必要的。

**Departure to Latent Space**
对于任意一个 likelihood-based model，学习过程可以大致分为两个阶段：1）perceptual compression，该阶段消除了高频细节信息，但仍然学到很少的语义变化；2）semantic compression，学习数据中的概念和语义信息。因此，我们的目标是首先找到一个感知上等效但计算上更合适的空间，在该空间上训练用于高分辨率图像合成的扩散模型。
根据之前的一些经验，我们将训练过程分为两个阶段：首先，训练一个 autoencoder，将图像从 RGB 高维空间映射到一个更低维的表示空间，该空间和原始的数据是 perceptually equivalent 的（感知等价，我的理解是虽然维度不同，但感知上并没有变化）。重要的是，与之前的工作相比，我们不需要依赖过度的空间压缩，因为我们在学习到的 latent 空间中训练 Diffusion Model，这在空间维度方面表现出更好的缩放特性。 降低复杂度还可以通过单次推理实现从 latent 空间高效生成图像（其实就是通过 decoder 将 latent 转为图像）。 我们将该方法称为 Latent Diffusion Models。
其中，autoencoder 仅需要训练一次即可，然后就可以复用在不同的训练实验或其它任务上（或者说，在训练 Diffusion Models 时，VAE 往往是冻结住不训练的，直接用之前训好的权重即可）。

## 4.相关工作

**Generative Models for Image Synthesis**
GANs 的主要缺点：1）优化困难；2）数据量比较大时，很难拟合整个数据分布；
VAE 的缺点：虽然能够生成高分辨率的图像，但是质量无法和 GANs 媲美；
autoregressive 自回归模型：有不错的生成质量，但计算量太大，只适用于低分辨率图像；

**Diffusion Probabilistic Models**
生成效果好，但是直接在 pixel space 计算，效率低。

**Two-Stage Image Synthesis**
VQ-VAEs、VQGANs。

## 5.方法
原始的 Diffusion Models 虽然通过对相应的 loss 进行降采样来降低计算量，但由于其仍是在 pixel 空间上计算，所以还是需要巨大的计算量。
本文提出，先使用一个 autoencoding model 将图像从 pixel space 映射到 latent space，且两者是 perceptually equivalent 的，重要的是该过程可以极大地减少计算复杂度。

### 5.1 Perceptual Image Compression
Perceptual Image Compression 在这里指的就是将图像从 pixel space 映射到 latent space，即感知压缩。
本文的 perceptual compression model 是基于 perceptual loss 和 patch-based adversarial objective 训练了一个 autoencoder，通过保证局部真实性限制了重建过程在图像流形内（写的太难懂了），也避免了 L1 或 L2 损失容易引入的模糊问题。

假设，给定一个 RGB 图像 $x ∈ \mathbb{R} ^{H\times W\times 3}$，encoder $\varepsilon$ 将 $x$ 编码到 latent 空间，表示为 $z = \varepsilon (x)$，而 decoder $\mathcal{D}$ 将 latent 空间中的向量重建为图像，表示为 $\tilde{x} = \mathcal{D} (z) = \mathcal{D} (\varepsilon (x))$，其中 $z ∈ \mathbb{R} ^{h\times w\times c}$。通常，将 $f = H / h = W / w$ 成为降采样率。

为了避免落入高方差的 latent 空间，本文实验了两种不同类型的正则化方法：
- KL-reg：在符合标准正态分布的 latent 上施加一个轻微的 KL 惩罚，类似于 VAE；
- VQ-reg：在 decoder 中使用一个向量量化层，类似于 VQGAN；

### 5.2 Latent Diffusion Models

Diffusion Models 是一种基于概率的生成模型，它们通过模拟数据的扩散过程来学习数据的分布。核心思想是将数据生成过程视为一个逆向的马尔可夫链过程，其中每一步都涉及到对数据添加噪声，直到数据完全转变为随机噪声。然后，模型通过逆向这个过程来学习如何从噪声中恢复出原始数据的分布。

用公式表示如下：
$L_{DM} = \mathbb{E} _{x,\epsilon ∼ \mathcal{N}(0,1),t } [\left \| \epsilon  - \epsilon_{\theta } (x_t,t)  \right \|_{2}^{2} ]$
其中 t 的取值为 {1,...T}。

Generative Modeling of Latent Representations 指的是在 latent 空间中从随机噪声到 latent 向量的去噪过程。低维的 latent 向量表示，保留了原始数据中重要的语义信息，且降低了计算复杂度；以 Unet 为主要结构，其归纳偏置使得最终生成的图像细节更好。
该过程用公式表示如下：
$L_{LDM} := \mathbb{E} _{\epsilon (x),\epsilon ∼ \mathcal{N}(0,1),t } [\left \| \epsilon  - \epsilon_{\theta } (z_t,t)  \right \|_{2}^{2}  ]$

### 5.3 Conditioning Mechanisms
Conditioning Mechanisms（条件机制）是指将额外信息或条件输入到模型中，以控制或引导生成过程的方法。这些条件可以是类别标签、文本描述、图像或其他任何能够影响生成结果的信息。条件机制使得模型能够生成特定于这些条件的数据样本，从而提高了模型的灵活性和应用范围。

实现方式：
- 在UNet架构中引入了 cross-attention 层，这些层能够将条件信息（如文本、图像或其他模态的数据）引入到UNet的不同中间层，从而影响生成过程。

假设，条件信息为 $y$，通过一个 encoder $\tau _{\theta }$ 将 $y$ 映射为 $\tau _{\theta } ∈ \mathbb{R} ^{M\times d_{\tau } }$，然后将其通过一个 cross-attention 层与 UNet 的中间层进行融合：
$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d} } )\cdot V$
其中，$Q=W_{Q}^{(i)} \cdot \varphi _{i} (z_t), K=W_{K}^{(i)}\cdot \tau _\theta (y), V=W_{V}^{(i)}\cdot \tau _\theta (y)$，而 $\varphi _i(z_t) ∈ \mathbb{R} ^{N\times d_{\epsilon }^{i} }$ 表示的是 UNet 的中间层特征。

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/66499f7e-bf5f-483e-b47b-5c7631dd7f84">
</div>

## 6.实验

### 6.1 On Perceptual Compression Tradeoffs
该部分在 vae 降采样率和生成质量之间做了权衡。

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/d5e6a1a6-c516-4ff9-a191-2cf8500120fd">
</div>

当 f = 4,8,16 时，速度和生成质量之间相对比较权衡。< 4 时，速度太慢，> 16 时，质量太差。

### 6.2 Image Generation with Latent Diffusion
unconditional image synthesis 无条件控制
<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/1e7611af-e6a0-4740-b3b3-6e3a8dbb3c55">
</div>
ProjectedGAN 的效果貌似很好。

### 6.3 Conditional Latent Diffusion
- text-conditional 文本控制
<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/9a08d526-4fae-4ac2-938e-244f3efb4fbb">
</div>

- class-conditional 类别控制
<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/06e29563-41e2-4c46-80de-0290522d68a2">
</div>

### 6.4 Super-Resolution with Latent Diffusion
LDMs 还可以通过 concatenation 操作对低分辨率图像进行超分重建。

### 6.5 Inpainting with Latent Diffusion
inpainting 指的是对图中 mask 区域进行填充，LDMs 也可以完成该任务。

结果略。

## 7.结论
本文提出了潜在扩散模型（LDMs），这是一种简单且高效的方法，显著提高了去噪扩散模型的训练和采样效率，同时不降低其质量。基于此，我们的跨注意力条件机制使得实验能够在广泛的条件图像合成任务上与最先进的方法相比取得有利的结果，而无需特定任务的架构。
