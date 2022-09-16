# DALL·E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents

[paper](https://arxiv.org/pdf/2204.06125.pdf) | [talk]() | [项目主页](https://openai.com/dall-e-2/)

## Part1.前言

DALL·E 2是OpenAI继DALL·E与GLIDE之后，有关文本图像生成工作的又一力作。

- DALL·E 2可以根据文本描述去生成**原创性**的图片（fake image），实现任意概念、属性或者风格的组合；
- DALL·E 2还可以根据文本对已有的图片进行编辑和修改，可以任意添加和移除图片中的物体，甚至可以考虑到阴影、反射、纹理等细节；
- 即便没有文本输入，DALL·E 2依旧可以做一些图像生成的工作，比如相同风格的图像生成。

从最近的DALL·E 2到google的Imagen，底层都是使用扩散模型（diffusion model）去做图像生成，相比于GAN其作为后起之秀，仍有很多值得去挖掘的地方。

## Part2.标题&作者

题目：使用CLIP训练好的特征，做层级式、依托于文本的图像生成。

Hierarchical（层级式）结构：DALL·E 2是先生成一个小分辨率的图片，比如64x64，然后再利用一个模型上采样到256x256，然后再使用一个模型上采样到1024x1024，最终生成一个高清大图。

Text-Conditional：DALL·E 2本身完成的任务就是根据文本去生成图像。

CLIP Latents：DALL·E 2是先训练好一个CLIP模型，找到图像与文本之间的相连关系，这时给定一个文本，通过CLIP的文本编码器将其映射为文本特征；然后DALL·E 2再训练一个prior模型，其输入为文本特征，输出为图像特征，然后将图像特征送给图像解码器，从而生成一个完整的图像。所以整个DALL·E 2模型是离不开CLIP特征的。

作者全部来自OpenAI，一作参与过CLIP与DALL·E的工作，Mark Chen也是DALL·E的原班作者，主要是做Codex与GPT3的，因为图像解码部分使用的是扩散模型（diffusion model），Prafulla Dhariwal与Alex Nichol都是这方面的专家，他们做过iGPT，最近又写了Imporve DDPM（denoising diffusion model）。

DALL·E 2可以看作是CLIP模型加上GLIDE模型，而GLIDE模型就是一个基于扩散模型的文本生成的方法。

## Part3.摘要

之前基于对比学习的方法如CLIP已经可以学得很稳健的图像特征，既能抓住语义信息又能抓住风格信息。为了能够借助这些特征来做图像生成，文章提出了一个两阶段模型，分别为**prior** 与 **decoder**。prior就是给定一个文本描述，生成一个类似于CLIP的图像特征，然后通过decoder生成一张图像。

从文本到图像的生成过程可以直观的理解为以下几个阶段：

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/189401469-2b90ad1b-4cf3-4195-a3cf-762afd2b587a.png">

作者发现，这种显式的生成图像特征的方式，能够显著的提升图像的diversity（多样性），说明了就是prior的必要性，而且对生成图像的写实程度和与文本的匹配程度都没有损失。相比而言，GAN生成的图像虽然也很逼真，但是都差不多，不太具备原创性。

基于扩散模型的图像decoder，可以基于给定的图像特征生成很多不同的图片，图片的语义和风格比较接近，只是细节不同。

因为DALL·E 2是一个从文本生成图像的任务，很容易地通过CLIP模型作为中间的桥梁，从而实现通过文本直接对图像进行编辑的功能，而且是zero-shot的，无需训练。

DALL·E 2的模型结构：decoder使用的是扩散模型，prior模型里尝试了auto regressive自回归模型与扩散模型，发现还是使用扩散模型的效率又高，效果又好。所以DALL·E 2整个都是扩散模型。

## Part4.引言

视觉领域最近的一些进展主要是使用了更大的模型和更大的数据集，数据集主要是一些图像、文本对。最典型的方法就是CLIP，其拥有很多特质，比如one-shot能力，而且学得的特征也很稳健，尤其对于分布偏移的数据，另外如果finetune一些下游任务效果也不错。

目前，扩散模型在图像生成领域也是一个很好的工具，在视频生成任务上也有很好的效果。扩散模型是一种概率分布模型，它生成的图片是从一个分布里去采样，多样性比较好，但它的保真度比不过GAN，因为GAN本身就是为了以假乱真去优化的。从2020年开始，一些列的工作把扩散模型的保真度做的更好，比如DDPM、Improved DDPM、Diffusion models beats GANs，GLIDE、DALL·E 2，这一系列工作采用了很多的技巧，其中最重要的一个是guidance technique（引导），它能够通过牺牲一部分多样性来达到更好的保真度，在指标上与GANs不相上下，也是这两年变得火热的原因。

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/189463970-6817af55-a88e-45a1-ab1b-3b187a602fd0.png">

模型结果图如上，上半部分其实就是CLIP，下半部分才是真正的DALL·E 2。

对于CLIP模型，给定一个文本及其对应的图像，分别通过一个文本编码器和图像编码器得到文本特征与图像特征，两者互为正样本，该文本与其它图像互为负样本，通过这种方式去做对比学习，将文本特征与图像特征拉入一个合并的、多模态的特征空间。训练完成CLIP模型之后，将其文本编码器与图像编码器freeze，可作为特征提取器使用，在DALL·E 2中CLIP全程都是freeze的。

DALL·E 2采用的是两阶段的训练方式，第一阶段为prior，第二阶段为decoder。文章提到，在生成图像时如果有一个显式的生成图像特征的过程，就是先从文本生成文本特征，再从文本特征生成图像特征，然后再由图像特征生成图像效果会好很多。具体地，在训练时：

- 给定一个文本&图像对，通过CLIP text-encoder(freeze)对文本提取文本特征；
- 通过CLIP image-encoder(freeze)对图像提取图像特征，作为prior的ground truth，也就是说给定文本特征，用prior预测图像特征，以此来训练得到prior模型（在推理时，给定一个文本特征，通过prior模型可以预测得到类似于CLIP的图像特征）；
- 通过prior得到图像特征之后，将其送入一个基于扩散模型的decoder，最终生成图像。

由上可知，DALL·E 2就是将CLIP与GLIDE叠加在了一起，但是里面有很多值得借鉴的技巧，而且大力出奇迹效果确实很好。另外文章中并没有说该方法为DALL·E 2，而是unCLIP，因为CLIP是从输入（文本、图像）到特征的过程，而unCLIP是从文本--文本特征--图像特征--图像，可看作是是从文本特征到图像的过程，类似于CLIP的反过程。

## Part5.模型方法

文章中比较简略的写了prior与decoder的实现细节，只看这一篇论文可能并不能对整体的框架有很好的了解。下面一部分会从GAN，VAE，VQVAE，DALL·E开始介绍，使得对图像生成有更全面的了解。

![image](https://user-images.githubusercontent.com/22740819/190552576-35f4c85f-d8be-4fbc-889d-6ccdeb4ca58b.png)

图片来自：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

### 5.1 GAN

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/189480961-083c5fc6-5e9e-433f-a3e6-144e1acb5b42.png">

GAN需要训练两个网络，生成器（generator）与判别器（discriminator）。给定一个随机噪声，通过生成器生成一张图片，然后将生成的图片送给判别器，同时给一些真实图片给判别器，然后让判别器判断哪些是真图片，哪些是假图片，最终其实是一个二分类问题。通过生成器与判别器之间的相互较量，生成器也不断的优化，使得生成比较真实的图片。因为GAN使用的目标函数就是为了以假乱真，所以GAN生成的图片保真度一般都比较高。

经过多年时间对GAN的改造，如今GAN也比较好用，所需的数据也不是很多，能够在各个场景下面使用，但是一个致命缺点是训练不稳定，主要原因是需要同时训练生成器与判别器两个网络，另外因为GAN的主要优化目标是让图片尽可能的真实，其多样性不太好，它的多样性主要就是来自于输入的随机噪声，简单来说就是它的创造性不好，不是一个概率模型，它的生成都是隐式的，就是通过一个网络去生成，不知道具体是遵循的什么分布，所以，GAN在数学上不如后续的VAE，扩散模型优美。

### 5.2 Auto-Encoder, DAE, VAE, VQ-VAE

<img width="525" alt="image" src="https://user-images.githubusercontent.com/22740819/189487359-ab20c07d-9aef-4ddf-8193-633a72ce7ca2.png">

**Auto-Encoder**: 给定一个输入x，通过一个编码器得到特征，该特征维度一般比较小，所以经常称之为bottleneck，然后再经过一个解码器，最终得到一个图像，训练时的目标函数是生成的图像尽可能的重建输入x，因为是自己重建自己，所以叫自编码器。

**Denoising Auto-Encoder**: DAE相对于AE是首先对输入x进行一定程度的打乱得到x_c，后续过程与AE相同，最终输出还是重建x，而不是x_c。这个改进证明非常有用，尤其是在视觉领域，会使得训练得到的模型更加鲁棒、不容易过拟合，部分原因是图像像素的冗余性太高了，即使把输入的图片x进行一定程度的污染，模型还是能抓住它的本质将其重建出来。

> 无论是AE、DAE还是MAE，其关键都是为了学习中间的bottleneck特征，然后用于分类、分割等下游任务，它并不是用来做生成的，因为学得的特征z并不是一个概率分布，没办法对其进行采样，并不像GAN中那样是一个随机噪声，它是一个专门用来重建的特征，但是这种encoder-decoder确实是一种很好的结构，为了使用这种结构用于图像生成，有了后续的VAE。

**Variational Auto-Encoder**: 

<img width="559" alt="image" src="https://user-images.githubusercontent.com/22740819/189488271-b9c3cfc0-51dd-4c32-9f7a-2f4fde18f902.png">

VAE与AE非常不一样，虽然整体框架仍然是encoder-decoder结构，但是它中间不再是学习一个固定的bottleneck特征，而是去学习一个分布，VAE中假设该分布是一个高斯分布，就可以用均值和方差来表述。具体地，在encoder得到特征之后，在后面加一些FC层，去预测均值和方差，在得到均值和方差之后，就用上面的公式采样得到一个z出来，这样的话VAE就可以用来做生成了。因为在训练完成VAE模型之后，就可以将前面的encoder丢掉，z可以是从高斯随机噪声中抽样出来的一个样本，然后通过一个decoder去生成一张图片。

> 由于VAE预测的是一个概率分布，从贝叶斯的角度看，前面的部分就是给定x得到z的过程，是一个后验概率，然后学习得到的distribution其实就是一个先验分布；后面部分是给定z预测得到图像x的过程，其实就是maxilize likelyhood，从数学上就优美了很多。而且VAE也有一些很不错的性质，因为它学得的是一个分布，然后从分布里抽样，它生成的图片多样性就比GAN好很多。

**VQ-VAE**:

<img width="654" alt="image" src="https://user-images.githubusercontent.com/22740819/189488422-02bd8037-7425-4dc8-8aa8-2ce3ddfc099a.png">

VQ-VAE整体上与VAE差不多，其中VQ就是vector quantization，就是把VAE量化。原因是生活中的图像、声音等这种信号都是连续的，大部分任务都是回归任务，但实际上将其表示出来去解决这些问题时其实都将其离散化了，图像就变成了像素，语音也都进行了抽样，大部分工作的比较好的模型也都是分类模型，又都从回归任务变成了分类任务。所以，用之前的VAE方式就不好把模型做大，而且分布z也不好学，所以取而代之的是不去做这个分布的预测，而是用一个codebook代替，这里的codebook可以理解为一个聚类中心，尺寸为KxD，一般K=8192，D=512或768，也就是有8192个聚类中心，如果给定一个图片，经过一个编码器得到特征f，然后将特征图f的向量去跟codebook的向量做对比，然后把距离最近的聚类中心对应的编码索引存放到z矩阵里，一旦做完特征匹配，通过z中的映射得到一个新的特征向量f_q（quantised feature），经过量化后的特征就会非常的可控，因为它全部来自于codebook，不是一个随机的东西，优化起来就会比较容易，后续通过decoder恢复得到图像的过程就和之前一样。

VQ-VAE其实非常有用，它后来不仅用在了DALL·E项目，还用在了视觉领域的自监督学习中，比如BEIT、VL-BEiT等。

**DALL·E**:

### 5.3 Diffusion Model

### 5.4 DALL·E 2

DALL·E 2的训练集也是图像&文本对。假设给定一个图片输入x，z_i与z_t分别表示CLIP生成的图像与文本特征，整个DALL·E 2模型可以分为两个部分：

- prior：根据一个文本y，生成一个图像特征z_i；
- decoder：输入z_i，通过decoder生成图像x；

#### 5.4.1 decoder

decoder其实是一个GLIDE的变体，首先它用了CLIP模型的guidance，只不过使用的形式在具体的操作上稍微有些变化，技术细节可以看代码。

DALL·E 2中也使用了classifier-free guidance，guidance信号要么来自于CLIP模型，要么来自于文本。实际使用中是10%的时间随机将CLIP特征设成0，训练时50%的时间把文本特征直接丢掉。

为了生成高分辨率的图片，decoder中做了级联式的结构，使用了两个difusion model做上采样，先将分辨率从64x64采样到256x256，再从256x256上采样到1024x1024。另外为了训练时的稳定性，在训练的过程中加入了很多的噪声，而且difusion model的结构主要基于U-Net，只使用了普通的卷积，没有attention操作，所以在推理时可以用在任何尺寸，而不用担心序列长度必须保持一致。

#### 5.4.2 prior

prior模型的作用是给定一个文本，去生成一个图像特征z_i，用于后续decoder的输入。DALL·E 2中针对prior采用了两种方案：

- Autoregressive prior: 该方式与DALL·E和GPT就比较像，输入为文本特征，同时也有CLIP生成的图像特征，然后将图像特征遮住去做自回归预测，这种方式一般训练效率比较低；
- Diffusion prior: 

无论是Autoregressive prior还是Diffusion prior，都用了classifier-free guidance，因为效果确实好。

对于Diffusion prior，实际是训练了一个transformer decoder，因为它的输入输出都是embedding，所以用U-Net不太合适，直接用transformer更合适。
