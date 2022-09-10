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

### 5.1 GAN
