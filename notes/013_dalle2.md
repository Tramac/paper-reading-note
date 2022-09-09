# DALL·E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents

[paper](https://arxiv.org/pdf/2204.06125.pdf) | [talk]() | [项目主页](https://openai.com/dall-e-2/)

## 前言

DALL·E 2是OpenAI继DALL·E与GLIDE之后，有关文本图像生成工作的又一力作。

- DALL·E 2可以根据文本描述去生成**原创性**的图片（fake image），实现任意概念、属性或者风格的组合；
- DALL·E 2还可以根据文本对已有的图片进行编辑和修改，可以任意添加和移除图片中的物体，甚至可以考虑到阴影、反射、纹理等细节；
- 即便没有文本输入，DALL·E 2依旧可以做一些图像生成的工作，比如相同风格的图像生成。

从最近的DALL·E 2到google的Imagen，底层都是使用扩散模型（diffusion model）去做图像生成，相比于GAN其作为后起之秀，仍有很多值得去挖掘的地方。

## 标题&作者

题目：使用CLIP训练好的特征，做层级式、依托于文本的图像生成。

Hierarchical（层级式）结构：DALL·E 2是先生成一个小分辨率的图片，比如64x64，然后再利用一个模型上采样到256x256，然后再使用一个模型上采样到1024x1024，最终生成一个高清大图。

Text-Conditional：DALL·E 2本身完成的任务就是根据文本去生成图像。

CLIP Latents：DALL·E 2是先训练好一个CLIP模型，找到图像与文本之间的相连关系，这时给定一个文本，通过CLIP的文本编码器将其映射为文本特征；然后DALL·E 2再训练一个prior模型，其输入为文本特征，输出为图像特征，然后将图像特征送给图像解码器，从而生成一个完整的图像。所以整个DALL·E 2模型是离不开CLIP特征的。

作者全部来自OpenAI，一作参与过CLIP与DALL·E的工作，Mark Chen也是DALL·E的原班作者，主要是做Codex与GPT3的，因为图像解码部分使用的是扩散模型（diffusion model），Prafulla Dhariwal与Alex Nichol都是这方面的专家，他们做过iGPT，最近又写了Imporve DDPM（denoising diffusion model）。

DALL·E 2可以看作是CLIP模型加上GLIDE模型，而GLIDE模型就是一个基于扩散模型的文本生成的方法。

## 摘要

之前基于对比学习的方法如CLIP已经可以学得很稳健的图像特征，既能抓住语义信息又能抓住风格信息。为了能够借助这些特征来做图像生成，文章提出了一个两阶段模型，分别为**prior** 与 **decoder**。prior就是给定一个文本描述，生成一个类似于CLIP的图像特征，然后通过decoder生成一张图像。

从文本到图像的生成过程可以直观的理解为以下几个阶段：

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22740819/189401469-2b90ad1b-4cf3-4195-a3cf-762afd2b587a.png">

作者发现，这种显式的生成图像特征的方式，能够显著的提升图像的diversity（多样性），说明了就是prior的必要性，而且对生成图像的写实程度和与文本的匹配程度都没有损失。相比而言，GAN生成的图像虽然也很逼真，但是都差不多，不太具备原创性。

基于扩散模型的图像decoder，可以基于给定的图像特征生成很多不同的图片，图片的语义和风格比较接近，只是细节不同。

因为DALL·E 2是一个从文本生成图像的任务，很容易地通过CLIP模型作为中间的桥梁，从而实现通过文本直接对图像进行编辑的功能，而且是zero-shot的，无需训练。

DALL·E 2的模型结构：decoder使用的是扩散模型，prior模型里尝试了auto regressive自回归模型与扩散模型，发现还是使用扩散模型的效率又高，效果又好。所以DALL·E 2整个都是扩散模型。

## 引言
