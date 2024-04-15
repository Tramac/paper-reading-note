# SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

SDXL 作为一个技术报告，我们直接上干货！

## 1 性能对比
首先对比一下 SDXL 相较于 SD 1.5 及 SD 2.1 的**性能优势**：

<div align="center">
<img width="400" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/577c1f50-8b70-4dc0-88b9-4b6247e91367">
</div>

- 单独的SDXL 1.0 Base模型的表现明显优于之前的所有SD版本，而完整的SDXL 1.0模型（Base模型 + Refiner模型）则实现了最佳的图像生成整体性能

## 3 模型对比

SDXL 相较于 SD 1.5 & 2.1 在模型结构上的差异：
<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/01620a72-8ba1-4458-a391-4b4a9f186869">
</div>

### 2.1 UNet 结构差异

U-Net 的 Encoder 和 Decoder 结构从之前系列的 4stage 改成3stage（[1,1,1,1] -> [0,2,10]），在第一个stage中不再使用Spatial Transformer Blocks，而在第二和第三个stage中大量增加了Spatial Transformer Blocks，并舍弃最后一个 stage；
> 这样做的原因：在具有更高特征维度的第一个stage中，不使用 Spatial Transformer block，可以明显减少显存占用和计算量。然后在第二和第三个stage这两个维度较小的feature map上使用数量较多的 Spatial Transformer block，能在大幅提升模型整体性能（学习能力和表达能力）的同时，优化了计算成本。舍弃掉最后一个具有最小维度 feature map 的 stage，是为了让SDXL的出图分辨率提升至1024x1024（理论上从 128x128 大小的特征上重建出 1024x1024 的图是要比 64x64 的特征上重建更简单）。
	
**UNet 模型（Base部分）参数量增加到2.6B，参数量为之前模型的3倍左右**。

### 2.2 Text Encoder 差异

SDXL 使用更强大的 pre-trained text encoder，并且使用了两个 text encoder，分别是 OpenCLIP ViT-bigG（694M）和OpenAI CLIP ViT-L/14（123.65M），大大增强了 SDXL 对文本的提取和理解能力，同时提高了输入文本和生成图片的一致性；
> 两个 text encoder 提取到的文本特征 concat 之后送入 UNet 中的 cross-attention 模块；

**Text encoder 部分参数量共 694+123=817M 左右**。

除了常规的 text embedding 之外，同时引入 pooled text embedding 作为文本控制条件；
> 具体地，对 OpenCLIP ViT-bigG 提取到的文本特征进行 max pooling 操作（[N, 77, 2048] -> [N, 1, 2048]）得到 pooled text emb，将其嵌入到Time Embeddings中（add操作），作为辅助约束条件（强化文本的整体语义信息），但是这种辅助条件的强度是较为微弱的。

### 2.3 VAE 差异

VAE 部分结构不变，只对训练过程做了优化。

SDXL 使用了和之前 Stable Diffusion 系列一样的 VAE 结构（KL-f8），但在训练中选择了更大的 batch size（256 vs 9），并且对模型进行指数滑动平均操作（**EMA**，exponential moving average），EMA对模型的参数做平均，提高了性能并增加模型鲁棒性。

下表是Stable Diffusion XL的VAE在COCO2017 验证集上，图像大小为256×256像素的情况下的性能：
<div align="center">
<img width="400" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/71fed60d-9b98-4902-98e8-9b1b7c725d95">
</div>

（注：SDXL 的 VAE 是从头开始训练的）

上面的表中的三个VAE模型结构是一样的，不同点在于SD 2.x VAE是基于SD 1.x VAE微调训练了Decoder部分，同时保持Encoder部分权重不变，使他们有相同的Latent特征分布，所以**SD 1.x和SD 2.x的VAE模型是互相兼容的**。而SDXL VAE是重新从头开始训练的，所以其Latent特征分布与之前的两者不同。

由于Latent特征分布产生了变化，SDXL VAE 的**缩放系数**也产生了变化。VAE 在将 Latent 特征送入U-Net之前，需要对Latent特征进行缩放让其标准差尽量为1，之前的Stable Diffusion系列采用的缩放系数为0.18215，由于SDXL的VAE进行了全面的重训练，所以缩放系数重新设置为0.13025。

**注意：由于缩放系数的改变，SDXL VAE 模型与之前的Stable Diffusion系列并不兼容。如果在SDXL上使用之前系列的VAE，会生成充满噪声的图片。**

与此同时，与Stable Diffusion一样，VAE模型在SDXL中除了能进行图像压缩和图像重建的工作外，通过**切换不同微调训练版本的VAE模型，能够改变生成图片的细节与整体颜色（更改生成图像的颜色表现，类似于色彩滤镜）。**

需要注意的是：原生 SDXL VAE采用 FP16 精度时会出现数值溢出成 NaNs 的情况，导致重建的图像是一个黑图，所以必须使用 FP32 精度进行推理重建。如果大家想要FP16精度进行推理，可以使用[sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)版本的SDXL VAE模型，其对FP16出现的NANs的情况进行了修复。

## 训练策略优化 Micro-Conditioning

SDXL 在训练阶段提出了很多优化方法，包括图像尺寸条件化、图像裁剪参数条件化以及多尺度训练策略。

- **size-conditioning**

	之前在Stable Diffusion的训练过程中，主要分成两个阶段，先在256x256的图像尺寸上进行预训练，然后在512x512的图像尺寸上继续训练。
	而这两个阶段的训练过程都要对最小图像尺寸进行约束。通常有两种处理方式：1）丢弃小于最小尺寸的数据；2）将小尺寸图像进行超分辨上采样。
	如果选择第一种处理方式，会导致训练数据中的大量数据被丢弃，从而很可能导致模型性能和泛化性的降低。下图展示了如果将尺寸小于256x256的图像丢弃，**整个数据集将减少39%的数据**。如果再丢弃尺寸小于512x512的图像，**未利用数据占整个数据集的百分比将更大**。
	<div align="center">
	<img width="300" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/ba74920b-35da-46bc-a8cd-a41f042426dc">
	</div>

	如果选择第二种处理方式，对于图像尺寸过小的场景，**目前的超分模型可能会在对图像超分的同时会引入一些噪声伪影，影响模型的训练，导致生成一些模糊的图像**。

	为了解决上述问题，SDXL 提出了 size-conditioning 策略，即将原始图像分辨率信息作为空间条件引入到 UNet 的训练中。具体地，使用 Fourier feature encoding 对 original height 和 original width 分别编码，然后将特征 concat 后加在Time Embedding上，从而**将图像尺寸信息作为条件控制引入训练过程**。这样以来，模型在训练过程中能够学习到图像的原始分辨率信息，从而在推理生成阶段，会根据用户设定的尺寸，更好地适应不同尺寸的图像生成，而不会产生噪声伪影的问题。
	
	下图展示了使用 size-conditioning 的效果：
	<div align="center">
	<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/c0ed7ee6-32d6-4771-8206-e0e69441d520">
	</div>
	如下图所示，在使用了 size-conditioning 策略后，Base模型已经对不同图像分辨率有了“自己的判断”。当输入低分辨率条件时，生成的图像较模糊；在不断增大分辨率条件时，生成的图像质量不断提升。

- crop-conditioning

	之前的Stable Diffusion系列模型，由于需要输入固定的图像尺寸用作训练，很多数据在预处理阶段会被裁剪。**生成式模型中典型的预处理方式是先调整图像尺寸，使得最短边与目标尺寸匹配，然后再沿较长边对图像进行随机裁剪或者中心裁剪**。虽然裁剪是一种数据增强方法，但是训练中对图像裁剪导致的图像特征丢失，**可能会导致AI绘画模型在图像生成过程中出现不符合训练数据分布的特征**。

	下图给出了一些使用 crop resize 造成的失败案例：
	<div align="center">
	<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/36ac7f70-5468-4d12-8057-edba0787077e">
	</div>
	
	为了解决上述问题，SDXL 提出了 crop-conditioning 策略，即在加载数据时，将左上角的裁剪坐标（c_top，c_left）利用 Fourier feature encoding 编码后加在Time Embedding上（实际应用时是先和 size-conditioning 特征 concat 之后在与 time embedding 相加），以此将 crop 信息作为控制条件引入 U-Net（Base）模型中，从而在训练过程中让模型学习到对“图像裁剪”的认识。
	
	下图展示了使用 crop-conditioning 策略的效果：
	<div align="center">
	<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/6a600da3-063d-4985-973b-2e9110a95d4a">
	</div>
	从上图中可以看到，将不同的 c_crop 坐标条件的生成图像进行了对比，当设置 c_crop=(0,0) 时可以生成主要物体居中并且无特征缺失的图像，而采用其它的坐标条件则会出现有裁剪效应的图像。

	size-conditioning 策略和 crop-conditioning 策略都能在 SDXL 训练过程中使用（在线方式应用），同时也可以很好的迁移到其他生成式模型的训练中。伪代码如下：
	<div align="center">
	<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/f621aa35-37a6-4cce-82a2-67fb82b4be59">
	</div>

- **Multi-Aspect Training**

	真实世界的图像是有一个非常广泛的尺寸和长宽比范围的，而通常 text-to-image 模型生成的图都是限制在 512x512 或 1024x1024 这种正方形的尺寸上，其实这是不太合理的。
	
	基于此，SDXL 提出了多尺度训练策略。首先采用 size-conditioning 和 crop-conditioning 这两种策略在256x256 和 512x512 的图像尺寸上分别预训练600000步和200000步（batch size = 2048），**总的数据量约等于 （600000 + 200000） x 2048 = 16.384亿**。

	接着Stable Diffusion XL在 1024x1024 的图像尺寸上采用多尺度方案来进行微调，并将数据分成不同纵横比的桶（bucket），并且**尽可能保持每个桶的像素数接近1024×1024，**同时相邻的bucket之间height或者width一般相差64像素左右，Stable Diffusion XL的具体分桶情况如下图所示：
	<div align="center">
	<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/f2f3a30a-a180-4f2d-9a7d-483cf3e55b2f">
	</div>
	其中Aspect Ratio = Height / Width，表示高宽比。

	在训练过程中，一个Batch从一个桶里的图像采样，并且在每个训练步骤（step）中可以在不同的桶之间交替切换。**除此之外，Aspect Ratio也会作为条件嵌入到U-Net（Base）模型中，嵌入方式和上面提到的其他条件嵌入方式一致，让模型能够更好地学习到“多尺度特征”。**

	与此同时，SDXL在多尺度微调阶段依然使用 crop-conditioning 策略，进一步增强 SDXL 对图像裁剪的敏感性。

	在完成了多尺度微调后，SDXL 就可以进行不同Aspect Ratio的图像生成了，**不过官方推荐生成尺寸默认为1024x1024。**

### **Refiner**

SDXL 中新增了一个重要模块：refiner。

SDXL 是一个**二阶段的级联扩散模型**，包括 Base 模型和 Refiner 模型。其中**Base模型的主要工作和Stable Diffusion 1.x-2.x一致**，具备文生图（txt2img）、图生图（img2img）、图像inpainting等能力。在Base模型之后，级联了Refiner模型，**对Base模型生成的图像Latent特征进行精细化提升，其本质上是在做图生图的工作**。

**SDXL Base模型由U-Net、VAE以及CLIP Text Encoder（两个）三个模块组成**，在FP16精度下Base模型大小6.94G（FP32：13.88G），其中U-Net占5.14G、VAE模型占167M以及两个CLIP Text Encoder一大一小（OpenCLIP ViT-bigG和OpenAI CLIP ViT-L）分别是1.39G和246M。

**SDXL Refiner模型同样由U-Net、VAE和CLIP Text Encoder（一个）三个模块组成**，在FP16精度下Refiner模型大小6.08G，其中U-Net占4.52G、VAE模型占167M（与Base模型共用）以及CLIP Text Encoder模型（OpenCLIP ViT-bigG）大小1.39G（与Base模型共用）。

比起Stable Diffusion 1.x-2.x，**SDXL的参数量增加到了66亿（Base模型35亿+Refiner模型31亿），并且先后发布了模型结构完全相同的0.9和1.0两个版本**。Stable Diffusion XL 1.0在0.9版本上使用**更多训练集+RLHF**来优化生成图像的色彩、对比度、光线以及阴影方面，使得生成图像的构图比0.9版本更加鲜明准确。

级联 refiner 之后的生成  pipeline 如下：
<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/1a44a5c0-34da-4363-95a5-1a96dd69e903">
</div>

#### Bonus

文中对一些闭源工作表达了不满（没错，Midjourney 就是你:），下面直接贴出文中原话：
> A major concern in the field of visual media creation is that while black-box-models are often recognized as state-of-the-art, the opacity of their architecture prevents faithfully assessing and validating their performance. This lack of transparency hampers reproducibility, stifles innovation, and prevents the community from building upon these models to further the progress of science and art. Moreover, these closed-source strategies make it challenging to assess the biases and limitations of these models in an impartial and objective way, which is crucial for their responsible and ethical deployment. With SDXL we are releasing an open model that achieves competitive performance with black-box image generation models.
