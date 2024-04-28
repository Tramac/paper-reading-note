# Stable Diffusion 2

[homepage](https://stability.ai/news/stable-diffusion-v2-release) | [code](https://github.com/Stability-AI/stablediffusion)

Stable Diffusion 2.x 是 Stable Diffusion 1.x 的改进版，原理上没有太大差异，主要是训练数据、训练过程以及模型选择的不同，这里就直接放 stable diffusion 2.x 的改进点。

## 1 Stable Diffusion 2.0 相较于 1.0 的改进点

### 1.1 模型结构

#### 1.1.1 Text Encoder

- 模型由 CLIP ViT-L/14 升级为 CLIP ViT-H/14
	Stable Diffusion 1.x 系列中的 Text Encoder 采用的是 OpenAI 开源的 CLIP ViT-L/14 模型，其模型参数量为123.65M；而Stable Diffusion V2系列则换成了新的 OpenCLIP 模型—— CLIP ViT-H/14 模型（基于LAION-2b数据集训练），其参数量为354.03M，比SD 1.x的Text Encoder模型大了3倍左右。具体对比如下表所示：

	| Text Encoder | emb dims |Params | ImageNet Top1 | Mscoco image retrieval at 5 | Flickr30k image retrieval at 5 |
	| --------------- | ------------ | --------- | ------------------ | -------------------------------- | --------------------|
	| Openai L/14 | 768  | 123.65M | 75.4% | 61% | 87% |
	| CLIP ViT-H/14 | 1024 | 354.03M | 78.0% | 73.4% | 94% |

	SD2.0 中所使用的新的 text encoder 模型在多个指标上均有明显的提升，表明 CLIP ViT-H/14 模型的 Text Encoder 能够输出更准确的文本语义信息。

- 使用Text Encoder倒数第二层的特征来作为U-Net模型的文本信息输入
	SD 1.x 使用的是 Text Encoder 倒数第一层的特征。Imagen 和 novelai 在训练时也采用了Text Encoder倒数第二层的特征，**因为倒数第一层的特征存在部分丢失细粒度文本信息的情况，而这些细粒度文本信息有助于SD模型更快地学习某些概念特征**。

#### 1.1.2 VAE

SD2.0 和 SD1.x 的 VAE 部分是一致的。

#### 1.1.3 UNet

SD 2.0 U-Net与SD 1.x U-Net的整体架构是一样的，但由于切换了 Text Encoder 模型，在 SD 2.0 中 U-Net 的 **cross attention dimension** 从 SD 1.x U-Net 的 768 变成了 1024，从而 U-Net 部分的整体参数量有一些增加（860M -> 865M）。与此同时，在 SD 2.0 U-Net 中不同 stage 的 attention 模块的 **attention head dim** 是不固定的（5、10、20、20），而 SD 1.x 则是不同 stage 的 attention 模块采用固定的 attention head 数量（8），这个改动不会影响模型参数量。

### 1.2 训练数据和训练过程

#### 1.2.1 训练数据

Stable Diffusion 2.0模型从头开始在LAION-5B数据集的子集（该子集通过LAION-NSFW分类器过滤掉了NSFW数据，过滤标准是punsafe=0.1和美学评分>= 4.5）上**以256x256的分辨率训练了550k步，**然后接着**以512x512的分辨率在同一数据集上进一步训练了850k步**。

SD 1.x系列模型主要采用LAION-5B中美学评分>= 5以上的子集来训练，而到了SD 2.0版本采用美学评分>= 4.5以上的子集，**这相当于扩大了训练数据集**。

#### 1.2.2 不同训练过程得到的不同版本的模型

- 512-base-ema.ckpt：SD 2.0的基础版本模型
- 768-v-ema.ckpt：在512-base-ema.ckpt模型的基础上，使用[v-objective（Progressive Distillation for Fast Sampling of Diffusion Models）损失函数先训练 150k 步，然后以 768x768 分辨率在 LAION-5B 数据集的子集上又进行了140k步的训练得到
- 512-depth-ema.ckpt：在512-base-ema.ckpt 模型的基础上继续进行 200k 步的微调训练。只不过在训练过程中增加了图像深度图（深度图信息由 MiDaS 算法生成）作为控制条件
- 512-inpainting-ema.ckpt： 在 512-base-ema.ckpt 模型的基础上继续训练了200k步。和 stable-diffusion-inpainting 模型一样，使用 LAMA 中提出的Mask生成策略，将Mask作为一个额外条件加入模型训练，从而获得一个图像inpainting模型
- x4-upscaling-ema.ckpt： stable-diffusion-x4-upscaler 模型是基于 Latent Diffusion 架构的4倍超分模型，采用了**基于VQ-reg正则的VAE模型**，下采样率设置为 f=4。这个模型使用LAION中分辨率大于 2048x2048 的子集（10M）作为训练集训练迭代了 1.25M 步，同时在训练过程中设置 512x512 的crop操作来降低显存占用与加速训练。如果我们用SD系列模型生成512x512分辨率的图像，再输入 stable-diffusion-x4-upscaler 模型就可以得到 2048x2048 分辨率的图像。

## 2 Stable Diffusion 2.1 相较于 2.0 的改进点

SD 2.0在训练过程中采用NSFW检测器过滤掉了可能包含安全风险的图像（punsafe=0.1），但是同时也过滤了很多人像图片，这导致SD 2.0在人像生成上效果并不理想，**所以SD 2.1在SD 2.0的基础上放开了过滤限制（punsafe=0.98），在SD 2.0的基础上继续进行微调训练**。

最终SD 2.1的人像的生成效果得到了优化和增强，同时与SD 2.0相比也提高了生成图片的整体质量，其base生成分辨率有512x512和768x768两个版本：

- 512-base-ema.ckpt（stable-diffusion-2-1-base模型）：在stable-diffusion-2-base（512-base-ema.ckpt） 模型的基础上，放开NSFW检测器限制（punsafe=0.98），使用相同的训练集继续微调训练220k步。
- 768-v-ema.ckpt（stable-diffusion-2-1模型）：在stable-diffusion-2（768-v-ema.ckpt 2.0）模型的基础上，使用相同NSFW检测器规则（punsafe=0.1）和相同数据集继续微调训练了55k步，然后放开NSFW检测器限制（punsafe=0.98），额外再微调训练了155k步。

### Ref:
- https://zhuanlan.zhihu.com/p/632809634
