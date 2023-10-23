
# EasyPhoto: Your Personal AI Photo Generator

[paper](https://arxiv.org/abs/2310.04672.pdf) | [sd-webui-EasyPhoto](https://github.com/aigc-apps/sd-webui-EasyPhoto) | [easyphoto](https://github.com/aigc-apps/easyphoto)

## 题目&作者

你的个人 AI 照片生成器。

作者团队全部来自阿里团队。

## 摘要

[Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 是一个基于 Gradio 搭建的 web 界面，提供了很多 stable diffusion 模型相关的功能。本文为该 UI 提供了一个新的插件「EasyPhoto」，可以实现 AI 人像生成。通过使用 5～20 张特定用户的相关图像对模型进行微调（通常是训练一个 LoRA），达到基于任意模板生成 AI 照片的目的。

我们目前的实现支持多人、不同照片风格的修改，此外，还允许用户使用强大的 SDXL 模型生成精彩的模板图像，以此来增强 EasyPhoto 提供更多样化和令人满意结果的能力。

EasyPhoto 的 pipeline 的能力还在继续扩展，期望后续不局限于人脸的生成。

## 1.引言

AIGC 真实人像写真，妙鸭相机作为 AIGC 领域一款收费产品成功的向大家展示了如何使用 AIGC 技术少量的人脸图片建模，快速提供**真、像、美**的个人写真，在极短的时间拥有大量的付费用户。

同时，随着 Stable Diffusion 领域开源社区的快速发展，社区也涌现出了类似 FaceChain 这样基于 Modelscope 开源社区结合 diffusers 的开源项目，用于指导用户快速开发个人写真。然而对于大量使用 SD web UI 的 AIGC 玩家，短时
间内却没有一个效果足够好的开源插件，去适配真人写真这一功能。

EasyPhoto 使用 SD 模型的 image-to-image 能力，保证生成图像的真实、准确性。通过用户上传几张同一个人的照片，快速训练 LoRA 模型，然后结合用户上传的模版图片，快速生成**真、像、美**的写真图片。

> 用户上传的几张照片被称为「digital doppelganger」。

在推理阶段，通过使用 ControlNet 来控制生成图像与模板图像之间的相似性。但是会存在几个问题：

- identity loss：身份丢失
- boundary artifacts：边界伪影

为了克服上面的问题，所以使用了 two-stage 的扩散过程。该方法确保了生成的图像保证了上传图像的身份正确（即肯定得是同一个人），并且极大限度的减少视觉不一致性。

本文主要内容：

- EasyPhoto 中提出了一个新的训练 LoRA 模型的方法，通过集成多个 LoRA 模型来保持生成结果的人脸保真度；
- 结合强化学习方法来优化 LoRA 模型的「facial identity rewards」，进一步提高生成结果与训练图像之间的身份相似度；
- 提出一个 tow-stage inpaint-based diffusion 过程，旨在生成具有高相似度和美观度的图像；图像预处理过程经过精心设计，为 ControlNet 创造合适的指导；此外，利用强大的 SDXL 模型来生成模板，从而产生各种多样化且真实的输出；

## 2.相关工作

### 2.1 Stable Diffusion

Stable Diffusion 作为 Stabibity-AI 开源的图像生成模型，分为 SD1.5 / SD2.1 / SDXL 等版本，是通过对海量的图像文本对（LAION-5B）进行训练结合文本引导的扩散模型，使用训练后的模型，通过对输入的文字进行特征提取，引导扩散模型在多次的迭代中生成高质量且符合输入语义的图像。

### 2.2 ControlNet

ControlNet 通过添加部分训练参数，对 Stable Diffusion 模型进行扩展，用于处理一些额外的输入信号，例如边缘、深度图、人体姿态等，从而完成利用这些额外输入的信号，引导扩散模型生成与信号相关的图像内容。

### 2.3 LoRA

LoRA 是一种基于低秩矩阵的对大模型进行少量参数微调训练的方法，广泛应用于各种大模型的下游微调任务。AI 真人写真需要保证最后生成的图像和我们想要的生成的人是相像的，这就需要我们使用 LoRA 技术，对输入的少量图片，进行一个简单的训练，从而使得我们可以得到一个小的指定人脸（ID）的模型。

### 2.4 人脸相关 AI 技术

针对AI写真这一特定领域，**如何使用尽量少的图片，快速的训练出又真实又相像的人脸Lora模型，是能够产出高质量AI写真的关键**，网络上也有大量的文章和视频资料为介绍如何训练。这里介绍一些在这个过程中，我们使用的开源AI模型，用于提升最后人脸LoRA训练的效果。

在这个过程中我们大量的使用了 [ModelScope](https://www.modelscope.cn) 和其他Github的开源模型，用于完成如下的人脸功能

| 人脸模型 | 模型卡片 | 功能 | 使用 |
|-------------|-------------|-------|-------|
| FaceID | https://github.com/deepinsight/insightface | 对矫正后的人脸提取特征，同一个人的特征距离会更接近 | EasyPhoto图片预处理,过滤非同ID人脸EasyPhoto训练中途验证模型效果EasyPhoto预测挑选基图片 |
| 人脸检测 | cv_resnet50_face | 输出一张图片中人脸的检测框和关键点 | 训练预处理，处理图片并抠图预测定位模板人脸和关键点 |
| 人脸分割 | cv_u2net_salient | 显著目标分割 | 训练预处理，处理图片并去除背景 |
| 人脸融合 | cv_unet-image-face-fusion | 融合两张输入的人脸图像 | 预测，用于融合挑选出的基图片和生成图片，使得图片更像ID对应的人 |
| 人脸美肤 | cv_unet_skin_retouching_torch | 对输入的人脸进行美肤 | 训练预处理：处理训练图片，提升图片质量预测：用于提升输出图片的质量 |

### 2.5 SD Web UI

[SD Web UI](https//github.com/AUTOMATIC1111/stable-diffusion-webui)] 是最常用的 Stable Diffusion 开发工具，我们提到的文图生成 / ControlNet /LoRA 等功能，都被社区开发者贡献到这一工具中，用于快速部署一个可以调试的文图生成服务，所以我们也在 SDWebUI 下实现了 EasyPhoto 插件，将上面提到的 人脸预处理/训练/艺术照生成全部集成到了这一插件中。

## 3.训练过程

AI 真人写真生成是一个基于 Stable Diffusion 和 AI人脸相关技术，实现的定制化人像 LoRA 模型训练和指定图像生成 pipeline 的集合。

**EasyPhoto生成**

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/d831cd70-b901-4e7b-8938-8b6457156dcf">
</div>

上图为 EasyPhoto 的生成（推理）阶段示意图，其主要步骤如下：

1. 首先使用人脸检测对输入的指定模板进行人脸检测（crop & warp），并结合数字分身进行模板替换；
2. 采用 FaceID 模型挑选用户输入的最佳 ID Photo 和模板照片进行人脸融合（face fusion）；
3. 使用融合后的图片作为基底图片，使用替换后的人脸作为 control 条件，加上数字分身对应的 LoRA，进行图到图局部重绘生成；
4. 采用基于 stable diffusion + 超分的方式进一步在保持 ID 的前提下生成高清结果图。

**EasyPhoto训练**

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/0062916b-c086-4fd5-a938-6f23763ee77e">
</div>

上图为 EasyPhoto 的训练示意图，整个 pipeline 基于开源模型 stable diffusion + 人物定制 LoRA + ControlNet 的方式完成图像生成。训练采用了大量的人脸预处理技术，用于把用户上传的图片进行筛选和预处理，并引入相关验证和模型融合技术。

主要步骤如下：

 1. 采用 FaceID 和图像质量分数对所有图片进行聚类和评分，筛选出非同 ID 照片；
 2. 采用人脸检测和主体分割（saliency detection），抠出第 1 步筛选后的人脸图片进行人脸检测抠图，并分割去除背景；
 3. 采用美肤模型优化部分低质量人脸，提升训练数据的图片质量；
 4. 采用单一标注的方案，对处理后的训练图片进行标注（EasyPhoto 中固定 prompt 为 "easyphoto_face, easyphoto, 1person"），并使用相关的 LoRA 训练；
 5. 训练过程中采用基于 FaceID 的验证步骤，间隔一定的 step 保存模型，并最后根据相似度融合模型；

> validation step 是一个比较重要的步骤，具体的操作是在训练过程中计算输入图像与生成图像（由训练的 LoRA 结合一个 canny ControlNet 生成）之间的 FaceID gap，该过程有助于 LoRA模型的融合，最终确保训练后的 LoRA 模型为用户期望得到的。另外，face_id score 最高的生成图像会被选作 face_id image，将用于增强推理生成的身份相似性。

训练时同时训练了多个 LoRA 模型，它们的训练目标以及集成方法暂不展开。

## 4. 推理过程

<div align="center">
<img width="900" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/a3963ba7-3abb-4966-b361-c88cf21b0517">
</div>

上图展示了单 ID 下的推理过程示意图。主要包含三个部分：1）人脸预处理，获取模型输入图像和 ControlNet 所需的输入；2）第一次 diffusion，生成类似于用户输入的粗略结果；3）第二次 diffusion，修复边界伪影，使结果更加真实。

推理过程的输入包括一个「inference template」 和一个「Face id Image」，其中 face_id image 是训练阶段的验证过程中生成的 face_id score 最大的图像。

### 4.1 人脸预处理

基于「inference template」生成 AI 写真的直接方法是用 SD 模型修复模板的人脸区域，此外，引入 ControlNet 可以显著增强生成图像与用户的身份一致性和相似性。

然而，直接应用 ControlNet 来修复人脸区域可能会引入「伪影」问题或以下问题：

- 「template image」与「input image」不一致：这里的 input image 应该指的是 face_id image，很明显，是因为模板图像中的关键点与 face_id image 中的关键点不兼容，因此，利用 face_id image 图像作为 ControlNet 的输入将不可避免的导致最终结果不一致。
- 修复区域缺陷：mask 人脸区域并用新的 face 修复会导致明显的缺陷，特别是边界区域，这件对生成结果的真实性影响很大。
- ControlNet 使用之后无法保证身份一致性：由于训练过程中未使用 ControlNet，因此在推理过程中使用 ControlNet 可能会破坏训练后的 LoRA 模型的保留用户身份 ID 的能力。

为了解决上述问题，本文提出了三种人脸预处理方法，以获得第一次 Diffusion 阶段的输入图像、mask以及 reference image。

- Align and Paste：针对模板与 face_id image 之间的 **face landmark 不匹配**问题，本文提出了一种「affine transformation」和「face-pasting」算法。首先，分别得到 template image 和 face_id image 的 face landmark；然后，计算出能够将两者 face landmark 对齐的仿射变换矩阵 M；最后，将 face_id image 应用该变换之后直接贴在 template image 上（仅人脸区域）。该过程保留了 face_id image 的 landmark 的同时还与 template image 完成了对齐。随后，替换人脸后的图像将作为 openpose ControlNet 的输入，既保证和 face_id image 的身份一致性以及也保留与 template image 的人脸结构相似性。
- Face Fuse：为了纠正由「mask inpainting」造成的**边界伪影**，本文提出了一种 canny ControlNet 方法，以指导图像生成过程并确保有和谐的边缘。但是，template image 和 target ID 的 canny 边缘之间可能存在兼容性问题，为了解决该问题，本文使用了一个 FaceFusion 算法，用来融合 template image 与 roop image。融合后的图像对边缘的稳定性有明显的改善，有助于减轻边界伪影问题，使得在第一次 diffusion 时得到更好的结果。
	> roop image 是一张用户的 ground truth 图片。

- ControlNet-based Validation：因为在训练阶段训练 LoRA 时并没有训 ControlNet，直接在推理阶段使用 ControlNet 可能会对 LoRA 模型的**身份保存能力**产生影响。为解决该问题，在训练阶段引入了 「ControlNet for validation」，验证图像是由训练图像通过 LoRA 模型和 ControlNet 模型结合生成得到，然后比较生成图像与训练图像之间的 face_id score，以有效的集成 LoRA 模型，通过合并不同阶段的模型并考虑 ControlNet 的影响，大大增强了模型的泛化能力。总之，这种验证方法保证了在推理阶段使用 ControlNet 时的模型保证身份一致性的问题。

### 4.2 第一次 Diffusion

第一次 diffusion 是基于 template image 生成一张与「input user id」一致的图像。模型的输入为 template image 与 user's image 融合后的图像，同时还有由 template image 得到的校准后的 face mask（扩展到包含耳朵）。

为了增强对图像生成过程的控制，集成了三个 ControlNet 单元：

- Canny ControlNet：通过使用融合图像的 canny edge 作为 ControlNet 的控制条件，指导模型使 template 与 face_id image 融合的更精确。与原始 template image 相比，融合图像已经结合了 id 信息，所以，使用融合图像的 canny edge 可以最大限度的减少**边界伪影**问题；
- Color ControlNet：由融合图像的 color 作为 ControlNet 控制条件，修复了区域内的**颜色分布**，确保一致性和连贯性；
- OpenPose ControlNet：由 replaced image 的 openpose 作为 ControlNet 的控制条件，保证用户的 face 一致性和 template image 的面部结构。

通过合并这些 ControlNet 单元，第一次扩散过程已经可以生成与用户指定 ID 非常相似的高保真结果了。

### 4.3 第二次 Diffusion

第二次 diffusion 致力于微调和细化人脸边界附近的伪影问题。此外，还提供了 mask 嘴部附近区域的选项，以增强特定区域内的生成效果。

与第一次 diffusion 类似，首先将上一阶段的输出图像与 user's image（roop image）进行融合，以此作为第二阶段的输入。同时，fuse image 也作为 canny ControlNet 的输入，更好的控制生成过程。此外，还使用了 tile ControlNet 来实现更高的分辨率，利用增强生成图像的细节和整体质量。
> user's image = roop image

在最后一步中，对生成的图像进行后处理，将其调整为与 inference template 同样的尺寸，这保证了生成图像与模板之间的一致性和兼容性。此外，还使用了美肤算法和人像增强算法，进一步提高最终结果的质量和外观。

### 4.4 Multi User Ids

EasyPhoto 还同时支持多个 user IDs 的生成，这是一个单 ID 功能的扩展。

<div align="center">
<img width="900" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/f9f2597f-74c5-4a48-b1cb-4635c477f615">
</div>

上图为「multi-user IDs」生成的推理过程。

首先在 inference template 上进行人脸检测，然后分为多个 mask，对于每个只包含单个人脸的 mask 其它人脸区域均被白色遮盖住，那么该问题就被简化为了单 user ID 生成问题。

## 5.Anything ID

Anything ID 不局限于某个特定的人脸，可以是任何的 ID 特征，在此不展开了。

## 6.实验

- 写实风

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/9a8515dc-4b6d-41d5-b468-b9ebe7a8bf75">
</div>

- 漫画风

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/67846a35-d32f-49cc-b653-57f102a64e92">
</div>

- 多 ID 生成

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/8b4758cb-7e47-4a53-b603-fb64e8f47764">
</div>

- SDXL 生成的图像作为 template image

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/9a9d5e44-6129-4aeb-996c-e67a8a7b2e55">
</div>
