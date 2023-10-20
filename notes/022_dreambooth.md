# DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

[paper](https://arxiv.org/pdf/2208.12242.pdf) | [code](https://link.zhihu.com/?target=https%3A//github.com/XavierXiao/Dreambooth-Stable-Diffusion)

## 标题&作者
DreamBooth：通过微调 text-to-image 扩散模型以实现 subject-driven 生成。
text-to-image 指的是常规的从文本到图像的扩散模型，根据文本提示生成高质量和多样化的图像，但是这些模型缺乏在指定场景中合成它们的能力。
subject-driven 可以理解为输入的少量图像所确定的主体（subject），比如示意图中的柯基犬。
DreamBooth 将两者结合，text 指定场景，image 给定主体，合成该主体在不同场景下的图像。

作者来自谷歌团队。

## 摘要
目前，大型的 text-to-image 模型已经能够根据给定的「text prompt」来生成高质量、多样化的图像，然而，这些模型缺乏**特定主体的再现能力**，特别是对个人定制的物体，很难实现个人物体在风格、造型等方面的再现能力。

DreamBooth 提出了一种具备「定制化」（personalization）能力的 text-to-image 方法：**通过一种基于「new autogenous class-specific prior preservation loss」的微调策略，训练少量需要定制化物体的图片，实现该生成物体在风格上、背景多样化的再创造**，在保证物体特征不变的前提下，姿势、视角和光照条件以及背景内容可以自定义（这些主要得益于预训练模型学到的先验知识）。
<div align="center">
<img width="800" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/54f61812-339d-4749-b921-97e7eda1c289">
</div>
如上图所示，输入 4 张柯基的照片，在保持柯基的外貌特征不变的前提下，可以将其背景改为在水里、狗窝等，充满创意。

通俗理解为，DreamBooth 的微调使模型将「unique identifier」与「specific subject」绑定在了一起，实现精准的生成目标内容。
> unique identifier：个人理解就是对指定主体的描述，包含在 text prompt 中，比如一只“柯基”的照片；
> specific subject：即少量图像中指定的主体，比如柯基犬。

因为，通常的文生图模型无法仅通过 text prompt 来精准的定位视觉生成空间的特定目标（比如柯基），只能在 dog 这个大类里面，带有随机性的生成各种各样的狗，即使 prompt 描述的非常精细，也无法与视觉空间形成精准的映射，那假如在视觉空间上放置一个精准定位词[V]，就像是在目标位置放置了一个 GPS，这样就可以在诸多的 dog 中定位到柯基这个特定类别了。 
<div align="center">
<img width=800" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/8ad2efbc-f0b3-491d-a607-89072e8e6880">
</div>

## 引言
目前，大型的 text-to-image 模型已经可以基于「text prompt」生成高质量、多样化的图像了，这得益于模型从大量的「image-caption pairs」数据中学得的强大的语义先验知识（semantic prior）。

举例说明，大量的数据训练使模型学得了将单词 “dog” 与各种各样的狗的图片（各种姿势、场景下）绑定在了一起，所以，当你让模型生成有关 “dog” 的图像时，即可得到不同姿势、场景、光照、视角下的新的图像。虽然，这些模型的综合能力是非常强的，但它无法精确地再现某个特定的主体，比如「柯基A」。主要原因是模型的「输出域 output domain」的表达能力是有限的，即使是 text prompt 再怎么详细，也可能产生具有不同外观的实例。此外，即使模型的「text embedding」在「language-vision」空间中是共享的，也无法准确的重建出给定的主体的外观，而只能创建出与给定主体相类似的内容，如下图所示：
<div align="center">
<img width="450" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/c9da5e22-5385-478d-847a-a226c70dbc29">
</div>
上图给出了某个特定闹钟的图像作为输入，其中 DALL-E2 和 Imagen 两个方法所生成的图像虽然形似，但都不是这个给定的闹钟，而 DreamBooth 生成的闹钟和给定的完全相同。

本文提出了一种具备「定制化」能力的 text-to-image 模型，目标是扩展模型的「language-vision」字典，使得能够「绑定」新的词汇与特定主体。
> 通俗一些，比如通常的文生图模型已经学得的「language-vision」已经包含了 *“dog” - 各类狗*之间的映射，现在 DreamBooth 可以新增一个 *“柯基A” - 用户指定的柯基犬* 的映射关系，使得当用户让模型生成有关“柯基A”的图像时，肯定就是这个用户所给定的这一只。

效果类似于magic photo booth “神奇照相亭” —— 一旦拍摄了几张某个对象的图像，照相亭就会根据简单直观的文本提示生成该对象在不同条件和场景下的照片。

DreamBooth 通过微调 text-to-image 模型，训练数据为指定图像与包含特定词的 text prompt 对，从而使特定实例「class-specific instance」与某个特定词「unique identifier」绑定在一起。为了防止语言漂移「language drift」导致模型将类名（例如“狗”）与特定实例相关联，本文提出了一种**autogenous, class-specific prior preservation loss**，它利用了预训练模型学得的语义先验知识，鼓励它生成与指定主体相同的类不同实例。
> 看到这里，对「unique identifier」的理解应该就比较明确了，字面意思为唯一标识符，直观意思就是对特定主体的表示，如“柯基”。那么，训练数据对就是：{image: 柯基图1, text prompt: 一张柯基的图片}。

## 3.方法

给定少量包含某个特定主体的图片，无需任何文本描述，目标是生成该主体在「 text prompt」指定场景下的新图像。
3.1 节介绍了 text-to-image diffusion models 的背景知识，3.2 节介绍了所提出的将「unique identifier」与「 specific object」绑定在一起的技术，3.3 节介绍了所提出的「class-specific prior-preservation loss」，用以解决微调时的「language drift」问题。

### 3.1 Text-to-Image Diffusion Models

扩散模型的背景知识比较复杂，论文中也没有展开，这里推荐苏剑林的 [blog](https://spaces.ac.cn/archives/9152) 中对 diffusion model 的解释，感兴趣的可以看看。

本文我们只针对 text-to-image 扩散模型进行介绍：假设已知一个扩散模型为 $\hat{x_{\theta} }$，给定一个初始噪声 ε（正态分布），以及一个条件向量 $c = Γ(P)$，其中 P 表示「 text prompt」， Γ 表示「 text encoder」，那么生成图像可以表示为： $x_{gen} = \hat{x_{\theta} }(\epsilon, c )$ ，优化该模型的损失函数为均方误差，如下：

$$
\mathbb{E}_{x,c,\epsilon ,t}[w_t||\hat{x}_{\theta}(\alpha_tx + \sigma_t\epsilon , c) - x||]
$$

其中，x 为真实图像 ground truth，c 为条件向量，由 text prompt 生成，$\alpha_tx$ 、$\sigma_t$ 和 $w_t$ 为控制噪声和采样质量的参数，随扩散时间 t 变化的函数。

### 3.2 Personalization of Text-to-Image Models

我们的第一个任务是将指定的主体实例「注入」到模型「输出域」中，以便在生成新图像时可以从模型输出域中「索引」到这个主体。

一个自然的想法是，使用少量的（few-shot）包含该主体的数据对模型进行微调，但是对生成模型来说，使用少量的样本进行微调时必须特别小心，因为很容易造成过拟合或「mode collapse」（模型坍塌）。目前已经有部分工作来避免这些问题，但是它们的目标都是寻求生成与给定主体类似的目标，而我们的目标是保留主体的外观。

对于「过拟合」和「模型坍塌」问题，本文观察到一个奇怪的发现：使用 3.1 中的损失函数，如果对微调进行谨慎的设置，并不会出现这些问题。（没明白这一句是要表达什么，感觉与前面内容有冲突。。。）

**Designing Prompts for Few-Shot Personalization**

为 few-shot 微调设计 prompt。
我们的目标是：**将一个新的（unique identifier, subject）关系对「注入」到模型“字典”中**。为了绕过为给定图像写详细描述的开销，本文选择一种更加简单的方法，即对所有输入图像均采用**a [identifier] [class noun]**的 prompt，其中

 - [identifier]：对指定主体的标识符，比较细粒度，比如柯基。
 - [class noun]：指定主体的更粗粒度的类别，比如 cat、dog 等。可以由用户提供或者通过一个分类器生成。

所设计的 prompt 中选择添加一个 [class noun] 来进行约束，我猜是为了和预训练时使用的 prompt 更加契合，并且发现如果不使用  [class noun] 约束会造成训练时间变长和 「language drift」，同时性能降低。本质上，我们想要寻求一种能够充分利用预训练模型学得的先验知识的方法，并将其与「unique identifier」结合。

**Rare-token Identifiers**

我们发现，如果选取现有的单词作为 [identifier] 并不是最优的，因为模型必须学会将它们与它们原始的含义分开，然后在将其与给定的主体结合起来。这就促使我们想要选取的 [identifier] 在预训练模型中是具有弱先验的，即**预训练模型不怎么认识这个词**，在这里被称作「rare-token」。
> 单词具有强先验 == 模型见过很多次这个词

一种危险的做法是选取英文中的随机字符并将它们连接起来生成罕见的 [identifier]，比如 “xxy5syt00”。但是，实际上这种特殊的单词可能被 tokenizer 分开，所分开的词仍然具有强语义性。

本文提供了一种寻找「rare-token」的方式：首先在词表中找出罕见的 token，然后将该 token 进行反转，作为最终的 identifier，以此来最大可能降低改词具有强先验的概率。具体的，本文采取的方式是**从 T5-XXL 模型词表的 5000~10000 区间内随机选取 1-3 个 token 进行组合作为 [identifier]**。

### 3.3 Class-specific Prior Preservation Loss

根据本文的实验，**finetune 模型的所有层可使主体的保真度更高**，但调整所有层时有可能导致以下两个问题：

- language drift：这是在语言模型中是一个经常遇到的问题，通常一个在大规模文本预料上训练好的语言模型，当迁移到下游任务进行 fine-tuning 时，有可能丢失该语言的语法和语义知识，本文也是第一个发现在 diffusion model 中也存在相似的现象，即**模型会慢慢遗忘如何生成与主体相同类别的图像**；
- reduced output diversity：即输出多样性可能减少。Text-to-image 扩散模型天然具有大量的输出多样性，当在少量样本上进行微调时，我们希望能够以新颖的视角、姿势及清晰度生成新的主体图像。但是由于样本少，存在输出
姿势、视角多样性变少的风险，尤其是当模型训练时间过长时。

为了缓解上述两个问题，本文提出一种「autogenous class-specific prior preservation loss」，以鼓励多样性并避免「language drift」。本质上，文本的方法是**用它自己生成的样本来监督模型，以便在小样本微调开始时保>留先验信息**，这使得它能够生成属于该大类（这里指的是 dog 这种大类）的不同图像，并保留有关该类先验的知识，以便与有关主体实例的知识结合使用。

具体来说，给定一个符合正态分布的随机噪声 $z_{t_1} \sim \mathcal{N}(0, I)$ 以及条件向量 $c_{pr} := \Gamma (f("a [class \ noun]"))$，使用一个固定的预训练好的扩散模型生成样本 $x_{pr} = \hat x(z_{t_1}, c_{pr})$，则损失函数为：

$$
\mathbb{E}_{x,c,\epsilon, \epsilon',t} [w_t||\hat x_\theta (\alpha _t + \sigma _t\epsilon , c) - x||_{2}^{2} +
 \lambda w_{t'}||\hat x_\theta (\alpha _{t'}x_{pr} + \sigma _{t'}\epsilon ', c_{pr}) - x_{pr}||_{2}^{2}] 
$$

其中第二项是用用自己生成的图像来监督模型，用于保留模型先验。

> 用模型本来生成的图像作为监督信号，当然是希望保留原来的信息了。

<div align="center">
<img width="457" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/581f1bec-5dc6-462b-8adb-fe46fc23be08">
</div>

上图为训练示意图，尽管很简单，但我们发现**这种保留先验「prior-preservation」的损失对于鼓励输出多样性和克服「language-drift」是很有效的，另外还发现我们可以训练模型进行更多迭代，而不会出现过度拟合的风险**。

经实验，以下参数即可达到很好的结果：

| 超参数       | 值                                           |
| ------------ | -------------------------------------------- |
| Items        | 1000                                         |
| Lambda       | 1                                            |
| Lr           | 1e-05 for Imagen，5e-06 for Stable Diffusion |
| dataset size | 3-5 images                                   |

在训练过程中，使用 prompt “a [class noun]” 生成了 ～1000个样本（用于计算损失第二项），但很少部分被用到。

使用 TPUv4 训练 Imagen 只需要 5 分钟，使用 NVIDIA A100 训练 stable diffusion 同样只需要 5 分钟。

## 4. 实验

**Evaluation Metrics**

-  CLIP-I：生成图像和真实图像 CLIP embeddings 之间的平均余弦相似度，但它不是为了区分可能具有高度相似的文本描述的不同主体而构建的；
- DINO：本文新提出的，生成图像和真实图像 ViTS/16 DINO embeddings 之间的余弦相似度。

**对比方法**

- DreamBooth using Imagen
- DreamBooth using Stable Diffusion
- Textual Inversion using Stable Diffusion
> Textual Inversion 是一个只是 finetune 模型的 text encoder 模块其中的一小部分的方法。

主体保真度与 prompt 保真度对比：
<div align="center">
<img width="468" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/16c04105-cd96-48a6-8873-a677ed639179">
</div>

实际生成效果对比：
<div align="center">
<img width="480" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/ae1e3551-eef9-4462-89e2-46ca863cce89">
</div>

- Textual Inversion 方法已经完全不是给定的物体了。

有无「prior-preservation loss」的效果对比：
<div align="center">
<img width="470" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/83ae4c51-789b-4104-98ce-538cbd9d2b2f">
</div>

- 使用「prior-preservation loss」明显增加了多样性。

## 5.局限性

<div align="center">
<img width="473" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/6c0db1ce-9815-4b35-a8e3-d0b7eadcbe4b">
</div>

上图给出了一些可能存在的问题：

- 与给定场景不符；
- 场景不准；
- 过拟合；
