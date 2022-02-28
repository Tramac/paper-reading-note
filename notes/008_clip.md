# CLIP

[Paper](https://arxiv.org/pdf/2103.00020.pdf) | [Talk](https://www.bilibili.com/video/BV1SL4y1s7LQ)

CLIP自提出以来就引起很大关注，其方法简单但又效果很好。CLIP的迁移学习能力非常强，预训练好的模型能够在任意一个视觉分类的数据集上取得不错的效果，而且是zero-shot的，即它完全没有在这些训练集上做过训练，就能得到很好的效果。文章在30多个数据集上做了测试，包括了OCR，视频动作检测，坐标定位以及一些细分类任务，涵盖面非常广，尤其是在ImageNet上的结果，CLIP在不使用ImageNet训练集的情况下，就可以取得和之前ResNet50在有监督情况下的结果。

## Part1.标题&作者

Learning Transferable Visual Models From Natural Language Supervision：利用自然语言处理来的一些监督信号，去学习一个可以迁移效果很好的视觉模型。

Transferable: 迁移性能好

## Part2.摘要

- 目前的视觉模型训练通常是使用提前定义好的一些物体类别的集合，然后模型通过去预测这些定义好的类别，从而完成模型的训练。提前定义好的类别集合：比如ImageNet 1000类，cifar10 10类，coco 80个类，cityscapes 19个类，kinetics 400个类等。这种提前定义好的样本集合可以大大简化模型学习的难度，但是也限制了模型的泛化性
- 直接从自然语言里获取一些监督信号，只要是语言描述过的物体，就有可能让视觉模型去识别到这个物体，而不仅仅是提前定义好的那些固定的类别。
- 本文证实了用一个简单的预训练任务，就可以高效的、可扩展的去学习到最好的图像表征。具体的，给定一些图片和一些文本，模型去判断图片和哪些文本是配对的。既然要做的是配对的任务，那么所需要的训练数据就应该是配对的，本文爬取了一个4亿大小的图片、文本对，然后用自监督的方式去预训练模型。
- CLIP是利用多模态的对比学习去完成模型预训练，然后利用自然语言去引导视觉模型去做物体分类。分类也不再局限于已经学到的视觉概念，还能扩展到新的类别，也就是这个预训练模型可以直接在新的下游任务上做zero-shot推理的。

## Part3.引言

- 直接从文本数据里去预训练一个模型已经在NLP里取得了革命性的成功，比如BERT、GPT等，不论是使用自回归预测的方式还是使用掩码完形填空的方式，都是一种自监督的训练方式。所以，它的目标函数是和下游任务无关的，只是想通过预训练得到一个泛化能力好的特征，这套框架基本都是"text-to-text"，即文字进文字出，其模型架构也是跟下游任务无关的，所以在处理下游任务时，无需再针对下游任务去设计head或者其它特殊处理。比如GPT3，可以做分类、翻译、生成等。
- VirTex、ICMLM、ConVIRT均是基于transformer的工作，不同的是VirTex是用自回归预测的方式做预训练，ICMLM是利用完形填空的方式，ConvVIRT与CLIP非常相似，但是只在医疗图像上做的实验。三者在模型或数据上都没有取得很大的规模。
- 之前的一些引入自然语言信号的弱监督预训练方法效果不好，基本都是因为训练数据量级小，而CLIP收集了400M（4亿）对图像/文本对作为训练样本。模型上从ResNet50、EfficientNet到Vit-L，在大数据大模型的加持下，CLIP在多个数据集上取得很好的效果。
- 迁移学习的效果基本是和模型的大小呈正相关的，模型逐步增大，迁移学习的效果也在逐步增高，是一个比较平滑的过程。
- 作者为了进一步验证CLIP模型学到的特征的有效性，做了类似对比学习中的linear-probe验证，效果也比之前在ImageNet上有监督的结果好，而且泛化性能更好。

## Part4.模型方法

**4.1 利用自然语言信号训练视觉模型的优点**

- 不需要再去标注数据，直接抓去图片/文本对即可，很容易把数据规模做大
- 监督信号是一个文本，不再是N选1的标签，模型的输入输出的自由度就会大很多
- 训练时，将图片与文本绑定在一起，学得的特征不再是单单的视觉特征，而是一个多模态特征，很容易去做zero-shot迁移学习

**4.2 常见的一些图片/文本数据集**

- MS-COCO
- Visual Genome
- YFCC100M

前两者数据标注质量高，但是规模小，约10w数据；YFCC100M虽有1亿数据，但是标注质量差，清洗后的规模只有1500w。

CLIP收集了一个4亿规模的数据集--WebImageText。

**4.3 预训练方法的选择**

***将预测性任务改为对比型任务***

视觉模型的训练代价一般都比较高，模型大，计算量高。如果引入自然语言信号，而且数据集更大，任务更难，代价将会更大。

CLIP首先尝试类似VirTex的方式，对于图像使用一个CNN，对于文本使用一个Transformer，均是从头开始训练，任务是给定一张图片，然后预测图片所对应的文本，此为预测性任务。

> 预测性任务的缺点：对于给定图片，去预测其文本描述，需要逐字逐句的去预测该文本，这个任务相对来说就太难了。因为，对于一张图片可以有很多不同的描述，而且文本描述之间可能差距很大，如果用预测性任务去预训练模型，就会有太多的可能性，导致模型的训练非常困难。
> 
> 对比学习的优点：对比学习任务只需要判断图片和文本是否配对，任务简单的很多，就不需要逐字逐句的预测文本了，约束放宽了很多，该监督信号也更加的合理。

<image src="https://user-images.githubusercontent.com/22740819/153564599-f9aabcd8-ba05-4c5c-a830-99977ba740a0.png" width=400>

CLIP将预测性任务替换为对比型的任务，训练效率提高了4倍。

***伪代码***
  
<image src="https://user-images.githubusercontent.com/22740819/153567259-8f7d45d3-15f2-4c7f-a622-ecf1ccd6bb3e.png" width=400>

- 给定一个mini-match的图片输入I与文本输入T，分别通过一个编码器得到对应的特征I_f与T_f；
- 对于两个特征分别通过一个投射层，然后再做L2归一化，得到特征I_e与T_e；
- 对两个特征计算余弦相似度，最终计算loss。

***一些细节***

- CLIP由于使用的数据集特别大，所以在训练时不存在over-fitting的问题，图片编码器与文本编码器都不需要提前进行预训练；
- 最后对特征进行投射时，并没有使用非线性投射层，使用的而是线形投射层，两者对效果的影响不大。猜测非线性投射层只对纯图片的单模态任务比较重要；
- CLIP使用的数据增强只有随机裁剪；
- 由于模型、数据规模很大，调参成本太大，CLIP在训练时将对比学习中的temperature参数设置成可学习的标量，该参数在训练过程中直接被优化了。

**4.4 训练**

- 对于视觉侧共训练了8个模型，包括5个ResNet与3个Vision Transformer，所有这些模型使用Adam训练32个epoch；
- 调参时，只使用ResNet50训练1个epoch进行对比；
- 训练时，batch size为32768，非常大；
  
## Part5.Zero-Shot与Prompt Engineering

**5.1 Zero-Shot Transfer**

之前自监督或无监督的方法，它们主要是研究的是特征学习能力，目标是学得一种泛化能力比较好的特征，比如MoCo，SimCLR等。但即便有了好的特征，当应用到下游任务时，还是需要有标签的数据进行finetune，这时又会面临一些问题，比如下游任务的数据不好收集，或者数据分布有偏移等。

Zero-Shot Transfer是说，我们致力于训练好一个模型，接下来就不用再去训练或finetune了。

***Zero-Shot Transfer具体是如何做的***

<image src="https://user-images.githubusercontent.com/22740819/155678059-efe6fd57-4490-4332-9d81-b06642390fef.png" width=600>

当CLIP预训练完之后会有两个编码器，即Image Encoder和Text Encoder。
  
  - 给定一张图像，通过Image Encoder得到图像特征I1；
  - 文本侧的输入则是一些感兴趣的标签，比如我们有plane、car、dog、bird四个标签，首先这些标签会通过prompt engineering生成四个句子，比如A photo of a {object}，然后将这四个句子分别通过Text Encoder得到四个文本特征T1，...，T4；
  - 然后计算四个文本特征T1~T4与图像特征I1之间的余弦相似性cosine similarity，最后得到的相似度通过softmax层得到概率分布，这时概率最大（相似度最高）的句子大概率就是在描述你的输入图像，也就是说句子中所包含的物体就是你输入图像里应该有的物体。

  e.g.以ImageNet为例，如果对ImageNet中所有图片进行推理：对每个类生成一个句子，则会生成1000个句子，然后对于每张图片与这1000个句子计算相似性。

 **5.2 Prompt Engineering And Ensembling**
  
  ***Prompt Engineering***
  
  Prompt: 可理解为“提示”的意思，prompt engineering主要起到提示的作用或者说文本引导作用。
  
  ***为什么要做Prompt Engineering？***
  
  - Polysemy：多义性，也就是一个单词同时可能拥有很多的含义。如果在做文本/图像匹配时，每次只用一个单词做文本特征提取，就有可能面临这种问题。比如在ImageNet中同时存在着两个类construction cranes与cranes，前者指起重机，后者是鹤，如果不加任何语境只用cranes提取特征，那么特征很可能是有歧义的。
  - 预训练时，文本侧一般也都是一个句子，如果在做推理时每次输入的只是一个单词，就会存在distribution gap问题，提出的特征可能就不是很好。
  
  基于以上两个原因，本文提出了prompt template的方式，也就是给定一个提示模板，把标签放入模板中形成一个句子。从单词到句子首先解决了distribution gap问题，而且模板一般是“A photo of a {label}."，在此语境中label一般是名词，同时缓解了歧义问题。
  
  ***Prompt Ensembling***
  
  Prompt Ensembling就是多用一些提示模板，做多次推理，然后把结果综合起来，ensemble一般都可以得到更好的结果。
  
## Part6.实验
  
  **6.1 CLIP vs ResNet50**
  
  <image src="https://user-images.githubusercontent.com/22740819/155922497-407b6990-34a5-4c29-806a-ebe12e0cbe16.png" width=300>
    
  其中，CLIP是预训练后直接做zero-shot transfer，ResNet50是只用resnet50提取特征，然后接线型分类头预测。结果显示，在27个数据上其中16个CLIP结果更好，一般对于给物体进行分类的数据集来说，CLIP结果一般会更好。
    
  对于一些更难的数据集，比如文本分类，用CLIP zero-shot transfer做不是很合理，应该用few-shot方式。
  
  **6.2 Zero-Shot CLIP vs Few-Shot CLIP vs other Few-Shot**
  
  <image src="https://user-images.githubusercontent.com/22740819/155923166-93f9779b-b35b-4598-8a12-40ae53eebe6b.png" width=300>
  
  Few-Shot：相比于Zero-Shot直接做预测，Linear Probe方式是将backbone冻住提取特征，然后用有标签数据训练得到分类头，由于也算是见到了样本，可以认为是Few-Shot。
  
  - Zero-Shot CLIP比之前专做Few-Shot迁移的BiT-M方法结果好；
  - 对CLIP来说，当样本比较少时(1～4)用了训练样本的Few-Shot反而不如Zero-Show效果好；
  - 随着训练样本的增多，Few-Shot CLIP的效果是最好的；
  
  **6.3 Zero-Shot vs Few-Shot vs 全部数据做linear probe**
  
  linear probe: 将预训练好的模型冻住，只拿全部数据训练一个分类头做预测。
    
  fine-tune: 不把backbone冻住，用全部数据微调所有参数。由于CLIP所使用的数据很大，fine-tune效果应该很好，而且CLIP的目标主要是做一个好的预训练模型，所以不用fine-tune做对比了。
  
  <image src="https://user-images.githubusercontent.com/22740819/155935791-25dc37a4-2bb1-4bda-b41e-f12df9de27f1.png" width=600>
  
  从右图结果可看，即便用所有数据训练，CLIP效果也是更好。
  
  **6.3 CLIP vs ImageNet上最好模型--EfficientNet L2 NS
  
  <image src="https://user-images.githubusercontent.com/22740819/155936322-54cc7e8c-6852-4fb1-a18b-5beaef91b9fe.png" width=300>

  结果显示，在27个数据集上，其中有21个CLIP的结果优于EfficientNet L2 noisy student。
  
  **6.4 CLIP预训练模型的鲁棒性**
    
  <image src="https://user-images.githubusercontent.com/22740819/155936717-0ea0d8db-2799-4718-9536-844f5a695a0d.png" width=600>
  
  当下游任务的数据与预训练数据分布有偏移时，CLIP的效果退化相对也不是很多。

## Part7.CLIP的局限性
  
  - CLIP与基线的ResNet50相比确实是有些优势，ImageNet上准确都是76.2+，但是现在一些SOAT方法比如MAE等都到了90+，这十几个点的差距靠扩大数据等方法还是有些吃力的；
  - CLIP zero-shot在某些数据集上效果其实不如ResNet50，比如细分类数据集或者比较抽象的数据，或者更难的下游任务上；
  - CLIP在做迁移时，如果下游任务的数据与预训练任务的数据分布差异特别大，泛化效果也不是很好，比如MNIST就不行；
  - CLIP虽然可以zero-shot预测结果，但还是从给定的固定类里去做选择，目前对生成式的任务没有好的方法；
  - 对数据的利用还不是很高效，需要大量的数据投喂，如果能减少数据用量也很好；
  - 实验过程中一直是在使用ImageNet和其它27个数据集做对比，无形之中引入了偏见；
  x
  
