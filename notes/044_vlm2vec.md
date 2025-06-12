# VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks

[paper](https://arxiv.org/pdf/2410.05160) | [code](https://github.com/TIGER-AI-Lab/VLM2Vec)

## Highlight

- 代码都开源了
- 框架比较通用

## 摘要

- 提出 MMEB，一个多模态 embedding benchmark，覆盖分类、VQA、多模态检索、visual grounding 四个任务，另外包括 20 个训练集和 16 个评估集，现在被广泛应用
- 开源 VLM2VEC，一个基于对比学习的训练框架，可以处理图像-文本的任意组合，生成固定维度的向量

## 引言

目前多模态 embedding 面临的主要问题：

- 现有的工作通常只评估视觉 embedding 的效果，比如 ImageNet 分类、COCO/Flickr 检索
- 诸如 CLIP、BLIP、SigLIP 等模型，要么是分别处理文本和图像，要么只对视觉与文本特征进行简单的融合，限制了捕捉图文关系的能力
- 泛化能力差，或者说 zero-shot 场景能力不足

## MMEB

MMEB 是一个用于在多个任务上评估多模态 embedding 能力的 benchmark。共 36 个数据集，包括 classification、VQA、retrieval、visual grounding 四个任务，每个任务都被转化为一个 ranking 问题。模型输入包括一个 instruction 和一个 query（可以是 image、text 的任意组合），任务就是从 candidates 中挑选正确的答案，其实就是选 TOP1。

36 个数据集可以分为两类：20 个 in-distribution 数据集用于训练，16 个 OOD 数据集用于评估。

MMEB 子数据集构成：

![image](https://github.com/user-attachments/assets/56f0e167-cc2c-44cb-9445-7bd24f14761e)

各数据集统计如下：

![image](https://github.com/user-attachments/assets/09af704c-f73d-4058-8626-32e04a55684e)

评估时，embedding 模型分别对 query 和 target candidates 提取 embedding，直接计算 Precision@1。candidate 的数量直接影响检索成本，数量越大评估成本越高，数量越小 benchmark 越简单，容易过拟合。所以 MMEB 取了一个相对平衡的值 1000。

各个任务的输入形态

- Classification：query 为 instruction 和 image，以及可选的相关文本，target 为 class label，candidates 数量等于类别数量
- VQA：query 包括 instruction、image 以及一个 question，target 为 answer，candidates 包括一个 gt 和 999 个干扰选项
- Retrieval：query 和 target 可以是 text、images、instruction 的任意组合，candidates 包括一个 gt 和 999 个干扰选项
- Visual Grounding：query 包括 instruction 和 image，candidates 包括一个 gt 和 999 个干扰选项

> 对于 query 输入，文本部分包括 instruction 和 text，两者可能比较容易迷惑，举个例子说明：
>
> 对于一个检索任务，Query Text 为：Find me an everyday image that matches the given caption. Man riding a motor bike on a dirt road on the countryside. 其中 Find me an everyday image that matches the given caption. 为 instruction，而 Man riding a motor bike on a dirt road on the countryside. 为 text。
>
> 相当于 instrcution 表明了任务，text 是具体的 query 内容。

## VLM2VEC

VLM2VEC 基于对比学习框架，可以将任意的 VLM 模型转换为 embedding 模型。

![image](https://github.com/user-attachments/assets/e209cb8d-4833-471f-9b7f-43a596b28466)

假设，一个相关的 query-target 对表示为 $$(q, t^+)$$，两者均可以是单张图、一条文本或者图-文组合。在 query 上加入 instruction 可以构成模型的输入：

$q_{inst} = [IMAGE_TOKEN] Instruct: \{task\_definition\} \n\  Query: \{q\}$

其中，task_definition 是一个占位符，用来填写用一句话描述的 embedding 任务。

给定一个 VLM 模型，将 query 和 target 送入模型并分别得到 embedding：$$(h_{q_{inst}}, h_{t^+})$$。我们取最后一层的最后一个 token 的特征作为最终 embedding。

损失函数为标准的 InfoNCE，每个 batch 内的其它样本作为负样本。

通过 GradCache 增大 batch size

因为难负样本的获取通常比较困难，所以增大 batch size 变得非常重要。增大 batch size 的一个瓶颈是 GPU 显存，本文采用了 GradCache，一种梯度缓存技术，将对比损失和编码器之间的反向传播解耦，沿 batch 维度删除编码器反向传播数据依赖关系。

## 实验

VLM2VEC 采用了 Phi-3.5-V 和 LLaVA-1.6 作为 backbone，做了 full model fine-tuning 和 LoRA。图像分辨率有 1344x1344和336x336两种设置。对于 20 个训练数据集，对于样本量超过 50K 的数据集，随机挑选出 50K 作为训练数据，总共有 662K 数据。使用 GradCache 后，对于全参数训练来说，batch-size 由 4 可以增大至 1024。

**结果对比**

![image](https://github.com/user-attachments/assets/91d8425d-2279-4cde-9898-4db852e4a9dc)

- Fine-tuning 之后，在 IND 上的结果才 67.5，是不是有点低？
- LoRA 的效果比全量FT更好，不太理解

**跨任务的泛化性**

![image](https://github.com/user-attachments/assets/304ea49b-e301-4438-8dd3-5edd43104096)

- 整体上，VLM2VEC 训练检索任务，在其他任务上的泛化能力更强一些

**Instruction 的影响**

![image](https://github.com/user-attachments/assets/400de692-fae4-4e1b-a761-cbfe00e5fc75)

- 对于 CLIP 来说，加入 instruction 效果变差
- 对于 VLM2VEC，加入 instruction 明显变好
