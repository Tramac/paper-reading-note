# MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval

[paper](https://arxiv.org/pdf/2412.14475) | [github](https://github.com/VectorSpaceLab/MegaPairs)

## Abstract

- 多模态检索效果不佳是因为**缺乏训练数据**
- 提出一种数据合成方法--**MegaPairs**，基于开源的图像数据集，利用 VLM 和 LLM 生成大量训练数据
- 使用 MegaPairs 生成的高质量数据集，效果优于 70 倍大小的已有数据
- 训练数据规模：50W、2600W

该方法的局限性：

- 数据合成 pipeline 未开源
- finetune 代码未开源
- 当前版本不支持中文：文本侧基座无论是 CLIP-based 和 MLLM-based 都是仅英文模型

## 1 Introduction

CLIP、ALIGN、SigLIP 们的局限性：

- 做不了组合图像检索、多模态文档检索等

MagicLens 等数据合成方法的局限性：

- 生成数据的可扩展性、质量、多样性、可用性较差
  - 可扩展性：网上的数据只有一小部分网页包含多个图像
  - 质量：多图时要么不相关，要么内容重复
  - 多样性：相关图像内容单调，多样性差
  - 可用性：生成的数据集不开源

## 2 方法

### 2.1 MegaPairs 数据构造

每个 data instance 是一个三元组：$$(\mathcal{I_q}, \mathcal{I_t}, \mathcal{T_{q\to t}})$$，其中$$\mathcal{I_q}$$为 query image，$$\mathcal{I_t}$$ 为 target image，$$ \mathcal{T_{q\to t}}$$ 描述了从 query 到 target 的过渡关系。

构造以上的三元组主要有两个挑战：

1. 保证图像对的相关性和多样性
2. 精确地 instruction 标注

#### 2.1.1 相关图像 pairs 挖掘

![image](https://github.com/user-attachments/assets/77d7cf18-e429-45b0-9667-8ff38b130e35)

如上图（a）中的 pipeline，首先有一个大规模的图像数据集（DataComp-1B，10亿），对于每一个 query 图像 $$(\mathcal{I_q}, \mathcal{C_q})$$，其中 $$\mathcal{C_q}$$ 是图像 $$\mathcal{I_q}$$ 所对应的 caption，先利用多个相似性模型（EVA-CLIP image encoder）来搜索一组不同的相关 target 图像，即正样本：$$(\mathcal{I_{t_1}}, \mathcal{I_{t_2}},...,\mathcal{I_{t_n}})$$。

相似性模型：

- EVA-CLIP image encoder 用于 visual-semantic 相关资源的挖掘
- DINOv2 用于visual-pattern 相关资源的挖掘
- EVA-CLIP text encoder 用于 caption 相关资源的挖掘

我们选择相似得分在(0.8, 0.96) 之间的样本作为 target 图像，<0.8 认为是不相关的样本，去除 >0.96 的样本是为了消除重复样本。

**难负样本**对于检索模型很重要。在有了上面的 query 和 targets 后，对于每一个 pair $$(\mathcal{I_q}, \mathcal{I_{t_i}})$$，MegaPairs 将 targets 中的 $${\{\mathcal{I_{t_j}} | t_j \ne  t_i}\}$$ 作为难负样本。

> 这里没太懂，理论上 query 和 targets 中的图都是互为正样本的，但为什么还能作为难负样本？

#### 2.1.2 开放式 instruction 生成

如上图（b）中的 pipeline，对于每一个图像 pair $$(\mathcal{I_q}, \mathcal{I_{t_i}})$$，先使用 MLLM（InternVL2-26B）生成两张图的共同概念和差异，然后使用 LLM（LLaMA3-8B）生成 instruction，使用 LLM 有效的增加了 instruction 的多样性。

最终，可以构建出三元组数据：$$(\mathcal{I_q},  \mathcal{T_{q\to {t_i}}}, \mathcal{I_{t_i}})$$，其中 $$(\mathcal{I_q},  \mathcal{T_{q\to {t_i}}})$$ 可以用来检索 $$\mathcal{I_{t_i}}$$。

### 2.2 MMRet Model

#### 2.2.1 CLIP-based MMRet

CLIP 本身是一个对偶 encoder 结构，两个独立的 encoder 分别编码 image 和 text，得到各自的 embedding，为了得到 multimodel embedding，使用一个 **score-fusion strategy**，直接将 image embedding 和 text embedding 相加，得到最终特征。

> 既然是直接相加，为什么叫 score-fusion strategy？需要看一下 UniIR

#### 2.2.2 MLLM-based MMRet

MLLM-based 方法通常是先用一个 ViT 提取得到 image token embedding，再将 image token 与 text token 同时送入 LLM（LLaVA-1.6），两者的融合靠 LLM 完成。

另外，MMRet 额外使用一个 task-specific instructions 作为输入，用以提高泛化性，整体的 query 输入如下：

$$<instruct> \{task\_inst\} <query> \{q_t\} \ \{q_i\} \ [EOS]$$

> task-specific instruction 和 query text 的区别：

其中，$$[EOS]$$ token 的最后一层的 normalized 输出作为最终特征。

#### 2.2.3 Multimodel 对比学习

MegaPairs 使用对比学习将原始的 CLIP 和 MLLM 转换为 MMRet 模型，即采用标准的 InfoNCE loss 作为训练目标。

和单模态不同的是，MMRet 中的 query 和 candidate 可以是 image、text、image-text 的任意组合。

> 训练数据是固定的三元组，各个模态的随机组合在训练时是如何处理的？

## 3 实验

### 3.1 在图像检索任务上的 zero-shot 能力

CLIP-based MMRet 的 batch size=2048，每个query 对应1个正样本对应4个难负样本，图像大小 resize 至 224x224。MLLM-based MMRet 的 batch-size=144，每个 query 对应1个正样本和3个难负样本，图像大小 resize 至 512x512，且使用 LoRA 训练。

![image](https://github.com/user-attachments/assets/c201dcf0-3004-4621-bff5-2df960b9533a)

- MMRet-MLLM 模型在四个 benchmark 中的三个中达到了 SOTA
- MMRet 在所有模型尺度上都表现出卓越的性能
- MMRet-Base 模型超越了更大的模型，证明了 MegaPairs 数据集的有效性

> 但是 R@x 的指标都很低啊

### 3.2 在 MMEB 上的多模态任务的表现

#### 3.2.1 Zero-shot 能力

MMRet-MLLM 在 MegaPairs 上训，在 MMEB 上测，结果如下：

![image](https://github.com/user-attachments/assets/1efe957c-ad76-4cda-bb9a-ff2fac8550bf)

#### 3.2.2 Supervised Fine-tuning 表现

MMRet-MLLM 先在 MegaPairs 上训，再在 MMEB 上微调一个 epoch，结果如下：

![image](https://github.com/user-attachments/assets/45eadc21-627d-4a62-8112-9192f908af2a)

> 这里没有说对比方法有没有在 MegaPairs 上训练。

### 3.3 MegaPairs 数据集的详细研究

#### 3.3.1 数据的可扩展性和质量

![image](https://github.com/user-attachments/assets/6e3fdabb-1dac-4ddf-ac4a-e38012133356)

- 随着训练数据量增多，性能也逐步提升，说明 MegaPairs 数据集的可扩展性
- 上图中虚线是 MagicLens-B 模型在其 36.7M 数据规模训练得到的结果，而使用 0.5M MegaPairs 数据即可达到相同的结果，说明了 MegaPairs 数据的高质量

#### 3.3.2 难负样本的影响

在 MegaPairs 中，对于一个 query，使用相似性模型检索得到的召回集后，对于那些不是 target 的图像会被当做难负样本。

训练中使用这种难负样本对效果的提升：

![image](https://github.com/user-attachments/assets/eeea1d39-eb66-4b03-8f37-aa9773cab9aa)

> Qry：query image negative 指的应该是同一个 batch 内除了自身外其它的 query image 可以作为负样本。

#### 3.3.3 构造三元组数据时的搜索策略

![image](https://github.com/user-attachments/assets/1f19ed6c-200d-4195-8bde-d1a4c8710b7c)

- 从结果看，使用相似性模型挖掘数据时，三者同时使用挖掘出的数据质量最好



数据集示例：

![image](https://github.com/user-attachments/assets/3312ae3e-0499-4204-a383-0f7ab5635960)
