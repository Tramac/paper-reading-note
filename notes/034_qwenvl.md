# Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

[paper](https://arxiv.org/pdf/2308.12966) | [code](https://github.com/QwenLM/Qwen-VL/tree/master)

## TL;DR

Qwen-VL: Qwen-VL 以 Qwen-7B 的预训练模型作为语言模型的初始化，并以 [Openclip ViT-bigG](https://github.com/mlfoundations/open_clip) 作为视觉编码器的初始化，中间加入单层随机初始化的 cross-attention，经过约1.5B的图文数据训练得到。最终图像输入分辨率为448。

Qwen-VL 给出了大量的数据清洗细节，这点非常赞，个人认为现阶段对数据有深入的认识和丰富的清洗经验非常重要，比调参更重要。（真的每当听到诸如 “你只是整了一些数据” 此类的话就非常反感：））

一些训练技巧：

- 大量的数据清洗方法
- 为了避免图像细节丢失，使用更高分辨率输入，为了缓解引入的计算量，使用 window attention
- 训练时始终使用一些纯文本数据，以维持原有的 LLM 能力

## Abstract

本文介绍了 Qwen-VL 系列工作，一组 vision-language 模型。以Qwen-LM 作为基座，通过以下设计赋予其视觉能力：1）visual receptor；2）input-output interface；3）3-stage training pipeline；4）multilingual multimodal cleaned corpus。除了传统的图像描述（image description）和问答（qution-answerning）能力，Qwen-VL 还通过对齐 image-caption-box 扩展出了 grounding 和 text-reading 能力。在众多的多模态模型对比中，Qwen-VL 系列取得了 SOTA 的效果。

## 1 Introduction

当前的开源 VLM 训练和优化还是不足的，和那些闭源模型比还有一定差距。

由于现实世界的视觉场景非常复杂，细粒度（**fine-grained**）的视觉理解对 VLM 非常重要，但在这个方向上的尝试还不够，大多数开源 VLM 仍然以粗粒度（coarse-grained）方法感知图像，缺乏细粒度感知的能力，例如 object grounding 或 text reading。

Qwen-VL 通过引入一个 **visual receptor** 使得 LLM 基座模型具备了视觉能力，visual receptor 由一个 language-aligned visual encoder 和一个 position-aware adapter 构成。

总结来说，Qwen-VL 系列模型具备以下特征：

- Leading performace：在相同尺寸的模型对比中，在多个 benchmark 上都取得了 SOTA 结果；
- Multi-lingual：支持多语种；
- Multi-image：允许任意的 image-text 数据交错输入；
- Fine-grained visual understanding：得益于更高分辨率的图像输入以及细粒度语料的使用，Qwen-VL 表现出极具竞争力的细粒度视觉理解能力，与现有的 VLM 相比，Qwen-VL 具有更好的 grounding、text-reading、text-oriented question answering 和 fine-grained dialog 能力。

## 2 Method

### 2.1 Model Architecture

![](https://github.com/user-attachments/assets/d2805997-8e11-462f-9c6c-af58f56713a1)

**Large Language Model**：Qwen-VL 采用 Qwen-7B 作为 language decoder。

**Visual Encoder**：visual encoder 采用的是 CLIP ViT-bigG。

**Position-aware Vision-Language Adapter**：

为了缓解长图像特征序列带来的效率问题，Qwen-VL 引入了一个 vision-language adapter 来压缩图像特征，其包括一个随机初始化的单层 cross-attention 模块。该模块使用一组可训练向量（Embeddings）作为 query vectors，由 visual encoder 得到的 image features 作为 key，以此将视觉特征序列压缩为 256 的固定长度。

（这里的 256 指的是 token embedding 维度还是 token 数量？）

此外，考虑到位置信息对细粒度图像理解的重要性，将 2D absolute positional encodings 合并到 cross-attention 的 query-key 中（具体如何实现？），以减轻压缩过程中位置细节的潜在损失。最终将长度为 256 的压缩图像特征序列输入到大型语言模型中。

### 2.2 Inputs and Outputs

**Image Input**

图像经过 visual encoder 和 adapter 处理之后得到固定长度的图像特征序列，为了区分图像特征输入和文本特征输入，使用两个 special tokens 进行区分：`<img> `和 `</img>`，标记图像特征的开始和结束。

**Bounding Box Input and Output**

为了增强模型的细粒度视觉理解和 grouding 能力，Qwen-VL 的训练涉及 region 描述、问题和 detections。与其它的多模态模型不同，Qwen-VL 的输入包括 bounding box 信息及其对应区域的描述。对于 box，首先用归一化对坐标进行了处理（处理至范围[0, 1000），不太理解？），并且转换为指定的字符串格式："(Xtoplef t, Ytoplef t), (Xbottomright, Ybottomright)"。坐标信息与其它文本信息一致，tokenizer 无需额外的 positional vocabulary。为了区分上面的 detection string 和常规的指令 text string，使用 `<box>` 和 `</box>` 两个 special tokens 进行区分。此外，为了将 region 描述信息与其对应的 box 对应起来，使用 `<ref>` 和 `</ref>` 两个 special tokens 对框内的描述内容进行标记。

（这里应该给出一条训练数据的示例）

## 3 Training

![](/home/zhaoxiangming/.config/Typora/typora-user-images/image-20241219100216907.png)

Qwen-VL 的训练 pipeline 包括三个阶段：前两个阶段为 pre-training，最后一个阶段为 instruction fine-tune。

### 3.1 Pre-training

第一个阶段的 pre-training，主要使用 large-scale，weakly labeled，web-crawled 的 image-text pairs 进行训练，数据包含了几个开源数据集（LAION-en、LAION-zh、LAION-COCO、DataComp、Coyo）以及一些内部数据，并对数据进行了清理，由原始的 5 billion 清理至 1.4 billion，其中 77.3% 为英文语料，22.7% 为中文语料，具体分布如下：

![](https://github.com/user-attachments/assets/14136fb9-ebf2-4d40-bb41-f530a9796a35)



在该阶段训练时，将 LLM 进行 freeze，仅优化 vision encoder 和 VL adapter 部分，图像输入分辨率为 224 x 224。训练目标为 minimize the cross-entropy of the text tokens。

### 3.2 Multi-task Pre-training

该阶段，Qwen-VL 使用了 high-quality、fine-gained 的数据进行训练，图像输入分辨率更高，且 image-text 交替输入（第一阶段应该只有单轮输入）。各任务所使用的数据集和样本量：

![](https://github.com/user-attachments/assets/0729b30a-e763-432a-a430-29630a2c9a89)

训练数据格式：

![](https://github.com/user-attachments/assets/5786b267-6441-43fa-bf58-c93212a0789c)

对于 text generation 任务（纯文本），使用了内部数据来维持 Qwen-VL 的 LLM 能力（应该是怕模型坍塌，使原本的语言能力丢失）。

该阶段，图像输入分辨率提高到 448 x 448，以此减少图像信息的丢失。

需要说明的是，使用高分辨率的 vision transformer 会带来计算量的增加，一种解法是在 vision transformer 中使用 **window attention**，比如每 4 个 layer 中，其中 3 个使用窗口大小为 224x224 的 window attention，1个使用原始的 attention。

训练时打开了 LLM，整个 Qwen-VL 均参与训练，训练目标和第一阶段一致，即 predict next token。

### 3.3 Supervised Fine-tuning

该阶段主要是在指令数据上进行微调以提高对话能力，得到的模型为 Qwen-VL-Chat。数据主要来自 caption data 和由 LLM self-instruction 生成的对话数据，通常只解决单图像对话和推理，仅限于图像内容理解。另外通过手动注释、模型生成和策略连接构建了一组额外的对话数据，以将引入 localization 和多图理解能力。此外，在训练期间混合多模态和纯文本对话数据，以确保模型在对话能力方面的泛化性。

最终使用的指令数据量为 350K，该阶段训练时 freeze 住 vision encoder，仅优化 LLM 和 VL adapter 部分。

训练数据格式：

![](https://github.com/user-attachments/assets/1f0a19c0-cfd5-442d-afbb-cf5fbe74779b)

训练数据示例如下：

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是Qwen-VL,一个支持视觉输入的大模型。"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n图中的狗是什么品种？"
      },
      {
        "from": "assistant",
        "value": "图中是一只拉布拉多犬。"
      },
      {
        "from": "user",
        "value": "框出图中的格子衬衫"
      },
      {
        "from": "assistant",
        "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
      }
    ]
  },
  { 
    "id": "identity_2",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>assets/mm_tutorial/Chongqing.jpeg</img>\nPicture 2: <img>assets/mm_tutorial/Beijing.jpeg</img>\n图中都是哪"
      },
      {
        "from": "assistant",
        "value": "第一张图片是重庆的城市天际线，第二张图片是北京的天际线。"
      }
    ]
  }
]
```

## 4 Evaluation

Qwen-VL 评估了 Image Caption、VQA、Text-oriented VQA、Refer Expression Comprehension、Few-shot Learning、Instruction Following 等，具体结果直接看 paper，不在这里贴了。

各阶段训练 setting

![](https://github.com/user-attachments/assets/eccdfd2d-92d2-4ed3-9464-b9289b6e05b6)

## Bonus

### 数据清洗

- Image-text pairs

  对于爬虫数据以及开源数据集（LAION-en、LAION-zh、LAION-COCO、DataComp、Coyo）处理如下：

  1. Removing pairs with too large aspect ratio of the image 删除长宽比过大的图像
  2. Removing pairs with too small image 删除太小的图
  3. Removing pairs with a harsh CLIP score (dataset-specific) 删除 CLIP 相关性得分较低的数据(特定数据集)
  4. Removing pairs with text containing non-English or non-Chinese characters 删除文本包含非英文和非中文的数据
  5. Removing pairs with text containing emoji characters 删除文本中包含表情符号的数据
  6. Removing pairs with text length too short or too long 删除文本太长或太短的数据
  7. Cleaning the text's HTML-tagged part 清理掉文本中 HTML 的内容
  8. Cleaning the text with certain unregular patterns 清理掉文本中某些非常规的 pattern

  对于学术 caption 数据集，删除文本中包含特殊 tag 的数据。

- Multi-task 数据

  - VQA

    对于 VQAv2 数据集，根据最大置信度选择 answer 作为标注，对于 VQA 数据集不做任何操作

  - Grounding

    对于 GRIT 数据集，发现 caption 经常包含 recursive grounding box，使用贪心算法来清理 caption，以确保每个图像都包含大多数没有 recursive box。对于其他 grounding 数据集，只需将名词/短语与各自的 box 坐标连接起来。

  - OCR

    - 我们使用 Synthdog 方法合成 OCR 数据集。具体的，使用 COCO 数据集中的图作为背景，然后选择 41 种英文字体和 11 种中文字体生成文本，其它的就使用 Synthdog 的默认参数。由于文本是生成的，所以我们知道它的坐标位置，可以将坐标直接作为 label。
    - 对于 PDF 数据，使用 PyMuPDF 按照以下步骤得到标注结果：
      1. Extracting all texts and their bounding boxes for each page.
      2. Rendering each page and save them as an image file.
      3. Removing too small image.
      4. Removing images with too many or too few characters.
      5. Removing images containing Unicode characters in the "Latin Extended-A" and "Latin Extended-B" blocks.
      6. Removing images containing Unicode characters in the "Private Use Area (PUA)" block.

    - 对于 HTML 网页数据，使用 Puppeteer 按照以下步骤处理：
      1. Extracting all texts for each webpage. 
      2. Rendering each page and save them as an image file.
      3. Removing too small image.
      4. Removing images with too many or too few characters.
      5. Removing images containing Unicode characters in the "Private Use Area (PUA)" block.
