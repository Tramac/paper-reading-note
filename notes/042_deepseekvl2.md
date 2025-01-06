# DeepSeek-VL2

[paper](https://arxiv.org/pdf/2412.10302) | [code](https://github.com/deepseek-ai/DeepSeek-VL2)

## TL;DR

- MoE Vision-Language Models
- 支持动态分辨率

## Abstract

DeepSeek-VL2 是一个 MoE 模型，相比于 DeepSeek-VL 主要有两个改进：1）视觉部分采用了一种 dynamic tiling 视觉编码策略，可以处理不同长宽比的高分辨率图像；2）语言部分更新至 DeepSeekMoE 模型，采用了 multi-head latent attention 机制，将 kv cache 压缩至 latent 空间，以实现高效的推理和高吞吐量。

基于改进的 vision-language 数据集训练，DeepSeek-VL2 在 VQA、OCR、document/table/chart 理解、visual grounding 等多种任务上表现出了很强的能力。

DeepSeek-VL2 系列共有 DeepSeek-VL2-Tiny、DeepSeek-VL2-Small 和 DeepSeek-VL2 三种变体，参数量分别为 1B、2.8B 和 4.5B。

## 1 Introduction

DeepSeek-VL2 利用了 Mixture-of-Experts（MoE）架构，相比于 DeepSeek-VL 主要有以下三个方面的改进：

1. 支持动态大小、高分辨率图像输入，增强了视觉理解能力；
2. 更强的 LLM，显著提高了训练和推理效率；
3. 改进 vision-language 数据构造 pipeline，不仅提高了整体性能，同时也支持视觉基础能力，如 visual grounding；

## 2 Model Architecture

<center>
    <img src="https://github.com/user-attachments/assets/35df17d8-d679-45d8-bf24-5b6ac0513772">
</center>

DeepSeek-VL2 包括三个核心模块：1）vision encoder；2）vision-language adaptor；3）MoE-based LLM。

**Dynamic Tiling Strategy**

DeepSeek-VL 中使用的是一个 hybrid vision encoder，分别用 SigLIP 和 SAM-B 处理 384×384 和 1024×1024 分辨率的图像，虽然这种方式适用于大多数任务，但是将图像分辨率限制在了 1024×1024 以内，对于具有更高分辨率或极端长宽比的图像不太友好。

受 InternVL 2 和 LLaVA-NeXT 的启发，DeepSeek-VL2 采用了一个 dynamic tiling strategy，将高分辨图像切分为块。这种方式可以实现仅用一个 SigLIP-SO400M-384 处理各种不同长宽比、分辨率的图像。原始 SigLIP 主要针对 384×384 分辨率的图像，为了使用不同长宽比，DeepSeek-VL2 设置了一组候选分辨率 $𝐶_R = {(𝑚 · 384, 𝑛 ·384) | 𝑚 ∈ N, 𝑛 ∈ N, 1 ≤ 𝑚, 𝑛, 𝑚𝑛 ≤ 9}$，其中 𝑚 : 𝑛 就是长宽比（aspect ratio）。

给定一张尺寸为 (𝐻, 𝑊) 的图，首先依据候选长宽比 𝐶𝑅 计算需要 padding 的区域，padding 之后将其 resize 到最合适的尺寸 (𝑚𝑖 · 384, 𝑛𝑖 · 384)。然后将图像切分为 𝑚𝑖 × 𝑛𝑖 个尺寸为 384×384 的块，然后再加一个全局小图（global thumbnail view）。最终，SigLIP 需要处理共 (1 + 𝑚𝑖 × 𝑛𝑖) 个图像块，每个图像块生成 27 × 27 = 729 个 维度为 1152 的 visual token embedding。为了提高计算效率和控制 context length（visual token 数量），在处理多个(> 2)图像时禁用 dynamic tiling strategy（？）。

**Vision-Language Adaptor**

<center>
    <img src="https://github.com/user-attachments/assets/f2f8748e-96f8-4e33-b910-6c7004ae3af0">
</center>

vision encoder 之后，先使用一个 2×2 pixel shuffle 操作将每个图像块的 visual tokens 数从 27×27 压缩至 14×14=196 个。

> 2×2 pixel shuffle

然后，使用三个 special tokens 分别标记不同类型的 visual tokens：

- 对于 global thumbnail tile 的 14×14 个 token，在每一行的末尾添加一个 `<tile_newline>` token，最终有 14×15=210 个 tokens；
- 对于 𝑚𝑖 × 𝑛𝑖 个 local tile 的 token，首先将它们组成形状为 (𝑚𝑖 · 14, 𝑛𝑖 · 14) 的 2D grid，然后在最后一列的末尾添加一个 `<tile_newline>` token，表示每个 local tile 每一行的结束；
- 此外，在 global thumbnail tile 和 local tiles 之间添加一个 `<view_separator>` 来区分两者；

最终，全部的 visual sequence 包括 210 + 1 + 𝑚𝑖 · 14 × (𝑛𝑖 · 14 + 1) 个 visual tokens。随后使用两层 MLP 投影到语言模型的特征空间中。

**DeepSeekMoE LLM**

LLM 使用的是 DeepSeekMoE，该模型使用了 Multi-head Latent Attention（MLA）机制，通过将 KV Cache 压缩至 latent 空间提高推理效率和吞吐率。同时，该模型还结合了 MoE 架构，允许通过稀疏计算进行有效的推理。训练 MoE 期间，为每个 expert 引入了一个 **global bias**，以改善 expert 之间的负载（cost-effectively）平衡。D eepSeek-VL2 最终有三个尺寸：1.0B、2.8B 和 4.5B。完整的结构参数如下：

<center>
    <img src="https://github.com/user-attachments/assets/8c9e9a5b-6973-4e61-8615-2fdf366c0199">
</center>

> Tiny 中并没有使用 MLA，猜测 MLA 相比 MHA 会有性能损耗？所以能不用就不用了？

## 3 Data Construction

DeepSeek-VL2 的训练分为三个阶段：1）VL alignment；2）VL pre-training；3）SFT。每个阶段都有特定的数据集。

### 3.1 Vision-Language Alignment Data

alignment 阶段仅训练 MLP adaptor，对齐 vision encoder 和 LLM 特征空间。该阶段使用 ShareGPT4V 数据集，共约 1.2M 样本。

### 3.2 Vision-Language Pretraining Data

pre-training 数据的特点是量大、覆盖范围广、质量较差。

pre-training 数据使用了 vision-language 和 text-only 数据，保持 VL 能力和纯文本性能之间的平衡。在 DeepSeek-VL2 中 VL data 和 text-only data 比例为 7:3。其中 VL data 可以划分为以下几类：

- Interleaved image-text data：图文交互数据
- Image captioning data：caption 数据
- Optical character recognition data：OCR
- Visual question-answering (QA) data：VQA
- Visual grounding data：目标及其坐标信息用 <|ref|> 特殊 token 区分
  - Prompt: Locate <|ref|><query><|/ref|> in the given image.
  - Response: <|ref|><query><|/ref|><|det|>[[x1, y1, x2, y2],...]<|/det|>
- Grounded conversation data
  - Prompt: <|grounding|>Can you describe the content of the image?
  - Response: Two <|ref|>dogs<|/ref|><|det|>[[x1, y1, x2, y2],...]<|/det|> are running on the grass.

### 3.3 Supervised Fine-tuning Data

SFT 数据的特点是量少，但质量高。

SFT 数据包括开源数据集以及高质量的私有 QA pairs 数据。

- General visual question-answering

  虽然公开的 VQA 数据集的多样性好，但是通常有以下问题：1）response 简短；2）OCR 质量差；3）幻觉内容严重。为了解决这些问题，根据原始问题、图像、OCR 信息重新生成 response（用什么方法？）。

  在开发过程中，我们观察到 DeepSeek-VL2 的早期版本，尤其是 DeepSeek-VL2 Tiny 偶尔会在中文响应中混入英文单词，但在更大的模型中没有出现该现象，这可能是 pre-training 阶段中&英文数据配比不平衡以及模型能力限制导致的。为了解决小模型中的这个问题，我们使用了一个私有的 Chinese QA 数据集（多大规模？），缓解了语言混合问题。此外，还创建了一个额外的内部数据集来补充现实世界的视觉知识，包括动漫、网红事物、美食和艺术。

- OCR and document understanding

  主要是对开源 OCR 数据集清洗，删除低质部分；对于 document understanding，从内部数据中构造出一个子数据集，然后，生成特定于文档理解的多轮会话 QA 对。

- Table and chart understanding

  基于公开数据集，重新生成 response。

- Reasoning, logic, and mathematics

  对公开的 reasoning-focused 数据集添加更详细的推理过程，并且标准化相应格式。实验过程中发现，详细的 reponse 对 Tiny 这种规模的模型帮助不大，可能受限于模型能力。

- Textbook and academic questions

  主要来自教科书内容，该数据集主要强调跨多个学术学科的大学级内容。

- Web-to-code and plot-to-Python generation

  对开源数据集的 response 进行了质量优化，同时收集了一些 web code、python plot code。

- Visual grounding

  来源于一些公开数据集，为了提高模型能力，我们将数据中的 query 部分内容翻译成了中文，并创建了额外的负样本。我们还添加了 in-context 视觉基础数据（我理解就是多图数据），其中该任务涉及跨多个图像定位同一类别的对象。

- Grounded conversation

- Text-Only datasets

## 4 Training Methodology

### 4.1 Training Pipelines

DeepSeek-VL2 训练包括三个阶段：1）initial stage 使用 image-text pairs 数据训练 vision encoder + adaptor；2）pre-training 阶段训练整个模型；3）SFT 阶段同样训练整个模型。所有阶段的任务均是 **next token prediction**。

**Vision-Language Alignment**

基于 pre-trained DeepSeekMoE 语言模型，该阶段的主要目标就是对齐视觉空间和语言空间，使得 LLM 能够有效的处理视觉输入。和 LLaVA、DeepSeek-VL 不同的是，该阶段把 vision encoder 也放开训练了。

**Vision-Language Pre-training**

pre-training 阶段主要是增强模型的多模态理解能力，每个任务不那么强但都要会，同时保证原始语言能力不退化。该阶段训练所有参数，共训了 ~800B tokens。

**Supervised Fine-Tuning**

该阶段主要是增强模型的指令遵循能力和对话能力。同样的，也是调整所有参数。

### 4.2 Hyperparameters and Infrastructures

训练超参数如下：

<center>
    <img src="https://github.com/user-attachments/assets/1df19c20-74ca-4702-9843-45dab56d20cc">
</center>

训练框架还是幻方自家的 High-flyer，并行计算时一个主要挑战是 vision encoder 和 LLM 计算量的不平衡，跨 GPU load 时需要注意 vision encoder，避免 pipeline bubbles 以及 GPU 利用率低。为了解决这个问题，我们在pipeline 并行策略中实现了 vision encoder 的 fine-grained layer 划分；此外，我们在前向和后向过程中跨不同数据并行等级执行图像 tile 负载平衡，以减轻动态分辨率策略引起的图像块数量不平衡。

DeepSeek-VL2 使用 16/33/42 个节点的集群在 7/10/14 天完成，每个节点配备 8 个 NVIDIA A100 GPU。

## 5 Evaluation

对比了在 OCR-related benchmarks 和 VQA、math-related benchmarks 上的对比效果：

- OCR-related

  <center>
      <img src="https://github.com/user-attachments/assets/d1984113-ecf8-4137-b192-499413a436e2">
  </center>

- general QA 和 math-related

  <center>
      <img src="https://github.com/user-attachments/assets/45c11231-360c-479c-9cf0-78dcb8e09e7f">
  </center>

从结果来开，同期的工作水平相差不大，如 Qwen2-VL 和 InternVL2

另外，paper 还放了一些示例，这里不就贴了，直接看论文吧。
