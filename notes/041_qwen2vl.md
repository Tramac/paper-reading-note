# Qwen2-VL

[paper](https://arxiv.org/pdf/2409.12191) | [code](https://github.com/QwenLM/Qwen2-VL) | [blog](https://qwenlm.github.io/zh/blog/qwen2-vl/)

## TL;DR

- 动态分辨率处理：引入naive dynamic resolution技术，能够灵活处理不同分辨率的输入
- 多模态位置编码：创新性提出多模态旋转位置编码（M-RoPE），促进跨文本、图像和视频的位置信息的有效融合
- 图像和视频的统一理解框架：图像被处理为两个相同帧，保持与视频处理的一致性，使用3D tubes替代2D patches处理方式
- 详细的对数据格式，基础设施的描述，很赞

## Abstract

Qwen2-VL 引入了动态分辨率机制，使模型能够动态地将不同分辨率的图像处理成不同数量的 visual tokens。使模型生成更高效和准确的视觉表示，模型同时还集成了 M-RoPE，促进跨文本、图像和视频的位置信息的有效融合。Qwen2-VL 采用统一的范式来处理图像和视频，增强了模型的视觉感知能力。为了探索 LVLMs 的潜力，Qwen2-VL 研究了 LVLMs 的 scaling laws。通过对模型参数量（2B、8B、 72B） 以及训练数据量的扩增，Qwen2-VL 系列实现了极具竞争力的性能。在各种多模态基准测试中取得了与 GPT-4o 和 Claude3.5-Sonnet 等领先模型相当的结果，优于其他通用模型。

## 1 Introduction

LVLMs 的一般范式是 visual encoder→cross-modal connector→LLM，基于 high-quality 数据集和 next-token prediction 的目标，完成模型的训练。很多方法都是在这个范式的基础之上做一些改进：更大的模型结构、更高的输入分辨率、MoE、模型集成、更复杂的 adaptor。

然而，当前的 LVLMs 通常局限于固定分辨率的输入。对于不同分辨率的图像，会先经过上采样或下采样至固定尺寸后再送给模型。虽然这种一刀切的策略能够处理不同分辨率的图像，但是限制了模型在不同尺度上捕获信息的能力，特别是会导致高分辨率图像中细节信息的丢失。

此外，大多数 LVLM 依赖于 frozen CLIP-style vision encoder，这种预训练模型的视觉表征能力通常比较有限，尤其是对复杂的推理任务或者需要理解图像中的复杂细节时。有些工作试图通过在 LVLM 训练过程中微调 vision encoder 来解决这些限制，为了进一步增强模型对不同分辨率的适应性，本文在 LVLM 训练过程中引入了动态分辨率训练。另外，还在 ViT 中使用 2D RoPE，使模型更好地捕捉不同空间尺度的信息。

对于视频 video，本质是一些图像帧序列，许多方法将其视为一个单独的模态。和 text 这种 1D 信息不同，使用 1D 位置编码对 video 这种具有 3D 空间特征的信息进行编码是不充分的，为了弥补这一差距，本文提出 Multimodal Rotary Position Embedding (M-RoPE)，使用单独的组件分别表示时序和空间信息，这使得模型能够自然地理解动态内容，例如视频或流数据，提高其理解和与世界交互的能力。

Qwen2-VL的关键进展包括：

- **读懂不同分辨率和不同长宽比的图片**：Qwen2-VL 在 MathVista、DocVQA、RealWorldQA、MTVQA 等视觉理解基准测试中取得了全球领先的表现。
- **理解20分钟以上的长视频**：Qwen2-VL 可理解长视频，并将其用于基于视频的问答、对话和内容创作等应用中。
- **能够操作手机和机器人的视觉智能体**：借助复杂推理和决策的能力，Qwen2-VL 可集成到手机、机器人等设备，根据视觉环境和文字指令进行自动操作。
- **多语言支持**：为了服务全球用户，除英语和中文外，Qwen2-VL 现在还支持理解图像中的多语言文本，包括大多数欧洲语言、日语、韩语、阿拉伯语、越南语等。

## 2 Approach

Qwen2-VL 系列由 3 个大小的模型组成，分别是 Qwen2-VL-2B、Qwen2-VL-7B 和 Qwen2-VL72B。

### 2.1 Model Architecture

<center>
    <img src="https://github.com/user-attachments/assets/fff6ab8c-fb5f-4186-b500-e98e7f13506c">
</center>

Qwen2-VL 保留了 Qwen-VL 的框架。为了适应于各种规模的 adaptor，Qwen2-VL 实现了一个参数量约为 675M 的 ViT，可以同时处理图像和视频输入。为了进一步增强模型在视频中有效感知和理解视觉信息的能力，Qwen2-VL 引入了几个关键升级：

- **Naive Dynamic Resolution**

  Qwen2-VL 现在可以处理任何分辨率的图像，将它们动态转换为可变数量的 visual tokens。为了支持此功能，我们将 ViT 中原始的绝对位置编码替换为 2D-RoPE，用于捕捉图像的二维位置信息。在推理阶段，不同分辨率的图像被打包成单个序列，为了减少每个图像的 visual tokens 数量，在ViT之后用了一个简单的 MLP 层，将相邻的 2 × 2 tokens 压缩为单个 token。使用特殊 token $<|vision_start|>$ 和 $<|vision_end|>$ 表示 visual tokens 的开始和结束。因此，分辨率为224 × 224的图像，使用 patch_size=14的 ViT 编码，在进入LLM之前将被压缩为 66 个 tokens。

  > 224/14=16，16/2=8（MLP压缩），8x8+2=66

- **Multimodal Rotary Position Embedding (M-RoPE)**

  1D-RoPE 只能用于编码 1D 位置信息，M-RoPE 有效的对多模态输入的位置信息进行了建模。通过对原始 RoPE 解构为三个组件来实现的：(temporal, height, width)。对于 text 输入，这些组件使用相同的位置 ID，使得 M-RoPE 在功能上等同于 1D-RoPE。对于图像，每个 visual token 的时间ID保持不变，而不同的 id 根据 token 在图像中的位置分配给 height 和 weight。

- **Unified Image and Video Understanding**

  Qwen2-VL 采用混合训练方案，结合图像和视频数据，使其同时具备图像理解和视频理解能力。

  > image understanding 和 video comprehension，单纯的记录一下 understanding 和 comprehension。。。

  为了尽可能完整地保留视频信息，我们以每秒两帧（fps=2）的速度对每个视频进行采样。然后使用 depth=2 的 3D 卷积处理视频输入，允许模型处理 3D tubes 而不是 2D patches，从而使其能够在不增加序列长度的情况下处理更多的视频帧。为了平衡对长视频处理的计算需求，我们动态调整每个视频帧的分辨率，将每个视频的总 tokens 数限制为 16384。这种训练方法在模型长视频理解能力和训练效率的能力之间取得了平衡。

2.2 Training

和 Qwen-VL 一样，还是采用三阶段的训练。

第一阶段使用 image-text pairs 仅训练 vision encoder 和 adaptor（paper 中并没有提这部分，但应该是训了的）；第二阶段训练整个模型；第三阶段在指令数据集上训练 adaptor+LLM。

Qwen2-VL 的预训练数据包括 image-text pairs、OCR、交错的 image-text 文章、VQA、视频对话、image knowledge 等类型。数据源主要来自 web pages、开源数据集、合成数据，**数据截止日期为2023年6月**（现在都这么来评估数据量了:)）。

在第一阶段 pre-training，Qwen2-VL 训练了大概 600B tokens，vision encoder 使用的是 DFN 中的 ViT，将其中的位置编码替换为了RoPE-2D，LLM 采用的 Qwen2。该阶段主要通过OCR和图像分类任务学习图像-文本关系、图像中的文本内容识别。这种基础训练有助于模型稳健地理解视觉-文本相关性以及与 LLM 空间对齐。

> 现在大模型的训练数据量大多用覆盖的 token 数衡量了，因为数据太多了，基本都是一个 epoch 或更少。

第二阶段的 pre-training，使用了另外 800B tokens，引入了更多的混合图像-文本数据，促进对视觉和文本信息之间相互作用的更细致的理解。同时还使用了纯文本数据，保证语言模型能力不退化。

在整个 pre-training 阶段，Qwen2-VL 总共处理了 1.4 trillion tokens，不仅包含 text tokens，还包含 image tokens。然而，在训练过程中，我们只为 text tokens 提供监督（因为回答只有 text）。

在第三阶段指令微调时，使用 ChatML 的格式构造指令遵循数据，该阶段的数据集不仅包含纯文本对话数据，还包括多模态对话数据，多模态数据部分包括 image QA、文档解析、多图比较、视频理解、视频对话、agent 交互等类型数据。

2.2.1 Data Format

与 Qwen-VL 一致，Qwen2-VL 仍然使用 special tokens 来区分视觉和文本输入。视觉 token 的起始和结束用 `<|vision_start|>` 和 `<|vision_end|>` 表示。

**Dialogue Data**

对于指令微调数据集中的对话数据格式，使用了 ChatML 格式，每个交互语句都标记了两个 special tokens（`<|im_start|>`  和 `<|im_end|>`）来表明对话终止（即：我说完了，该你说了:)）。下面示例蓝色标记的部分表示会监督的部分：

<center>
    <img src="https://github.com/user-attachments/assets/d86fd959-c407-4eb7-a1c3-220f1b4bd40d">
</center>

**Visual Grounding**

为了赋予模型 visual grounding 能力，bounding box 坐标归一化至 [0, 1000)，用 `(X_topleft， Y_topleft), (X_bottomright, Y_bottomright)` 四元组表示。然后使用 special tokens `<|box_start|>` 和 `<|box_end|>` 表示坐标内容的开始和结束。为了准确地将边界框与其文本描述对应起来，还引入了 `<|object_ref_start|>` 和 `<|object_ref_end|>` 来表示框内容描述的起始和结束，示例如下：

<center>
    <img src="https://github.com/user-attachments/assets/3f404dd9-9b63-4466-afb4-129a75d8fb8c">
</center>

**Visual Agent**

为了可以使 Qwen2-VL 作为通用 VL-Agent，我们将 UI 操作、机器人控制、游戏和导航等各种代理任务视为顺序决策问题，使 Qwen2-VL 通过多步动作执行来完成任务。对于每个任务，我们首先定义一组可执行的动作和 keywords pattern (图中下划线部分) 用于函数调用，然后 Qwen2-VL 分析 observations、执行推理和规划、执行所选 action 并与环境交互以获得新的 observations（听起来像是强化学习）。这个循环迭代地重复，直到任务成功完成。数据格式如下：

<center>
    <img src="https://github.com/user-attachments/assets/b48afb83-4164-43bb-a77f-ca8a707b1631">
</center>

2.3 Multimodal Model Infrastructure

Qwen2-VL 的训练是基于阿里云的 PAI-Lingjun Intelligent Computing Service 平台完成的。

> scalable computing（可扩展计算）、auto resuming（自动恢复）、straggler detection（掉队检测），可见稳定性对于大模型训练的重要程度。

**Storage 存储**

数据存储采用阿里云的 ultra-speed CPFS（Cloud Parallel File Storage）。文本数据和视觉数据的存储是解耦的，文本数据只存储在 CPFS 上，并使用 mmap 提高访问效率；视觉数据使用阿里云的 OSS（Object Storage Service）进行存储。在训练期间，我们通过 OS S的 python-client 访问视觉数据，并调整并发性和重试参数，以避免达到 QPS 限制。我们还发现**视频数据解码是主要的瓶颈**，尤其是对于长视频，经过一些开源技术和阿里内部工具的试用，最终采用了 **caching decoding technique**。

**Parallelism 并行化**

Qwen2-VL 使用了 3D parallelism，结合了 data parallelism (DP)，tensor parallelism (TP) 和 pipeline parallelism (PP)。另外还利用了 deepspeed 的 zero-1 redundancy optimizer 以节省内存。Sequence parallelism (SP) 用来挑选 checkpoint 也可以节省内存。

在启用 TP 训练时，我们总是将 vision encoder 和 LLM 分片在一起，而不是将所有的 vision encoder 合并在一起，因为它的参数相对较少。同时我们还发现由于卷积算子的不确定性行为，TP 训练会导致不同的模型 shared-weights，为了解决这个问题，通过离线减少共享权重，从而避免额外的 all-reduce 通信步骤，这种方法对性能的影响很小。

对于 Qwen2-VL 72B 使用了 1F1B PP 技术，我们将 vision encoder、adaptor 和 LLM 的几个 decoder layers 合成一个 stage，剩余的 decoder layer 均匀切分。

**Software**

PyTorch 2.1.2 + CUDA 11.8、flash-attention、LayerNorm、RMSNorm、Adam。

> 这一章节展示了很多数据细节和训练细节，有很多不是太懂，但能感受到 Qwen 团队很想把所有踩到的坑都告诉你，很赞的开源精神。
>
> 感觉现在大模型的壁垒很大程度上就是数据+训练技巧+计算资源。

3 Experiments

3.1 Compare to SOTAs

<center>
    <img src="https://github.com/user-attachments/assets/a88840c5-45b1-458b-b8d1-46ed7d123f33">
</center>

其它的 agent 能力等就不在这里展示了，详细的结果直接看论文吧。
