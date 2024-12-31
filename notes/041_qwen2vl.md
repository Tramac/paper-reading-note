Qwen2-VL

[paper](https://arxiv.org/pdf/2409.12191) | [code](https://github.com/QwenLM/Qwen2-VL) | [blog](https://qwenlm.github.io/zh/blog/qwen2-vl/)

TL;DR

- 详细的对数据格式，基础设施的描述，很赞



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
