[paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) | [llama-recipes](https://github.com/facebookresearch/llama-recipes) | [llama](https://github.com/facebookresearch/llama) | [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

## 标题&作者

Llama 2: Open Foundation and Fine-Tuned Chat Models

LLaMA2 是一系列开源的基座模型以及对话模型。

作者团队来自 Meta AI，一作与 LLaMA 相同。

## 摘要

LLaMA2 是一系列经过预训练和微调的大语言模型（LLMs），参数规模从 7B 到 70B 不等。本文中将微调后的 LLMs 称为 LLaMA2-Chat，针对于对话场景。

LLaMA2 在大多数 benchmark 上的表现都优于其它的开源对话模型，经过人类对其有用性和安全性的评估，很可能作为闭源模型的合适替代品。本文提供了 LLaMA2-Chat 关于微调和安全改进的详细描述。

## 1.引言

虽然目前已经有了很多优秀的开源模型，比如 BLOOM、LLaMA、Falcon 等，但是仍然无法作为那些闭源模型（ChatGPT、BARD、Claude）的合适替代品。这些闭源的工作为了更加符合人类偏好，在微调阶段花了很大功夫，极大的增强了模型的可用性和安全性。微调阶段需要大量的计算资源和人工标注数据，并且这部分资源通常不被公开，限制了语言大模型的发展。

本文开源了以下模型：

- LLaMA2：LLaMA 1的升级版本，在新的公开可用数据上进行训练得到，相比于 LLaMA1 有以下更新
  - 预训练数据增加 40%；
  - 上下文窗口大小翻倍，提升至 4K；
  - 使用分组查询注意力机制（grouped-query attention）；
  - 开源了 7B、13B 和 70B 模型；
- LLaMA2-Chat
  - 基于 LLaMA2 进行微调，得到的对话模型；
  - 开源了 7B、13B 和 70B 模型；

## 2.预训练

LLaMA2 相比于 LLaMA1 的改进在引言部分已经有了大概的说明，下表中给了一个更直观的对比：

<img width="637" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/4afaac6b-c27d-4e8e-b4a6-498139318aab">



### 2.1 预训练数据

LLaMA2 的训练语料是由公开可用的数据混合得到，其中不包括来自 Meta 产品或服务的数据，并且尽力删除了大量包含个人信息的网站数据。考虑到性能和成本之间的 trade-off，LLaMA2 共在 2 万亿的 tokens 上完成预训练，为了增加知识并减少幻觉，对最有事实性的数据进行了上采样。总结如下：

- 不包含 Meta 自己的数据（更方便大家复现）；
- 移除包含私人信息的数据；
- 对事实性数据进行上采样，以增加模型知识，减少幻觉；

### 2.2 训练细节

LLaMA2 保留了 LLaMA1 中大部分的预训练设置及模型架构，它们均使用标准的 Transformer 架构、RMSNorm 归一化、SwiGLU 激活函数、旋转位置编码。

两者之间的差异见下表：

<img width="637" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/27a7f112-cceb-441d-b688-0b105220ea2c">

**超参数**：

- AdamW 优化器， 其中 β1 = 0.9，β2 = 0.95，eps = 10−5；
- 余弦学习率衰减方式，warmup 2000 个 iter，最终学习率减小到初始学习率的 10%；
- Weight decay 0.1，并使用梯度裁剪；

LLaMA2 在以上超参数设置下，随着 tokens 处理数量的 loss 曲线变化：

<img width="637" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/3a28735c-853d-4e8f-891d-7480261bd267">

**Tokenizer**：

- 分词采用 SentencePiece 实现的 BPE 算法；
- 所有数值被分为独立的数字；
- 未知 UTF-8 表示为 bytes；
- 词表大小为 32K；

**模型评估**：

能力维度和评估数据集：

- Code（代码）：HumanEval、MBPP；
- Commonsense Reasoning（常识推理）：PIQA、SIQA、HellaSwag、WinoGrande、ARC、OpenBookQA、CommonsenseQA；
- World Knowledge：NaturalQuestions、TriviaQA；
- Reading Comprehension（阅读理解）：SQuAD、QuAC、BoolQ；
- Math（数学）：GSM8K、MATH；
- Popular Aggregated Benchmarks（学术界较流行的 bmk）：MMLU、Big Bench Hard、AGI Eval；

对比方法：

- 开源方法：LLaMA1、MosaicML Pretrained Transformer（MPT）、Falcon
- 闭源方法：GPT-3.5、GPT-4、PaLM、PaLM-2-L

效果对比见下表：

<img width="634" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/fcc674b4-aa67-41de-9a0c-ed893cad2ae1">

- 与开源模型相比，在同等模型参数量规模下，LLaMA2 所有能力都更优（除了 code 能力）；

<img width="631" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/555e1130-76f7-4349-a96d-364982c7e447">

- 与闭源模型相比，LLaMA2 比 GPT-4 和 PaLM-2-L 还有差距；

## 3.Fine-tuning

基于 LLaMA2，使用 instruction tuning 和 RLHF 技术继续进行微调，得到了 LLaMA2-Chat。

### 3.1 Supervised Fine-Tuning (SFT)

<img width="631" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/3a7ad336-6446-460b-bc51-7da4987497d4">

上图给出了一个有用性和安全性的 SFT 标注数据示例。

#### 3.1.1 数据质量很重要

- 有很多不同来源的第三方 SFT 数据集，但是这些数据很多都缺乏多样性且质量较差。而且发现，使用少量高质量的样本的效果比使用几百万质量较差的样本更好，所以 LLaMA2 的 SFT 数据是人工精标了 27540 条高质量样本；
- 人工标注的数据集质量也参差不齐，需要做好质检工作；
- 本文发现使用 SFT 模型采样出来的数据质量和人工标注的差不多，所以可以把更多精力投入到 RLHF 得数据标注上。

#### 3.1.2 训练细节

- cosine 学习率衰减方式，初始值 2 x 10e-5；
- weight decay 0.1；
- batch size 64；
- sequence length 4096；
- 将样本的 prompts 和 answers concatenate 起来（通过一个特殊的 token 连接） ，保证 sequence length 是4096；
- 计算 loss 时 mask 掉 user prompt ，只对 answer tokens 进行反向传播；
- 模型训练 2 个 epoch；

### 3.2 Reinforcement Learning with Human Feedback (RLHF)

基于人类反馈的强化学习。

RLHF 环节是继续对模型进行微调，训练一个**奖励模型**，使得模型输出与人类偏好（human preferences）对齐（align），具有指令遵循（instruction following）的能力。

#### 3.2.1 人类偏好数据收集

人类偏好数据的标注采用二元比较（binary com）的方式，这种方式可以最大化所收集数据的多样性。标注过程如下：

- Step1：令标注员先写下一个 prompt；

- Step2：采样两个模型，分别得到该 prompt 的响应（为了最大化多样性，生成响应时对模型参数以及 temperature 超参也做了调整）；

- Step3：让标注员对两个响应进行不对比评估，包括明显更好、更好、稍微更好、仅好一点/不确定。

  > 这一步就有一个人类偏好的感觉了：给定两个答案，选择一个更加偏好的。

**人类偏好数据的采集注重有用性和安全性**，有用性指的是 Llama 2-Chat 的回应是否满足用户的请求；安全性指的是 Llama 2-Chat 的回应是否安全，例如，“给出制作炸弹的详细指示”可以被认为是有用的，但其结果是不安全的，所以这两方面缺一不可。

除了基于以上的标注准则之外，在安全性数据采集阶段，还额外标注了一个安全标签（safety label），分为以下三个类别：1）其中更好的答案是安全的，而另一个是不安全的；2）两个答案都是安全的；3）两个答案都是不安全的。经过统计，三种类别的占比分别为 18%，47% 和 35%。需要注意的时，样本中不包含“被选择的答案是不安全的，另一个是安全的”（这里论文里写的有些不好理解，其实被选择的答案就是前面提到的被评估为更好的答案），因为我们通常认为不存在更好的答案反而不安全的情况。

<img width="675" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/00b7bb97-488f-44fe-80df-3cb84415936e">

LLaMA2 一共标注了 140W 条偏序对，这个数据量比大多数开源数据集都多。

#### 3.2.2 奖励模型

奖励模型以模型的 response 和 prompt（包括前几轮的上下文）作为输入，输出一个分数表示模型的生成质量（有用性或安全性）。利用这种 response score 作为奖励，可以在 RLHF 过程中优化 LLaMA2-Chat，以实现更好的人类偏好对齐、提高有用性和安全性。

有研究表明，**使用单个奖励模型同时优化有用性和安全性是比较困难的，所以LLaMA2-Chat中训练了两个单独的奖励模型，分别优化两者**。

**3.2.2.1 训练目标**

上面有提到，LLaMA2 的 RLHF 阶段训练数据也是采用的二偏序对，即 A>B 的方式标注形式，不过做了一些改进。传统的「二偏序对」标注只标注出「答案A」比「答案B」好，但不知道「号多少」，但**「好多少」这个信息却是 Reward Model 训练中的一个关键信息**。

InstructGPT 中训练 RM 所使用的 ranking loss 如下：

<img width="300" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/b08a346c-09a7-4235-96cd-67a99ce6a37a">

该函数的目标是：尽可能的「最大化」这个「好答案得分」和「坏答案得分」差值。但不会管「好答案」比「坏答案」好多少，而是无脑将给的两个答案之间的得分拉开，这样就会造成，如果给定 2 对不同的偏序：

1. A>>A'：A（好答案）远远好于 A'（坏答案）
2. B>B'：B（好答案）略微好于 B'（坏答案）

我们其实是希望：**B 和 B' 之间的得分差要比 A 和 A‘ 之间的得分差要小一些，但使用传统「一视同仁」的方法是无法做到这一点的**。

为了解决上面的问题，LLaMA2 在进行数据标注时，不止标注了 A 好于 B，还将「好多少」分成了 4 个等级：

1. 显著好于（significantly better）
2. 好于（better）
3. 略微好于（slightly better）
4. 几乎持平（negligibly better / unsure）

对于每一个「二序便对」，标注人员都需要从 4 个选项里面选择一个，从而**表明答案之间的好坏差异程度**。

基于上面的讨论，LLaMA2 中 RLHF 阶段所使用的 loss 添加了好坏程度系数 margin：

<img width="300" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/4f05ea8d-40cb-476e-ab92-3e42c005ed6f">

该函数可以通过修改 m(r) 的值来控制偏序对的得分差异，当 A 只是略微优于 B 时，给一个较小的 margin，反之，给一个较大的 margin。

> 因为 loss 最外层是一个 -log 函数，意味着 r(a) - r(b) - m(r) 越小 loss 越大，因此差异较大的答案就需要减一个较大的 margin，从而提升 loss 值。

那么这个 margin 值怎么定呢？

LLaMA2 给出了 2 种不同的常数配置，一种差距较大，一种差距较小：

<img width="550" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/b51514f4-8057-4694-8a5b-be2a9358e930">

通过添加 margin，RM 能够更好区分「显著好于」这类的样本，而对于「差异值不大」样本提升较小：

<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/ac5368a7-5ccd-47e9-b939-2f41e47ae0e7">

**3.2.2.2 数据构成**

LLaMA2 不止使用了上述的标注数据，还使用了一些其它开源数据，提升 Reward Model 泛化性。

前面也提到，由于 LLaMA2 的目标是同时提升「有用性」和「安全性」，但这两个标准有时是存在冲突的，使用一个 Reward Model 很难同时兼顾这两点，因此，**LLaMA2 训练了 2 个单独的 RM，分别用于「有用性」打分和「安全性」打分**。其中：

- **有用性数据集**（Helpfulness）：有用性数据集（占50%）+ 安全性数据集（占 50%，一半来自于 Meta Safety，另一半来自开源数据集）；
- **安全性数据集**（Safety）：安全性数据集（占90%，由 Meta Safety 和 Anthropic Harmless 共同组成）+ 有用性数据（占 10%，由 Meta Helpfuness 和开源数据集组成）。

由此可以看出以下两点：

1. 训练「有用性」时，掺入一半的「安全性」数据有利于模型学习；
2. 训练「安全性」时，应该以「安全性数据为主」，混入少量的「有用性」数据（约10%）能够有利于模型在 2 个答案都安全的情况下做出正确选择；

**3.2.2.3 训练细节**

超参数：

| 超参数      | 值                           |
| ----------- | ---------------------------- |
| train epoch | 1 epoch                      |
| max_lr      | 5e-6（70B）、1e-5（7B，13B） |
| lr schedule | cosine, down to 10%          |
| warm up     | 3% of total steps            |
| batch size  | 512 pairs（1024 rows）       |

训练结果：

<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/1df64376-e62c-4c7d-9ae6-48398396a8ca">

由结果可以看出：

- 有用性 RM 在有用性数据测试上的 Acc 更高，安全性 RM 在安全性数据集上 Acc 更高；
- RM 对显著差异的样本的 Acc 明显高于不显著差异的样本（上图下半部分，差距可高达40%，94.3 -> 55.3）；

**3.2.2.4 模型趋势**

对于规模大小而言，论文中提到：**使用更大的「模型规模」+ 「数据规模」能够让模型有更高的acc**：

<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/fde4b621-62e0-4443-816c-6b02966b3e98">

**尽管 Llama 已经使用了百万级别的数据量训练，但 RM 似乎还没有达到饱和状态**，这意味着 meta 可能会继续增加标注数据量。

#### 3.2.3 迭代微调

LLaMA2 用了两种主要算法进行 RLHF 微调：

- Proximal Policy Optimization (PPO)
- Rejection Sampling fine-tuning

> 在 RLHF v4 版本之前，LLaMA2 只使用了「Rejection Sampling」微调，之后采取了两者顺序组合的方式，在再次 Sampling 之前，先使用 PPO 训练一段时间，再开始 Samling。

**3.2.3.1 Reject Sampling**

拒绝采样（Reject Sampling）是指让一个模型针对同一个 prompt 生成 K 个答案，然后利用 RM 为这 K 个答案打分，选出最优的答案，再去对原本的模型进行 SFT，以增强模型能力。

值的关注的有两点：

1. LLaMA2 只在 70B 的模型上做了 Reject Sampling，小模型（7B、13B）则是使用 70B 模型的生成结果进行蒸馏的。这么做可能是因为，越大的模型更容易产生好的候选回答，且多样性也更丰富些。
2. 由于 Reject Sampling 是多个阶段的，即模型一次又一次更新的，我们在选取最优样本的时候应该搜集每个阶段的最优样本，而不是仅仅只是最近一次迭代模型产生的最优样本。

<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/26cad9ed-bff7-4697-a4f9-d60b5df7a623">

上图是使用「最优分数样本」和「分数居中样本」训练模型的 reward 变化趋势图。可以看到，使用「最优样本」训练模型能够帮助模型达到更好的上限，而使用「居中分数样本」则提升并不大。

最优样本是从候选样本里通过 RM 获得的，那么候选样本是如何选取的呢？

主要方法还是通过调整 temperature 来实现生成不同的样本，不同的 temperature 对训练的结果也有一定影响：

<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/cb67af6f-37c6-4eda-8156-d53d77f8cacb">

从上图可以看到：

1. Temperature 过小（生成答案过于固定，探索率过小）会导致最终上限越低（蓝色线条）。这不难理解，因为限制了模型的探索，那么发现优秀样本的机会就小了；
2. Temperature 过大（生成答案过于随机，探索率过大）也会导致模型的上限略低（黄色线条）。因为，过大的 temperature 会让模型忽略已经学到的概率分布，生成一些很不合理的答案。

因此，选择一个适中的 temperature 对整个采样训练来讲是非常重要的，设置为 0.8~0.9 比较合适。

**3.2.3.2 PPO**

给 PPO 训练的 reward 如下：

<img width="300" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/cb222c1d-dd0f-4581-b96e-9da77c2e6ba6">

大体来讲是在 reward 上减掉了和 reference model 之间的 KL，用于稳定模型训练，这和主流方法一致。

结合上面提到的两个 Reward Model，这里的 reward 计算方式：

<img width="400" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/3be33b14-887d-4f76-a8ba-a8d1d0c94887">

从公式可以看出，最终的 reward 是从 2 个 reward model 中选择某一个得到的，具体来说，当「安全性 RM」过低时，使用「安全性RM」的得分用于惩罚模型，除此之外均使用「有用性RM」的得分作为答案的打分。

可以理解为：在优先保证模型不产生「有害答案」的前提下，尽可能提升模型「有用答案」的生成能力。

训练超参数：

| 超参数            | 值                                  |
| ----------------- | ----------------------------------- |
| weight decay      | 0.1                                 |
| gradient clipping | 1.0                                 |
| lr                | 1e-06（constant）                   |
| clip ratio（PPO） | 0.2                                 |
| batch size        | 512                                 |
| mini batch size   | 64                                  |
| KL penalty        | 1e-02（7B、13B），5e-03（34B、70B） |

**3.2.3.3 训练结果**

为了验证效果，论文将不同版本的 LLaMA2 和 ChatGPT 之间做对比，计算获胜率（win-rate）：

<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/2094c37c-ea47-43c1-b1e2-25b731a3ff6a">

可以看出，在使用 PPO之前（v4 之前），Reject Sampling 的上限大概在 50% 左右，而加入 PPO （v5）之后，模型的上限提升到了 60%。从这里可以看出，PPO 对整个模型的上限有着很重要的作用。

除了模型评估之外，论文中还列举了和多个其他模型之间的人工评价结果，基本在各个任务上都是完胜：

<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/fe97ea34-e344-4e19-8eaf-5206889340d6">

在论文的第 4 章和第 5 章还有一些对 RLHF 的讨论，包括：安全、蒸馏、Reward 分布变化的细节等等在此不做展开。



Ref：

- https://zhuanlan.zhihu.com/p/660058778
