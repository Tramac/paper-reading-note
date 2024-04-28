[paper](https://arxiv.org/pdf/2310.04378.pdf) | [code](https://github.com/luosiallen/latent-consistency-model)

## 题目&作者
latent 一致性模型，在只有几步迭代的情况下快速生成高分辨率的图像。
作者团队来自清华大学，一作[主页](https://luosiallen.github.io/)。

## 摘要
Latent Diffusion models（LDMs）在合成高分辨率图像方面取得了显著的成果，但是其迭代采样过程计算量大，导致生成速度慢。为了克服这个缺点，本文提出了 Latent Consistency Models（LCMs），可以在任何 pre-trained LDMs 上以最少的步骤实现快速推理生成，基于 pre-trained LDMs 蒸馏训练一个 LCMs 模型，在 768x768 分辨率仅需 32 个 A100 GPU 时。另外，本文还提出了一种 Latent Consistency Fine-tuning（LCF），一种专为在定制图像数据集上微调 LCM 的方法。

## 1 引言
Diffusion models 作为一类强大的生成模型，在多个领域取得了显著成果，特别是在高分辨率文本到图像的合成任务中表现出色。但 Diffusion models 有一个明显的缺点：由于迭代采样过程导致生成速度慢，-   限制了它们在实时应用中的可行性。

为了提高采样速度，研究者们提出了多种方法，其中最受关注的一个工作的是 Consistency Models。然而，Consistency models 仅限于像素空间图像生成任务，使其不适合合成高分辨率图像。 此外，尚未探索 conditional diffusion model 的应用和 classifier-free guidance 的结合，使得该方法不适合 text-to-image 任务。

类比于 Latent Diffusion Models 和 Diffusion Models 的关系，本文提出的 Latent Consistency Models 也是将 Consistency Models 在像素空间的操作通过一个 auto-encoder 转移至 Latent 空间。

本文主要贡献：
- 提出 Latent Consistency Models（LCMs），实现快速高分辨图像生成。LCMs 将 consistency models 应用于 latent 空间，可以在 pre-trained LDMs 上实现几步甚至一步生成；
- 提出一个简单高效的 guided consistency distillation 方法，实现从 Stable Diffusion 的蒸馏训练。同时提出了 skipping-step 策略加速收敛；
- 提出了一种针对 LCMs 的 fine-tuning 方法，使预训练的 LCMs 能够有效地适应定制数据集，同时保留快速推理的能力。

## 2 相关工作
主要是简述了 Diffusion Models、Accelerating DMs、Latent Diffusion Models 以及 Consistency Models 等工作，在此就不展开了。

## 3 题要
本节主要简单回顾了 Diffusion Models 和 Consistency Models 的定义。
